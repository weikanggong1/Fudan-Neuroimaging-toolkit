"""Prepare T1w-space surfaces and FS-to-fsLR initialization from recon-all."""

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np

from ..msm.prepare import _write_gifti


@dataclass(frozen=True)
class T1SurfacePair:
    white: Path
    pial: Path
    midthickness: Path
    vertex_count: int
    midthickness_source: Path | None = None


@dataclass(frozen=True)
class T1SurfaceGeometry:
    left: T1SurfacePair
    right: T1SurfacePair


def _apply_affine(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def load_fsnative_to_t1w(value):
    """Read a forward fsnative scanner-RAS to T1w scanner-RAS affine."""
    from ..flirt.coordinates import _numpy_affine
    if value is None:
        return np.eye(4, dtype=np.float64)
    if isinstance(value, (str, Path)):
        value = np.loadtxt(value)
    return _numpy_affine(value, "fsnative_to_t1w")


def prepare_t1w_surface_geometry(subject_dir: str | Path, output_dir: str | Path,
                                 *, fsnative_to_t1w=None,
                                 overwrite: bool = False) -> T1SurfaceGeometry:
    """Convert recon-all white/pial meshes from tkRAS to scanner T1w RAS.

    The returned pairs have native FreeSurfer vertex order. This only reads
    recon-all files with nibabel; no FreeSurfer executable is invoked. Both
    hemispheres are validated before any final file is written; failed
    publication restores previous outputs.
    """
    from .surface_fmriprep import _publish_projection

    subject = Path(subject_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    destinations = {
        hemi: {name: output / f"{hemi}.{name}.T1w.native.surf.gii"
               for name in ("white", "pial", "midthickness")}
        for hemi in ("lh", "rh")
    }
    for paths in destinations.values():
        for path in paths.values():
            if path.is_dir():
                raise ValueError(f"output file is a directory: {path}")
            if (path.exists() or path.is_symlink()) and not overwrite:
                raise FileExistsError(path)
    orig = nib.load(str(subject / "mri/orig.mgz"))
    if not isinstance(orig, nib.MGHImage):
        raise ValueError("mri/orig.mgz must be an MGH image")
    transform = (load_fsnative_to_t1w(fsnative_to_t1w) @ orig.affine
                 @ np.linalg.inv(orig.header.get_vox2ras_tkr()))
    if not np.isfinite(transform).all() or abs(np.linalg.det(transform[:3, :3])) < 1e-10:
        raise ValueError("orig.mgz and fsnative_to_t1w must define an invertible finite world affine")
    result = {}
    converted = {}
    for hemi in ("lh", "rh"):
        paths = destinations[hemi]
        white, faces = fsio.read_geometry(str(subject / "surf" / f"{hemi}.white"))
        pial, pial_faces = fsio.read_geometry(str(subject / "surf" / f"{hemi}.pial"))
        if (white.ndim != 2 or white.shape[1] != 3 or not len(white)
                or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
                or not np.issubdtype(faces.dtype, np.integer)
                or faces.min() < 0 or faces.max() >= len(white)):
            raise ValueError(f"{hemi} white has invalid vertices or triangle indices")
        if not np.array_equal(faces, pial_faces):
            raise ValueError(f"{hemi} white and pial topology differ")
        mid_path = next((subject / "surf" / f"{hemi}.{name}"
                         for name in ("midthickness", "graymid")
                         if (subject / "surf" / f"{hemi}.{name}").is_file()), None)
        if mid_path is None:
            raise FileNotFoundError(
                f"{hemi}.midthickness or {hemi}.graymid is required in recon-all surf/; "
                "provide the existing sMRIPrep/FreeSurfer surface with the recon-all inputs"
            )
        mid, mid_faces = fsio.read_geometry(str(mid_path))
        if not np.array_equal(faces, mid_faces):
            raise ValueError(f"{hemi} midthickness topology differs from white/pial")
        for name, values in (("white", white), ("pial", pial), ("midthickness", mid)):
            if values.shape != white.shape or not np.isfinite(values).all():
                raise ValueError(f"{hemi} {name} has invalid vertices")
        white = _apply_affine(white, transform)
        pial = _apply_affine(pial, transform)
        mid = _apply_affine(mid, transform)
        converted[hemi] = ({"white": white, "pial": pial, "midthickness": mid}, faces)
        result[hemi] = T1SurfacePair(paths["white"], paths["pial"],
                                      paths["midthickness"], len(white), mid_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".fnit-t1-surfaces-", dir=output.parent) as directory:
        staging = Path(directory)
        names = []
        for hemi, (vertices, faces) in converted.items():
            for name, values in vertices.items():
                destination = destinations[hemi][name]
                _write_gifti(staging / destination.name, values, faces, hemi)
                names.append(destination.name)
        _publish_projection(staging, output, tuple(names), overwrite)
    return T1SurfaceGeometry(result["lh"], result["rh"])


@dataclass(frozen=True)
class T1SurfacePreparation:
    geometry: T1SurfaceGeometry
    initial_spheres: tuple[Path, Path]
    individual_rois: tuple[Path, Path]


def _preparation_names():
    return tuple(f"native/{hemi}.{name}.T1w.native.surf.gii"
                 for hemi in ("lh", "rh") for name in ("white", "pial", "midthickness")) + tuple(
        f"{hemi}.{name}" for hemi in ("L", "R") for name in (
            "sphere.FS.native.surf.gii", "sphere.FS_to_fsLR.native.surf.gii",
            "roi.thickness.native.shape.gii", "roi.filled.native.shape.gii",
            "roi.individual.native.shape.gii",
        )
    )


def prepare_fmriprep_surface_inputs(
    subject_dir: str | Path, hcp_assets_dir: str | Path, output_dir: str | Path,
    *, wb_command: str | Path = "wb_command", fsnative_to_t1w=None,
    overwrite: bool = False,
) -> T1SurfacePreparation:
    """Prepare T1w native meshes, FS-to-fsLR spheres and cortex ROIs.

    Requires existing recon-all ``orig.mgz``, white/pial, midthickness or
    graymid, sphere.reg and thickness files. All 16 generated files are
    protected by ``overwrite`` and staged together before publication.
    Does not create MNI surfaces or resample wmparc.
    """
    import shutil
    import subprocess
    from .surface_fmriprep import _publish_projection

    subject = Path(subject_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    names = _preparation_names()
    for name in names:
        path = output / name
        if path.is_dir():
            raise ValueError(f"output file is a directory: {path}")
        if (path.exists() or path.is_symlink()) and not overwrite:
            raise FileExistsError(path)
    mesh = Path(hcp_assets_dir).expanduser().resolve() / "global/templates/standard_mesh_atlases"
    executable = shutil.which(str(wb_command))
    if executable is None:
        raise FileNotFoundError(wb_command)
    def command(destination, *args):
        subprocess.run([executable, *map(str, args)], check=True,
                       capture_output=True, text=True)
        if not destination.is_file():
            raise RuntimeError(f"Workbench did not create {destination}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".fnit-surface-preparation-", dir=output.parent) as directory:
        staging = Path(directory)
        geometry = prepare_t1w_surface_geometry(
            subject, staging / "native", fsnative_to_t1w=fsnative_to_t1w,
        )
        for hemi, fs_hemi, pair in (("L", "lh", geometry.left),
                                    ("R", "rh", geometry.right)):
            vertices, faces = fsio.read_geometry(str(subject / "surf" / f"{fs_hemi}.sphere.reg"))
            native_faces = np.asarray(nib.load(str(pair.white)).darrays[1].data)
            if (vertices.shape != (pair.vertex_count, 3) or not np.isfinite(vertices).all()
                    or not np.array_equal(faces, native_faces)):
                raise ValueError(f"{fs_hemi}.sphere.reg topology differs from white/pial or has invalid vertices")
            sphere = staging / f"{hemi}.sphere.FS.native.surf.gii"
            _write_gifti(sphere, vertices, faces, fs_hemi)
            average = mesh / f"fs_{hemi}/fsaverage.{hemi}.sphere.164k_fs_{hemi}.surf.gii"
            transform = mesh / (
                f"fs_{hemi}/fs_{hemi}-to-fs_LR_fsaverage.{hemi}_LR."
                f"spherical_std.164k_fs_{hemi}.surf.gii"
            )
            registered = staging / f"{hemi}.sphere.FS_to_fsLR.native.surf.gii"
            command(registered, "-surface-sphere-project-unproject", sphere,
                    average, transform, registered)
            thickness = np.asarray(fsio.read_morph_data(
                str(subject / "surf" / f"{fs_hemi}.thickness")), dtype=np.float32)
            if len(thickness) != pair.vertex_count or not np.isfinite(thickness).all():
                raise ValueError(f"{fs_hemi}.thickness is invalid")
            raw = staging / f"{hemi}.roi.thickness.native.shape.gii"
            filled = staging / f"{hemi}.roi.filled.native.shape.gii"
            individual = staging / f"{hemi}.roi.individual.native.shape.gii"
            nib.save(nib.GiftiImage(darrays=[nib.gifti.GiftiDataArray(
                (np.abs(thickness) > 0).astype(np.float32), intent="NIFTI_INTENT_SHAPE"
            )]), str(raw))
            command(filled, "-metric-fill-holes", pair.midthickness, raw, filled)
            command(individual, "-metric-remove-islands", pair.midthickness,
                    filled, individual)
        _publish_projection(staging, output, names, overwrite)

    def published_pair(pair):
        return T1SurfacePair(*(output / "native" / path.name for path in (
            pair.white, pair.pial, pair.midthickness)), pair.vertex_count, pair.midthickness_source)

    published = T1SurfaceGeometry(published_pair(geometry.left), published_pair(geometry.right))
    return T1SurfacePreparation(
        published,
        tuple(output / f"{hemi}.sphere.FS_to_fsLR.native.surf.gii" for hemi in ("L", "R")),
        tuple(output / f"{hemi}.roi.individual.native.shape.gii" for hemi in ("L", "R")),
    )
