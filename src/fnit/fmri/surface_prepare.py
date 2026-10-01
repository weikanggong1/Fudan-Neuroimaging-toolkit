"""Prepare T1w-space surfaces and FS-to-fsLR initialization from recon-all."""

from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class T1SurfaceGeometry:
    left: T1SurfacePair
    right: T1SurfacePair


def _apply_affine(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def prepare_t1w_surface_geometry(subject_dir: str | Path, output_dir: str | Path,
                                 *, overwrite: bool = False) -> T1SurfaceGeometry:
    """Convert recon-all white/pial meshes from tkRAS to scanner T1w RAS.

    The returned pairs have native FreeSurfer vertex order. This only reads
    recon-all files with nibabel; no FreeSurfer executable is invoked.
    """
    subject = Path(subject_dir).expanduser().resolve()
    orig = nib.load(str(subject / "mri/orig.mgz"))
    if not isinstance(orig, nib.MGHImage):
        raise ValueError("mri/orig.mgz must be an MGH image")
    transform = orig.affine @ np.linalg.inv(orig.header.get_vox2ras_tkr())
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = {}
    for hemi in ("lh", "rh"):
        paths = {name: output / f"{hemi}.{name}.T1w.native.surf.gii"
                 for name in ("white", "pial", "midthickness")}
        if any(path.exists() for path in paths.values()) and not overwrite:
            raise FileExistsError(paths["white"])
        white, faces = fsio.read_geometry(str(subject / "surf" / f"{hemi}.white"))
        pial, pial_faces = fsio.read_geometry(str(subject / "surf" / f"{hemi}.pial"))
        if not np.array_equal(faces, pial_faces):
            raise ValueError(f"{hemi} white and pial topology differ")
        white = _apply_affine(white, transform)
        pial = _apply_affine(pial, transform)
        for name, vertices in (("white", white), ("pial", pial),
                               ("midthickness", (white + pial) / 2)):
            _write_gifti(paths[name], vertices, faces, hemi)
        result[hemi] = T1SurfacePair(paths["white"], paths["pial"],
                                      paths["midthickness"], len(white))
    return T1SurfaceGeometry(result["lh"], result["rh"])


@dataclass(frozen=True)
class T1SurfacePreparation:
    geometry: T1SurfaceGeometry
    initial_spheres: tuple[Path, Path]
    individual_rois: tuple[Path, Path]


def prepare_fmriprep_surface_inputs(
    subject_dir: str | Path, hcp_assets_dir: str | Path, output_dir: str | Path,
    *, wb_command: str | Path = "wb_command", overwrite: bool = False,
) -> T1SurfacePreparation:
    """Prepare T1w native meshes, FS-to-fsLR spheres and cortex ROIs.

    Requires existing recon-all ``orig.mgz``, white/pial, sphere.reg and
    thickness files. Does not create MNI surfaces or resample wmparc.
    """
    import shutil
    import subprocess

    subject = Path(subject_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    mesh = Path(hcp_assets_dir).expanduser().resolve() / "global/templates/standard_mesh_atlases"
    executable = shutil.which(str(wb_command))
    if executable is None:
        raise FileNotFoundError(wb_command)
    geometry = prepare_t1w_surface_geometry(
        subject, output / "native", overwrite=overwrite,
    )

    def command(destination, *args):
        subprocess.run([executable, *map(str, args)], check=True,
                       capture_output=True, text=True)
        if not destination.is_file():
            raise RuntimeError(f"Workbench did not create {destination}")

    spheres = []
    rois = []
    for hemi, fs_hemi, pair in (("L", "lh", geometry.left),
                                ("R", "rh", geometry.right)):
        vertices, faces = fsio.read_geometry(str(subject / "surf" / f"{fs_hemi}.sphere.reg"))
        native_faces = np.asarray(nib.load(str(pair.white)).darrays[1].data)
        if len(vertices) != pair.vertex_count or not np.array_equal(faces, native_faces):
            raise ValueError(f"{fs_hemi}.sphere.reg topology differs from white/pial")
        sphere = output / f"{hemi}.sphere.FS.native.surf.gii"
        _write_gifti(sphere, vertices, faces, fs_hemi)
        average = mesh / f"fs_{hemi}/fsaverage.{hemi}.sphere.164k_fs_{hemi}.surf.gii"
        transform = mesh / (
            f"fs_{hemi}/fs_{hemi}-to-fs_LR_fsaverage.{hemi}_LR."
            f"spherical_std.164k_fs_{hemi}.surf.gii"
        )
        registered = output / f"{hemi}.sphere.FS_to_fsLR.native.surf.gii"
        command(registered, "-surface-sphere-project-unproject", sphere,
                average, transform, registered)
        thickness = np.asarray(fsio.read_morph_data(
            str(subject / "surf" / f"{fs_hemi}.thickness")), dtype=np.float32)
        if len(thickness) != pair.vertex_count or not np.isfinite(thickness).all():
            raise ValueError(f"{fs_hemi}.thickness is invalid")
        raw = output / f"{hemi}.roi.thickness.native.shape.gii"
        filled = output / f"{hemi}.roi.filled.native.shape.gii"
        individual = output / f"{hemi}.roi.individual.native.shape.gii"
        nib.save(nib.GiftiImage(darrays=[nib.gifti.GiftiDataArray(
            (np.abs(thickness) > 0).astype(np.float32), intent="NIFTI_INTENT_SHAPE"
        )]), str(raw))
        command(filled, "-metric-fill-holes", pair.midthickness, raw, filled)
        command(individual, "-metric-remove-islands", pair.midthickness,
                filled, individual)
        spheres.append(registered)
        rois.append(individual)
    return T1SurfacePreparation(geometry, tuple(spheres), tuple(rois))
