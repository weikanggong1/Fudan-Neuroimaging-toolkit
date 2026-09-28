"""Prepare the native sulcal metrics and rotated spheres used by MSMSulc."""

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np

from .surface_prepare import _write_gifti
from .surface import SurfaceHemisphere


@dataclass(frozen=True)
class MSMSulcInputs:
    native_sphere: Path
    rotated_sphere: Path
    native_sulc: Path
    reference_sphere: Path
    reference_sulc: Path
    affine: Path


def prepare_msmsulc_inputs(
    subject_dir: str | Path,
    initial_spheres: tuple[str | Path, str | Path],
    hcp_assets_dir: str | Path,
    output_dir: str | Path,
    *,
    wb_command: str | Path = "wb_command",
) -> dict[str, MSMSulcInputs]:
    """Prepare HCP MSMSulc inputs for both hemispheres without running MSM.

    ``initial_spheres`` are the L/R native-topology FS-to-fsLR spheres from
    ``prepare_fs_sphere_projection_inputs``. FreeSurfer sulcal depth is
    negated to match HCP's native GIFTI convention. Workbench performs the
    same spherical affine regression and radius normalization as HCP v4.7.0.
    """
    subject = Path(subject_dir).expanduser().resolve()
    assets = Path(hcp_assets_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    executable = shutil.which(str(wb_command))
    if executable is None:
        raise FileNotFoundError(f"Connectome Workbench not found: {wb_command}")
    if len(initial_spheres) != 2:
        raise ValueError("initial_spheres must contain left and right paths")
    output.mkdir(parents=True, exist_ok=True)
    prepared = {}
    for hemi, fs_hemi, initial in zip(("L", "R"), ("lh", "rh"), initial_spheres):
        sphere_file = subject / "surf" / f"{fs_hemi}.sphere"
        sulc_file = subject / "surf" / f"{fs_hemi}.sulc"
        reference_sphere = assets / "global/templates/standard_mesh_atlases" / (
            f"fsaverage.{hemi}_LR.spherical_std.164k_fs_LR.surf.gii"
        )
        reference_sulc = assets / "global/templates/standard_mesh_atlases" / (
            f"{hemi}.refsulc.164k_fs_LR.shape.gii"
        )
        for path in (sphere_file, sulc_file, reference_sphere, reference_sulc, Path(initial)):
            if not path.is_file():
                raise FileNotFoundError(path)
        vertices, faces = fsio.read_geometry(str(sphere_file))
        initial_image = nib.load(str(initial))
        initial_vertices = next(
            np.asarray(array.data) for array in initial_image.darrays if array.intent == 1008
        )
        initial_faces = next(
            np.asarray(array.data) for array in initial_image.darrays if array.intent == 1009
        )
        sulc = np.asarray(fsio.read_morph_data(str(sulc_file)), dtype=np.float32)
        if (len(vertices) != len(initial_vertices) or not np.array_equal(faces, initial_faces)
                or sulc.shape != (len(vertices),) or not np.isfinite(sulc).all()):
            raise ValueError(f"{fs_hemi} sphere, initial registration, and sulc differ in topology")
        native_sphere = output / f"{hemi}.sphere.native.surf.gii"
        native_sulc = output / f"{hemi}.sulc.native.shape.gii"
        affine = output / f"{hemi}.sphere_rot.mat"
        unscaled = output / f"{hemi}.sphere_rot.unscaled.surf.gii"
        rotated = output / f"{hemi}.sphere_rot.surf.gii"
        _write_gifti(native_sphere, vertices, faces, fs_hemi)
        structure = "CortexLeft" if hemi == "L" else "CortexRight"
        nib.save(nib.GiftiImage(
            darrays=[nib.gifti.GiftiDataArray(-sulc, intent="NIFTI_INTENT_SHAPE")],
            meta=nib.gifti.GiftiMetaData({"AnatomicalStructurePrimary": structure}),
        ), str(native_sulc))
        for args in (
            ("-surface-affine-regression", native_sphere, initial, affine),
            ("-surface-apply-affine", native_sphere, affine, unscaled),
            ("-surface-modify-sphere", unscaled, 100, rotated),
        ):
            subprocess.run([executable, *map(str, args)], check=True, capture_output=True, text=True)
        prepared[hemi] = MSMSulcInputs(
            native_sphere, rotated, native_sulc, reference_sphere, reference_sulc, affine
        )
    return prepared


def apply_dedrift(
    registered_spheres: tuple[str | Path, str | Path],
    dedrift_spheres: tuple[str | Path, str | Path],
    hcp_assets_dir: str | Path,
    output_dir: str | Path,
    *,
    wb_command: str | Path = "wb_command",
) -> dict[str, Path]:
    """Compose native MSMAll spheres with explicit 164k group DeDrift spheres.

    UKB and HCP use different group transforms. The caller must provide the
    intended L/R files; this function never substitutes one for the other.
    The composition follows ``bb_dedrift_resample`` using Workbench.
    """
    if len(registered_spheres) != 2 or len(dedrift_spheres) != 2:
        raise ValueError("registered_spheres and dedrift_spheres need L/R paths")
    executable = shutil.which(str(wb_command))
    if executable is None:
        raise FileNotFoundError(f"Connectome Workbench not found: {wb_command}")
    atlas = Path(hcp_assets_dir).expanduser().resolve() / "global/templates/standard_mesh_atlases"
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = {}
    for hemi, registered, dedrift in zip(("L", "R"), registered_spheres, dedrift_spheres):
        registered = Path(registered).expanduser().resolve()
        dedrift = Path(dedrift).expanduser().resolve()
        reference = atlas / f"fsaverage.{hemi}_LR.spherical_std.164k_fs_LR.surf.gii"
        for path in (registered, dedrift, reference):
            if not path.is_file():
                raise FileNotFoundError(path)
        subject_image = nib.load(str(registered))
        drift_image = nib.load(str(dedrift))
        reference_image = nib.load(str(reference))
        if (len(drift_image.darrays[0].data) != len(reference_image.darrays[0].data)
                or not np.array_equal(drift_image.darrays[1].data,
                                      reference_image.darrays[1].data)
                or not len(subject_image.darrays[0].data)):
            raise ValueError(f"{hemi} DeDrift and reference spheres differ in topology")
        destination = output / f"{hemi}.sphere.MSMAll_DeDrift.native.surf.gii"
        subprocess.run([
            executable, "-surface-sphere-project-unproject", str(registered),
            str(reference), str(dedrift), str(destination),
        ], check=True, capture_output=True, text=True)
        result[hemi] = destination
    return result


def prepare_registered_projection(
    hemisphere: SurfaceHemisphere,
    individual_roi: str | Path,
    registered_sphere: str | Path,
    reference_sphere_164k: str | Path,
    reference_roi_164k: str | Path,
    output_dir: str | Path,
    *,
    wb_command: str | Path = "wb_command",
) -> SurfaceHemisphere:
    """Rebuild native ROI and 32k midthickness for a new registration sphere."""
    executable = shutil.which(str(wb_command))
    if executable is None:
        raise FileNotFoundError(f"Connectome Workbench not found: {wb_command}")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = [Path(path).expanduser().resolve() for path in (
        hemisphere.white, hemisphere.pial, hemisphere.midthickness,
        hemisphere.atlas_sphere, hemisphere.atlas_roi, individual_roi,
        registered_sphere, reference_sphere_164k, reference_roi_164k,
    )]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    nvertices = len(nib.load(str(paths[2])).darrays[0].data)
    if (len(nib.load(str(paths[5])).darrays[0].data) != nvertices
            or len(nib.load(str(paths[6])).darrays[0].data) != nvertices):
        raise ValueError("registered sphere and individual ROI must match native surface")
    projected = output / "atlasroi.native_projected.shape.gii"
    roi = output / "roi.native.shape.gii"
    atlas_mid = output / "midthickness.32k_fsLR.surf.gii"
    commands = (
        ("-metric-resample", paths[8], paths[7], paths[6], "BARYCENTRIC",
         projected, "-largest"),
        ("-metric-math", "(atlas + individual) > 0", roi, "-var", "atlas",
         projected, "-var", "individual", paths[5]),
        ("-surface-resample", paths[2], paths[6], paths[3], "BARYCENTRIC", atlas_mid),
    )
    for args in commands:
        subprocess.run([executable, *map(str, args)], check=True,
                       capture_output=True, text=True)
    return SurfaceHemisphere(
        white=paths[0], pial=paths[1], midthickness=paths[2],
        registered_sphere=paths[6], native_roi=roi,
        atlas_sphere=paths[3], atlas_midthickness=atlas_mid,
        atlas_roi=paths[4],
    )
