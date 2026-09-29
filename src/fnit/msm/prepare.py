"""Prepare the native sulcal metrics and rotated spheres used by MSMSulc."""

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np



def _write_gifti(path, points, faces, hemi):
    image = nib.GiftiImage(
        darrays=[
            nib.gifti.GiftiDataArray(np.asarray(points, dtype=np.float32),
                                     intent="NIFTI_INTENT_POINTSET"),
            nib.gifti.GiftiDataArray(np.asarray(faces, dtype=np.int32),
                                     intent="NIFTI_INTENT_TRIANGLE"),
        ],
        meta=nib.gifti.GiftiMetaData({
            "AnatomicalStructurePrimary": "CortexLeft" if hemi == "lh" else "CortexRight"
        }),
    )
    nib.save(image, str(path))


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
    ``prepare_fmriprep_surface_inputs``. FreeSurfer sulcal depth is
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
