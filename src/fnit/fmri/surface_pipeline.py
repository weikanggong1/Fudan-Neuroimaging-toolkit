"""Connect the MNI volume pipeline to UKB-style fsLR32k projection."""

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import shutil
import zipfile

import nibabel as nib
import numpy as np

from ..flirt.coordinates import flirt_to_world_affine
from .surface import SurfaceHemisphere, SurfaceProjectionResult, run_surface_projection
from .surface_prepare import prepare_fs_sphere_projection_inputs
from .surface_qc import SurfaceQCResult, make_ribbon_goodvoxels
from .surface_registration import apply_dedrift, prepare_registered_projection


@dataclass(frozen=True)
class SurfacePipelineInputs:
    """Subject MNI surfaces and ROI images prepared by a structural pipeline."""

    left: SurfaceHemisphere
    right: SurfaceHemisphere
    subject_rois: str | Path
    atlas_rois: str | Path
    wb_command: str | Path = "wb_command"
    goodvoxels: str | Path | None = None


@dataclass(frozen=True)
class SurfacePipelineResult:
    projection: SurfaceProjectionResult
    qc: SurfaceQCResult | None


def run_surface_from_mni(
    clean_mni: str | Path,
    mni_reference: str | Path,
    inputs: SurfacePipelineInputs,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> SurfacePipelineResult:
    """Project an existing cleaned MNI BOLD with registered subject surfaces.

    When ``goodvoxels`` is absent, estimate it from the white/pial ribbon and
    cleaned fMRI. A supplied mask permits a fixed-input comparison. Sphere
    registration and subject/atlas ROI creation remain upstream inputs.
    """
    output = Path(output_dir).expanduser().resolve()
    qc = None
    goodvoxels = inputs.goodvoxels
    if goodvoxels is None:
        qc = make_ribbon_goodvoxels(
            clean_bold=clean_mni,
            reference=mni_reference,
            left_white=inputs.left.white,
            left_pial=inputs.left.pial,
            right_white=inputs.right.white,
            right_pial=inputs.right.pial,
            output_dir=output / "qc",
            wb_command=inputs.wb_command,
        )
        goodvoxels = qc.goodvoxels
    projection = run_surface_projection(
        clean_mni=clean_mni,
        mni_reference=mni_reference,
        goodvoxels=goodvoxels,
        subject_rois=inputs.subject_rois,
        atlas_rois=inputs.atlas_rois,
        left=inputs.left,
        right=inputs.right,
        output_dir=output / "projection",
        wb_command=inputs.wb_command,
        overwrite=overwrite,
    )
    return SurfacePipelineResult(projection=projection, qc=qc)


def run_surface_from_volume(
    volume_dir: str | Path,
    recon_all: str | Path,
    hcp_assets_dir: str | Path,
    output_dir: str | Path,
    *,
    wb_command: str | Path = "wb_command",
    device: str = "cpu",
    overwrite: bool = False,
    registered_spheres: tuple[str | Path, str | Path] | None = None,
    dedrift_spheres: tuple[str | Path, str | Path] | None = None,
) -> SurfacePipelineResult:
    """Map confound-cleaned MNI BOLD with FS or supplied L/R registration spheres.

    ``registered_spheres`` may contain independently estimated MSMSulc or
    MSMAll native-topology spheres. ``dedrift_spheres`` must contain the
    corresponding L/R 164k group transforms and requires registered spheres.
    """
    if dedrift_spheres is not None and registered_spheres is None:
        raise ValueError("dedrift_spheres requires registered_spheres")
    if registered_spheres is not None and len(registered_spheres) != 2:
        raise ValueError("registered_spheres must contain left and right paths")
    volume = Path(volume_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    clean = volume / "filtered_func_data_clean_MNI152_2mm.nii.gz"
    report_path = volume / "pipeline_report.json"
    t1_path = volume / "T1_brain.nii.gz"
    reference_path = volume / "MNI152_2mm_brain.nii.gz"
    pull_path = volume / "reg/MNI152_2mm_to_T1_pull_ras.nii.gz"
    affine_path = volume / "reg/T1_to_MNI152_2mm_affine.mat"
    for path in (clean, report_path, t1_path, reference_path, pull_path, affine_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("outputs", {}).get("clean_mni") != clean.name or not any(
        report.get("wm_csf_motion_regression", {}).get(name, False)
        for name in ("wm", "csf", "motion")
    ):
        raise ValueError("volume output must contain a completed WM, CSF, or motion regression")
    t1 = nib.load(str(t1_path))
    reference = nib.load(str(reference_path))
    bold = nib.load(str(clean))
    if bold.ndim != 4 or bold.shape[:3] != reference.shape or not np.allclose(
        bold.affine, reference.affine, rtol=0, atol=1e-4
    ):
        raise ValueError("clean MNI BOLD must match the saved MNI reference grid")
    forward = flirt_to_world_affine(
        np.loadtxt(affine_path), t1.affine, reference.affine,
        t1.shape[:3], reference.shape,
        t1.header.get_zooms()[:3], reference.header.get_zooms()[:3],
    )
    output.mkdir(parents=True, exist_ok=True)
    source = Path(recon_all).expanduser().resolve()
    with TemporaryDirectory(prefix="recon_all_", dir=output) as temporary:
        if source.is_file() and source.suffix.lower() == ".zip":
            subject = Path(temporary) / "FreeSurfer"
            required = ("mri/orig.mgz", "mri/orig/001.mgz", "mri/wmparc.mgz") + tuple(
                f"surf/{hemi}.{name}"
                for hemi in ("lh", "rh")
                for name in ("white", "pial", "sphere.reg", "thickness")
            )
            with zipfile.ZipFile(source) as archive:
                for name in required:
                    member = f"FreeSurfer/{name}"
                    target = subject / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as reader, target.open("wb") as writer:
                        shutil.copyfileobj(reader, writer)
        else:
            subject = source / "FreeSurfer" if (source / "FreeSurfer").is_dir() else source
        scanner = nib.load(str(subject / "mri/orig/001.mgz"))
        if scanner.shape != t1.shape or not np.allclose(
            scanner.affine, t1.affine, rtol=0, atol=1e-4
        ):
            raise ValueError("recon-all scanner T1 and volume-pipeline T1 have different grids")
        prepared = prepare_fs_sphere_projection_inputs(
            subject_dir=subject,
            pull_ras=pull_path,
            initial_t1_to_mni_world=forward,
            mni_reference=reference_path,
            hcp_assets_dir=hcp_assets_dir,
            output_dir=output / "prepared",
            wb_command=wb_command,
            device=device,
            overwrite=overwrite,
        )
        hemispheres = {"L": prepared.left, "R": prepared.right}
        if registered_spheres is not None:
            spheres = dict(zip(("L", "R"), registered_spheres))
            if dedrift_spheres is not None:
                spheres = apply_dedrift(
                    registered_spheres=registered_spheres,
                    dedrift_spheres=dedrift_spheres,
                    hcp_assets_dir=hcp_assets_dir,
                    output_dir=output / "dedrift",
                    wb_command=wb_command,
                )
            mesh = Path(hcp_assets_dir).expanduser().resolve() / (
                "global/templates/standard_mesh_atlases"
            )
            for hemi in ("L", "R"):
                hemispheres[hemi] = prepare_registered_projection(
                    hemisphere=hemispheres[hemi],
                    individual_roi=output / "prepared" / f"{hemi}.roi.individual.native.shape.gii",
                    registered_sphere=spheres[hemi],
                    reference_sphere_164k=mesh / (
                        f"fsaverage.{hemi}_LR.spherical_std.164k_fs_LR.surf.gii"
                    ),
                    reference_roi_164k=mesh / f"{hemi}.atlasroi.164k_fs_LR.shape.gii",
                    output_dir=output / "registered" / hemi,
                    wb_command=wb_command,
                )
        return run_surface_from_mni(
            clean_mni=clean,
            mni_reference=reference_path,
            inputs=SurfacePipelineInputs(
                left=hemispheres["L"],
                right=hemispheres["R"],
                subject_rois=prepared.subject_rois,
                atlas_rois=prepared.atlas_rois,
                wb_command=wb_command,
            ),
            output_dir=output,
            overwrite=overwrite,
        )
