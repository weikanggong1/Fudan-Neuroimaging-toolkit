"""Connect cleaned volume BOLD to fsLR32k projection."""

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import shutil
import subprocess
import zipfile

import nibabel as nib
import numpy as np

from ..flirt.coordinates import flirt_to_world_affine
from .surface import SurfaceHemisphere, SurfaceProjectionResult, run_surface_projection
from .surface_prepare import prepare_fmriprep_surface_inputs
from .surface_qc import SurfaceQCResult, make_ribbon_goodvoxels
from .surface_registration import prepare_msmsulc_inputs
from .surface_msmsulc import run_msmsulc
from .surface_fmriprep import run_fmriprep_surface_projection
from .normalization import resample_world


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
    registration: str = "msmsulc",
    registered_spheres: tuple[str | Path, str | Path] | None = None,
    goodvoxels: str | Path | None = None,
) -> SurfacePipelineResult:
    """Map confound-cleaned BOLD to fsLR32k with fMRIPrep's projection order.

    Native EPI BOLD is interpolated once to T1w for cortical sampling. The
    existing 2-mm MNI152NLin6Asym BOLD supplies subcortical CIFTI voxels.
    ``registered_spheres`` may supply external L/R MSMSulc spheres for an
    exact fixed-sphere comparison; otherwise FNIT estimates them with Torch.
    """
    if registration not in ("msmsulc", "fs"):
        raise ValueError("registration must be 'msmsulc' or 'fs'")
    if registered_spheres is not None and len(registered_spheres) != 2:
        raise ValueError("registered_spheres must contain left and right paths")
    volume = Path(volume_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    assets = Path(hcp_assets_dir).expanduser().resolve()
    clean_mni = volume / "filtered_func_data_clean_MNI152_2mm.nii.gz"
    t1 = volume / "T1_brain.nii.gz"
    report_file = volume / "pipeline_report.json"
    bbr_file = volume / "reg/example_func2highres.mat"
    dseg = assets / "fmriprep/tpl-MNI152NLin6Asym_res-02_atlas-HCP_dseg.nii.gz"
    for path in (clean_mni, t1, report_file, bbr_file, dseg):
        if not path.is_file():
            raise FileNotFoundError(path)
    report = json.loads(report_file.read_text(encoding="utf-8"))
    if not any(report.get("wm_csf_motion_regression", {}).get(key, False)
               for key in ("wm", "csf", "motion")):
        raise ValueError("volume output lacks completed WM/CSF/motion regression")
    native = volume / report["outputs"]["aroma_clean_native"]
    if not native.is_file():
        raise FileNotFoundError(native)
    epi_image = nib.load(str(native))
    t1_image = nib.load(str(t1))
    mni_image = nib.load(str(clean_mni))
    label_image = nib.load(str(dseg))
    canonical_mni = nib.as_closest_canonical(mni_image)
    canonical_label = nib.as_closest_canonical(label_image)
    if (epi_image.ndim != 4 or mni_image.ndim != 4 or
            epi_image.shape[3] != mni_image.shape[3] or
            canonical_mni.shape[:3] != canonical_label.shape or
            not np.allclose(canonical_mni.affine, canonical_label.affine,
                            rtol=0, atol=1e-4)):
        raise ValueError("volume BOLD and TemplateFlow HCP 2-mm atlas are incompatible")
    epi_to_t1 = flirt_to_world_affine(
        np.loadtxt(bbr_file), epi_image.affine, t1_image.affine,
        epi_image.shape[:3], t1_image.shape,
        epi_image.header.get_zooms()[:3], t1_image.header.get_zooms()[:3],
    )
    dtseries = output / "projection/space-fsLR_den-91k_bold.dtseries.nii"
    if dtseries.exists() and not overwrite:
        raise FileExistsError(dtseries)
    output.mkdir(parents=True, exist_ok=True)
    t1_bold = output / "clean_T1w.nii.gz"
    if not t1_bold.exists() or overwrite:
        resample_world(native, t1, np.linalg.inv(epi_to_t1), t1_bold,
                       device=device)
    source = Path(recon_all).expanduser().resolve()
    with TemporaryDirectory(prefix="recon_all_", dir=output) as temporary:
        if source.is_file() and source.suffix.lower() == ".zip":
            subject = Path(temporary) / "FreeSurfer"
            required = ("mri/orig.mgz", "mri/orig/001.mgz") + tuple(
                f"surf/{hemi}.{name}"
                for hemi in ("lh", "rh")
                for name in (("white", "pial", "sphere.reg", "thickness", "sphere", "sulc")
                             if registration == "msmsulc" and registered_spheres is None
                             else ("white", "pial", "sphere.reg", "thickness"))
            )
            with zipfile.ZipFile(source) as archive:
                for name in required:
                    target = subject / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(f"FreeSurfer/{name}") as reader, target.open("wb") as writer:
                        shutil.copyfileobj(reader, writer)
        else:
            subject = source / "FreeSurfer" if (source / "FreeSurfer").is_dir() else source
        scanner = nib.load(str(subject / "mri/orig/001.mgz"))
        if scanner.shape != t1_image.shape or not np.allclose(
                scanner.affine, t1_image.affine, atol=1e-4, rtol=0):
            raise ValueError("recon-all scanner T1 and volume-pipeline T1 have different grids")
        prepared = prepare_fmriprep_surface_inputs(
            subject_dir=subject, hcp_assets_dir=assets,
            output_dir=output / "prepared", wb_command=wb_command,
            overwrite=overwrite,
        )
        if registration == "msmsulc" and registered_spheres is None:
            sulc_inputs = prepare_msmsulc_inputs(
                subject_dir=subject,
                initial_spheres=prepared.initial_spheres,
                hcp_assets_dir=assets, output_dir=output / "msmsulc_inputs",
                wb_command=wb_command,
            )
            spheres = run_msmsulc(sulc_inputs, output / "msmsulc", device=device)
            registered_spheres = (spheres["L"], spheres["R"])
        if registered_spheres is None:
            registered_spheres = prepared.initial_spheres
        mesh = assets / "global/templates/standard_mesh_atlases"
        executable = shutil.which(str(wb_command))
        if executable is None:
            raise FileNotFoundError(wb_command)
        hemispheres = {}
        for hemi, geometry, individual_roi, sphere in zip(
            ("L", "R"), (prepared.geometry.left, prepared.geometry.right),
            prepared.individual_rois, registered_spheres,
        ):
            atlas_mid = output / "registered" / hemi / "midthickness.32k_fsLR.surf.gii"
            atlas_mid.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([
                executable, "-surface-resample", str(geometry.midthickness),
                str(sphere), str(mesh / f"{hemi}.sphere.32k_fs_LR.surf.gii"),
                "BARYCENTRIC", str(atlas_mid),
            ], check=True, capture_output=True, text=True)
            hemispheres[hemi] = SurfaceHemisphere(
                white=geometry.white, pial=geometry.pial,
                midthickness=geometry.midthickness,
                registered_sphere=sphere,
                native_roi=individual_roi,
                atlas_sphere=mesh / f"{hemi}.sphere.32k_fs_LR.surf.gii",
                atlas_midthickness=atlas_mid,
                atlas_roi=mesh / f"{hemi}.atlasroi.32k_fs_LR.shape.gii",
            )
        projection = run_fmriprep_surface_projection(
            clean_t1w=t1_bold, clean_mni=clean_mni,
            left=hemispheres["L"], right=hemispheres["R"],
            left_label=mesh / "L.atlasroi.32k_fs_LR.shape.gii",
            right_label=mesh / "R.atlasroi.32k_fs_LR.shape.gii",
            hcp_dseg=dseg, output_dir=output / "projection",
            goodvoxels=goodvoxels, wb_command=wb_command,
            overwrite=overwrite,
        )
    return SurfacePipelineResult(projection, None)
