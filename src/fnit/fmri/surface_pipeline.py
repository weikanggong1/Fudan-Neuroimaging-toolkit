"""Map a BIDS Derivatives volume run to fsLR32k with MSMSulc."""

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import shutil
import subprocess
import time
import zipfile

import nibabel as nib
import numpy as np

from ..flirt.coordinates import flirt_to_world_affine
from ..msm import prepare_msmsulc_inputs, run_msmsulc
from ..msm.config import MSMSulcConfig
from .assets_setup import BASE_URL, MESH
from .bids import locate_bids_inputs
from .derivatives import ensure_derivative_dataset, fmri_derivative_paths, sidecar, write_json
from .normalization import resample_world
from .surface import SurfaceHemisphere
from .surface_fmriprep import run_fmriprep_surface_projection
from .surface_prepare import prepare_fmriprep_surface_inputs


@dataclass(frozen=True)
class FMRISurfaceResult:
    """Persistent fsLR32k GIFTI and 91k CIFTI BIDS Derivatives paths."""

    left: Path
    right: Path
    dtseries: Path
    metadata: Path
    timing_seconds: dict[str, float]


def fMRISurface_pipeline(
    bids_root: str | Path,
    derivatives_root: str | Path,
    *,
    subject: str,
    recon_all: str | Path,
    hcp_assets_dir: str | Path,
    session: str | None = None,
    task: str = "rest",
    run: str | None = None,
    acquisition: str | None = None,
    direction: str | None = None,
    reconstruction: str | None = None,
    echo: str | None = None,
    wb_command: str | Path = "wb_command",
    device: str = "cuda:0",
    overwrite: bool = False,
    registered_spheres: tuple[str | Path, str | Path] | None = None,
    msm_config: MSMSulcConfig | str | Path | None = None,
    msm_execution: str = "optimized",
    goodvoxels: str | Path | None = None,
) -> FMRISurfaceResult:
    """Project completed volume derivatives using matching T1 recon-all surfaces."""
    started = time.perf_counter()
    if registered_spheres is not None and len(registered_spheres) != 2:
        raise ValueError("registered_spheres must contain left and right paths")
    if registered_spheres is not None and msm_config is not None:
        raise ValueError("msm_config cannot be applied to supplied registered_spheres")
    if msm_execution not in ("optimized", "reference"):
        raise ValueError("msm_execution must be 'optimized' or 'reference'")
    if msm_config is None:
        configuration = MSMSulcConfig()
    elif isinstance(msm_config, MSMSulcConfig):
        configuration = msm_config
    elif isinstance(msm_config, (str, Path)):
        configuration = MSMSulcConfig.from_file(msm_config)
    else:
        raise TypeError("msm_config must be MSMSulcConfig or a config path")
    registration = {"Method": "provided spheres"}
    registration_seconds = None
    inputs = locate_bids_inputs(
        bids_root, subject=subject, session=session, task=task, run=run,
        acquisition=acquisition, direction=direction,
        reconstruction=reconstruction, echo=echo,
    )
    mni_sidecar = None
    paths = None
    for t1_candidate in inputs.t1w_images:
        candidate = fmri_derivative_paths(inputs, t1_candidate, derivatives_root)
        if candidate.clean_mni.is_file():
            paths = candidate
            mni_sidecar = sidecar(candidate.clean_mni)
            break
    if paths is None:
        raise FileNotFoundError("completed FNIT volume BIDS derivative not found")
    ensure_derivative_dataset(paths.root, inputs.bids_root)
    for path in (paths.clean_native, paths.clean_mni, paths.t1_brain,
                 paths.bbr_matrix, mni_sidecar):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = json.loads(mni_sidecar.read_text(encoding="utf-8"))
    source_t1 = inputs.bids_root / metadata.get("FNIT", {}).get("SourceT1w", "")
    if source_t1.resolve() not in [p.resolve() for p in inputs.t1w_images]:
        raise ValueError("volume derivative T1w does not belong to this BIDS subject")
    paths = fmri_derivative_paths(inputs, source_t1, derivatives_root)
    if not any(metadata.get("FNIT", {}).get("ConfoundRegression", {}).get(key, False)
               for key in ("wm", "csf", "motion")):
        raise ValueError("volume derivative lacks completed WM, CSF or motion regression")
    if paths.dtseries.exists() and not overwrite:
        raise FileExistsError(paths.dtseries)
    assets = Path(hcp_assets_dir).expanduser().resolve()
    dseg = assets / "fmriprep/tpl-MNI152NLin6Asym_res-02_atlas-HCP_dseg.nii.gz"
    if not dseg.is_file():
        raise FileNotFoundError(dseg)
    native = paths.clean_native
    clean_mni = paths.clean_mni
    t1 = paths.t1_brain
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
        np.loadtxt(paths.bbr_matrix), epi_image.affine, t1_image.affine,
        epi_image.shape[:3], t1_image.shape,
        epi_image.header.get_zooms()[:3], t1_image.header.get_zooms()[:3],
    )
    with TemporaryDirectory(prefix="fnit-surface-") as work_dir:
        output = Path(work_dir)
        t1_bold = output / "clean_T1w.nii.gz"
        resample_world(native, t1, np.linalg.inv(epi_to_t1), t1_bold,
                       device=device)
        source = Path(recon_all).expanduser().resolve()
        with TemporaryDirectory(prefix="recon_all_", dir=output) as temporary:
            if source.is_file() and source.suffix.lower() == ".zip":
                subject_dir = Path(temporary) / "FreeSurfer"
                required = ("mri/orig.mgz", "mri/orig/001.mgz") + tuple(
                    f"surf/{hemi}.{name}"
                    for hemi in ("lh", "rh")
                    for name in ("white", "pial", "sphere.reg", "thickness", "sphere", "sulc")
                )
                with zipfile.ZipFile(source) as archive:
                    for name in required:
                        target = subject_dir / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(f"FreeSurfer/{name}") as reader, target.open("wb") as writer:
                            shutil.copyfileobj(reader, writer)
            else:
                subject_dir = source / "FreeSurfer" if (source / "FreeSurfer").is_dir() else source
            scanner = nib.load(str(subject_dir / "mri/orig/001.mgz"))
            if scanner.shape != t1_image.shape or not np.allclose(
                    scanner.affine, t1_image.affine, atol=1e-4, rtol=0):
                raise ValueError("recon-all scanner T1 and volume-pipeline T1 have different grids")
            prepared = prepare_fmriprep_surface_inputs(
                subject_dir=subject_dir, hcp_assets_dir=assets,
                output_dir=output / "prepared", wb_command=wb_command,
            )
            if registered_spheres is None:
                registration_started = time.perf_counter()
                sulc_inputs = prepare_msmsulc_inputs(
                    subject_dir=subject_dir,
                    initial_spheres=prepared.initial_spheres,
                    hcp_assets_dir=assets, output_dir=output / "msmsulc_inputs",
                    wb_command=wb_command,
                )
                spheres = run_msmsulc(sulc_inputs, output / "msmsulc", device=device,
                                     config=configuration, execution=msm_execution)
                spheres = (spheres["L"], spheres["R"])
                registration_seconds = time.perf_counter() - registration_started
                report = json.loads((output / "msmsulc/registration_report.json").read_text(
                    encoding="utf-8"))
                registration = {
                    "Method": "FNIT MSMSulc-HOCR-FastPD",
                    "Configuration": configuration.to_dict(),
                    "Execution": msm_execution,
                    "Hemispheres": {
                        hemi: {key: report[hemi][key] for key in (
                            "seconds", "peak_allocated_gb", "folded_output_faces",
                        ) if key in report[hemi]} for hemi in ("L", "R")
                    },
                }
            else:
                spheres = registered_spheres
            mesh = assets / "global/templates/standard_mesh_atlases"
            executable = shutil.which(str(wb_command))
            if executable is None:
                raise FileNotFoundError(wb_command)
            hemispheres = {}
            for hemi, geometry, individual_roi, sphere in zip(
                ("L", "R"), (prepared.geometry.left, prepared.geometry.right),
                prepared.individual_rois, spheres,
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
            )
        for current, destination in ((projection.left_metric, paths.left),
                                     (projection.right_metric, paths.right),
                                     (projection.dtseries, paths.dtseries)):
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(current, destination)
        timing = {**projection.timing_seconds, "total": time.perf_counter() - started}
        if registration_seconds is not None:
            timing["msmsulc_preparation_and_registration"] = registration_seconds
        details = {**{key: metadata[key] for key in (
                       "TaskName", "EchoTime", "FlipAngle", "MagneticFieldStrength",
                       "Manufacturer", "PhaseEncodingDirection", "Units",
                   ) if key in metadata},
                   "Sources": [
                       f"bids::{paths.clean_native.relative_to(paths.root).as_posix()}",
                       f"bids::{paths.clean_mni.relative_to(paths.root).as_posix()}",
                   ],
                   "RepetitionTime": inputs.tr,
                   "SkullStripped": True,
                   "FNIT": {"Registration": registration["Method"],
                            "RegistrationDetails": registration,
                            "Projection": "fMRIPrep-style T1w cortex + MNI subcortex",
                            "TimingSeconds": timing,
                            "Coverage": json.loads(projection.coverage_report.read_text(encoding="utf-8"))}}
        for path in (paths.left, paths.right, paths.dtseries):
            item = dict(details)
            item["Density"] = ("32,492 vertices per hemisphere" if path != paths.dtseries
                               else "91,282 grayordinates; 32,492 fsLR vertices per hemisphere")
            if path == paths.dtseries:
                item["SpatialReference"] = {
                    "VolumeReference": "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin6Asym_res-02_T1w.nii.gz",
                    "CIFTI_STRUCTURE_CORTEX_LEFT": BASE_URL + MESH + "L.sphere.32k_fs_LR.surf.gii",
                    "CIFTI_STRUCTURE_CORTEX_RIGHT": BASE_URL + MESH + "R.sphere.32k_fs_LR.surf.gii",
                }
            else:
                hemi = "L" if path == paths.left else "R"
                item["SpatialReference"] = BASE_URL + MESH + f"{hemi}.sphere.32k_fs_LR.surf.gii"
            write_json(sidecar(path), item)
    return FMRISurfaceResult(paths.left, paths.right, paths.dtseries,
                             sidecar(paths.dtseries), timing)
