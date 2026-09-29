"""UKB-style MNI volume to fsLR32k projection using Connectome Workbench.

This is the projection and dense-timeseries stage with supplied, already
registered surfaces and ROI images. It does not estimate MSMSulc, MSMAll,
DeDrift, or the UKB goodvoxels mask.
"""

from dataclasses import dataclass
from pathlib import Path
import json
import math
import os
import shutil
import subprocess
import time

import nibabel as nib
import numpy as np


@dataclass(frozen=True)
class SurfaceHemisphere:
    """Paths for one hemisphere, with matching native and fsLR32k topology."""

    white: str | Path
    pial: str | Path
    midthickness: str | Path
    registered_sphere: str | Path
    native_roi: str | Path
    atlas_sphere: str | Path
    atlas_midthickness: str | Path
    atlas_roi: str | Path


@dataclass(frozen=True)
class SurfaceProjectionResult:
    """The dense timeseries, cortical metrics, volume source and timing."""

    dtseries: Path
    left_metric: Path
    right_metric: Path
    subcortical_volume: Path
    timing_seconds: dict[str, float]
    coverage_report: Path


def _nifti(path, name, ndim):
    image = nib.load(str(path))
    if not isinstance(image, nib.Nifti1Image) or image.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}D NIfTI image")
    return image


def _matching_grid(image, reference, name):
    if image.shape[:3] != reference.shape[:3] or not np.allclose(
        image.affine, reference.affine, atol=1e-4, rtol=0
    ):
        raise ValueError(f"{name} must match the MNI reference grid")


def _gifti_count(path, name, *, metric=False):
    image = nib.load(str(path))
    if not isinstance(image, nib.GiftiImage):
        raise ValueError(f"{name} must be a GIFTI file")
    arrays = image.darrays if metric else [item for item in image.darrays if item.intent == 1008]
    if len(arrays) != 1:
        raise ValueError(f"{name} must have exactly one {'metric' if metric else 'pointset'} array")
    data = np.asarray(arrays[0].data)
    if data.ndim != (1 if metric else 2) or (not metric and data.shape[1] != 3):
        raise ValueError(f"{name} has invalid vertex data")
    if not np.isfinite(data).all():
        raise ValueError(f"{name} contains nonfinite values")
    return len(data), int(np.count_nonzero(data)) if metric else None


def _hemisphere_paths(hemi, name):
    paths = {field: Path(getattr(hemi, field)).expanduser().resolve()
             for field in SurfaceHemisphere.__dataclass_fields__}
    for field, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}.{field}: {path}")
    native = [_gifti_count(paths[field], f"{name}.{field}")[0]
              for field in ("white", "pial", "midthickness", "registered_sphere")]
    native_roi, native_nonzero = _gifti_count(paths["native_roi"], f"{name}.native_roi", metric=True)
    atlas = [_gifti_count(paths[field], f"{name}.{field}")[0]
             for field in ("atlas_sphere", "atlas_midthickness")]
    atlas_roi, atlas_nonzero = _gifti_count(paths["atlas_roi"], f"{name}.atlas_roi", metric=True)
    if len(set((*native, native_roi))) != 1 or len(set((*atlas, atlas_roi))) != 1:
        raise ValueError(f"{name} surface and ROI vertex counts do not match")
    if atlas_roi != 32492 or not native_nonzero or not atlas_nonzero:
        raise ValueError(f"{name} requires nonempty native ROI and fsLR32k atlas ROI")

    def triangles(field, vertices):
        image = nib.load(str(paths[field]))
        arrays = [item for item in image.darrays if item.intent == 1009]
        if len(arrays) != 1:
            raise ValueError(f"{name}.{field} must have one triangle array")
        faces = np.asarray(arrays[0].data)
        if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
                or faces.min() < 0 or faces.max() >= vertices):
            raise ValueError(f"{name}.{field} has invalid triangle indices")
        return faces

    native_faces = triangles("white", native[0])
    for field in ("pial", "midthickness", "registered_sphere"):
        if not np.array_equal(triangles(field, native[0]), native_faces):
            raise ValueError(f"{name}.{field} does not share native vertex order")
    atlas_faces = triangles("atlas_sphere", atlas[0])
    if not np.array_equal(triangles("atlas_midthickness", atlas[0]), atlas_faces):
        raise ValueError(f"{name}.atlas_midthickness does not share atlas vertex order")
    return paths


def run_surface_projection(
    clean_mni: str | Path,
    mni_reference: str | Path,
    goodvoxels: str | Path,
    subject_rois: str | Path,
    atlas_rois: str | Path,
    left: SurfaceHemisphere,
    right: SurfaceHemisphere,
    output_dir: str | Path,
    *,
    wb_command: str | Path = "wb_command",
    overwrite: bool = False,
) -> SurfaceProjectionResult:
    """Project cleaned MNI BOLD to fsLR32k cortex and atlas subcortex.

    The input MNI BOLD/reference/goodvoxels/subject ROIs and atlas ROIs must
    share a 2-mm grid. Native GIFTIs are already in MNI scanner RAS; their
    registered spheres must correspond to the supplied fsLR32k atlas spheres.
    The explicit goodvoxels and native ROI files must come from upstream UKB
    dropout/ribbon and structural-surface processing, respectively.
    Workbench performs ribbon-constrained sampling, area-corrected spherical
    resampling, 2-mm FWHM smoothing, and dense CIFTI creation. This function
    does not infer the sphere registration method from filenames.
    """
    source = _nifti(clean_mni, "clean_mni", 4)
    reference = _nifti(mni_reference, "mni_reference", 3)
    for name, image in (("clean_mni", source),
                        ("goodvoxels", _nifti(goodvoxels, "goodvoxels", 3)),
                        ("subject_rois", _nifti(subject_rois, "subject_rois", 3)),
                        ("atlas_rois", _nifti(atlas_rois, "atlas_rois", 3))):
        _matching_grid(image, reference, name)
    if not np.allclose(reference.header.get_zooms()[:3], 2.0, atol=0.01, rtol=0):
        raise ValueError("mni_reference must have 2-mm voxels")
    for name, path in (("goodvoxels", goodvoxels), ("subject_rois", subject_rois),
                       ("atlas_rois", atlas_rois)):
        values = np.asarray(nib.load(str(path)).dataobj)
        if not np.isfinite(values).all() or not np.any(values > 0):
            raise ValueError(f"{name} must be finite and nonempty")
    unit = source.header.get_xyzt_units()[1]
    if unit not in ("sec", "msec", "usec"):
        raise ValueError("clean_mni must record its TR in seconds, msec, or usec")
    tr = float(source.header.get_zooms()[3]) * {"sec": 1.0, "msec": 0.001, "usec": 1e-6}[unit]
    if not math.isfinite(tr) or tr <= 0:
        raise ValueError("clean_mni must have a positive finite TR")
    surfaces = {"L": _hemisphere_paths(left, "left"),
                "R": _hemisphere_paths(right, "right")}
    executable = shutil.which(str(wb_command))
    if executable is None:
        raise FileNotFoundError(f"Connectome Workbench executable not found: {wb_command}")
    output = Path(output_dir).expanduser().resolve()
    dtseries = output / "clean_MNI_Atlas_registered_sphere_s2.dtseries.nii"
    if dtseries.exists() and not overwrite:
        raise FileExistsError(dtseries)
    output.mkdir(parents=True, exist_ok=True)
    work = output / "work"
    work.mkdir(exist_ok=True)
    timing = {}
    wb_env = os.environ.copy()
    wb_env.setdefault("OMP_NUM_THREADS", str(min(os.cpu_count() or 8, 8)))

    def command(stage, destination, *arguments):
        destination.unlink(missing_ok=True)
        started = time.perf_counter()
        subprocess.run([executable, *map(str, arguments)], check=True,
                       capture_output=True, text=True, env=wb_env)
        if not destination.is_file():
            raise RuntimeError(f"Workbench {stage} did not create {destination}")
        timing[stage] = time.perf_counter() - started

    sigma = 2.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    metrics = {}
    for hemi, paths in surfaces.items():
        native = work / f"{hemi}.native.func.gii"
        native_dilated = work / f"{hemi}.native_dilated.func.gii"
        native_masked = work / f"{hemi}.native_masked.func.gii"
        atlas = work / f"{hemi}.32k.func.gii"
        atlas_dilated = work / f"{hemi}.32k_dilated.func.gii"
        atlas_masked = work / f"{hemi}.32k_masked.func.gii"
        smoothed = output / f"{hemi}.32k_registered_sphere_s2.func.gii"
        command(f"{hemi}_ribbon", native, "-volume-to-surface-mapping", clean_mni,
                paths["midthickness"], native, "-ribbon-constrained",
                paths["white"], paths["pial"], "-volume-roi", goodvoxels)
        command(f"{hemi}_native_dilate", native_dilated, "-metric-dilate", native,
                paths["midthickness"], 10, native_dilated, "-nearest")
        command(f"{hemi}_native_mask", native_masked, "-metric-mask",
                native_dilated, paths["native_roi"], native_masked)
        command(f"{hemi}_resample", atlas, "-metric-resample", native_masked,
                paths["registered_sphere"], paths["atlas_sphere"],
                "ADAP_BARY_AREA", atlas, "-area-surfs", paths["midthickness"],
                paths["atlas_midthickness"], "-current-roi", paths["native_roi"])
        command(f"{hemi}_atlas_dilate", atlas_dilated, "-metric-dilate", atlas,
                paths["atlas_midthickness"], 30, atlas_dilated, "-nearest")
        command(f"{hemi}_atlas_mask", atlas_masked, "-metric-mask",
                atlas_dilated, paths["atlas_roi"], atlas_masked)
        command(f"{hemi}_smooth", smoothed, "-metric-smoothing",
                paths["atlas_midthickness"], atlas_masked, sigma, smoothed,
                "-roi", paths["atlas_roi"])
        metrics[hemi] = smoothed

    subject = work / "subcortical_subject.dtseries.nii"
    dilated = work / "subcortical_subject_dilated.dtseries.nii"
    smooth = work / "subcortical_subject_smoothed.dtseries.nii"
    atlas_template = work / "subcortical_atlas_template.dlabel.nii"
    resampled = work / "subcortical_atlas.dtseries.nii"
    atlas_dilated = work / "subcortical_atlas_dilated.dtseries.nii"
    subcortical = output / "subcortical_MNI_s2.nii.gz"
    command("subcortical_subject", subject, "-cifti-create-dense-timeseries",
            subject, "-volume", clean_mni, subject_rois, "-timestep", tr)
    command("subcortical_dilate", dilated, "-cifti-dilate", subject,
            "COLUMN", 0, 30, dilated)
    command("subcortical_smooth", smooth, "-cifti-smoothing", dilated,
            0, sigma, "COLUMN", smooth, "-fix-zeros-volume")
    command("subcortical_template", atlas_template, "-cifti-create-label",
            atlas_template, "-volume", atlas_rois, atlas_rois)
    command("subcortical_resample", resampled, "-cifti-resample", smooth,
            "COLUMN", atlas_template, "COLUMN", "ADAP_BARY_AREA", "CUBIC",
            resampled, "-volume-predilate", 10)
    command("subcortical_atlas_dilate", atlas_dilated, "-cifti-dilate",
            resampled, "COLUMN", 0, 30, atlas_dilated)
    command("subcortical_separate", subcortical, "-cifti-separate",
            atlas_dilated, "COLUMN", "-volume-all", subcortical)
    command("dense_timeseries", dtseries, "-cifti-create-dense-timeseries",
            dtseries, "-volume", subcortical, atlas_rois,
            "-left-metric", metrics["L"], "-roi-left", surfaces["L"]["atlas_roi"],
            "-right-metric", metrics["R"], "-roi-right", surfaces["R"]["atlas_roi"],
            "-timestep", tr)
    subcortical_image = _nifti(subcortical, "subcortical output", 4)
    if subcortical_image.shape != (*reference.shape, source.shape[3]):
        raise RuntimeError("Workbench subcortical output has the wrong grid or frame count")
    _matching_grid(subcortical_image, reference, "subcortical output")
    final = nib.load(str(dtseries))
    if not isinstance(final, nib.Cifti2Image) or final.shape[0] != source.shape[3]:
        raise RuntimeError("Workbench output is not a CIFTI dtseries with matching frames")
    axis = final.header.get_axis(0)
    if not isinstance(axis, nib.cifti2.cifti2_axes.SeriesAxis) or not np.isclose(axis.step, tr):
        raise RuntimeError("Workbench dtseries has the wrong time axis")
    structures = list(final.header.get_axis(1).iter_structures())
    names = {name for name, _, _ in structures}
    cortex = {"CIFTI_STRUCTURE_CORTEX_LEFT", "CIFTI_STRUCTURE_CORTEX_RIGHT"}
    if not cortex <= names or not names - cortex:
        raise RuntimeError("Workbench dtseries is missing cortex or subcortex")
    series = np.asarray(final.dataobj, dtype=np.float32)
    if not np.isfinite(series).all():
        raise RuntimeError("Workbench dtseries contains nonfinite values")
    varying = np.ptp(series, axis=0) > 0
    structure_coverage = {name: {
        "varying_grayordinates": int(varying[indices].sum()),
        "grayordinates": int(len(varying[indices])),
    } for name, indices, _ in structures}
    coverage_path = output / "dense_coverage_report.json"
    coverage_path.write_text(json.dumps({
        "frames": int(series.shape[0]),
        "tr_seconds": tr,
        "varying_grayordinates": int(varying.sum()),
        "grayordinates": int(varying.size),
        "coverage_percent": 100.0 * float(varying.mean()),
        "by_structure": structure_coverage,
    }, indent=2) + "\n", encoding="utf-8")
    return SurfaceProjectionResult(dtseries, metrics["L"], metrics["R"],
                                   subcortical, timing, coverage_path)
