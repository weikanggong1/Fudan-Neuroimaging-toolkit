"""fMRIPrep-style T1w ribbon projection and 91k CIFTI assembly.

Uses Workbench and nibabel; fMRIPrep, Nipype and FreeSurfer executables are
not runtime dependencies. Inputs must already be registered to their spaces.
"""

from pathlib import Path
import json
import shutil
import subprocess
import time

import nibabel as nib
from nibabel.cifti2.cifti2_axes import BrainModelAxis, SeriesAxis
import numpy as np

from .surface import SurfaceHemisphere, SurfaceProjectionResult, _hemisphere_paths


# NiWorkflows GenerateCifti structure and HCP-dseg label order for 91k.
_SUBCORTEX = (
    ("ACCUMBENS_LEFT", 26), ("ACCUMBENS_RIGHT", 58),
    ("AMYGDALA_LEFT", 18), ("AMYGDALA_RIGHT", 54),
    ("BRAIN_STEM", 16), ("CAUDATE_LEFT", 11), ("CAUDATE_RIGHT", 50),
    ("CEREBELLUM_LEFT", 8), ("CEREBELLUM_RIGHT", 47),
    ("DIENCEPHALON_VENTRAL_LEFT", 28), ("DIENCEPHALON_VENTRAL_RIGHT", 60),
    ("HIPPOCAMPUS_LEFT", 17), ("HIPPOCAMPUS_RIGHT", 53),
    ("PALLIDUM_LEFT", 13), ("PALLIDUM_RIGHT", 52),
    ("PUTAMEN_LEFT", 12), ("PUTAMEN_RIGHT", 51),
    ("THALAMUS_LEFT", 10), ("THALAMUS_RIGHT", 49),
)


def _las(image):
    current = nib.orientations.io_orientation(image.affine)
    target = nib.orientations.axcodes2ornt(("L", "A", "S"))
    return image.as_reoriented(nib.orientations.ornt_transform(current, target))


def create_fmriprep_cifti(
    clean_mni, left_metric, right_metric, left_label, right_label,
    hcp_dseg, output_file,
):
    """Assemble 91k CIFTI using NiWorkflows structure/voxel ordering.

    ``clean_mni`` is 4D MNI152NLin6Asym 2-mm BOLD. ``left/right_metric``
    are 32k fsLR time series; ``left/right_label`` are TemplateFlow's
    non-medial-wall fsLR32k labels; ``hcp_dseg`` is the matching TemplateFlow
    MNI152NLin6Asym HCP atlas. Returns the CIFTI path.
    """
    bold = _las(nib.load(str(clean_mni)))
    labels = _las(nib.load(str(hcp_dseg)))
    if bold.ndim != 4 or labels.ndim != 3 or bold.shape[:3] != labels.shape:
        raise ValueError("MNI BOLD and HCP dseg must share a 3D 2-mm grid")
    if not np.allclose(bold.affine, labels.affine, atol=1e-4, rtol=0):
        raise ValueError("MNI BOLD and HCP dseg affines differ")
    unit = bold.header.get_xyzt_units()[1]
    if unit not in ("sec", "msec", "usec"):
        raise ValueError("MNI BOLD must encode its TR")
    tr = float(bold.header.get_zooms()[3]) * {"sec": 1, "msec": .001, "usec": 1e-6}[unit]
    nframes = bold.shape[3]
    models = []
    arrays = []
    for hemi, metric_file, label_file in (
        ("LEFT", left_metric, left_label), ("RIGHT", right_metric, right_label)
    ):
        metric = nib.load(str(metric_file))
        label = nib.load(str(label_file))
        if len(metric.darrays) != nframes or len(label.darrays) != 1:
            raise ValueError(f"{hemi} metric frames or medial-wall label are invalid")
        vertices = np.flatnonzero(np.asarray(label.darrays[0].data))
        if len(label.darrays[0].data) != 32492 or not len(vertices):
            raise ValueError(f"{hemi} requires a nonempty fsLR32k label")
        data = np.stack([np.asarray(frame.data, dtype=np.float32)[vertices]
                         for frame in metric.darrays], axis=0)
        models.append(BrainModelAxis.from_surface(
            vertices, 32492, name=f"CIFTI_STRUCTURE_CORTEX_{hemi}"
        ))
        arrays.append(data)
    label_data = np.asarray(labels.dataobj, dtype=np.int16)
    bold_data = np.asarray(bold.dataobj, dtype=np.float32)
    for name, value in _SUBCORTEX:
        # NiWorkflows indexes each structure in Fortran voxel order.
        k, j, i = np.nonzero(label_data.T == value)
        if not len(i):
            continue
        voxels = np.stack((i, j, k), axis=1)
        models.append(BrainModelAxis(
            f"CIFTI_STRUCTURE_{name}", voxel=voxels,
            affine=bold.affine, volume_shape=bold.shape[:3],
        ))
        arrays.append(bold_data[i, j, k].T)
    axis = models[0]
    for model in models[1:]:
        axis += model
    data = np.concatenate(arrays, axis=1)
    output = Path(output_file).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Cifti2Image(data, header=nib.Cifti2Header.from_axes((SeriesAxis(0, tr, nframes), axis)))
    image.nifti_header.set_intent("NIFTI_INTENT_CONNECTIVITY_DENSE_SERIES")
    nib.save(image, str(output))
    return output


def run_fmriprep_surface_projection(
    clean_t1w: str | Path,
    clean_mni: str | Path,
    left: SurfaceHemisphere,
    right: SurfaceHemisphere,
    left_label: str | Path,
    right_label: str | Path,
    hcp_dseg: str | Path,
    output_dir: str | Path,
    *,
    goodvoxels: str | Path | None = None,
    wb_command: str | Path = "wb_command",
    overwrite: bool = False,
) -> SurfaceProjectionResult:
    """Project T1w BOLD to fsLR32k and combine with MNI BOLD as 91k CIFTI.

    ``left/right`` hold T1w-space white/pial/midthickness, MSMSulc sphere,
    native cortex mask, fsLR sphere/midthickness and atlas ROI. Optional
    ``goodvoxels`` is a T1w-grid volume ROI. Output is a 91k dtseries, two
    full 32k functional GIFTIs and a coverage JSON.
    """
    t1w = nib.load(str(clean_t1w))
    mni = nib.load(str(clean_mni))
    if t1w.ndim != 4 or mni.ndim != 4 or t1w.shape[3] != mni.shape[3]:
        raise ValueError("T1w and MNI inputs must be 4D with equal frame counts")
    if goodvoxels is not None:
        roi = nib.load(str(goodvoxels))
        if roi.shape != t1w.shape[:3] or not np.allclose(roi.affine, t1w.affine, atol=1e-4):
            raise ValueError("goodvoxels must share the T1w BOLD grid")
    wb = shutil.which(str(wb_command))
    if wb is None:
        raise FileNotFoundError(wb_command)
    hemis = {"L": _hemisphere_paths(left, "left"),
             "R": _hemisphere_paths(right, "right")}
    output = Path(output_dir).expanduser().resolve()
    dtseries = output / "space-fsLR_den-91k_bold.dtseries.nii"
    if dtseries.exists() and not overwrite:
        raise FileExistsError(dtseries)
    output.mkdir(parents=True, exist_ok=True)
    timing = {}
    metrics = {}

    def command(key, *args):
        start = time.perf_counter()
        subprocess.run([wb, *map(str, args)], check=True, capture_output=True, text=True)
        timing[key] = time.perf_counter() - start

    for hemi, paths in hemis.items():
        native = output / f"{hemi}.native.func.gii"
        dilated = output / f"{hemi}.native_dilated.func.gii"
        masked = output / f"{hemi}.native_masked.func.gii"
        atlas = output / f"{hemi}.atlas.func.gii"
        final = output / f"{hemi}.32k.func.gii"
        ribbon = ["-volume-to-surface-mapping", clean_t1w, paths["midthickness"],
                  native, "-ribbon-constrained", paths["white"], paths["pial"]]
        if goodvoxels is not None:
            ribbon.extend(("-volume-roi", goodvoxels))
        command(f"{hemi}_ribbon", *ribbon)
        command(f"{hemi}_dilate", "-metric-dilate", native,
                paths["midthickness"], 10, dilated, "-nearest")
        command(f"{hemi}_native_mask", "-metric-mask", dilated,
                paths["native_roi"], masked)
        command(f"{hemi}_resample", "-metric-resample", masked,
                paths["registered_sphere"], paths["atlas_sphere"],
                "ADAP_BARY_AREA", atlas, "-area-surfs", paths["midthickness"],
                paths["atlas_midthickness"], "-current-roi", paths["native_roi"])
        command(f"{hemi}_atlas_mask", "-metric-mask", atlas,
                paths["atlas_roi"], final)
        metrics[hemi] = final
    start = time.perf_counter()
    create_fmriprep_cifti(clean_mni, metrics["L"], metrics["R"],
                          left_label, right_label, hcp_dseg, dtseries)
    timing["cifti"] = time.perf_counter() - start
    image = nib.load(str(dtseries))
    data = np.asarray(image.dataobj, dtype=np.float32)
    varying = np.ptp(data, axis=0) > 0
    report = output / "coverage.json"
    report.write_text(json.dumps({
        "frames": int(data.shape[0]), "grayordinates": int(data.shape[1]),
        "varying_grayordinates": int(varying.sum()),
        "tr_seconds": float(image.header.get_axis(0).step),
        "timing_seconds": timing,
    }, indent=2) + "\n", encoding="utf-8")
    return SurfaceProjectionResult(dtseries, metrics["L"], metrics["R"],
                                   Path(clean_mni), timing, report)
