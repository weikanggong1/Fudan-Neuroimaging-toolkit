"""fMRIPrep-style T1w ribbon projection and validated 91k CIFTI assembly.

Uses Workbench and nibabel; fMRIPrep, Nipype and FreeSurfer executables are
not runtime dependencies. Inputs must already be registered to their spaces.

Modified in FNIT (2026): direct nibabel assembly, input validation and atomic
publication. CIFTI ordering/metadata derive from NiWorkflows 1.14.4,
Copyright 2021 The NiPreps Developers, under Apache License 2.0.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import os
import shutil
import subprocess
import time
from types import SimpleNamespace

import nibabel as nib
from nibabel.cifti2.cifti2_axes import BrainModelAxis, SeriesAxis
import numpy as np

from .._hemisphere_parallel import map_hemispheres, resolve_cpu_threads, workbench_environment
from .assets_setup import ASSETS, FMRIPREP_ASSETS, FMRIPREP_BASE, _sha256
from .surface import (
    SurfaceHemisphere, SurfaceProjectionResult, _check_gifti_hemisphere,
    _gifti_count, _hemisphere_paths,
)


# NiWorkflows 1.14.4 GenerateCifti structure and HCP-dseg label order for 91k.
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


def fmriprep_cifti_metadata():
    """Return the 91k Density/SpatialReference used by NiWorkflows 1.14.4."""
    return {
        "Density": (
            "91,282 grayordinates corresponding to all of the grey matter sampled at a "
            "2mm average vertex spacing on the surface and as 2mm voxels subcortically"
        ),
        "SpatialReference": {
            "VolumeReference": FMRIPREP_BASE + "tpl-MNI152NLin6Asym_res-02_T1w.nii.gz",
            "CIFTI_STRUCTURE_LEFT_CORTEX": (
                "https://templateflow.s3.amazonaws.com/tpl-fsLR/"
                "tpl-fsLR_den-32k_hemi-L_midthickness.surf.gii"
            ),
            "CIFTI_STRUCTURE_RIGHT_CORTEX": (
                "https://templateflow.s3.amazonaws.com/tpl-fsLR/"
                "tpl-fsLR_den-32k_hemi-R_midthickness.surf.gii"
            ),
        },
    }


def _las(image):
    current = nib.orientations.io_orientation(image.affine)
    target = nib.orientations.axcodes2ornt(("L", "A", "S"))
    return image.as_reoriented(nib.orientations.ornt_transform(current, target))


def _las_grid(image):
    """Inspect the LAS grid without decoding or reorienting 4D image data."""
    current = nib.orientations.io_orientation(image.affine)
    target = nib.orientations.axcodes2ornt(("L", "A", "S"))
    orientation = nib.orientations.ornt_transform(current, target)
    order = np.argsort(orientation[:, 0]).astype(int)
    shape = tuple(image.shape[index] for index in order) + tuple(image.shape[3:])
    affine = image.affine @ nib.orientations.inv_ornt_aff(orientation, image.shape[:3])
    return SimpleNamespace(shape=shape, ndim=image.ndim, affine=affine, header=image.header)


def _tr_seconds(image, name, expected=None):
    if image.ndim != 4 or image.shape[3] < 1:
        raise ValueError(f"{name} must be a nonempty 4D BOLD image")
    unit = image.header.get_xyzt_units()[1]
    if unit not in ("sec", "msec", "usec"):
        raise ValueError(f"{name} must encode its TR in seconds, milliseconds or microseconds")
    seconds = float(image.header.get_zooms()[3]) * {"sec": 1, "msec": .001, "usec": 1e-6}[unit]
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"{name} TR must be positive and finite")
    if expected is None:
        return seconds
    if isinstance(expected, (bool, np.bool_)) or not np.isfinite(expected) or expected <= 0:
        raise ValueError("tr_seconds must be positive and finite")
    # NIfTI-1 stores pixdim as float32, including its TR.
    if not np.isclose(seconds, expected, rtol=1e-6, atol=1e-7):
        raise ValueError(f"{name} TR differs from the requested time axis")
    return float(expected)


def _cifti_assets(left_label, right_label, hcp_dseg):
    checksums = dict(ASSETS + FMRIPREP_ASSETS)
    vertices = {}
    for hemisphere, path in (("LEFT", left_label), ("RIGHT", right_label)):
        _gifti_count(path, f"{hemisphere} atlas ROI", metric=True, hemisphere=hemisphere)
        expected = checksums[
            f"global/templates/standard_mesh_atlases/{hemisphere[0]}.atlasroi.32k_fs_LR.shape.gii"
        ]
        if _sha256(Path(path)) != expected:
            raise ValueError(f"{hemisphere} atlas ROI does not match the fixed fsLR32k asset SHA-256")
        values = np.asarray(nib.load(str(path)).darrays[0].data)
        vertices[hemisphere] = np.flatnonzero(values > 0)
    expected = checksums["fmriprep/tpl-MNI152NLin6Asym_res-02_atlas-HCP_dseg.nii.gz"]
    if _sha256(Path(hcp_dseg)) != expected:
        raise ValueError("HCP dseg does not match the fixed MNI152NLin6Asym asset SHA-256")
    labels = _las(nib.load(str(hcp_dseg)))
    values = np.asarray(labels.dataobj)
    if (labels.ndim != 3 or not np.isfinite(values).all()
            or (values < 0).any() or not np.equal(values, np.floor(values)).all()
            or not np.allclose(nib.affines.voxel_sizes(labels.affine), 2, rtol=0, atol=1e-5)):
        raise ValueError("HCP dseg must contain finite nonnegative integer labels on its 2-mm grid")
    counts = [int(np.count_nonzero(values == label)) for _, label in _SUBCORTEX]
    if (any(count == 0 for count in counts) or sum(counts) != 31870
            or sum(map(len, vertices.values())) != 59412):
        raise ValueError("Fixed 91k assets must provide both cortical ROIs and all 19 subcortical structures")
    return labels, vertices


def _mni_grid(bold, labels):
    if bold.ndim != 4 or bold.shape[:3] != labels.shape:
        raise ValueError("MNI BOLD and HCP dseg must share the fixed 3D 2-mm grid")
    if (not np.isfinite(bold.affine).all()
            or not np.allclose(bold.affine, labels.affine, atol=1e-4, rtol=0)
            or bold.header.get_xyzt_units()[0] != "mm"):
        raise ValueError("MNI BOLD must use the HCP dseg affine in millimeters")


def _millimeter_affine(image, name):
    if (not np.isfinite(image.affine).all()
            or abs(np.linalg.det(image.affine[:3, :3])) < 1e-10
            or image.header.get_xyzt_units()[0] != "mm"):
        raise ValueError(f"{name} must have a valid millimeter world affine")


def _metric_frames(path, hemisphere, nframes):
    image = nib.load(str(path))
    if not isinstance(image, nib.GiftiImage):
        raise ValueError(f"{hemisphere} time series must be a GIFTI file")
    _check_gifti_hemisphere(image, str(path), hemisphere)
    if len(image.darrays) != nframes:
        raise ValueError(f"{hemisphere} metric frame count differs from BOLD")
    frames = []
    for array in image.darrays:
        values = np.asarray(array.data, dtype=np.float32)
        if values.shape != (32492,) or not np.isfinite(values).all():
            raise ValueError(f"{hemisphere} metric must contain 32,492 finite values in every frame")
        frames.append(values)
    return frames


def create_fmriprep_cifti(
    clean_mni, left_metric, right_metric, left_label, right_label,
    hcp_dseg, output_file, *, tr_seconds=None, parallel=True, cpu_threads=None,
):
    """Assemble fixed 91k CIFTI using NiWorkflows 1.14.4 ordering.

    ``clean_mni`` is 4D MNI152NLin6Asym 2-mm BOLD. ``left/right_metric``
    are finite 32k fsLR time series; the ROIs and ``hcp_dseg`` must match
    the installer SHA-256 manifest. ``tr_seconds`` supplies the original
    BIDS TR and must agree with the NIfTI header after unit conversion.
    Without it, the validated MNI header provides the time step.
    ``parallel`` reads the independent cortical metrics concurrently while
    preserving L/R axis order. ``cpu_threads`` is the total CPU budget.
    """
    bold = _las(nib.load(str(clean_mni)))
    tr = _tr_seconds(bold, "MNI BOLD", tr_seconds)
    labels, vertices = _cifti_assets(left_label, right_label, hcp_dseg)
    _mni_grid(bold, labels)
    nframes = bold.shape[3]
    models = []
    arrays = []
    def cortex(hemisphere, threads):
        hemisphere = "LEFT" if hemisphere == "L" else "RIGHT"
        metric_file = left_metric if hemisphere == "LEFT" else right_metric
        frames = _metric_frames(metric_file, hemisphere, nframes)
        selected = vertices[hemisphere]
        values = np.stack([frame[selected] for frame in frames], axis=0)
        model = BrainModelAxis.from_surface(
            selected, 32492, name=f"CIFTI_STRUCTURE_CORTEX_{hemisphere}"
        )
        return values, model

    for values, model in map_hemispheres(cortex, parallel=parallel, cpu_threads=cpu_threads):
        arrays.append(values)
        models.append(model)
    label_data = np.asarray(labels.dataobj, dtype=np.int16)
    bold_data = np.asarray(bold.dataobj, dtype=np.float32)
    if not np.isfinite(bold_data).all():
        raise ValueError("MNI BOLD contains nonfinite values")
    for name, value in _SUBCORTEX:
        k, j, i = np.nonzero(label_data.T == value)
        models.append(BrainModelAxis(
            f"CIFTI_STRUCTURE_{name}", voxel=np.stack((i, j, k), axis=1),
            affine=bold.affine, volume_shape=bold.shape[:3],
        ))
        arrays.append(bold_data[i, j, k].T)
    axis = models[0]
    for model in models[1:]:
        axis += model
    data = np.concatenate(arrays, axis=1)
    if data.shape != (nframes, 91282):
        raise ValueError("The assembled CIFTI must contain exactly 91,282 grayordinates")
    image = nib.Cifti2Image(
        data, header=nib.Cifti2Header.from_axes((SeriesAxis(0, tr, nframes), axis))
    )
    image.header.matrix.metadata = nib.cifti2.Cifti2MetaData(fmriprep_cifti_metadata())
    image.nifti_header.set_intent("NIFTI_INTENT_CONNECTIVITY_DENSE_SERIES")
    output = Path(output_file).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # A failed save never exposes a truncated final dtseries.
    with TemporaryDirectory(prefix=".fnit-cifti-", dir=output.parent) as temporary:
        candidate = Path(temporary) / output.name
        nib.save(image, str(candidate))
        os.replace(candidate, output)
    return output


def _projection_names():
    return tuple(
        f"{hemisphere}.{metric}.func.gii"
        for hemisphere in ("L", "R")
        for metric in ("native", "native_dilated", "native_masked", "atlas", "32k")
    ) + ("space-fsLR_den-91k_bold.dtseries.nii", "coverage.json")


def _publish_projection(staging, output, names, overwrite):
    output.mkdir(parents=True, exist_ok=True)
    backup = staging / "previous"
    backup.mkdir()
    published = []
    saved = []
    try:
        for name in names:
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_dir():
                raise ValueError(f"output file is a directory: {target}")
            if target.exists() or target.is_symlink():
                if not overwrite:
                    raise FileExistsError(target)
                (backup / name).parent.mkdir(parents=True, exist_ok=True)
                os.replace(target, backup / name)
                saved.append(name)
            if overwrite:
                os.replace(staging / name, target)
                published.append(name)
            else:
                # Both paths are on the output filesystem. link() creates
                # each destination atomically and fails even if an existing
                # file or dangling symlink appeared after the initial check.
                os.link(staging / name, target)
                published.append(name)
                (staging / name).unlink()
    except BaseException:
        for name in reversed(published):
            (output / name).unlink()
        for name in saved:
            os.replace(backup / name, output / name)
        raise


def run_fmriprep_surface_projection(
    clean_t1w: str | Path, clean_mni: str | Path,
    left: SurfaceHemisphere, right: SurfaceHemisphere,
    left_label: str | Path, right_label: str | Path,
    hcp_dseg: str | Path, output_dir: str | Path, *,
    tr_seconds: float | None = None,
    goodvoxels: str | Path | None = None,
    wb_command: str | Path = "wb_command", overwrite: bool = False,
    parallel: bool = True, cpu_threads: int | None = None,
) -> SurfaceProjectionResult:
    """Project matched T1w/MNI BOLD and publish the complete fsLR32k/91k result.

    ``tr_seconds`` is the source BIDS TR; both volume headers must agree.
    Existing generated files are protected unless ``overwrite=True``.
    Workbench failure or invalid output leaves previous results intact.
    ``parallel=False`` selects serial hemispheres; ``cpu_threads`` is the
    total budget, shared by the L/R Workbench processes. ``None`` uses
    OMP_NUM_THREADS or PyTorch's existing thread count. No parent process
    thread setting is changed.
    """
    output = Path(output_dir).expanduser().resolve()
    budget = resolve_cpu_threads(cpu_threads)
    names = _projection_names()
    for name in names:
        target = output / name
        if target.is_dir():
            raise ValueError(f"output file is a directory: {target}")
        if (target.exists() or target.is_symlink()) and not overwrite:
            raise FileExistsError(target)
    t1w = nib.load(str(clean_t1w))
    mni = nib.load(str(clean_mni))
    tr = _tr_seconds(mni, "MNI BOLD", tr_seconds)
    _tr_seconds(t1w, "T1w BOLD", tr)
    _millimeter_affine(t1w, "T1w BOLD")
    if t1w.shape[3] != mni.shape[3]:
        raise ValueError("T1w and MNI inputs must have equal frame counts")
    labels, _ = _cifti_assets(left_label, right_label, hcp_dseg)
    _mni_grid(_las_grid(mni), labels)
    if goodvoxels is not None:
        roi = nib.load(str(goodvoxels))
        values = np.asarray(roi.dataobj)
        if (roi.shape != t1w.shape[:3]
                or not np.allclose(roi.affine, t1w.affine, atol=1e-4, rtol=0)
                or not np.isfinite(values).all() or (values < 0).any() or not (values > 0).any()):
            raise ValueError("goodvoxels must be a finite nonnegative nonempty ROI on the T1w BOLD grid")
    wb = shutil.which(str(wb_command))
    if wb is None:
        raise FileNotFoundError(wb_command)
    hemis = {"L": _hemisphere_paths(left, "left"),
             "R": _hemisphere_paths(right, "right")}
    output.parent.mkdir(parents=True, exist_ok=True)
    timing = {}
    with TemporaryDirectory(prefix=".fnit-surface-", dir=output.parent) as temporary:
        staging = Path(temporary)

        def project(hemisphere, threads):
            paths = hemis[hemisphere]
            local_timing = {}
            environment = workbench_environment(threads)

            def command(key, *args):
                start = time.perf_counter()
                subprocess.run([wb, *map(str, args)], check=True, capture_output=True,
                               text=True, env=environment)
                local_timing[key] = time.perf_counter() - start

            native = staging / f"{hemisphere}.native.func.gii"
            dilated = staging / f"{hemisphere}.native_dilated.func.gii"
            masked = staging / f"{hemisphere}.native_masked.func.gii"
            atlas = staging / f"{hemisphere}.atlas.func.gii"
            final = staging / f"{hemisphere}.32k.func.gii"
            ribbon = ["-volume-to-surface-mapping", clean_t1w, paths["midthickness"],
                      native, "-ribbon-constrained", paths["white"], paths["pial"]]
            if goodvoxels is not None:
                ribbon.extend(("-volume-roi", goodvoxels))
            command(f"{hemisphere}_ribbon", *ribbon)
            command(f"{hemisphere}_dilate", "-metric-dilate", native,
                    paths["midthickness"], 10, dilated, "-nearest")
            command(f"{hemisphere}_native_mask", "-metric-mask", dilated,
                    paths["native_roi"], masked)
            command(f"{hemisphere}_resample", "-metric-resample", masked,
                    paths["registered_sphere"], paths["atlas_sphere"],
                    "ADAP_BARY_AREA", atlas, "-area-surfs", paths["midthickness"],
                    paths["atlas_midthickness"], "-current-roi", paths["native_roi"])
            command(f"{hemisphere}_atlas_mask", "-metric-mask", atlas,
                    paths["atlas_roi"], final)
            return local_timing

        projection_started = time.perf_counter()
        for local_timing in map_hemispheres(project, parallel=parallel, cpu_threads=budget):
            timing.update(local_timing)
        timing["hemisphere_projection"] = time.perf_counter() - projection_started
        start = time.perf_counter()
        dtseries = staging / "space-fsLR_den-91k_bold.dtseries.nii"
        create_fmriprep_cifti(
            clean_mni, staging / "L.32k.func.gii", staging / "R.32k.func.gii",
            left_label, right_label, hcp_dseg, dtseries, tr_seconds=tr,
            parallel=parallel, cpu_threads=budget,
        )
        timing["cifti"] = time.perf_counter() - start
        # QC must own its data before staging is unlinked. A float32
        # np.asarray of an uncompressed CIFTI otherwise retains a live mmap,
        # keeping NFS .nfs files open until after TemporaryDirectory exits.
        image = nib.load(str(dtseries), mmap=False, keep_file_open=False)
        data = np.asarray(image.dataobj, dtype=np.float32)
        report = staging / "coverage.json"
        report.write_text(json.dumps({
            "frames": int(data.shape[0]), "grayordinates": int(data.shape[1]),
            "varying_grayordinates": int((np.ptp(data, axis=0) > 0).sum()),
            "nonfinite_values": int(np.count_nonzero(~np.isfinite(data))),
            "tr_seconds": float(image.header.get_axis(0).step),
            "execution": {"parallel": parallel and budget > 1, "cpu_threads": budget},
            "timing_seconds": timing,
        }, indent=2) + "\n", encoding="utf-8")
        _publish_projection(staging, output, names, overwrite)
    return SurfaceProjectionResult(
        output / "space-fsLR_den-91k_bold.dtseries.nii", output / "L.32k.func.gii",
        output / "R.32k.func.gii", Path(clean_mni), timing, output / "coverage.json",
    )
