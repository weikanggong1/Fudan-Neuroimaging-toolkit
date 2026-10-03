#!/usr/bin/env python3
"""比较具名的完整公开 raw→volume/recon/surface 产物，不运行生产算法。

CLI: --manifest PRIVATE.json --output-root NEW_DIRECTORY
私有 manifest 包含 cohort_id、case_id、frames、tr_seconds、raw.{t1w,bold}（path/sha256）、
brain_mask（path/sha256）、cifti_axis_assets.{left_roi,right_roi,dseg}（path/sha256）、
candidate/reference（report/files 的 JSON 路径）。
files JSON 必须具名 preproc_mni/dtseries；不按 glob 寻找或替换失败结果。
公开输出仅含许可允许的公共 case ID、数值、定义及哈希，逐点图数组保持私有。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

import nibabel as nib
from nibabel.cifti2.cifti2_axes import BrainModelAxis, SeriesAxis
import numpy as np


MASK_SHA256 = "e4e2b284170271afdafe26ac0997b2af5a0f5ddac35e28a7e796b52e8bc5adb1"
AXIS_ASSET_SHA256 = {
    "left_roi": "4ac9199dab151ccdc2a35bdddb5bac4f4907da7dfebd90807c09b82e2eb9d512",
    "right_roi": "698f46b399f5a89829f83cc697e32dc8841031fbf31ffef75aaaa0c4a16c3015",
    "dseg": "9c25e63edec37b3876756b749a3f0127511c6b63bf2855060a44007bb479b987",
}
CASES = ("CON01", "CON03", "CON04", "CON05", "CON06", "CON07", "CON08", "CON09", "CON10", "CON11")
SUBCORTEX = (
    "ACCUMBENS_LEFT", "ACCUMBENS_RIGHT", "AMYGDALA_LEFT", "AMYGDALA_RIGHT",
    "BRAIN_STEM", "CAUDATE_LEFT", "CAUDATE_RIGHT", "CEREBELLUM_LEFT", "CEREBELLUM_RIGHT",
    "DIENCEPHALON_VENTRAL_LEFT", "DIENCEPHALON_VENTRAL_RIGHT", "HIPPOCAMPUS_LEFT",
    "HIPPOCAMPUS_RIGHT", "PALLIDUM_LEFT", "PALLIDUM_RIGHT", "PUTAMEN_LEFT",
    "PUTAMEN_RIGHT", "THALAMUS_LEFT", "THALAMUS_RIGHT",
)
STRUCTURES = {"CIFTI_STRUCTURE_" + name for name in ("CORTEX_LEFT", "CORTEX_RIGHT", *SUBCORTEX)}
SUBCORTEX_LABELS = (26, 58, 18, 54, 16, 11, 50, 8, 47, 28, 60, 17, 53, 13, 52, 12, 51, 10, 49)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("manifest/report must be a JSON object")
    return value


def read_bound_json(path):
    payload = Path(path).read_bytes()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("manifest/report must be a JSON object")
    return value, hashlib.sha256(payload).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class NonfiniteInput(ValueError):
    def __init__(self, side, count, start, end):
        super().__init__(f"{side} has {count} nonfinite values in frames {start}:{end}")
        self.public = {"side": side, "nonfinite_values_in_failing_chunk": count,
                       "first_frame": start, "last_frame_exclusive": end}


def checked_file(entry):
    path = Path(entry["path"]).expanduser().resolve()
    digest = sha256(path)
    if digest != entry["sha256"]:
        raise ValueError("manifest file SHA-256 mismatch")
    return path, digest


def cohort_identity(value):
    if not isinstance(value, str) or not value or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for character in value):
        raise ValueError("comparison must declare a public cohort identifier without paths")
    return value


RUN_TIMING_BOUNDARIES = {
    "continuous_api_wall_seconds": "FNIT API call and final target-device synchronization after imports/CUDA initialization; includes automatic volume ICA/AROMA and pipeline publication/cleanup; excludes subsequent wrapper validation",
    "driver_through_saved_output_validation_seconds": "FNIT API-start boundary through saved-image validation, private output binding and raw/config/metadata rechecks; excludes imports/CUDA initialization, monitor join, final source checks and final report save",
    "container_process_wall_seconds": "Official container process creation through exit; wrapper validation and saved QC excluded",
    "continuous_wall_through_saved_QC_seconds": "Official container-start boundary through workflow, runtime extraction, wrapper validation/input-source rechecks and saved primary QC report; preflight and later schema adapter excluded",
    "wall_seconds": "Official alias of continuous_wall_through_saved_QC_seconds; not summed with that same interval",
    "process_wall_seconds": "FNIT queue process creation through driver exit; includes imports, initialization, API, wrapper validation and final report/finalization",
    "schema_adapter_seconds": "Independent later reference schema recheck/adaptation; excluded from original workflow wall",
    "recovered_QC_seconds": "Independent later saved-output verification and binding recovery; original MRI workflow clocks unchanged; waiting gap excluded",
    "recovery_gap_since_original_end_seconds": "Wall span from original wrapper end to later recovery completion; includes recovered_QC_seconds and any intervening wait, not MRI compute time; never add it to recovered_QC_seconds",
    "failed_attempt_wall_seconds": "Failed FNIT attempt from API-start boundary to caught exception; may be null when failure precedes API start",
}


def timing_boundaries(report, timings):
    """Keep the original passed/failed wrapper boundary and later QC separate."""
    boundaries = {key: RUN_TIMING_BOUNDARIES[key] for key in timings}
    if "recovered_QC_seconds" in report:
        state = report.get("continuous_wall_through_saved_QC_boundary_status")
        if state not in ("original filename-check failed", "original QC passed"):
            raise ValueError("recovered reference must declare the unchanged original wrapper boundary")
        if state == "original filename-check failed":
            for key in ("continuous_wall_through_saved_QC_seconds", "wall_seconds"):
                if key in boundaries:
                    boundaries[key] = "Original container-start through wrapper filename-binding/QC failure and saved original report; later complete output recovery and elapsed recovery span excluded"
    return boundaries


def reference_recovery_provenance(report):
    if "recovered_QC_seconds" not in report:
        return None
    recovered = report.get("recovered_saved_QC", {})
    original_status = report.get("original_harness_status")
    errors = report.get("original_harness_QC_errors")
    boundary = report.get("continuous_wall_through_saved_QC_boundary_status")
    failed_binding = (original_status == "failed" and errors == ["Expected two fsnative BOLD GIFTI outputs, observed 0"]
                      and boundary == "original filename-check failed")
    passed_binding = original_status == "complete" and errors == [] and boundary == "original QC passed"
    if (not (failed_binding or passed_binding)
            or report.get("status") != "complete"
            or report.get("command_exit_code") != 0 or report.get("container_exit_code") != 0
            or report.get("source_unchanged_during_run") is not True
            or report.get("input_unchanged_during_run") is not True
            or recovered.get("status") != "passed"
            or recovered.get("case_id") != report.get("subject")
            or recovered.get("original_report_sha256") != report.get("original_report_sha256")
            or any(recovered.get(key) is not True for key in
                   ("saved_file_guards_equal", "source_guards_equal", "original_report_guard_equal"))
            or recovered.get("frames") != report.get("frames")
            or not np.isclose(recovered.get("repetition_time", np.nan), report.get("repetition_time", np.nan))):
        raise ValueError("reference recovery must bind complete saved checks and the unchanged passed/failed original report")
    for digest in (report.get("original_report_sha256"), recovered.get("validator_sha256")):
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("reference recovery must bind valid original-report and validator SHA-256")
    if "corrective_QC_seconds" in report and report["corrective_QC_seconds"] != report["recovered_QC_seconds"]:
        raise ValueError("corrective QC alias differs from its original recovered duration")
    provenance = {"original_report_sha256": report["original_report_sha256"],
            "original_harness_status": original_status, "recovered_saved_output_status": "passed",
            "validator_sha256": recovered["validator_sha256"],
            "saved_file_guards_equal": True, "source_guards_equal": True,
            "original_report_guard_equal": True, "original_continuous_boundary": boundary,
            "timing_scope": "later full saved-output checks; original workflow clocks immutable"}
    for key in ("corrective_harness_sha256", "corrective_validator_sha256"):
        if key in report:
            provenance[key] = report[key]
    if "corrective_QC_seconds" in report:
        provenance["corrective_QC_seconds_alias"] = "identical to recovered_QC_seconds; never summed"
    return provenance


def run_files(spec, raw_hashes, frames, tr, *, reference=False, case_id=None):
    report_path, files_path = (Path(spec[key]).expanduser().resolve() for key in ("report", "files"))
    report, report_digest = read_bound_json(report_path)
    files, files_digest = read_bound_json(files_path)
    if ((spec.get("report_sha256") is not None and spec["report_sha256"] != report_digest)
            or (spec.get("files_sha256") is not None and spec["files_sha256"] != files_digest)):
        raise ValueError("completed report/files differ from their explicit cohort bindings")
    if case_id is not None and report.get("subject") != case_id:
        raise ValueError("completed run subject differs from its declared public comparison case")
    if report.get("status") != "complete" or report.get("source_unchanged_during_run") is not True:
        raise ValueError("comparison requires a completed run with unchanged execution sources")
    if report.get("input_sha256") != raw_hashes:
        raise ValueError("completed run raw T1w/BOLD hashes differ from the paired manifest")
    if report.get("frames") != frames or not np.isclose(report.get("repetition_time", np.nan), tr, rtol=1e-6, atol=1e-7):
        raise ValueError("completed run does not cover the full raw frame count/TR")
    if not reference and (report.get("volume_executed") is not True or report.get("backend") != "fnit"):
        raise ValueError("candidate must execute automatic volume and FNIT reconstruction from raw inputs")
    if not reference:
        for key in ("raw_inputs_unchanged", "configuration_unchanged", "driver_unchanged"):
            if key in report and report[key] is not True:
                raise ValueError("completed candidate contradicts its named input/configuration/driver guard")
    if reference and report.get("command_exit_code") != 0:
        raise ValueError("reference official complete command must exit zero")
    if reference and report.get("input_unchanged_during_run") is not True:
        raise ValueError("reference must verify unchanged raw inputs during its complete workflow")
    versions = report.get("software_versions", {})
    if reference and (versions.get("fmriprep") != "25.2.4" or versions.get("freesurfer") != "7.3.2"):
        raise ValueError("reference report must verify fMRIPrep 25.2.4 and FreeSurfer 7.3.2")
    outputs, output_digests = {}, {}
    for key in ("preproc_mni", "dtseries"):
        path = Path(files[key]).expanduser().resolve()
        digest = sha256(path)
        if report.get("output_checks", {}).get(key, {}).get("sha256") != digest:
            raise ValueError("named completed output does not match its report SHA-256")
        outputs[key] = path
        output_digests[key] = digest
    timings = {}
    for key in ("continuous_api_wall_seconds", "driver_through_saved_output_validation_seconds",
                "container_process_wall_seconds", "wall_seconds", "continuous_wall_through_saved_QC_seconds", "schema_adapter_seconds",
                "recovered_QC_seconds", "recovery_gap_since_original_end_seconds"):
        if key in report:
            value = report[key]
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value) or value < 0:
                raise ValueError("completed report has an invalid named duration")
            timings[key] = value
    provenance = {"report_sha256": report_digest, "files_manifest_sha256": files_digest,
                     "source_revision": report.get("source_revision"),
                     "driver_sha256": report.get("driver_sha256"),
                     "driver_sha256_after": report.get("driver_sha256_after"),
                     "configuration_sha256": report.get("configuration_sha256"),
                     "execution_guards": {key: report[key] for key in
                         ("raw_inputs_unchanged", "configuration_unchanged", "driver_unchanged") if key in report},
                     "launcher_sha256": report.get("launcher_sha256"),
                     "container_SIF_sha256": report.get("SIF_sha256"),
                     "canonical_command_sha256": report.get("canonical_command_sha256"),
                     "software_versions": {key: versions[key] for key in
                         ("fmriprep", "freesurfer", "smriprep", "nipype", "nibabel", "python") if key in versions},
                     "reported_timing_seconds": timings,
                     "timing_boundaries": timing_boundaries(report, timings),
                     "timing_boundary": (
                         "Official MRI process completed; original wrapper filename/QC binding failed, and later corrective QC is a separate interval with waiting gap excluded"
                         if reference and report.get("continuous_wall_through_saved_QC_boundary_status") == "original filename-check failed" else
                         "Official continuous workflow through completed wrapper output validation and saved primary QC; preflight and later schema adaptation excluded"
                         if reference else
                         "FNIT API and driver-through-validation intervals reported separately; both start after imports/CUDA initialization. Automatic mature volume includes ICA/AROMA clean work. The API excludes subsequent wrapper validation; the driver validation interval still excludes monitor/source finalization and final report save"),
                     "output_sha256": output_digests}
    if "recovered_QC_seconds" in report:
        provenance["reference_qc_recovery"] = reference_recovery_provenance(report)
    return outputs, provenance, (report_path, files_path)


def fixed_brain_axis(assets):
    """Bind index identities to fixed HCP ROIs and the original TF dseg.

    Brain model order is NiWorkflows' documented cortex-L, cortex-R and 19
    structure order; subcortical indices advance i before j before k in LAS.
    This checks the published coordinate contract without importing FNIT.
    """
    paths = {}
    for key, expected in AXIS_ASSET_SHA256.items():
        paths[key], actual = checked_file(assets[key])
        if actual != expected:
            raise ValueError("CIFTI axis asset differs from its fixed original-source SHA-256")
    models = []
    for hemi, key, count in (("LEFT", "left_roi", 29696), ("RIGHT", "right_roi", 29716)):
        image = nib.load(str(paths[key]))
        if not isinstance(image, nib.GiftiImage) or len(image.darrays) != 1:
            raise ValueError("fixed cortical ROI must be a single-array GIFTI")
        values = np.asarray(image.darrays[0].data)
        if values.shape != (32492,) or not np.isfinite(values).all():
            raise ValueError("fixed cortical ROI must contain finite fsLR32k vertex values")
        vertices = np.flatnonzero(values > 0)
        if vertices.size != count:
            raise ValueError("fixed cortical ROI has an invalid vertex count")
        models.append(BrainModelAxis.from_surface(vertices, 32492, name=f"CIFTI_STRUCTURE_CORTEX_{hemi}"))
    image = nib.load(str(paths["dseg"]))
    orientation = nib.orientations.ornt_transform(nib.orientations.io_orientation(image.affine),
                                                  nib.orientations.axcodes2ornt(("L", "A", "S")))
    labels = image.as_reoriented(orientation)
    values = np.asarray(labels.dataobj)
    if labels.ndim != 3 or not np.isfinite(values).all() or (values < 0).any() or not np.equal(values, np.floor(values)).all():
        raise ValueError("fixed HCP dseg must contain finite nonnegative integer labels")
    for name, label in zip(SUBCORTEX, SUBCORTEX_LABELS):
        # np.argwhere on transposed data gives the documented k/j/i order.
        voxels = np.argwhere(values.transpose(2, 1, 0) == label)[:, ::-1]
        if not len(voxels):
            raise ValueError("fixed HCP dseg lacks a required subcortical structure")
        models.append(BrainModelAxis(f"CIFTI_STRUCTURE_{name}", voxel=voxels,
                                    affine=labels.affine, volume_shape=labels.shape))
    axis = models[0]
    for model in models[1:]:
        axis = axis + model
    if len(axis) != 91282:
        raise ValueError("fixed CIFTI assets do not define exactly 91,282 grayordinates")
    return axis, paths


def brain_axes_equal(a, b):
    return (np.array_equal(a.name, b.name) and np.array_equal(a.vertex, b.vertex)
            and np.array_equal(a.voxel, b.voxel) and a.nvertices == b.nvertices
            and a.volume_shape == b.volume_shape and np.array_equal(a.affine, b.affine))


def finite(values, side, start, end):
    count = int(values.size - np.count_nonzero(np.isfinite(values)))
    if count:
        raise NonfiniteInput(side, count, start, end)


def distribution(values):
    values = np.asarray(values, np.float64)
    if not values.size:
        return {"count": 0, "mean": None, "median": None, "p05": None, "p95": None,
                "minimum": None, "maximum": None}
    return {"count": int(values.size), "mean": float(values.mean()), "median": float(np.median(values)),
            "p05": float(np.percentile(values, 5)), "p95": float(np.percentile(values, 95)),
            "minimum": float(values.min()), "maximum": float(values.max())}


def stream_statistics(reader, points, frames, *, chunk_size=8):
    """Stable merge of per-chunk centered moments; all point/frame errors retained.

    reader(start,end) returns paired point×frame arrays. With 228,483 voxels
    and 180 frames, exact absolute-error percentiles need about 329 MB FP64.
    The routine uses CPU validation arrays and never allocates a GPU tensor.
    """
    mx, my, xx, yy, xy, squared_error = [np.zeros(points, np.float64) for _ in range(6)]
    low_x, low_y = np.full(points, np.inf), np.full(points, np.inf)
    high_x, high_y = np.full(points, -np.inf), np.full(points, -np.inf)
    identical = np.ones(points, bool)
    absolute = np.empty((points, frames), np.float64)
    count = 0
    for start in range(0, frames, chunk_size):
        end = min(start + chunk_size, frames)
        x, y = (np.asarray(value, np.float64) for value in reader(start, end))
        if x.shape != (points, end - start) or y.shape != x.shape:
            raise ValueError("paired full-frame reader returned incompatible dimensions")
        finite(x, "candidate", start, end)
        finite(y, "reference", start, end)
        bx, by = x.mean(axis=1), y.mean(axis=1)
        dx, dy = bx - mx, by - my
        xc, yc = x - bx[:, None], y - by[:, None]
        width = end - start
        weight = count * width / (count + width)
        xx += np.einsum("ij,ij->i", xc, xc) + dx * dx * weight
        yy += np.einsum("ij,ij->i", yc, yc) + dy * dy * weight
        xy += np.einsum("ij,ij->i", xc, yc) + dx * dy * weight
        mx += dx * width / (count + width)
        my += dy * width / (count + width)
        difference = x - y
        absolute[:, start:end] = np.abs(difference)
        squared_error += np.einsum("ij,ij->i", difference, difference)
        low_x = np.minimum(low_x, x.min(axis=1)); high_x = np.maximum(high_x, x.max(axis=1))
        low_y = np.minimum(low_y, y.min(axis=1)); high_y = np.maximum(high_y, y.max(axis=1))
        identical &= np.all(x == y, axis=1)
        count += width
    constant_x, constant_y = low_x == high_x, low_y == high_y
    valid = ~constant_x & ~constant_y & (xx > 0) & (yy > 0)
    r = np.full(points, np.nan)
    r[valid] = np.clip(xy[valid] / np.sqrt(xx[valid] * yy[valid]), -1, 1)
    r[valid & identical] = 1.
    sd_x, sd_y = np.sqrt(np.maximum(xx, 0) / frames), np.sqrt(np.maximum(yy, 0) / frames)
    tsnr_x, tsnr_y = np.full(points, np.nan), np.full(points, np.nan)
    np.divide(mx, sd_x, out=tsnr_x, where=~constant_x & (sd_x > 0))
    np.divide(my, sd_y, out=tsnr_y, where=~constant_y & (sd_y > 0))
    return {"r": r, "mean_candidate": mx, "mean_reference": my,
            "sd_candidate": sd_x, "sd_reference": sd_y, "tsnr_candidate": tsnr_x,
            "tsnr_reference": tsnr_y, "constant_candidate": constant_x,
            "constant_reference": constant_y, "zero_candidate": (low_x == 0) & (high_x == 0),
            "zero_reference": (low_y == 0) & (high_y == 0), "absolute": absolute,
            "squared_error": squared_error, "reference_energy": yy + frames * my**2,
            "frames": frames}


def summarize(stats, selection=slice(None)):
    r = stats["r"][selection]
    cx, cy = (stats[key][selection] for key in ("constant_candidate", "constant_reference"))
    zx, zy = (stats[key][selection] for key in ("zero_candidate", "zero_reference"))
    error = stats["absolute"][selection]
    mse = float(stats["squared_error"][selection].sum()) / error.size
    ref_rms = np.sqrt(float(stats["reference_energy"][selection].sum()) / error.size)
    mean_difference = stats["mean_candidate"][selection] - stats["mean_reference"][selection]
    tx, ty = (stats[key][selection] for key in ("tsnr_candidate", "tsnr_reference"))
    tv = np.isfinite(tx) & np.isfinite(ty)
    return {"points": int(r.size), "frames": stats["frames"], "values": int(error.size),
            "nonfinite_candidate_values": 0, "nonfinite_reference_values": 0,
            "constant_candidate": int(cx.sum()), "constant_reference": int(cy.sum()),
            "both_constant": int((cx & cy).sum()), "one_side_constant": int((cx ^ cy).sum()),
            "zero_candidate": int(zx.sum()), "zero_reference": int(zy.sum()),
            "both_zero": int((zx & zy).sum()), "one_side_zero": int((zx ^ zy).sum()),
            "temporal_r": distribution(r[np.isfinite(r)]),
            "rmse": float(np.sqrt(mse)), "reference_rms": float(ref_rms),
            "normalized_rmse": float(np.sqrt(mse) / ref_rms) if ref_rms > 0 else None,
            "absolute_difference": {"mean": float(error.mean()), "p99": float(np.percentile(error, 99)),
                                    "maximum": float(error.max())},
            "temporal_mean_bias": {"signed": distribution(mean_difference),
                                   "mean_absolute": float(np.abs(mean_difference).mean())},
            "tsnr": {"candidate": distribution(tx[np.isfinite(tx)]),
                     "reference": distribution(ty[np.isfinite(ty)]), "valid_pairs": int(tv.sum()),
                     "undefined_pairs": int((~tv).sum()), "bias": distribution(tx[tv] - ty[tv])}}


def ras_orientation(image):
    return nib.orientations.ornt_transform(nib.orientations.io_orientation(image.affine),
                                           nib.orientations.axcodes2ornt(("R", "A", "S")))


def volume_grid(image):
    orientation = ras_orientation(image)
    order = np.argsort(orientation[:, 0]).astype(int)
    return tuple(image.shape[i] for i in order), image.affine @ nib.orientations.inv_ornt_aff(orientation, image.shape[:3])


def validate_volume(image, frames, tr):
    if image.ndim != 4 or image.shape[3] != frames:
        raise ValueError("MNI final BOLD must retain all 180 raw frames")
    xyz, unit = image.header.get_xyzt_units()
    scale = {"sec": 1., "msec": .001, "usec": .000001}.get(unit)
    if xyz != "mm" or scale is None or not np.isclose(image.header.get_zooms()[3] * scale, tr, rtol=1e-6, atol=1e-7):
        raise ValueError("MNI final BOLD must declare millimeter geometry and the actual raw TR")


def volume_statistics(candidate, reference, mask, frames, tr):
    x, y = (nib.load(str(path), keep_file_open=True) for path in (candidate, reference))
    for image in (x, y):
        validate_volume(image, frames, tr)
        shape, affine = volume_grid(image)
        if shape != mask.shape or not np.allclose(affine, mask.affine, rtol=0, atol=1e-4):
            raise ValueError("canonical RAS MNI output and fixed brain mask physical grids differ")
    brain = np.asarray(mask.dataobj) > 0
    if not brain.any():
        raise ValueError("fixed MNI brain mask is empty")
    def reader(start, end):
        blocks = []
        for side, image in (("candidate_MNI_full_image", x), ("reference_MNI_full_image", y)):
            values = np.asanyarray(image.dataobj[..., start:end])
            finite(values, side, start, end)
            canonical = nib.orientations.apply_orientation(values, ras_orientation(image))
            blocks.append(canonical[brain])
        return blocks
    stats = stream_statistics(reader, int(brain.sum()), frames)
    arrays = {"brain_mask": brain, "volume_affine": mask.affine}
    for output, key in (("volume_temporal_r", "r"), ("volume_candidate_mean", "mean_candidate"),
                        ("volume_reference_mean", "mean_reference")):
        data = np.full(brain.shape, np.nan if key == "r" else 0., np.float32)
        data[brain] = stats[key].astype(np.float32)
        arrays[output] = data
    report = summarize(stats)
    report.update(domain="entire fixed TemplateFlow MNI152NLin6Asym res-02 brain mask, including zero and constant timeseries",
                  canonical_ras_shape=list(mask.shape), canonical_ras_affine=mask.affine.tolist(),
                  physical_grid_equal=True, index_reorientation_only=True, spatial_interpolation_performed=False)
    report["output_metadata"] = {side: {"stored_dtype": str(image.get_data_dtype()),
        "original_shape": list(image.shape), "original_axis_codes": list(nib.aff2axcodes(image.affine)),
        "spatial_unit": image.header.get_xyzt_units()[0], "temporal_unit": image.header.get_xyzt_units()[1],
        "stored_time_step": float(image.header.get_zooms()[3])}
        for side, image in (("candidate", x), ("reference", y))}
    return report, arrays


def cifti_statistics(candidate, reference, frames, tr, *, mni_grid=None, expected_axis=None):
    x, y = (nib.load(str(path)) for path in (candidate, reference))
    axes = []
    for image in (x, y):
        if not isinstance(image, nib.Cifti2Image) or image.shape != (frames, 91282):
            raise ValueError("final CIFTI must have full 180-by-91282 shape")
        series, brain = image.header.get_axis(0), image.header.get_axis(1)
        if (not isinstance(series, SeriesAxis) or not isinstance(brain, BrainModelAxis)
                or series.unit != "SECOND" or series.start != 0 or series.size != frames
                or not np.isclose(series.step, tr, rtol=1e-6, atol=1e-7)):
            raise ValueError("CIFTI series axis does not match the full raw time axis")
        if (brain.nvertices.get("CIFTI_STRUCTURE_CORTEX_LEFT") != 32492
                or brain.nvertices.get("CIFTI_STRUCTURE_CORTEX_RIGHT") != 32492):
            raise ValueError("CIFTI cortical brain models must use actual fsLR32k vertex identities")
        if {str(name) for name, _, _ in brain.iter_structures()} != STRUCTURES:
            raise ValueError("CIFTI must contain both cortices and all 19 subcortical structures")
        counts = {str(name): len(model) for name, _, model in brain.iter_structures()}
        if (counts["CIFTI_STRUCTURE_CORTEX_LEFT"] != 29696
                or counts["CIFTI_STRUCTURE_CORTEX_RIGHT"] != 29716
                or sum(counts["CIFTI_STRUCTURE_" + name] for name in SUBCORTEX) != 31870):
            raise ValueError("CIFTI brain models must retain the fixed cortical/subcortical 91k counts")
        if mni_grid is not None:
            grid_shape, grid_affine = volume_grid(SimpleNamespace(shape=brain.volume_shape, affine=brain.affine))
            if grid_shape != mni_grid[0] or not np.allclose(grid_affine, mni_grid[1], rtol=0, atol=1e-4):
                raise ValueError("CIFTI subcortical physical volume grid differs from fixed MNI brain-mask grid")
        if expected_axis is not None and not brain_axes_equal(brain, expected_axis):
            raise ValueError("CIFTI brain model indices/order differ from the fixed HCP/TemplateFlow 91k assets")
        axes.append((series, brain))
    a, b = axes[0][1], axes[1][1]
    if not brain_axes_equal(a, b) or axes[0][0] != axes[1][0]:
        raise ValueError("CIFTI time/brain axes differ; shape-only pairing is forbidden")
    stats = stream_statistics(lambda start, end: (x.dataobj[start:end, :].T, y.dataobj[start:end, :].T), 91282, frames)
    report = summarize(stats)
    report.update(brain_axis_exactly_equal=True, series_axis_exactly_equal=True,
                  fixed_original_assets_axis_exactly_equal=expected_axis is not None,
                  time_axis={"start": float(axes[0][0].start), "step": float(axes[0][0].step),
                             "unit": axes[0][0].unit, "size": int(axes[0][0].size)},
                  output_metadata={side: {"stored_dtype": str(image.get_data_dtype()), "shape": list(image.shape)}
                      for side, image in (("candidate", x), ("reference", y))},
                  per_structure={}, cortical_structures=2, subcortical_structures=19)
    arrays = {"cifti_temporal_r": stats["r"].astype(np.float32)}
    axis_digest = hashlib.sha256()
    for value in (a.name.astype("U").tobytes(), a.vertex.astype("<i8").tobytes(),
                  a.voxel.astype("<i8").tobytes(), np.asarray(a.affine, dtype="<f8").tobytes()):
        axis_digest.update(value)
    axis_digest.update(json.dumps({"volume_shape": list(map(int, a.volume_shape)),
                                   "nvertices": {key: int(value) for key, value in a.nvertices.items()}},
                                  sort_keys=True).encode("utf-8"))
    report["brain_axis_identity_sha256"] = axis_digest.hexdigest()
    for name, selection, model in a.iter_structures():
        report["per_structure"][str(name)] = summarize(stats, selection)
        if str(name).startswith("CIFTI_STRUCTURE_CORTEX_"):
            hemi = "left" if str(name).endswith("LEFT") else "right"
            indices = model.vertex
            if len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= 32492):
                raise ValueError("cortical brain-model vertex indices are invalid or duplicated")
            values = np.full(32492, np.nan, np.float32)
            values[indices] = stats["r"][selection].astype(np.float32)
            present = np.zeros(32492, bool); present[indices] = True
            arrays[f"cortex_{hemi}_temporal_r"] = values
            arrays[f"cortex_{hemi}_brain_axis_mask"] = present
    return report, arrays


def compare(manifest, output):
    started = time.perf_counter()
    cohort = cohort_identity(manifest["cohort_id"])
    source_revision = manifest["source_revision"]
    if not isinstance(source_revision, str) or len(source_revision) != 40 or any(c not in "0123456789abcdef" for c in source_revision):
        raise ValueError("comparison must bind the exact candidate scientific source revision")
    candidate_root = Path(manifest["candidate_root"]).expanduser().resolve()
    for key in ("report", "files"):
        if not Path(manifest["candidate"][key]).expanduser().resolve().is_relative_to(candidate_root):
            raise ValueError("candidate report/files escape the explicitly bound cohort root")
    case, frames, tr = manifest["case_id"], manifest["frames"], float(manifest["tr_seconds"])
    if case not in CASES or frames != 180 or not np.isclose(tr, 2.1 if case in CASES[:3] else 2.4):
        raise ValueError("benchmark manifest must bind one of the ten public full-180-frame runs and its true TR")
    script_hash = sha256(__file__)
    raw_files, raw_hashes = {}, {}
    for key in ("t1w", "bold"):
        raw_files[key], raw_hashes[key] = checked_file(manifest["raw"][key])
    raw = nib.load(str(raw_files["bold"]), keep_file_open=True)
    raw_t1 = nib.load(str(raw_files["t1w"]))
    if raw.ndim != 4 or raw.shape[3] != frames or raw_t1.ndim != 3:
        raise ValueError("raw paired T1w/BOLD shapes do not match the complete benchmark")
    finite(np.asarray(raw_t1.dataobj), "raw_T1w_single_volume", 0, 1)
    for first in range(0, frames, 8):
        finite(np.asanyarray(raw.dataobj[..., first:first + 8]), "raw_BOLD", first, min(first + 8, frames))
    if "bold_json" in manifest["raw"]:
        json_path, _ = checked_file(manifest["raw"]["bold_json"])
        if not np.isclose(read_json(json_path).get("RepetitionTime", np.nan), tr, rtol=1e-6, atol=1e-7):
            raise ValueError("raw BOLD JSON TR differs from the full time axis")
        raw_files["bold_json"] = json_path
    mask_path, mask_hash = checked_file(manifest["brain_mask"])
    if mask_hash != MASK_SHA256:
        raise ValueError("comparison mask must be the fixed original TemplateFlow MNI6 res-02 brain mask")
    mask = nib.as_closest_canonical(nib.load(str(mask_path)))
    values = np.asarray(mask.dataobj)
    if mask.ndim != 3 or not np.isfinite(values).all() or not np.isin(values, (0, 1)).all():
        raise ValueError("fixed brain mask must contain finite binary 3D values")
    candidate, candidate_provenance, candidate_records = run_files(manifest["candidate"], raw_hashes, frames, tr, case_id=case)
    if candidate_provenance["source_revision"] != source_revision:
        raise ValueError("completed candidate differs from the explicitly bound scientific source revision")
    if any(not path.is_relative_to(candidate_root) for path in candidate.values()):
        raise ValueError("candidate final outputs escape the explicitly bound cohort root")
    reference, reference_provenance, reference_records = run_files(manifest["reference"], raw_hashes, frames, tr, reference=True, case_id=case)
    expected_axis, axis_assets = fixed_brain_axis(manifest["cifti_axis_assets"])
    for key in candidate:
        if candidate[key].samefile(reference[key]):
            raise ValueError("candidate/reference final outputs must be independent files")
    watched = (*raw_files.values(), mask_path, *candidate.values(), *reference.values(),
               *candidate_records, *reference_records, *axis_assets.values(), Path(__file__).resolve())
    before = {path: sha256(path) for path in watched}
    expected_before = {path: manifest["raw"][key]["sha256"] for key, path in raw_files.items()}
    expected_before[mask_path] = mask_hash
    expected_before.update({path: AXIS_ASSET_SHA256[key] for key, path in axis_assets.items()})
    expected_before[Path(__file__).resolve()] = script_hash
    for outputs, provenance, records in ((candidate, candidate_provenance, candidate_records),
                                          (reference, reference_provenance, reference_records)):
        expected_before.update({path: provenance["output_sha256"][key] for key, path in outputs.items()})
        expected_before[records[0]] = provenance["report_sha256"]
        expected_before[records[1]] = provenance["files_manifest_sha256"]
    if any(before[path] != digest for path, digest in expected_before.items()):
        raise RuntimeError("input/output/report changed between initial identity validation and comparison snapshot")
    volume, arrays = volume_statistics(candidate["preproc_mni"], reference["preproc_mni"], mask, frames, tr)
    cifti, cortical_arrays = cifti_statistics(candidate["dtseries"], reference["dtseries"], frames, tr,
                                             mni_grid=(mask.shape, mask.affine), expected_axis=expected_axis)
    arrays.update(cortical_arrays)
    arrays.update(case_id=np.asarray(case), frames=np.asarray(frames), tr_seconds=np.asarray(tr))
    if any(sha256(path) != digest for path, digest in before.items()):
        raise RuntimeError("input/output/source changed during comparison")
    arrays_path = output / "arrays.private.npz"
    np.savez_compressed(arrays_path, **arrays)
    return {"schema_version": 1, "cohort_id": cohort, "candidate_source_revision_bound": source_revision, "case_id": case, "status": "complete", "frames": frames,
            "dataset": {"id": "ds001226", "version": "5.0.1", "license": "CC0",
                        "git_commit": "359d372c5e972a161966312128adb365870df949"},
            "tr_seconds": tr, "raw_input_sha256": raw_hashes, "fixed_brain_mask_sha256": mask_hash,
            "fixed_cifti_axis_assets_sha256": {key: before[path] for key, path in axis_assets.items()},
            "candidate": candidate_provenance, "reference": reference_provenance,
            "comparison_script_sha256": script_hash, "sources_unchanged_during_comparison": True,
            "software_versions": {"nibabel": nib.__version__, "numpy": np.__version__},
            "method": {"signal": "full minimally preprocessed final outputs",
                       "protocol_difference": "FNIT recon-all fixed FreeSurfer 8.2 algorithms plus FNIRT versus complete fMRIPrep 25.2.4 with FreeSurfer 7.3.2 and ANTs; scientific protocols/algorithm versions differ",
                       "timed_work_difference": "FNIT automatic mature volume computes/saves both preproc and ICA/AROMA clean outputs and all four volumes are checked by its runner; official complete fMRIPrep does not denoise BOLD, and its default anatomical workflow also registers internal MNI152NLin2009cAsym in addition to the requested MNI152NLin6Asym outputs. Accuracy compares only preproc and corresponding CIFTI. Raw whole durations and corresponding steps remain separate; no general speedup inferred",
                       "strict_numerical_equivalence": "not_assessed; metrics describe complete protocol differences",
                       "pearson": "each complete 180-frame nonconstant pair; undefined constant pairs counted, never assigned r=0 or removed from error metrics",
                       "rmse": "sqrt(mean((candidate-reference)^2)) over every point/frame in each named domain, including zeros/constants",
                       "normalized_rmse": "RMSE / sqrt(mean(reference^2)); null if reference RMS=0",
                       "temporal_mean_bias": "candidate temporal mean minus reference temporal mean, per point",
                       "tsnr": "temporal mean / population temporal SD (ddof=0); constant-series tSNR undefined and counted",
                       "absolute_p99": "exact percentile across all point/frame absolute errors",
                       "memory": "CPU NumPy validation with 8-frame reads and bounded exact-error buffers; no production GPU timing included"},
            "volume": volume, "cifti": cifti,
            "reconstruction_geometry": {"status": "not_assessed_here",
                                        "reason": "independent native reconstructions are not assumed to share topology or vertex correspondence; same-index mesh and sphere-angle errors require a separate validated geometry comparison"},
            "derived_arrays": {"file": arrays_path.name, "sha256": sha256(arrays_path),
                               "publication": "private derived arrays; only CC0 brain-masked figures and aggregate numbers approved for publication"},
            "comparison_wall_seconds": time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    manifest = {}
    try:
        manifest_hash = sha256(args.manifest)
        manifest = read_json(args.manifest)
        report = compare(manifest, args.output_root)
        if sha256(args.manifest) != manifest_hash:
            raise RuntimeError("comparison manifest changed during execution")
        report["private_manifest_sha256"] = manifest_hash
    except Exception as error:
        (args.output_root / "failure.private.txt").write_text(traceback.format_exc())
        report = {"schema_version": 1, "status": "failed",
                  "case_id": manifest.get("case_id") if manifest.get("case_id") in CASES else None,
                  "comparison_script_sha256": sha256(__file__),
                  "failure": {"type": type(error).__name__, "details": "failure.private.txt"}}
        try:
            report["cohort_id"] = cohort_identity(manifest.get("cohort_id"))
        except ValueError:
            pass
        if isinstance(error, NonfiniteInput):
            report["failure"].update(error.public)
    write_json(args.output_root / "comparison.public.json", report)
    print(json.dumps(report, allow_nan=False))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
