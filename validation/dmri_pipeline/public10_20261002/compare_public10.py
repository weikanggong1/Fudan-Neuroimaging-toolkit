#!/usr/bin/env python3
"""公开十人双分支的独立输出比较及完整计划汇总。

只读取已生成的影像、梯度和报告，不运行模型、不插值、不选择成功病例。
compare（可省略子命令）比较单人单分支；aggregate 读取 planned_cases 清单。
报告只保存匿名 caseNN、相对角色、数值与哈希，不保存输入目录或异常原文。
完整 ROI 的误差是主要结果，共同非零支持仅作补充；不预设数值等价阈值。
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import sys

import nibabel as nib
import numpy as np


MAP_NAMES = ("FA", "MD", "L1", "L2", "L3", "MO", "ICVF", "OD", "ISOVF")
BACKENDS = ("tbss", "mmorf")
AFFINE_TOLERANCE = 1e-5
P95_MAX_SAMPLES = 1_000_000
P95_SEED = 1729


def clean_numbers(value):
    if isinstance(value, dict):
        return {key: clean_numbers(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [clean_numbers(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_image(path):
    try:
        image = nib.load(str(path))
        # 解码 slope/intercept；一次读取 gzip 4D，避免每卷反复解压。
        values = np.asarray(image.dataobj, dtype=np.float32)
        if values.ndim == 5 and values.shape[3] == 1:
            # NIfTI 向量常以 (x,y,z,1,channels) 保存；只移除单例轴。
            values = values[:, :, :, 0, :]
        if values.ndim == 4 and values.shape[-1] == 1:
            values = values[..., 0]
        metadata = {
            "available": True, "raw_shape": list(image.shape),
            "comparison_shape": list(values.shape), "stored_dtype": str(image.get_data_dtype()),
            "decoded_dtype": "float32", "affine": image.affine.tolist(),
            "voxel_sizes_mm": list(map(float, image.header.get_zooms()[:3])),
            "spatial_units": image.header.get_xyzt_units()[0],
            "nonfinite_elements": int(np.count_nonzero(~np.isfinite(values))),
            "size_bytes": Path(path).stat().st_size, "sha256": file_hash(path),
            "proxy_slope": float(getattr(image.dataobj, "slope", 1.0)),
            "proxy_intercept": float(getattr(image.dataobj, "inter", 0.0)),
        }
        for form in ("qform", "sform"):
            matrix, code = getattr(image, "get_" + form)(coded=True)
            metadata[form + "_code"] = int(code)
            metadata[form] = matrix.tolist() if matrix is not None else None
        return {"image": image, "values": values, "metadata": clean_numbers(metadata)}
    except Exception as error:
        return {"image": None, "values": None,
                "metadata": {"available": False, "error_type": type(error).__name__}}


def geometry_gate(left, right, *, spatial_only=False):
    if left["values"] is None or right["values"] is None:
        return {"passed": False, "reasons": ["image_unavailable"]}
    left_shape, right_shape = left["values"].shape, right["values"].shape
    a, b = left["image"].affine, right["image"].affine
    finite_affine = bool(np.isfinite(a).all() and np.isfinite(b).all())
    checks = {
        "shape_equal": left_shape[:3] == right_shape[:3] if spatial_only else left_shape == right_shape,
        "affine_finite": finite_affine,
        "affine_equal": bool(finite_affine and np.allclose(a, b, atol=AFFINE_TOLERANCE, rtol=0)),
        "decoded_values_finite": left["metadata"]["nonfinite_elements"] == right["metadata"]["nonfinite_elements"] == 0,
    }
    return {"passed": all(checks.values()), **checks,
            "reasons": [key for key, passed in checks.items() if not passed]}


def support(left, right):
    a, b = int(left.sum()), int(right.sum())
    intersection = int(np.count_nonzero(left & right))
    union = a + b - intersection
    return {"candidate_voxels": a, "reference_voxels": b,
            "intersection_voxels": intersection, "union_voxels": union,
            "dice": 2.0 * intersection / (a + b) if a + b else 1.0,
            "jaccard": intersection / union if union else 1.0,
            "both_empty": a + b == 0}


class Statistics:
    """按卷累计精确中心矩，只有巨大 4D 的 p95 使用固定抽样。"""
    def __init__(self, total, *, sample_limit=P95_MAX_SAMPLES):
        self.total = int(total)
        n = min(self.total, sample_limit)
        self.positions = np.arange(n, dtype=np.int64) if n == self.total else np.sort(
            np.random.default_rng(P95_SEED).choice(self.total, n, replace=False))
        self.samples = np.empty(n, dtype=np.float64)
        self.count = self.different = 0
        self.mx = self.my = self.m2x = self.m2y = self.cross = 0.0
        self.sum_abs = self.sum_sq = self.sum_diff = self.sum_ref_sq = self.maximum = 0.0

    def update(self, left, right):
        x, y = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
        if not x.size:
            return
        n, previous = x.size, self.count
        mx, my = float(x.mean()), float(y.mean())
        dx, dy = x - mx, y - my
        delta_x, delta_y = mx - self.mx, my - self.my
        combined = previous + n
        weight = previous * n / combined
        self.m2x += float(np.dot(dx, dx)) + delta_x**2 * weight
        self.m2y += float(np.dot(dy, dy)) + delta_y**2 * weight
        self.cross += float(np.dot(dx, dy)) + delta_x * delta_y * weight
        self.mx += delta_x * n / combined
        self.my += delta_y * n / combined
        difference = x - y
        absolute = np.abs(difference)
        self.sum_abs += float(absolute.sum())
        self.sum_sq += float(np.dot(difference, difference))
        self.sum_diff += float(difference.sum())
        self.sum_ref_sq += float(np.dot(y, y))
        self.maximum = max(self.maximum, float(absolute.max()))
        self.different += int(np.count_nonzero(difference))
        start, end = np.searchsorted(self.positions, [previous, combined])
        self.samples[start:end] = absolute[self.positions[start:end] - previous]
        self.count = combined

    def result(self):
        if self.count != self.total:
            raise ValueError("statistics_count_mismatch")
        varying = self.count >= 2 and self.m2x > 0 and self.m2y > 0
        rmse = math.sqrt(self.sum_sq / self.count) if self.count else None
        reference_rms = math.sqrt(self.sum_ref_sq / self.count) if self.count else None
        return {
            "elements": self.count, "different_elements": self.different,
            "values_exact": self.different == 0, "candidate_mean": self.mx if self.count else None,
            "reference_mean": self.my if self.count else None,
            "pearson_r": float(np.clip(self.cross / math.sqrt(self.m2x * self.m2y), -1, 1)) if varying else None,
            "correlation_status": "computed" if varying else "empty_constant_or_single_element",
            "mae": self.sum_abs / self.count if self.count else None,
            "rmse": rmse, "mean_signed_difference": self.sum_diff / self.count if self.count else None,
            "reference_rms": reference_rms,
            "rmse_over_reference_rms": rmse / reference_rms if reference_rms else None,
            "max_absdiff": self.maximum if self.count else None,
            "p95_absdiff": float(np.percentile(self.samples, 95)) if self.samples.size else None,
            "p95_sampling": {"method": "all_elements" if self.positions.size == self.total else
                             "uniform_without_replacement_over_selected_voxel_volume_elements",
                             "sample_elements": int(self.positions.size), "population_elements": self.total,
                             "seed": None if self.positions.size == self.total else P95_SEED,
                             "other_statistics_use_all_elements": True},
        }


def volume_arrays(values):
    if values.ndim == 3:
        yield values
    else:
        for indices in np.ndindex(values.shape[3:]):
            yield values[(slice(None),) * 3 + indices]


def statistics(left, right, regions):
    counts = {name: 0 for name in regions}
    for x, y in zip(volume_arrays(left), volume_arrays(right)):
        for name, mask in regions.items():
            selected = ((x != 0) | (y != 0)) if mask is None else mask
            counts[name] += int(selected.sum())
    # 3D 脑区图的 p95 也用全体素；大 4D 才抽样。
    limit = max(left.size, 1) if left.ndim == 3 else P95_MAX_SAMPLES
    accumulators = {name: Statistics(count, sample_limit=limit) for name, count in counts.items()}
    for x, y in zip(volume_arrays(left), volume_arrays(right)):
        for name, mask in regions.items():
            selected = ((x != 0) | (y != 0)) if mask is None else mask
            accumulators[name].update(x[selected], y[selected])
    return {name: accumulator.result() for name, accumulator in accumulators.items()}


def image_pair(left, right, region=None, *, common_region=None):
    gate = geometry_gate(left, right)
    result = {"candidate": left["metadata"], "reference": right["metadata"],
              "gate": gate, "statistics": None, "support": None}
    if region is not None:
        roi_gate = geometry_gate(left, region, spatial_only=True)
        gate["fixed_roi_grid_and_finite"] = roi_gate["passed"]
        if region["values"] is not None and region["values"].ndim != 3:
            roi_gate["passed"] = False
        if not roi_gate["passed"]:
            gate["passed"] = False
            gate["reasons"].append("fixed_roi_grid_or_finite_failed")
        elif not np.count_nonzero(region["values"]):
            gate["passed"] = False
            gate["reasons"].append("fixed_roi_empty")
    if not gate["passed"]:
        return result
    x, y = left["values"], right["values"]
    fixed = region["values"].astype(bool) if region is not None else np.ones(x.shape[:3], bool)
    regions = {"fixed_roi": fixed, "nonzero_union": None}
    if x.ndim == 3:
        regions["common_nonzero_support_in_fixed_roi"] = fixed & (x != 0) & (y != 0)
        result["support"] = {"whole_grid": support(x != 0, y != 0),
                             "fixed_roi": support((x != 0) & fixed, (y != 0) & fixed)}
    if common_region is not None:
        common_gate = geometry_gate(left, common_region, spatial_only=True)
        result["common_brain_roi_gate"] = common_gate
        if common_gate["passed"]:
            regions["common_brain_masks"] = common_region["values"].astype(bool)
    result["statistics"] = statistics(x, y, regions)
    return result


def derived_mask(source, values):
    if (values is None or source["values"] is None
            or source["metadata"].get("nonfinite_elements", 1) != 0):
        return {"image": None, "values": None, "metadata": {"available": False}}
    return {"image": source["image"], "values": values.astype(bool),
            "metadata": {**source["metadata"], "nonfinite_elements": 0, "derived_boolean_mask": True}}


def compare_mask(left, right):
    gate = geometry_gate(left, right)
    return {"candidate": left["metadata"], "reference": right["metadata"], "gate": gate,
            "threshold": 0.5,
            "support": support(left["values"] > .5, right["values"] > .5) if gate["passed"] else None}


def map_path(root, backend, space, name, *, reference=False):
    if space == "native":
        prefix = "NODDI_" if name in ("ICVF", "OD", "ISOVF") else "dti_"
        return root / "native" / f"{prefix}{name}.nii.gz"
    if reference and backend == "tbss":
        suffix = "_skeletonised" if space == "skeleton" else ""
        return root / "tbss" / "stats" / f"all_{name}{suffix}.nii.gz"
    if reference and backend == "mmorf":
        return root / "mmorf" / space / f"{name}.nii.gz"
    return root / "registration" / space / f"{name}.nii.gz"


def text_pair(left_path, right_path):
    try:
        x, y = (np.loadtxt(path, dtype=np.float64, ndmin=2) for path in (left_path, right_path))
        valid = x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all()
        return {"status": "compared" if valid else "shape_or_finite_failed",
                "candidate_shape": list(x.shape), "reference_shape": list(y.shape),
                "values_exact": bool(np.array_equal(x, y)) if valid else None,
                "max_absdiff": float(np.abs(x - y).max()) if valid and x.size else None}
    except Exception as error:
        return {"status": "unavailable", "error_type": type(error).__name__}


def gradients(candidate, reference, bvals_path):
    try:
        bvals = np.loadtxt(bvals_path, dtype=np.float64).reshape(-1)
        arrays = [np.loadtxt(root / "eddy/data.eddy_rotated_bvecs", dtype=np.float64, ndmin=2)
                  for root in (candidate, reference)]
        arrays = [array.T if array.shape == (bvals.size, 3) and array.shape != (3, bvals.size)
                  else array for array in arrays]
        if any(array.shape != (3, bvals.size) or not np.isfinite(array).all() for array in arrays) or not np.isfinite(bvals).all():
            return {"status": "shape_or_finite_failed"}
        x, y = arrays
        nx, ny = np.linalg.norm(x, axis=0), np.linalg.norm(y, axis=0)
        dw = bvals > 100
        selected = dw & (nx > 0) & (ny > 0)
        dot = np.sum(x[:, selected] * y[:, selected], axis=0) / (nx[selected] * ny[selected])
        direct = np.degrees(np.arccos(np.clip(dot, -1, 1)))
        antipodal = np.degrees(np.arccos(np.clip(np.abs(dot), 0, 1)))
        def angle_summary(values):
            return {"mean": float(values.mean()), "rms": float(np.sqrt(np.mean(values**2))),
                    "p95": float(np.percentile(values, 95)), "max": float(values.max())} if values.size else None
        return {"status": "compared", "bval_volumes": int(bvals.size),
                "selected_dwi_volumes": int(dw.sum()), "compared_dwi_volumes": int(selected.sum()),
                "excluded_zero_direction_volumes": int((dw & ~selected).sum()),
                "volume_indices": np.flatnonzero(selected).tolist(),
                "directed_angle_degrees": angle_summary(direct),
                "antipodal_angle_degrees": angle_summary(antipodal),
                "directed_angle_degrees_by_volume": direct.tolist(),
                "antipodal_angle_degrees_by_volume": antipodal.tolist(),
                "rule": "bval >100; unit-normalize; antipodal acos(abs(dot)) respects diffusion-axis sign symmetry"}
    except Exception as error:
        return {"status": "unavailable", "error_type": type(error).__name__}


def scaled_mm(image):
    sizes = np.asarray(image.header.get_zooms()[:3], dtype=np.float64)
    matrix = np.diag([*sizes, 1.0])
    if np.linalg.det(image.affine[:3, :3]) > 0:
        matrix[0, 0] *= -1
        matrix[0, 3] = (image.shape[0] - 1) * sizes[0]
    return matrix


def affine_pair(left_path, right_path, moving, fixed):
    result = text_pair(left_path, right_path)
    result["coordinates"] = "moving-to-fixed FSL scaled-mm"
    if result["status"] != "compared":
        return result
    x, y = (np.loadtxt(path) for path in (left_path, right_path))
    if x.shape != (4, 4) or not all(np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0) for matrix in (x, y)):
        result["status"] = "invalid_homogeneous_matrix"
        return result
    result.update(candidate_matrix=x.tolist(), reference_matrix=y.tolist())
    if moving is None or fixed is None:
        result["physical_displacement"] = {"status": "geometry_unavailable"}
        return result
    last = np.array(moving.shape[:3], dtype=np.float64) - 1
    points = np.array([*itertools.product(*[(0., float(v)) for v in last]), tuple(last / 2)])
    source = np.vstack((points.T, np.ones(9)))
    a = fixed.affine @ np.linalg.inv(scaled_mm(fixed)) @ x @ scaled_mm(moving) @ source
    b = fixed.affine @ np.linalg.inv(scaled_mm(fixed)) @ y @ scaled_mm(moving) @ source
    distances = np.linalg.norm((a - b)[:3], axis=0)
    result["physical_displacement"] = {
        "status": "compared", "point_count": 9,
        "rule": "eight acquired voxel-center corners and image center; not whole-brain RMS or FSL rmsdiff",
        "units": "NIfTI world RAS millimetres", "distances_mm": distances.tolist(),
        "mean_mm": float(distances.mean()), "rms_mm": float(np.sqrt(np.mean(distances**2))),
        "max_mm": float(distances.max())}
    return result


def compare_case(args):
    candidate, reference = args.candidate_dir, args.reference_dir
    cmask = load_image(candidate / "eddy/nodif_brain_mask.nii.gz")
    rmask = load_image(reference / "eddy/nodif_brain_mask.nii.gz")
    masks_gate = geometry_gate(cmask, rmask)
    native_roi = derived_mask(rmask, rmask["values"] > .5 if rmask["values"] is not None else None)
    common = derived_mask(rmask, (rmask["values"] > .5) & (cmask["values"] > .5)
                          if masks_gate["passed"] else None)
    template = load_image(args.reference_roi)
    template_roi = derived_mask(template, template["values"] != 0 if template["values"] is not None else None)
    spaces = ["native", "standard"] + (["skeleton"] if args.backend == "tbss" else [])
    rois = {"native": native_roi, "standard": template_roi}
    if args.backend == "tbss":
        skeleton = load_image(args.fa_skeleton)
        valid = geometry_gate(template, skeleton)["passed"]
        rois["skeleton"] = derived_mask(template,
            (template["values"] != 0) & (skeleton["values"] >= args.skeleton_threshold) if valid else None)
    report = {
        "schema": "fnit_public10_pair_comparison_v1", "case_id": args.case_id,
        "registration_backend": args.backend, "accuracy_thresholds_applied": False,
        "affine_tolerance_mm": AFFINE_TOLERANCE,
        "script_sha256": file_hash(__file__),
        "fixed_roi_rule": {"native": "independent original EDDY brain mask >0.5",
                           "standard": "specified reference template or mask !=0",
                           "skeleton": "specified template !=0 and skeleton >= configured threshold"},
        "common_support_note": "supplement only; zero-valued or missing candidate support remains in primary fixed ROI",
        "skeleton_threshold": args.skeleton_threshold if args.backend == "tbss" else None,
        "template_provenance": template["metadata"], "maps": {},
        "masks": {"native_eddy": compare_mask(cmask, rmask)},
        "statistics_precision": "float32 decoded images; float64 validation accumulation; no inference or resampling",
        "p95_policy": "exact for 3D scalar maps; fixed-seed uniform <=1e6 samples for larger 4D, all other statistics exact over selected elements",
    }
    if args.backend == "tbss":
        report["skeleton_provenance"] = skeleton["metadata"]
    for space in spaces:
        report["maps"][space] = {}
        for name in MAP_NAMES:
            left = load_image(map_path(candidate, args.backend, space, name))
            right = load_image(map_path(reference, args.backend, space, name, reference=True))
            report["maps"][space][name] = image_pair(left, right, rois[space], common_region=common if space == "native" else None)
    upstream = {}
    for role, relative in (
        ("selected_topup_pair", "topup/B0_AP_PA.nii.gz"),
        ("topup_field_hz", "topup/fieldmap_fout.nii.gz"),
        ("topup_corrected_b0", "topup/fieldmap_iout.nii.gz"),
        ("eddy_complete_dwi", "eddy/data.nii.gz")):
        left, right = load_image(candidate / relative), load_image(reference / relative)
        upstream[role] = image_pair(left, right, None if role == "selected_topup_pair" else native_roi,
                                  common_region=None if role == "selected_topup_pair" else common)
    upstream["acquisition_parameters"] = text_pair(candidate / "topup/acqparams.txt", reference / "topup/acqparams.txt")
    upstream["eddy_index"] = text_pair(candidate / "eddy/eddy_index.txt", reference / "eddy/eddy_index.txt")
    upstream["rotated_gradients"] = gradients(candidate, reference, args.bvals) if args.bvals else {
        "status": "unavailable", "reason": "original_AP_bvals_not_supplied"}
    report["upstream"] = upstream
    # 只比较同一公开数据、相同网格上的矩阵；模板为各图共同输出网格。
    fa_source = load_image(candidate / "native/dti_FA.nii.gz")["image"]
    if args.backend == "tbss":
        right_mat = reference / "tbss/FA/dti_FA_to_MNI_affine.mat"
    else:
        right_mat = reference / "mmorf/dti_FA_to_MNI_affine.mat"
    report["affines"] = {"FA": affine_pair(candidate / "registration/dti_FA_to_MNI_affine.mat", right_mat,
                                          fa_source, template["image"])}
    if args.backend == "mmorf":
        ct1 = load_image(candidate / "registration/t1_brain_mask.nii.gz")
        rt1 = load_image(reference / "mmorf/t1_brain_mask.nii.gz")
        report["masks"]["T1_synthstrip"] = compare_mask(ct1, rt1)
        report["affines"]["T1"] = affine_pair(candidate / "registration/t1_to_MNI_affine.mat",
            reference / "mmorf/t1_to_MNI_affine.mat", ct1["image"], template["image"])
        registration = {}
        for role in ("warp", "jacobian", "warped_scalar", "warped_tensor"):
            left = load_image(candidate / "registration" / f"mmorf_{role}.nii.gz")
            right = load_image(reference / "mmorf" / f"mmorf_{role}.nii.gz")
            result = image_pair(left, right, template_roi)
            if role == "warp" and result["gate"]["passed"] and left["values"].shape[-1:] == (3,):
                delta = left["values"][template_roi["values"]].astype(np.float64) - right["values"][template_roi["values"]].astype(np.float64)
                lengths = np.linalg.norm(delta, axis=-1)
                result["vector_difference_mm"] = {
                    "coordinate_convention": "MMORF reference-axis relative millimetres; same convention, no FNIRT coefficient comparison",
                    "voxel_count": int(lengths.size), "mean": float(lengths.mean()) if lengths.size else None,
                    "rms": float(np.sqrt(np.mean(lengths**2))) if lengths.size else None,
                    "p95": float(np.percentile(lengths, 95)) if lengths.size else None,
                    "max": float(lengths.max()) if lengths.size else None}
            if role == "jacobian" and result["gate"]["passed"]:
                region = template_roi["values"]
                result["nonpositive_jacobian_fraction"] = {
                    key: float(np.mean(image["values"][region] <= 0)) if region.any() else None
                    for key, image in (("candidate", left), ("reference", right))}
            if role == "warped_tensor":
                result["interpretation_limit"] = "reference convenience tensor is original spline applywarp without extra reorientation; not a tensor solver equivalence gate"
            registration[role] = result
        report["mmorf_registration"] = registration
    pairs = [pair for maps in report["maps"].values() for pair in maps.values()]
    passed = sum(pair["gate"]["passed"] for pair in pairs)
    report.update(map_pairs_expected=len(spaces) * len(MAP_NAMES), map_pairs_checked=len(pairs),
                  map_geometry_and_finite_pairs_passed=passed,
                  all_required_map_geometry_and_finite_passed=passed == len(pairs),
                  status="complete" if passed == len(pairs) else "required_map_gate_failed",
                  numerical_equivalence_claimed=False)
    return clean_numbers(report)


def summary(values):
    array = np.asarray([value for value in values if isinstance(value, (int, float)) and np.isfinite(value)], dtype=float)
    if not array.size:
        return {"n": 0, "median": None, "q25": None, "q75": None, "iqr": None,
                "min": None, "max": None, "mean": None}
    q25, median, q75 = np.percentile(array, [25, 50, 75])
    return {"n": int(array.size), "median": float(median), "q25": float(q25),
            "q75": float(q75), "iqr": float(q75 - q25), "min": float(array.min()),
            "max": float(array.max()), "mean": float(array.mean())}


def read_report(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("report_not_object")
        return value, {"available": True, "sha256": file_hash(path)}
    except Exception as error:
        return {}, {"available": False, "error_type": type(error).__name__}


def get_nested(value, *keys):
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def process_metrics(path):
    if not path:
        return {"status": "unavailable"}
    try:
        text = Path(path).read_text(encoding="utf-8")
        try:
            data = json.loads(text)
            result = {"status": "loaded", "wall_seconds": data.get("wall_seconds"),
                      "observer_wall_seconds": data.get("wall_seconds"),
                      "wall_time_source": "observer_json_includes_observer_join",
                      "max_rss_kib": data.get("max_rss_kib"),
                      "exit_status": data.get("exit_status", data.get("exit_code")),
                      "sampled_own_gpu_memory_peak_mib": data.get("sampled_own_gpu_memory_peak_mib"),
                      "sampled_other_gpu_memory_peak_mib": data.get("sampled_other_gpu_memory_peak_mib"),
                      "gpu_memory_scope": "sum of this job's descendant native/PyTorch GPU processes at 5-second samples, includes CUDA context; other peak is concurrent unrelated use on the selected GPU",
                      "sha256": file_hash(path)}
            sibling = Path(path).with_name("time.txt")
            if sibling != Path(path) and sibling.is_file():
                gnu = process_metrics(sibling)
                if gnu.get("wall_seconds") is not None:
                    result.update(wall_seconds=gnu["wall_seconds"], wall_time_source="gnu_time_v",
                                  gnu_time_sha256=gnu["sha256"])
                for key in ("max_rss_kib", "exit_status"):
                    if gnu.get(key) is not None:
                        result[key] = gnu[key]
            return result
        except json.JSONDecodeError:
            result = {"status": "loaded", "sha256": file_hash(path), "wall_time_source": "gnu_time_v"}
            for line in text.splitlines():
                if "Elapsed (wall clock) time" in line:
                    elapsed = line.rsplit(": ", 1)[-1].strip()
                    seconds = 0.0
                    for part in elapsed.split(":"):
                        seconds = seconds * 60 + float(part)
                    result["wall_seconds"] = seconds
                elif "Maximum resident set size (kbytes)" in line:
                    result["max_rss_kib"] = int(line.rsplit(":", 1)[-1])
                elif "Exit status:" in line:
                    result["exit_status"] = int(line.rsplit(":", 1)[-1])
            return result
    except Exception as error:
        return {"status": "unavailable", "error_type": type(error).__name__}


def finite_positive(value):
    return isinstance(value, (int, float)) and np.isfinite(value) and value > 0


def comparison_statistics_report(report):
    return {space: {name: {
        "geometry_and_finite_passed": pair.get("gate", {}).get("passed", False),
        "fixed_roi": get_nested(pair, "statistics", "fixed_roi"),
        "support": get_nested(pair, "support", "fixed_roi"),
    } for name, pair in maps.items()} for space, maps in report.get("maps", {}).items()}


def paired_input_binding(candidate, reference, backend):
    """Compare exactly the roles used by this branch, including templates/checkpoint."""
    provenance = candidate.get("input_and_resource_provenance") or {}
    raw = reference.get("input_files") or {}
    bindings = {}
    for stem in ("AP", "PA"):
        for suffix in ("nii.gz", "bval", "bvec", "json"):
            role = f"{stem}.{suffix}"
            values = (get_nested(provenance, "raw_" + role, "sha256"),
                      get_nested(raw, role, "sha256"))
            if role != "PA.bvec" or any(values):
                bindings[role] = values
    bindings["FA_template"] = (get_nested(provenance, "fa_template", "sha256"),
                               get_nested(reference, "templates", "FA_reference", "sha256"))
    bindings["SynthStrip_weights"] = (get_nested(provenance, "synthstrip_weights", "sha256"),
                                      get_nested(reference, "SynthStrip", "weights", "sha256"))
    if backend == "tbss":
        bindings["FA_skeleton"] = (get_nested(provenance, "fa_skeleton", "sha256"),
                                   get_nested(reference, "templates", "FA_skeleton", "sha256"))
    else:
        for role, source, target in (("T1_input", "t1", None),
                                     ("T1_template", "t1_template", "T1_reference"),
                                     ("tensor_template", "tensor_template", "tensor_reference")):
            bindings[role] = (get_nested(provenance, source, "sha256"),
                              get_nested(raw, "T1w.nii.gz", "sha256") if target is None else
                              get_nested(reference, "templates", target, "sha256"))
    checks = {}
    for role, (left, right) in bindings.items():
        valid_left = isinstance(left, str) and re.fullmatch(r"[0-9a-f]{64}", left) is not None
        valid_right = isinstance(right, str) and re.fullmatch(r"[0-9a-f]{64}", right) is not None
        checks[role] = {"candidate_sha256": left if valid_left else None,
                        "reference_sha256": right if valid_right else None,
                        "status": "matched" if valid_left and valid_right and left == right else
                        "mismatched" if valid_left and valid_right else "unavailable"}
    statuses = {entry["status"] for entry in checks.values()}
    return {"status": "mismatched" if "mismatched" in statuses else
            "unavailable" if "unavailable" in statuses else "matched", "roles": checks}


def public_identifier(value, default=None):
    """Status/type labels only; never copy a command or raw exception text."""
    return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", value) else default


def public_rerun_reason(value):
    if not isinstance(value, str):
        return None
    # This is the benchmark plan's public explanation, not an exception message.
    reason = re.sub(r"(?<![A-Za-z0-9_])(?:/[\S]+|[A-Za-z]:[\\/][\S]+)",
                    "[private filesystem path]", value)
    return reason.replace("\r", " ").replace("\n", " ")[:1000]


def retained_initial_reference(row, manifest_root):
    report_path = row.get("initial_failed_reference_report")
    metrics_path = row.get("initial_failed_reference_process_metrics")
    if not report_path and not metrics_path:
        return None
    report, record = read_report(manifest_root / report_path) if report_path else (
        {}, {"available": False, "reason": "report_not_supplied"})
    metrics = process_metrics(manifest_root / metrics_path) if metrics_path else process_metrics(None)
    failure = get_nested(report, "failure", "exception_type") or report.get("failure_type")
    processing = report.get("total_processing_seconds")
    return {
        "status": public_identifier(report.get("status"), "unavailable"),
        "failure_type": public_identifier(failure), "report": record,
        "timing_seconds": {
            "processing": processing if isinstance(processing, (int, float)) and np.isfinite(processing) else None,
            "full_process": metrics.get("wall_seconds")},
        "memory": {"max_rss_kib": metrics.get("max_rss_kib"),
                   "own_gpu_sampled_peak_mib": metrics.get("sampled_own_gpu_memory_peak_mib"),
                   "other_gpu_sampled_peak_mib": metrics.get("sampled_other_gpu_memory_peak_mib")},
        "external_process_metrics": metrics,
        "reference_rerun_reason": public_rerun_reason(row.get("reference_rerun_reason")),
        "included_in_paired_speed_ratios": False,
    }


def retained_initial_candidate(row, manifest_root):
    report_path = row.get("initial_failed_candidate_report")
    metrics_path = row.get("initial_failed_candidate_process_metrics")
    if not report_path and not metrics_path:
        return None
    report, record = read_report(manifest_root / report_path) if report_path else (
        {}, {"available": False, "reason": "report_not_supplied"})
    metrics = process_metrics(manifest_root / metrics_path) if metrics_path else process_metrics(None)
    failure = get_nested(report, "failure", "exception_type") or report.get("failure_type")
    source_hashes = report.get("source_python_sha256") or {}
    source_digest = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest() if source_hashes else None
    return {
        "status": public_identifier(report.get("status"), "unavailable"),
        "failure_type": public_identifier(failure), "report": record,
        "timing_seconds": {"processing": finite_number(get_nested(report, "timing", "api_wall_seconds")),
                           "full_process": metrics.get("wall_seconds")},
        "memory": {"max_rss_kib": metrics.get("max_rss_kib"),
                   "own_gpu_sampled_peak_mib": metrics.get("sampled_own_gpu_memory_peak_mib"),
                   "other_gpu_sampled_peak_mib": metrics.get("sampled_other_gpu_memory_peak_mib"),
                   "peak_allocated_bytes": finite_number(get_nested(report, "memory", "peak_allocated_bytes")),
                   "peak_reserved_bytes": finite_number(get_nested(report, "memory", "peak_reserved_bytes"))},
        "source_commit": public_digest(report.get("source_commit_supplied"), length=40),
        "source_python_hash_count": len(source_hashes), "source_python_manifest_sha256": source_digest,
        "external_process_metrics": metrics,
        "candidate_rerun_reason": public_rerun_reason(row.get("candidate_rerun_reason")),
        "included_in_paired_speed_ratios": False,
    }


def reference_attempt_state(status, record, metrics, failure_type=None):
    observed = record.get("available", False) or metrics.get("status") == "loaded"
    if not observed:
        return None
    code = metrics.get("exit_status")
    if code not in (None, 0) or failure_type or status in ("failed", "nonfinite_outputs", "output_contract_failed"):
        return "failed"
    return "successful" if status == "complete" else "pending_or_unknown"


def reference_attempt_summary(rows):
    initial, rerun, primary = [], [], []
    retained_requested = 0
    for row in rows:
        selected = reference_attempt_state(row["run_status"]["reference"], row["reports"]["reference"],
                                            row["external_process_metrics"]["reference"],
                                            row["failure_types"].get("reference"))
        if selected is not None:
            primary.append(selected)
        retained = row["retained_initial_reference_attempt"]
        if retained is None:
            if selected is not None:
                initial.append(selected)
        else:
            retained_requested += 1
            original = reference_attempt_state(retained["status"], retained["report"],
                                               retained["external_process_metrics"], retained["failure_type"])
            if original is not None:
                initial.append(original)
            if selected is not None:
                rerun.append(selected)
    all_attempts = initial + rerun
    def counts(values):
        return {"observed_attempts": len(values), "successful_attempts": values.count("successful"),
                "failed_attempts": values.count("failed"),
                "pending_or_unknown_attempts": values.count("pending_or_unknown"),
                "success_fraction": values.count("successful") / len(values) if values else None}
    return {"planned_primary_cases": len(rows), "retained_initial_attempts_requested": retained_requested,
            "initial": counts(initial), "rerun": counts(rerun), "selected_primary": counts(primary),
            "all": counts(all_attempts),
            "counting_rule": "initial attempt per case is retained original if configured, otherwise selected primary; selected replacements are reruns; selected primary reports define paired speed ratios, failed original clocks stay separate"}


def candidate_attempt_summary(rows):
    adapted = [{
        "run_status": {"reference": row["run_status"]["candidate"]},
        "reports": {"reference": row["reports"]["candidate"]},
        "external_process_metrics": {"reference": row["external_process_metrics"]["candidate"]},
        "failure_types": {"reference": row["failure_types"].get("candidate")},
        "retained_initial_reference_attempt": row["retained_initial_candidate_attempt"],
    } for row in rows]
    result = reference_attempt_summary(adapted)
    result["counting_rule"] = (
        "only selected primary FNIT runs and explicitly retained failed candidate attempts are counted; "
        "other successful old-source diagnostic runs are outside this count; retained failed clocks "
        "and old source are separate from fresh selected primary clocks and source")
    return result


def finite_number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value) else None


def public_digest(value, length=64):
    return value if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{" + str(length) + "}", value) else None


def mmorf_startup_entries(entries):
    """Native program starts only; discard commands, PIDs, logs and exception text."""
    attempts = []
    if not isinstance(entries, list):
        return attempts
    for step in entries:
        if not isinstance(step, dict):
            continue
        name = step.get("name")
        if name == "mmorf":
            attempt = 1
        elif isinstance(name, str) and re.fullmatch(r"mmorf_startup_retry_[1-9][0-9]*", name):
            attempt = int(name.rsplit("_", 1)[-1])
        else:
            continue
        retry = step.get("startup_retry")
        public_retry = None
        if isinstance(retry, dict):
            public_retry = {
                "attempt": finite_number(retry.get("attempt")),
                "retry_allowed": retry.get("retry_allowed") if isinstance(retry.get("retry_allowed"), bool) else None,
                "log_sha256": public_digest(retry.get("log_sha256")),
                "reason": public_identifier(retry.get("reason")),
            }
        attempts.append({
            "attempt": attempt, "name": name,
            "seconds": finite_number(step.get("seconds")),
            "exit_code": finite_number(step.get("exit_code")),
            "sampled_process_gpu_memory_peak_mib": finite_number(step.get("sampled_process_gpu_memory_peak_mib")),
            "retry_allowed": public_retry["retry_allowed"] if public_retry else None,
            "startup_retry": public_retry,
        })
    return attempts


def reference_mmorf_startups(row, reference, backend, manifest_root):
    if backend != "mmorf":
        return None
    directory = row.get("reference_dir")
    helper, helper_record = read_report(manifest_root / directory / "mmorf/official_mmorf_report.json") if directory else (
        {}, {"available": False, "reason": "reference_dir_not_supplied"})
    attempts = mmorf_startup_entries(helper.get("subprocess_steps"))
    if attempts:
        source = "private_helper_report"
    else:
        attempts = mmorf_startup_entries(get_nested(reference, "MMORF", "native_gpu_observations"))
        source = "outer_reference_public_observations" if attempts else "unavailable"
    program = get_nested(helper, "binaries", "mmorf") or get_nested(reference, "FSL_binaries", "mmorf") or {}
    version = get_nested(helper, "reference", "MMORF")
    version = version if isinstance(version, str) and re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}", version) else None
    raw_policy = helper.get("startup_retry_policy")
    policy = None
    if isinstance(raw_policy, dict):
        policy = {
            "maximum_attempts": finite_number(raw_policy.get("maximum_attempts")),
            "delay_seconds": finite_number(raw_policy.get("delay_seconds")),
            "scope": public_rerun_reason(raw_policy.get("scope")),
            "all_attempts_included_in_processing_time": raw_policy.get("all_attempts_included_in_processing_time")
            if isinstance(raw_policy.get("all_attempts_included_in_processing_time"), bool) else None,
            "numerical_parameters_changed": raw_policy.get("numerical_parameters_changed")
            if isinstance(raw_policy.get("numerical_parameters_changed"), bool) else None,
        }
    return {
        "available": bool(attempts), "source": source,
        "helper_report": helper_record,
        "native_program": {
            "sha256": public_digest(program.get("sha256")),
            "size_bytes": finite_number(program.get("bytes", program.get("size_bytes"))),
            "version": version,
            "source_commit": public_digest(get_nested(helper, "reference", "MMORF_source_commit"), length=40),
        },
        "startup_retry_policy": policy, "attempts": attempts,
        "timing_scope_note": "native constructor startup attempts and delays remain inside the selected whole pipeline clock; this does not replace or add to full-reference rerun counts",
    }


def aggregate_plan(manifest, *, expected_case_count=10, manifest_root=Path(".")):
    rows = manifest.get("planned_cases")
    if not isinstance(rows, list):
        raise ValueError("planned_cases_must_be_list")
    expected = {(f"case{index:02d}", backend) for index in range(1, expected_case_count + 1) for backend in BACKENDS}
    indexed = {}
    for row in rows:
        key = (row.get("case_id"), row.get("backend"))
        if key not in expected or key in indexed:
            raise ValueError("unexpected_or_duplicate_planned_case")
        indexed[key] = row
    results = []
    for case_id, backend in sorted(expected):
        row = indexed.get((case_id, backend), {})
        loaded, records = {}, {}
        for role in ("candidate", "reference", "comparison"):
            relative = row.get(role + "_report")
            loaded[role], records[role] = read_report(manifest_root / relative) if relative else (
                {}, {"available": False, "reason": "not_in_plan_or_report_not_supplied"})
        candidate, reference, comparison = (loaded[role] for role in ("candidate", "reference", "comparison"))
        binding_errors = []
        for role, value in loaded.items():
            if value.get("case_id") not in (None, case_id):
                binding_errors.append(role + "_case_id_mismatch")
            if value.get("registration_backend") not in (None, backend):
                binding_errors.append(role + "_backend_mismatch")
        input_binding = paired_input_binding(candidate, reference, backend)
        if input_binding["status"] == "mismatched":
            binding_errors.append("paired_input_or_resource_sha256_mismatch")
        if (candidate.get("status") == reference.get("status") == "complete"
                and input_binding["status"] == "unavailable"):
            binding_errors.append("paired_input_or_resource_provenance_unavailable")
        cp_seed, rp_seed = get_nested(candidate, "parameters", "eddy_gp_seed"), get_nested(reference, "parameters", "gp_seed")
        if cp_seed is not None and rp_seed is not None and cp_seed != rp_seed:
            binding_errors.append("eddy_seed_mismatch")
        statuses = {role: value.get("status", "unavailable") for role, value in loaded.items()}
        complete_runs = statuses["candidate"] == statuses["reference"] == "complete" and not binding_errors
        comparison_complete = statuses["comparison"] == "complete" and not binding_errors
        cm = process_metrics(manifest_root / row["candidate_process_metrics"]) if row.get("candidate_process_metrics") else process_metrics(None)
        rm = process_metrics(manifest_root / row["reference_process_metrics"]) if row.get("reference_process_metrics") else process_metrics(None)
        if any(metrics.get("exit_status") not in (None, 0) for metrics in (cm, rm)):
            complete_runs = False
        cp = get_nested(candidate, "timing", "api_wall_seconds")
        rp = reference.get("total_processing_seconds")
        ratios = {}
        for scope, left, right in (("processing", cp, rp), ("full_process", cm.get("wall_seconds"), rm.get("wall_seconds"))):
            ratios[scope + "_reference_over_candidate"] = right / left if complete_runs and finite_positive(left) and finite_positive(right) else None
        stages_candidate = get_nested(candidate, "qc", "timings_seconds") or {}
        stages_reference = reference.get("stages_seconds") or {}
        candidate_detail = {}
        for event in candidate.get("stage_events", []):
            if event.get("status") == "complete" and finite_positive(event.get("seconds")):
                candidate_detail[event["stage"]] = candidate_detail.get(event["stage"], 0) + event["seconds"]
        reference_detail = get_nested(reference, "MMORF", "stages_seconds") or reference.get("TBSS_stages_seconds") or {}
        reference_gpu_peaks = [entry.get("sampled_process_gpu_memory_peak_mib") for entry in
                               get_nested(reference, "MMORF", "native_gpu_observations") or []]
        reference_gpu_peaks = [value for value in reference_gpu_peaks if isinstance(value, (int, float)) and np.isfinite(value)]
        failures = {role: get_nested(value, "failure", "exception_type") or value.get("failure_type")
                    for role, value in loaded.items() if statuses[role] not in ("complete", "unavailable")}
        source_hashes = candidate.get("source_python_sha256") or {}
        source_digest = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest() if source_hashes else None
        results.append({
            "case_id": case_id, "backend": backend, "plan_row_present": bool(row),
            "run_status": statuses, "binding_errors": binding_errors,
            "paired_input_and_resource_binding": input_binding,
            "paired_runs_complete": complete_runs, "paired_comparison_complete": comparison_complete,
            "reports": records, "failure_types": failures,
            "retained_initial_reference_attempt": retained_initial_reference(row, manifest_root),
            "reference_rerun_reason": public_rerun_reason(row.get("reference_rerun_reason")),
            "retained_initial_candidate_attempt": retained_initial_candidate(row, manifest_root),
            "candidate_rerun_reason": public_rerun_reason(row.get("candidate_rerun_reason")),
            "reference_mmorf_startup_attempts": reference_mmorf_startups(row, reference, backend, manifest_root),
            "source_commit": candidate.get("source_commit_supplied"),
            "source_python_hash_count": len(source_hashes), "source_python_manifest_sha256": source_digest,
            "timing_seconds": {"candidate_processing": cp, "reference_processing": rp,
                               "candidate_full_process": cm.get("wall_seconds"), "reference_full_process": rm.get("wall_seconds")},
            "paired_ratios": ratios, "candidate_stages_seconds": stages_candidate,
            "reference_stages_seconds": stages_reference,
            "candidate_nested_events_summed_by_name_seconds": candidate_detail,
            "reference_registration_detail_seconds": reference_detail,
            "memory": {"candidate_peak_allocated_bytes": get_nested(candidate, "memory", "peak_allocated_bytes"),
                       "candidate_peak_reserved_bytes": get_nested(candidate, "memory", "peak_reserved_bytes"),
                       "candidate_max_rss_kib": cm.get("max_rss_kib"), "reference_max_rss_kib": rm.get("max_rss_kib"),
                       "reference_mmorf_gpu_sampled_peak_mib": max(reference_gpu_peaks, default=None),
                       "candidate_own_gpu_sampled_peak_mib": cm.get("sampled_own_gpu_memory_peak_mib"),
                       "reference_own_gpu_sampled_peak_mib": rm.get("sampled_own_gpu_memory_peak_mib"),
                       "candidate_other_gpu_sampled_peak_mib": cm.get("sampled_other_gpu_memory_peak_mib"),
                       "reference_other_gpu_sampled_peak_mib": rm.get("sampled_other_gpu_memory_peak_mib")},
            "external_process_metrics": {"candidate": cm, "reference": rm},
            "map_gate_counts": {"expected": comparison.get("map_pairs_expected"),
                                "passed": comparison.get("map_geometry_and_finite_pairs_passed")},
            "maps": comparison_statistics_report(comparison),
            "native_mask_dice": get_nested(comparison, "masks", "native_eddy", "support", "dice"),
            "T1_mask_dice": get_nested(comparison, "masks", "T1_synthstrip", "support", "dice"),
        })
    branches = {}
    for backend in BACKENDS:
        selected = [row for row in results if row["backend"] == backend]
        completed = [row for row in selected if row["paired_runs_complete"]]
        all_stages = {side: sorted({key for row in completed for key in row[side + "_stages_seconds"]})
                      for side in ("candidate", "reference")}
        branch = {
            "planned_subjects": expected_case_count, "paired_runs_complete": len(completed),
            "paired_comparisons_complete": sum(row["paired_comparison_complete"] for row in selected),
            "paired_inputs_and_resources_matched": sum(row["paired_input_and_resource_binding"]["status"] == "matched" for row in selected),
            "reference_attempts": reference_attempt_summary(selected),
            "candidate_attempts": candidate_attempt_summary(selected),
            "missing_or_failed_case_ids": [row["case_id"] for row in selected if not row["paired_runs_complete"]],
            "accuracy_incomplete_case_ids": [row["case_id"] for row in selected if not row["paired_comparison_complete"]],
            "timing_seconds": {key: summary([row["timing_seconds"][key] for row in completed])
                               for key in ("candidate_processing", "reference_processing", "candidate_full_process", "reference_full_process")},
            "paired_ratios": {key: summary([row["paired_ratios"][key] for row in completed])
                              for key in ("processing_reference_over_candidate", "full_process_reference_over_candidate")},
            "stages_seconds": {side: {name: summary([row[side + "_stages_seconds"].get(name) for row in completed])
                                      for name in names} for side, names in all_stages.items()},
            "memory": {key: summary([row["memory"][key] for row in completed]) for key in
                       ("candidate_peak_allocated_bytes", "candidate_peak_reserved_bytes", "candidate_max_rss_kib", "reference_max_rss_kib", "reference_mmorf_gpu_sampled_peak_mib", "candidate_own_gpu_sampled_peak_mib", "reference_own_gpu_sampled_peak_mib", "candidate_other_gpu_sampled_peak_mib", "reference_other_gpu_sampled_peak_mib")},
            "native_mask_dice": summary([row["native_mask_dice"] for row in selected]),
            "T1_mask_dice": summary([row["T1_mask_dice"] for row in selected]),
            "maps": {},
        }
        spaces = ("native", "standard", "skeleton") if backend == "tbss" else ("native", "standard")
        for space in spaces:
            branch["maps"][space] = {}
            for name in MAP_NAMES:
                values = [get_nested(row, "maps", space, name) or {} for row in selected]
                branch["maps"][space][name] = {
                    "planned_pairs": expected_case_count,
                    "geometry_and_finite_pairs_passed": sum(value.get("geometry_and_finite_passed", False) for value in values),
                    "fixed_roi": {key: summary([get_nested(value, "fixed_roi", key) for value in values])
                                  for key in ("pearson_r", "mae", "rmse", "rmse_over_reference_rms", "p95_absdiff", "max_absdiff", "mean_signed_difference")},
                    "support_dice": summary([get_nested(value, "support", "dice") for value in values]),
                }
        branches[backend] = branch
        for key in ("candidate_nested_events_summed_by_name_seconds", "reference_registration_detail_seconds"):
            names = sorted({name for row in completed for name in row[key]})
            branch[key] = {name: summary([row[key].get(name) for row in completed]) for name in names}
    all_complete = all(row["paired_runs_complete"] and row["paired_comparison_complete"] for row in results)
    return clean_numbers({
        "schema": "fnit_public10_aggregate_v1", "script_sha256": file_hash(__file__),
        "expected_subjects": expected_case_count, "expected_branch_pairs": len(expected),
        "manifest_rows": len(rows), "status": "complete" if all_complete else "incomplete",
        "all_planned_results_complete": all_complete, "cases": results, "branches": branches,
        "reference_attempts": reference_attempt_summary(results),
        "candidate_attempts": candidate_attempt_summary(results),
        "source_commits": sorted({row["source_commit"] for row in results if row["source_commit"]}),
        "source_manifest_hashes": sorted({row["source_python_manifest_sha256"] for row in results if row["source_python_manifest_sha256"]}),
        "summary_policy": "each planned subject retained; timing uses every completed paired run regardless of accuracy; available map metrics retain per-map n; missing/failing pairs explicit; no success-only cohort claim",
        "speed_ratio_rule": "within-subject original seconds divided by FNIT seconds; median of paired ratios, not ratio of cohort medians",
        "iqr_rule": "numpy percentile linear; q75-q25",
        "accuracy_thresholds_applied": False, "numerical_equivalence_claimed": False,
        "timing_tree_note": "nested instrumentation is not added to public stages; candidate and original stage boundaries may differ; total clocks kept separate",
        "memory_scope_note": "candidate allocator peaks cover its entire pipeline; original MMORF own-PID memory is sampled every 5 s and excludes original EDDY; absent telemetry is null, never zero; GNU time RSS is a process-tree high-water mark",
    })


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="mode", required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("--case-id", required=True)
    compare.add_argument("--candidate-dir", type=Path, required=True)
    compare.add_argument("--reference-dir", type=Path, required=True)
    compare.add_argument("--backend", choices=BACKENDS, required=True)
    compare.add_argument("--reference-roi", type=Path, required=True)
    compare.add_argument("--fa-skeleton", type=Path)
    compare.add_argument("--skeleton-threshold", type=float, default=2000)
    compare.add_argument("--bvals", type=Path)
    compare.add_argument("--report", type=Path, required=True)
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--manifest", type=Path, required=True)
    aggregate.add_argument("--expected-case-count", type=int, default=10)
    aggregate.add_argument("--report", type=Path, required=True)
    return result


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0].startswith("--"):
        arguments.insert(0, "compare")
    argument_parser = parser()
    args = argument_parser.parse_args(arguments)
    if args.report.exists():
        argument_parser.error("report already exists; preserve the previous comparison")
    if args.mode == "compare":
        if not re.fullmatch(r"case[0-9]{2,3}", args.case_id):
            argument_parser.error("case-id must be anonymous caseNN")
        if args.backend == "tbss" and args.fa_skeleton is None:
            argument_parser.error("TBSS comparison requires fa-skeleton")
        report = compare_case(args)
    else:
        if args.expected_case_count < 1:
            argument_parser.error("expected-case-count must be positive")
        report = aggregate_plan(json.loads(args.manifest.read_text(encoding="utf-8")),
                                expected_case_count=args.expected_case_count,
                                manifest_root=args.manifest.parent)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "anonymous_report_written": True,
                      "numerical_equivalence_claimed": False}, ensure_ascii=False))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
