#!/usr/bin/env python3
"""核验本次独立 raw-to-TBSS 流程的上游输出，只写匿名 JSON。

输入为 FNIT 与官方流程各自新生成的目录及原 AP bval。比较选中的
TOPUP pair、acqp/index、Hz field、完整 EDDY DWI、旋转梯度、outlier map
和 FLIRT affine；不运行模型、不调用原软件、不重采样、不导出 NIfTI。

示例：python compare_upstream_end_to_end.py --fnit-root "$fnit_output_dir" \
    --reference-root "$official_output_dir" --bvals "$raw_ap_bvals" \
    --output "$anonymous_report_json"

EDDY 两侧完整数据各读取一次为 float32。相关、MAE、RMSE 和最大差值按卷
用 float64 累计；p95 在超过 100 万个元素时使用固定种子的均匀无放回抽样，
报告样本量与抽样方法。抽样仅影响 p95，不影响其余统计。该程序不计 benchmark
耗时、不设置精度通过阈值，不比较不同网格上的 FNIRT 系数。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import itertools
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np


AFFINE_TOLERANCE = 1e-5
PERCENTILE_MAX_SAMPLES = 1_000_000
PERCENTILE_SEED = 1729
OUTLIER_HEADER = "One row per scan, one column per slice. Outlier: 1, Non-outlier: 0"


@dataclass
class Image:
    image: object | None
    values: np.ndarray | None
    metadata: dict


def _geometry(image):
    affine = np.asarray(image.affine, dtype=np.float64)
    return {"shape": list(image.shape), "affine": affine.tolist(),
            "affine_finite": bool(np.isfinite(affine).all()),
            "voxel_sizes_mm": [float(value) for value in image.header.get_zooms()[:3]],
            "stored_dtype": str(image.get_data_dtype()),
            "spatial_units": image.header.get_xyzt_units()[0]}


def _volumes(values):
    if values.ndim == 3:
        yield values
    elif values.ndim == 4:
        for index in range(values.shape[3]):
            yield values[..., index]
    else:
        raise ValueError("expected 3D or 4D values")


def _load_image(path, *, squeeze_singleton=False):
    try:
        image = nib.load(str(path))
        values = np.asarray(image.dataobj, dtype=np.float32)
        if squeeze_singleton and values.ndim == 4 and values.shape[3] == 1:
            values = values[..., 0]
        metadata = _geometry(image)
        metadata.update(available=True, decoded_dtype="float32", comparison_shape=list(values.shape),
                        nonfinite_elements=sum(int(np.count_nonzero(~np.isfinite(volume)))
                                               for volume in _volumes(values)))
        return Image(image, values, metadata)
    except Exception as error:
        # 异常原文可能带病例名或私有绝对路径，报告只记录错误类型。
        return Image(None, None, {"available": False, "load_error_type": type(error).__name__})


def _gate(left, right, *, spatial_only=False):
    if left.values is None or right.values is None:
        return {"passed": False, "reasons": ["image_unavailable"]}
    left_shape, right_shape = left.values.shape, right.values.shape
    checks = {
        "shape_equal": bool(left_shape[:3] == right_shape[:3] if spatial_only
                            else left_shape == right_shape),
        "affine_finite": bool(np.isfinite(left.image.affine).all()
                              and np.isfinite(right.image.affine).all()),
        "affine_equal_within_tolerance": bool(np.allclose(
            left.image.affine, right.image.affine, rtol=0, atol=AFFINE_TOLERANCE)),
        "decoded_values_finite": bool(left.metadata["nonfinite_elements"] == 0
                                      and right.metadata["nonfinite_elements"] == 0),
    }
    return {"passed": all(checks.values()), **checks,
            "reasons": [name for name, passed in checks.items() if not passed]}


class _Statistics:
    """每卷合并中心矩；不创建完整 4D float64 数组。"""
    def __init__(self, total):
        self.total = int(total)
        sample_count = min(self.total, PERCENTILE_MAX_SAMPLES)
        if self.total <= PERCENTILE_MAX_SAMPLES:
            self.positions = np.arange(self.total, dtype=np.int64)
        else:
            generator = np.random.default_rng(PERCENTILE_SEED)
            self.positions = np.sort(generator.choice(self.total, size=sample_count, replace=False))
        self.samples = np.empty(sample_count, dtype=np.float64)
        self.count = 0
        self.mean_x = self.mean_y = self.m2_x = self.m2_y = self.cross = 0.0
        self.absolute_sum = self.square_sum = self.maximum = 0.0
        self.different = 0

    def update(self, x, y):
        # x/y 已由一个 3D volume 的 region 提取，临时 float64 大小仅为一卷。
        x, y = x.astype(np.float64), y.astype(np.float64)
        count = x.size
        if not count:
            return
        mean_x, mean_y = float(x.mean()), float(y.mean())
        centered_x, centered_y = x - mean_x, y - mean_y
        delta_x, delta_y = mean_x - self.mean_x, mean_y - self.mean_y
        combined = self.count + count
        weight = self.count * count / combined
        self.m2_x += float(np.dot(centered_x, centered_x)) + delta_x ** 2 * weight
        self.m2_y += float(np.dot(centered_y, centered_y)) + delta_y ** 2 * weight
        self.cross += float(np.dot(centered_x, centered_y)) + delta_x * delta_y * weight
        self.mean_x += delta_x * count / combined
        self.mean_y += delta_y * count / combined
        difference = x - y
        absolute = np.abs(difference)
        self.absolute_sum += float(absolute.sum(dtype=np.float64))
        self.square_sum += float(np.dot(difference, difference))
        self.maximum = max(self.maximum, float(absolute.max()))
        self.different += int(np.count_nonzero(difference))
        start, end = np.searchsorted(self.positions, [self.count, combined])
        self.samples[start:end] = absolute[self.positions[start:end] - self.count]
        self.count = combined

    def result(self):
        if self.count != self.total:
            raise ValueError("statistics element count differs from region prepass")
        varying = self.count >= 2 and self.m2_x > 0 and self.m2_y > 0
        correlation = float(np.clip(self.cross / np.sqrt(self.m2_x * self.m2_y), -1, 1)) if varying else None
        return {
            "elements": self.count, "different_elements": self.different,
            "values_exact": self.different == 0,
            "pearson_r": correlation,
            "correlation_status": "computed" if varying else "empty_constant_or_single_element",
            "mae": self.absolute_sum / self.count if self.count else None,
            "rmse": float(np.sqrt(self.square_sum / self.count)) if self.count else None,
            "max_absdiff": self.maximum if self.count else None,
            "p95_absdiff": float(np.percentile(self.samples, 95)) if self.samples.size else None,
            "p95_sampling": {"method": "all_elements" if self.count <= PERCENTILE_MAX_SAMPLES
                             else "uniform_without_replacement_over_valid_voxel_volume_elements",
                             "sample_elements": int(self.samples.size), "population_elements": self.count,
                             "seed": None if self.count <= PERCENTILE_MAX_SAMPLES else PERCENTILE_SEED,
                             "numpy_percentile_method": "linear",
                             "other_statistics_use_all_elements": True},
        }


def _statistics_by_volume(left, right, regions):
    """region=None 表示每一卷 x!=0 或 y!=0 的逐元素 union。"""
    counts = {name: 0 for name in regions}
    for x, y in zip(_volumes(left), _volumes(right)):
        for name, mask in regions.items():
            selected = (x != 0) | (y != 0) if mask is None else mask
            counts[name] += int(np.count_nonzero(selected))
    accumulators = {name: _Statistics(count) for name, count in counts.items()}
    for x, y in zip(_volumes(left), _volumes(right)):
        for name, mask in regions.items():
            selected = (x != 0) | (y != 0) if mask is None else mask
            accumulators[name].update(x[selected], y[selected])
    return {name: accumulator.result() for name, accumulator in accumulators.items()}


def _image_pair(left, right):
    return {"fnit": left.metadata, "reference": right.metadata,
            "gate": _gate(left, right), "statistics": None}


def _selected_pair(fnit_root, reference_root):
    left = _load_image(fnit_root / "topup/B0_AP_PA.nii.gz")
    right = _load_image(reference_root / "topup/B0_AP_PA.nii.gz")
    result = _image_pair(left, right)
    if (left.values is not None and right.values is not None
            and (left.values.ndim != 4 or right.values.ndim != 4
                 or left.values.shape[3] != 2 or right.values.shape[3] != 2)):
        result["gate"]["passed"] = False
        result["gate"]["reasons"].append("selected_TOPUP_pair_must_have_two_volumes")
    if left.values is not None and right.values is not None:
        result["affine_exact"] = bool(np.array_equal(left.image.affine, right.image.affine))
        if left.values.shape == right.values.shape:
            result["decoded_values_exact"] = bool(np.array_equal(left.values, right.values))
    if result["gate"]["passed"]:
        result["statistics"] = _statistics_by_volume(
            left.values, right.values, {"whole_selected_pair": np.ones(left.values.shape[:3], dtype=bool)})
    return result


def _text_numbers(left_path, right_path):
    try:
        left, right = (np.loadtxt(path, dtype=np.float64, ndmin=2) for path in (left_path, right_path))
        result = {"status": "compared", "fnit": left.tolist(), "reference": right.tolist(),
                  "shape_equal": left.shape == right.shape,
                  "values_finite": bool(np.isfinite(left).all() and np.isfinite(right).all())}
        if not result["values_finite"]:
            return {"status": "skipped", "reason": "nonfinite_numeric_text"}
        if result["shape_equal"]:
            result.update(values_exact=bool(np.array_equal(left, right)),
                          max_absdiff=float(np.abs(left - right).max()) if left.size else None)
        return result
    except Exception as error:
        return {"status": "skipped", "error_type": type(error).__name__}


def _compare_dwi_or_field(fnit_root, reference_root, relative_path, masks, *, four_dimensional):
    left = _load_image(fnit_root / relative_path, squeeze_singleton=not four_dimensional)
    right = _load_image(reference_root / relative_path, squeeze_singleton=not four_dimensional)
    result = _image_pair(left, right)
    expected_dimension = 4 if four_dimensional else 3
    if (left.values is not None and right.values is not None
            and (left.values.ndim != expected_dimension or right.values.ndim != expected_dimension)):
        result["gate"]["passed"] = False
        result["gate"]["reasons"].append("unexpected_number_of_dimensions")
    if not result["gate"]["passed"]:
        return result
    regions = {"nonzero_union": None}
    mask_gates = {name: _gate(left, image, spatial_only=True) for name, image in masks.items()}
    result["brain_mask_gates"] = mask_gates
    # 固定官方 ROI 仅依赖官方 mask；另一侧 mask 不可读时仍能核验此 ROI。
    if four_dimensional:
        if mask_gates["reference"]["passed"] and masks["reference"].values.ndim == 3:
            regions["fixed_official_brain"] = masks["reference"].values > 0.5
        else:
            result["fixed_official_brain_skipped_reason"] = "official_mask_grid_dimension_or_finite_gate_failed"
    if all(gate["passed"] for gate in mask_gates.values()):
        fnit_brain, reference_brain = masks["fnit"].values > 0.5, masks["reference"].values > 0.5
        if fnit_brain.ndim != 3 or reference_brain.ndim != 3:
            result["common_brain_intersection_skipped_reason"] = "brain_masks_must_be_3D"
        else:
            regions["common_brain_intersection"] = fnit_brain & reference_brain
    else:
        result["common_brain_intersection_skipped_reason"] = "brain_mask_grid_or_finite_gate_failed"
    result["statistics"] = _statistics_by_volume(left.values, right.values, regions)
    result["value_units"] = "DWI stored signal units" if four_dimensional else "Hz"
    result["region_rule"] = "brain masks >0.5; nonzero_union=(fnit!=0)|(reference!=0) per voxel and volume"
    if four_dimensional:
        result["volumes_compared"] = int(left.values.shape[3])
    return result


def _gradients(fnit_root, reference_root, bvals_path):
    bvals = np.loadtxt(bvals_path, dtype=np.float64).reshape(-1)
    if not np.isfinite(bvals).all():
        return {"status": "skipped", "reason": "nonfinite_bvals"}
    matrices = []
    for root in (fnit_root, reference_root):
        values = np.loadtxt(root / "eddy/data.eddy_rotated_bvecs", dtype=np.float64, ndmin=2)
        if values.shape == (bvals.size, 3) and values.shape != (3, bvals.size):
            values = values.T
        if values.shape != (3, bvals.size) or not np.isfinite(values).all():
            return {"status": "skipped", "reason": "gradient_shape_or_finite_failed"}
        matrices.append(values)
    left, right = matrices
    left_norm, right_norm = np.linalg.norm(left, axis=0), np.linalg.norm(right, axis=0)
    dw = bvals > 100
    keep = dw & (left_norm > 0) & (right_norm > 0)
    dot = np.sum((left[:, keep] / left_norm[keep]) * (right[:, keep] / right_norm[keep]), axis=0)
    angles = np.degrees(np.arccos(np.clip(dot, -1, 1)))
    antipodal = np.degrees(np.arccos(np.clip(np.abs(dot), 0, 1)))
    def summary(values):
        return {"mean": float(values.mean()) if values.size else None,
                "rms": float(np.sqrt(np.mean(values ** 2))) if values.size else None,
                "p95": float(np.percentile(values, 95)) if values.size else None,
                "max": float(values.max()) if values.size else None}
    return {"status": "compared", "rule": "original AP bval >100; unit-normalize both nonzero rotated columns",
            "bval_volumes": int(bvals.size), "selected_dwi_volumes": int(dw.sum()),
            "compared_dwi_volumes": int(keep.sum()), "excluded_zero_direction_volumes": int((dw & ~keep).sum()),
            "original_volume_indices": np.flatnonzero(keep).tolist(),
            "directed_angle_degrees": summary(angles), "directed_angle_degrees_by_volume": angles.tolist(),
            "antipodal_angle_degrees": summary(antipodal),
            "antipodal_angle_degrees_by_volume": antipodal.tolist(),
            "antipodal_rule": "acos(abs(dot)); diffusion gradient axes are invariant to sign",
            "fnit_nonzero_direction_norm_range": [float(left_norm[keep].min()), float(left_norm[keep].max())] if keep.any() else None,
            "reference_nonzero_direction_norm_range": [float(right_norm[keep].min()), float(right_norm[keep].max())] if keep.any() else None}


def _read_outliers(path, expected_shape):
    try:
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        if not lines or lines[0].lstrip("# ").strip() != OUTLIER_HEADER:
            return None, {"status": "skipped", "reason": "unrecognized_first_line_header"}
        values = np.loadtxt(io.StringIO("\n".join(lines[1:])), dtype=np.float64, ndmin=2)
        if values.shape != expected_shape:
            return None, {"status": "skipped", "reason": "unexpected_scan_by_slice_shape",
                          "shape": list(values.shape), "expected_shape": list(expected_shape)}
        if not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
            return None, {"status": "skipped", "reason": "outlier_values_are_not_finite_binary"}
        return values.astype(bool), {"status": "recognized", "format": "one header line; rows=scans; columns=z-slices",
                                    "shape": list(values.shape), "outlier_entries": int(values.sum())}
    except Exception as error:
        return None, {"status": "skipped", "error_type": type(error).__name__}


def _outliers(fnit_root, reference_root):
    shape = nib.load(str(reference_root / "eddy/data.nii.gz")).shape
    if len(shape) != 4:
        return {"status": "skipped", "reason": "reference_DWI_must_be_4D"}
    expected = (shape[3], shape[2])
    left, left_record = _read_outliers(fnit_root / "eddy/data.eddy_outlier_map", expected)
    right, right_record = _read_outliers(reference_root / "eddy/data.eddy_outlier_map", expected)
    result = {"fnit": left_record, "reference": right_record, "status": "skipped"}
    if left is not None and right is not None:
        both = int(np.count_nonzero(left & right))
        positive = int(left.sum()) + int(right.sum())
        result.update(status="compared", values_exact=bool(np.array_equal(left, right)),
                      different_entries=int(np.count_nonzero(left != right)),
                      both_outlier=both, fnit_only_outlier=int(np.count_nonzero(left & ~right)),
                      reference_only_outlier=int(np.count_nonzero(~left & right)),
                      both_nonoutlier=int(np.count_nonzero(~left & ~right)),
                      outlier_dice=2 * both / positive if positive else 1.0)
    return result


def _scaled_mm(image):
    sizes = np.asarray(image.header.get_zooms()[:3], dtype=np.float64)
    affine = np.asarray(image.affine, dtype=np.float64)
    if (not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-10
            or not np.isfinite(sizes).all() or np.any(sizes <= 0)):
        raise ValueError("invalid physical image geometry")
    matrix = np.diag([*sizes, 1.0])
    if np.linalg.det(affine[:3, :3]) > 0:
        matrix[0, 0] *= -1
        matrix[0, 3] = (image.shape[0] - 1) * sizes[0]
    return matrix


def _flirt(fnit_root, reference_root):
    left = np.loadtxt(fnit_root / "registration/dti_FA_to_MNI_affine.mat", dtype=np.float64)
    right = np.loadtxt(reference_root / "tbss/FA/dti_FA_to_MNI_affine.mat", dtype=np.float64)
    if (left.shape != (4, 4) or right.shape != (4, 4)
            or not np.isfinite(left).all() or not np.isfinite(right).all()
            or not np.allclose(left[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(right[3], [0, 0, 0, 1], atol=1e-8, rtol=0)):
        return {"status": "skipped", "reason": "invalid_4x4_homogeneous_matrix"}
    result = {"status": "compared", "matrix_coordinates": "moving-to-fixed FSL scaled-mm",
              "fnit": left.tolist(), "reference": right.tolist(), "fnit_minus_reference": (left - right).tolist(),
              "matrix_max_absdiff": float(np.abs(left - right).max()), "physical_point_displacement": None}
    try:
        moving = [nib.load(str(fnit_root / "registration/dti_FA_preprocessed.nii.gz")),
                  nib.load(str(reference_root / "native/dti_FA_preprocessed.nii.gz"))]
        fixed = [nib.load(str(fnit_root / "registration/standard/FA.nii.gz")),
                 nib.load(str(reference_root / "tbss/FA/MNI.nii.gz"))]
    except Exception as error:
        result["physical_points_skipped_error_type"] = type(error).__name__
        return result
    result["image_geometry"] = {"fnit_moving": _geometry(moving[0]), "reference_moving": _geometry(moving[1]),
                                "fnit_fixed": _geometry(fixed[0]), "reference_fixed": _geometry(fixed[1])}
    matching = all(a.shape[:3] == b.shape[:3] and np.allclose(a.affine, b.affine, atol=AFFINE_TOLERANCE, rtol=0)
                   for a, b in (moving, fixed))
    if not matching:
        result["physical_points_skipped_reason"] = "input_or_reference_geometry_differs"
        return result
    if any(image.header.get_xyzt_units()[0] not in ("mm", "unknown") for image in (*moving, *fixed)):
        result["physical_points_skipped_reason"] = "header_spatial_units_are_not_mm_or_unknown"
        return result
    last = np.asarray(moving[0].shape[:3], dtype=np.float64) - 1
    corners = list(itertools.product(*[(0.0, float(size)) for size in last]))
    points = np.asarray([*corners, tuple(last / 2)], dtype=np.float64)
    homogeneous = np.vstack((points.T, np.ones(9)))
    source_world = moving[0].affine @ homogeneous
    target_world = []
    for matrix, source, target in zip((left, right), moving, fixed):
        forward_world = target.affine @ np.linalg.inv(_scaled_mm(target)) @ matrix @ _scaled_mm(source) @ np.linalg.inv(source.affine)
        target_world.append((forward_world @ source_world)[:3].T)
    distances = np.linalg.norm(target_world[0] - target_world[1], axis=1)
    result["physical_point_displacement"] = {
        "rule": "same physical input points: eight voxel-center corners and image center; FSL scaled-mm with x flip when det(affine)>0",
        "world_coordinate_system": "NIfTI world-RAS millimetres",
        "units_rule": "stored affine and pixdim are interpreted in millimetres, following FSL; original header units recorded above",
        "input_voxel_points": points.tolist(), "fnit_fixed_world_points_mm": target_world[0].tolist(),
        "reference_fixed_world_points_mm": target_world[1].tolist(), "distances_mm": distances.tolist(),
        "mean_mm": float(distances.mean()), "rms_mm": float(np.sqrt(np.mean(distances ** 2))),
        "mean_square_mm2": float(np.mean(distances ** 2)), "max_mm": float(distances.max()),
        "point_count": 9, "whole_brain_RMS_or_FSL_rmsdiff": False,
    }
    return result


def _safe(function):
    try:
        return function()
    except Exception as error:
        return {"status": "skipped", "error_type": type(error).__name__}


def _json_safe(value):
    """损坏几何里的 NaN/Infinity 记作 null，保留 gate 的失败原因。"""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fnit-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--bvals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.fnit_root.is_dir() or not args.reference_root.is_dir() or not args.bvals.is_file():
        parser.error("两侧 root 必须是已有目录，bvals 必须是已有文件")
    if args.output.exists():
        parser.error("output 必须是新 JSON 文件")
    masks = {name: _load_image(root / "eddy/nodif_brain_mask.nii.gz", squeeze_singleton=True)
             for name, root in (("fnit", args.fnit_root), ("reference", args.reference_root))}
    report = {
        "schema_version": 1, "scope": "upstream outputs of the current independent real end-to-end runs",
        "accuracy_thresholds_applied": False, "affine_tolerance": AFFINE_TOLERANCE,
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "nibabel": nib.__version__},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "statistics_scope": "complete volumes; float32 decoded images, float64 per-volume centered-moment accumulation",
        "percentile_policy": {"max_samples": PERCENTILE_MAX_SAMPLES, "seed": PERCENTILE_SEED,
                              "sampling_applies_only_to_p95_absdiff": True},
        "brain_masks": {name: image.metadata for name, image in masks.items()},
        "selected_topup_pair": _safe(lambda: _selected_pair(args.fnit_root, args.reference_root)),
        "acqp": _text_numbers(args.fnit_root / "topup/acqparams.txt", args.reference_root / "topup/acqparams.txt"),
        "eddy_index": _text_numbers(args.fnit_root / "eddy/eddy_index.txt", args.reference_root / "eddy/eddy_index.txt"),
        "topup_fout_hz": _safe(lambda: _compare_dwi_or_field(
            args.fnit_root, args.reference_root, "topup/fieldmap_fout.nii.gz", masks, four_dimensional=False)),
        "eddy_complete_dwi": _safe(lambda: _compare_dwi_or_field(
            args.fnit_root, args.reference_root, "eddy/data.nii.gz", masks, four_dimensional=True)),
        "eddy_rotated_directions": _safe(lambda: _gradients(args.fnit_root, args.reference_root, args.bvals)),
        "eddy_outlier_map": _safe(lambda: _outliers(args.fnit_root, args.reference_root)),
        "flirt_affine": _safe(lambda: _flirt(args.fnit_root, args.reference_root)),
        "fnirt_coefficients_compared": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"anonymous_report_written": True, "model_runs": 0,
                      "new_nifti_files": 0, "accuracy_thresholds_applied": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
