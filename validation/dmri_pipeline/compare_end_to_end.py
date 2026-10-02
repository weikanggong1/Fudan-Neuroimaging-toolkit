#!/usr/bin/env python3
"""比较真实 raw-to-TBSS 整链结果；只输出匿名 JSON 与可选脑内 PNG。

FNIT: native/{dti_,NODDI_}*.nii.gz、registration/{standard,skeleton}/*.nii.gz。
官方: native/ 同名九图、tbss/stats/all_*.nii.gz 和 *_skeletonised.nii.gz。
本程序不运行配准、不重采样、不导出 NIfTI，也不设置精度通过阈值。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np


MAP_NAMES = ("FA", "MD", "L1", "L2", "L3", "MO", "ICVF", "OD", "ISOVF")
AFFINE_TOLERANCE = 1e-5


def _json_numbers(value):
    """非有限 header 数字转 null，不向 JSON 写 NaN/Infinity。"""
    array = np.asarray(value)
    if array.ndim:
        return [_json_numbers(item) for item in array]
    number = float(array)
    return number if np.isfinite(number) else None


def _normalized_data(image):
    values = np.asanyarray(image.dataobj)
    if values.ndim == 4 and values.shape[-1] == 1:
        values = values[..., 0]
    return values


def _metadata(image, values):
    header = image.header
    forms = {}
    for form_name in ("qform", "sform"):
        try:
            matrix, code = getattr(image, "get_" + form_name)(coded=True)
            forms[form_name] = _json_numbers(matrix) if matrix is not None else None
            forms[form_name + "_code"] = int(code)
        except (ValueError, np.linalg.LinAlgError):
            forms[form_name] = None
            forms[form_name + "_code"] = int(header[form_name + "_code"])
            forms[form_name + "_read_error"] = True
    slope, intercept = header.get_slope_inter()
    return {
        "available": True,
        "image_class": type(image).__name__,
        "raw_shape": list(image.shape),
        "comparison_shape": list(values.shape),
        "singleton_volume_squeezed": bool(tuple(image.shape) != values.shape),
        "stored_dtype": str(image.get_data_dtype()),
        "decoded_dtype": str(values.dtype),
        "pixdim": _json_numbers(header["pixdim"]),
        "affine": _json_numbers(image.affine),
        "units": list(header.get_xyzt_units()),
        "intent_code": int(header["intent_code"]),
        "header_slope": _json_numbers(slope) if slope is not None else None,
        "header_intercept": _json_numbers(intercept) if intercept is not None else None,
        "proxy_slope": _json_numbers(getattr(image.dataobj, "slope", 1.0)),
        "proxy_intercept": _json_numbers(getattr(image.dataobj, "inter", 0.0)),
        "nonfinite_voxels": int(np.count_nonzero(~np.isfinite(values))),
        **forms,
    }


@dataclass
class LoadedImage:
    image: object | None
    values: np.ndarray | None
    metadata: dict


def _load(path):
    try:
        image = nib.load(str(path))
        values = _normalized_data(image)
        return LoadedImage(image, values, _metadata(image, values))
    except Exception as error:
        # 不复制异常消息：其中可能含私有绝对路径或病例标识。
        return LoadedImage(None, None, {
            "available": False,
            "load_error_type": type(error).__name__,
        })


def _geometry_gate(left, right):
    if left.values is None or right.values is None:
        return {"passed": False, "reasons": ["image_unavailable"]}
    shape_equal = bool(left.values.ndim == right.values.ndim == 3
                       and left.values.shape == right.values.shape)
    affine_finite = bool(np.isfinite(left.image.affine).all()
                         and np.isfinite(right.image.affine).all())
    affine_equal = bool(affine_finite and np.allclose(
        left.image.affine, right.image.affine, rtol=0, atol=AFFINE_TOLERANCE
    ))
    finite = bool(left.metadata["nonfinite_voxels"] == 0
                  and right.metadata["nonfinite_voxels"] == 0)
    checks = {"shape_equal_after_singleton_squeeze": shape_equal,
              "affine_finite": affine_finite, "affine_equal": affine_equal,
              "decoded_values_finite": finite}
    return {"passed": all(checks.values()), **checks,
            "reasons": [name for name, passed in checks.items() if not passed]}


def _support(left, right):
    left_count, right_count = int(left.sum()), int(right.sum())
    intersection = int(np.count_nonzero(left & right))
    union = left_count + right_count - intersection
    return {
        "fnit_voxels": left_count, "reference_voxels": right_count,
        "intersection_voxels": intersection, "union_voxels": union,
        "dice": 2.0 * intersection / (left_count + right_count)
        if left_count + right_count else 1.0,
        "both_empty": left_count + right_count == 0,
    }


def _statistics(left, right, mask):
    count = int(np.count_nonzero(mask))
    if not count:
        return {"voxels": 0, "pearson_r": None, "correlation_status": "empty_region",
                "mae": None, "rmse": None, "max_absdiff": None, "p95_absdiff": None}
    x, y = left[mask].astype(np.float64), right[mask].astype(np.float64)
    difference = x - y
    absolute = np.abs(difference)
    has_variation = count >= 2 and np.std(x) > 0 and np.std(y) > 0
    correlation = float(np.corrcoef(x, y)[0, 1]) if has_variation else None
    return {
        "voxels": count, "pearson_r": correlation,
        "correlation_status": "computed" if has_variation else "constant_or_single_voxel",
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "max_absdiff": float(absolute.max()),
        "p95_absdiff": float(np.percentile(absolute, 95)),
    }


def _derived_mask(source, mask, description):
    if (source.values is None or mask is None
            or source.metadata.get("nonfinite_voxels", 1) != 0):
        return LoadedImage(None, None, {"available": False, "derived_from": description})
    return LoadedImage(source.image, mask, {
        **source.metadata, "decoded_dtype": "bool", "nonfinite_voxels": 0,
        "derived_boolean_mask": True, "derived_from": description,
    })


def _compare_images(left, right, fixed_region, region_name, common_region=None):
    gate = _geometry_gate(left, right)
    fixed_gate = _geometry_gate(left, fixed_region)
    gate["fixed_region_grid_and_finite"] = fixed_gate["passed"]
    if not fixed_gate["passed"]:
        gate["passed"] = False
        gate["reasons"].append("fixed_region_grid_or_finite_failed")
    result = {"fnit": left.metadata, "reference": right.metadata,
              "gate": gate, "statistics": None, "support": None}
    if common_region is not None:
        common_gate = _geometry_gate(left, common_region)
        reference_common_gate = _geometry_gate(right, common_region)
        common_gate["reference_grid_and_finite"] = reference_common_gate["passed"]
        if not reference_common_gate["passed"]:
            common_gate["passed"] = False
            common_gate["reasons"].append("reference_common_region_grid_or_finite_failed")
        result["common_native_brain_gate"] = common_gate
    if not gate["passed"]:
        return result
    x, y = left.values, right.values
    left_support, right_support = x != 0, y != 0
    fixed = fixed_region.values
    result["statistics"] = {
        "nonzero_union": _statistics(x, y, left_support | right_support),
        region_name: _statistics(x, y, fixed),
    }
    result["support"] = {
        "whole_grid": _support(left_support, right_support),
        region_name: _support(left_support & fixed, right_support & fixed),
    }
    if common_region is not None:
        common_passed = result["common_native_brain_gate"]["passed"]
        common = common_region.values
        result["statistics"]["common_native_brain"] = (
            _statistics(x, y, common) if common_passed else None
        )
        result["support"]["common_native_brain"] = (
            _support(left_support & common, right_support & common) if common_passed else None
        )
    return result


def _compare_masks(left, right):
    gate = _geometry_gate(left, right)
    return {"fnit": left.metadata, "reference": right.metadata,
            "gate": gate, "mask_threshold": 0.5,
            "support": _support(left.values > 0.5, right.values > 0.5)
            if gate["passed"] else None}


def _map_path(root, stage, name, official=False):
    if stage == "native":
        prefix = "NODDI_" if name in ("ICVF", "OD", "ISOVF") else "dti_"
        return root / "native" / f"{prefix}{name}.nii.gz"
    if official:
        suffix = "_skeletonised" if stage == "skeleton" else ""
        return root / "tbss" / "stats" / f"all_{name}{suffix}.nii.gz"
    return root / "registration" / stage / f"{name}.nii.gz"


def _figure(path, pairs, brain):
    """只画共同标准网格的脑内指标，三种显示共用每张指标的数值标尺。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coverage = brain.sum(axis=(0, 1))
    selected = []
    for index in np.argsort(-coverage, kind="stable"):
        if coverage[index] and all(abs(int(index) - z) >= 5 for z in selected):
            selected.append(int(index))
        if len(selected) == 3:
            break
    if not selected:
        raise ValueError("empty_template_brain")
    slices = sorted(selected)
    figure, axes = plt.subplots(3, 3 * len(slices),
                               figsize=(4.8 * len(slices), 5.6), squeeze=False)
    limits = {}
    for row, name in enumerate(("FA", "MD", "OD")):
        left, right = pairs[name]
        difference = np.abs(left.values.astype(np.float64) - right.values)
        minimum = min(0.0, float(left.values[brain].min()), float(right.values[brain].min()))
        maximum = max(float(left.values[brain].max()), float(right.values[brain].max()),
                      float(difference[brain].max()), minimum + 1e-12)
        limits[name] = {"vmin": minimum, "vmax": maximum,
                        "same_scale_for_fnit_reference_absdiff": True}
        for slice_index, z in enumerate(slices):
            for column, (label, volume) in enumerate((
                ("FNIT", left.values), ("FSL+AMICO", right.values), ("|Δ|", difference)
            )):
                axis = axes[row, slice_index * 3 + column]
                inside = brain[:, :, z].T
                display = np.ma.array(volume[:, :, z].T, mask=~inside)
                colormap = plt.get_cmap("viridis").copy()
                colormap.set_bad("black")
                rendered = axis.imshow(display, origin="lower", cmap=colormap,
                                       vmin=minimum, vmax=maximum, interpolation="nearest")
                axis.set_title(f"{label} · z={z}", fontsize=8)
                if slice_index == column == 0:
                    axis.text(-0.07, 0.5, name, transform=axis.transAxes,
                              rotation=90, va="center", ha="center", fontsize=10)
                axis.set_axis_off()
        figure.colorbar(rendered, ax=list(axes[row]), fraction=0.015, pad=0.015,
                        label="mm²/s" if name == "MD" else "dimensionless")
    figure.suptitle("Real end-to-end TBSS comparison — template brain only", fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(path, dpi=180, facecolor="white", metadata={"Software": "FNIT validation"})
    finally:
        plt.close(figure)
    return {"written": True, "z_voxel_indices": slices,
            "slice_selection": "template brain coverage; separated by at least 5 voxels",
            "orientation": "native template voxel x/y axes; no resampling",
            "brain_only": True, "color_limits": limits}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fnit-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--fa-reference", type=Path, required=True)
    parser.add_argument("--fa-skeleton", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, help="可选脑内 FA/MD/OD PNG")
    args = parser.parse_args(argv)
    if args.figure is not None and args.figure.suffix.lower() != ".png":
        parser.error("--figure must be a PNG")
    fa_reference, fa_skeleton = _load(args.fa_reference), _load(args.fa_skeleton)
    templates_gate = _geometry_gate(fa_reference, fa_skeleton)
    brain = fa_reference.values != 0 if templates_gate["passed"] else None
    fixed_skeleton = (fa_skeleton.values >= 2000) & brain if brain is not None else None
    fnit_native_mask = _load(args.fnit_root / "eddy" / "nodif_brain_mask.nii.gz")
    official_native_mask = _load(args.reference_root / "eddy" / "nodif_brain_mask.nii.gz")
    native_masks_gate = _geometry_gate(fnit_native_mask, official_native_mask)
    common_native_region = _derived_mask(
        fnit_native_mask,
        (fnit_native_mask.values > 0.5) & (official_native_mask.values > 0.5)
        if native_masks_gate["passed"] else None,
        "FNIT_and_official_eddy_brain_masks_gt_0_5",
    )
    regions = {
        "native": (_derived_mask(official_native_mask,
                   official_native_mask.values > 0.5 if official_native_mask.values is not None else None,
                   "official_eddy_brain_mask"), "fixed_official_native_brain"),
        "standard": (_derived_mask(fa_reference, brain, "FA_template_nonzero"),
                     "fixed_template_brain"),
        "skeleton": (_derived_mask(fa_reference, fixed_skeleton,
                     "FA_template_nonzero_and_skeleton_ge_2000"), "fixed_template_skeleton"),
    }
    report = {
        "schema_version": 1,
        "scope": "completed real raw AP/PA to independent FNIT and official TBSS outputs",
        "accuracy_thresholds_applied": False,
        "affine_tolerance": AFFINE_TOLERANCE,
        "comparison_dtype": "float64 statistics from original decoded values",
        "native_fixed_region": "official EDDY native mask > 0.5; no template resampling",
        "native_common_region": "FNIT EDDY mask > 0.5 and official EDDY mask > 0.5; no resampling",
        "template_brain_rule": "FA reference != 0",
        "template_skeleton_rule": "FA skeleton >= 2000 and FA reference != 0",
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "nibabel": nib.__version__},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "templates": {"fa_reference": fa_reference.metadata, "fa_skeleton": fa_skeleton.metadata,
                      "gate": templates_gate},
        "maps": {}, "masks": {}, "figure": {"requested": args.figure is not None, "written": False},
    }
    plotting_pairs = {}
    for stage in ("native", "standard", "skeleton"):
        report["maps"][stage] = {}
        for name in MAP_NAMES:
            left = _load(_map_path(args.fnit_root, stage, name))
            right = _load(_map_path(args.reference_root, stage, name, official=True))
            result = _compare_images(left, right, *regions[stage],
                                     common_region=common_native_region if stage == "native" else None)
            report["maps"][stage][name] = result
            if stage == "standard" and name in ("FA", "MD", "OD"):
                plotting_pairs[name] = (left, right)
    report["masks"]["native_eddy_brain"] = _compare_masks(fnit_native_mask, official_native_mask)
    fnit_fa = plotting_pairs["FA"][0]
    fnit_valid = (fnit_fa.values != 0) & brain if (
        report["maps"]["standard"]["FA"]["gate"]["passed"] and brain is not None
    ) else None
    report["masks"]["standard_fa_valid"] = _compare_masks(
        _derived_mask(fnit_fa, fnit_valid, "standard_FA_nonzero_and_template_brain"),
        _load(args.reference_root / "tbss" / "stats" / "mean_FA_mask.nii.gz"),
    )
    report["masks"]["skeleton"] = _compare_masks(
        _derived_mask(fnit_fa, fnit_valid & fixed_skeleton if fnit_valid is not None else None,
                      "standard_FA_valid_and_fixed_template_skeleton"),
        _load(args.reference_root / "tbss" / "stats" / "mean_FA_skeleton_mask.nii.gz"),
    )
    gates = [value["gate"]["passed"] for maps in report["maps"].values() for value in maps.values()]
    gates.extend(value["common_native_brain_gate"]["passed"]
                 for value in report["maps"]["native"].values())
    gates.extend(value["gate"]["passed"] for value in report["masks"].values())
    gates.append(templates_gate["passed"])
    report["all_geometry_and_finite_gates_passed"] = all(gates)
    report["map_pairs_passed"] = sum(value["gate"]["passed"] for maps in report["maps"].values()
                                     for value in maps.values())
    figure_success = True
    if args.figure is not None:
        if brain is not None and all(report["maps"]["standard"][name]["gate"]["passed"]
                                     for name in ("FA", "MD", "OD")):
            try:
                report["figure"].update(_figure(args.figure, plotting_pairs, brain))
            except Exception as error:
                report["figure"]["error_type"] = type(error).__name__
                figure_success = False
        else:
            report["figure"]["reason"] = "standard_map_or_template_gate_failed"
            figure_success = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"map_pairs_passed": report["map_pairs_passed"], "map_pairs_total": 27,
                      "all_geometry_and_finite_gates_passed": all(gates),
                      "figure_written": report["figure"]["written"],
                      "accuracy_thresholds_applied": False}))
    return 0 if all(gates) and figure_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
