#!/usr/bin/env python3
"""Reject numerical regressions between full real-data FLIRT benchmark runs.

Usage::

    python tools/check_flirt_gpu_parity.py --baseline-dir BASELINE \
        --optimized-dir OPTIMIZED --output-json paired_gate.json

Both directories must contain benchmark_flirt_gpu.py's report.private.json,
cold_0.mat/.nii.gz and warm_1.mat/.nii.gz.  The gate compares saved artifacts
bitwise, not the gzip container timestamp.  FLIRTResult.save writes matrices
with 12 significant digits; this gate does not establish bitwise identity of
the unsaved double-precision matrix.  Tiny-image tests exercise the gate only
and do not constitute registration benchmarks.  FSL paired metrics are checked
as reported summaries; this script neither runs FSL nor recomputes those metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

import nibabel as nib
import numpy as np


_RUNS = ("cold_0", "warm_1")
_SCHEDULE = "FSL default 8/4/2/1 mm"


def _bits_equal(first, second):
    first, second = np.asarray(first), np.asarray(second)
    return (first.shape == second.shape and first.dtype == second.dtype
            and np.array_equal(np.ascontiguousarray(first).view(np.uint8),
                               np.ascontiguousarray(second).view(np.uint8)))


def _float_bits(value):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("reported cost must be finite")
    return struct.pack("<d", value)


def _extension_signature(image):
    return [(extension.get_code(), hashlib.sha256(extension._raw).hexdigest())
            for extension in image.header.extensions]


def _image_checks(first_file, second_file):
    first, second = nib.load(first_file), nib.load(second_file)
    first_values = np.asarray(first.dataobj.get_unscaled())
    second_values = np.asarray(second.dataobj.get_unscaled())
    if not np.isfinite(first_values).all() or not np.isfinite(second_values).all():
        raise ValueError("moved image must contain only finite values")
    return {
        "voxels_bitwise_equal": _bits_equal(first_values, second_values),
        "header_bitwise_equal": (
            type(first.header) is type(second.header)
            and first.header.binaryblock == second.header.binaryblock
            and _extension_signature(first) == _extension_signature(second)
            and _float_bits(first.dataobj.slope) == _float_bits(second.dataobj.slope)
            and _float_bits(first.dataobj.inter) == _float_bits(second.dataobj.inter)
        ),
        "affine_bitwise_equal": _bits_equal(first.affine, second.affine),
    }


def _report_runs(report):
    runs = report["runs"]
    if len(runs) < 2 or runs[0]["condition"] != "cold" or runs[1]["condition"] != "warm":
        raise ValueError("report must contain a complete cold and warm registration")
    return runs[:2]


def _reported_fsl_metrics_equal(first, second):
    """Gate reported FSL summaries without executing the official program."""
    tolerances = {
        "world_grid_displacement_mm": dict.fromkeys(
            ("mean", "median", "p95", "minimum", "maximum", "rms"), 1e-10),
        "warped": {"pearson_r": 1e-12, "mae": 1e-10,
                   "rmse": 1e-10, "support_dice": 1e-12},
    }
    try:
        for group, fields in tolerances.items():
            for field, tolerance in fields.items():
                values = (float(first[group][field]), float(second[group][field]))
                if not all(np.isfinite(values)) or abs(values[0] - values[1]) > tolerance:
                    return False
        for field in ("comparison_mask", "voxels"):
            if first["warped"][field] != second["warped"][field]:
                return False
        required_header = {"shape_equal", "affine_max_abs", "dtype_equal",
                           "qform_code_equal", "sform_code_equal"}
        headers = (first["grid_header"], second["grid_header"])
        return (all(required_header <= header.keys() for header in headers)
                and all(np.isfinite(float(header["affine_max_abs"])) for header in headers)
                and headers[0] == headers[1])
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def check_parity(baseline_dir, optimized_dir):
    """Return a compact pass/fail record; all comparisons default to bitwise."""
    directories = (Path(baseline_dir), Path(optimized_dir))
    summary = {"schema_version": 1, "status": "fail", "comparison": "saved artifacts, bitwise",
               "conditions_equal": False, "runs": {}, "angular_search": {"status": "not captured"},
               "fsl_paired_scope": "reported benchmark summaries; no FSL execution or metric recomputation",
               "errors": []}
    errors = summary["errors"]
    try:
        reports = [json.loads((directory / "report.private.json").read_text())
                   for directory in directories]
        baseline, optimized = reports
        if any(report.get("schema_version") != 1 for report in reports):
            errors.append("unsupported report schema")
        inputs = baseline["input_sha256"]
        required = {"moving", "reference", "fsl_matrix", "fsl_moved"}
        if not required <= inputs.keys() or any(not inputs[key] for key in required):
            errors.append("required input fingerprints missing")
        for field in ("input_sha256", "dof", "cost"):
            if baseline[field] != optimized[field]:
                errors.append(f"{field} differs")
        if (baseline["dof"], baseline["cost"]) not in ((6, "normmi"), (12, "corratio")):
            errors.append("unsupported DOF/cost profile")
        run_pairs = zip(_report_runs(baseline), _report_runs(optimized))
        for name, (first_record, second_record) in zip(_RUNS, run_pairs):
            checks = {}
            for field in ("initial_matrix_used", "input_weight_used", "reference_weight_used", "source_versions"):
                if (first_record["qc"].get(field) != second_record["qc"].get(field)):
                    errors.append(f"{name}: {field} scientific condition differs")
            for label, record in (("baseline", first_record), ("optimized", second_record)):
                qc = record["qc"]
                expected_cost = ("FSL normalized mutual information" if baseline["dof"] == 6
                                 else "FSL correlation ratio")
                valid_definition = (
                    qc["schedule"] == _SCHEDULE and qc["angular_search"] is True
                    and qc["degrees_of_freedom"] == baseline["dof"]
                    and qc["cost"] == expected_cost
                    and qc["search_cost"] == "FSL correlation ratio"
                    and qc["optimizer"] == "MISCMATHS Brent coordinate search"
                    and qc["matrix_coordinate_system"] == "FSL scaled-mm"
                    and qc["matrix_direction"] == "moving/input-to-fixed/reference"
                )
                if not valid_definition:
                    errors.append(f"{name}: {label} does not report the complete reference schedule/definition")
                if (_float_bits(qc["cost_value"]) != _float_bits(record["paired"]["cost"])
                        or qc["cost_evaluations"] != record["paired"]["cost_evaluations"]):
                    errors.append(f"{name}: {label} QC and paired records disagree")
                if (not isinstance(qc["cost_evaluations"], int)
                        or qc["cost_evaluations"] <= 0):
                    errors.append(f"{name}: {label} invalid evaluation count")
            matrices = [np.loadtxt(directory / f"{name}.mat", dtype=np.float64)
                        for directory in directories]
            if any(matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                   for matrix in matrices):
                raise ValueError("saved matrix must be a finite 4x4 affine")
            checks["matrix_values_bitwise_equal"] = _bits_equal(*matrices)
            checks["matrix_file_bytes_equal"] = (
                (directories[0] / f"{name}.mat").read_bytes()
                == (directories[1] / f"{name}.mat").read_bytes()
            )
            checks.update(_image_checks(*(str(directory / f"{name}.nii.gz")
                                         for directory in directories)))
            checks["cost_bitwise_equal"] = (_float_bits(first_record["qc"]["cost_value"])
                                             == _float_bits(second_record["qc"]["cost_value"]))
            checks["evaluations_equal"] = (first_record["qc"]["cost_evaluations"]
                                           == second_record["qc"]["cost_evaluations"])
            checks["fsl_paired_metrics_unchanged"] = _reported_fsl_metrics_equal(
                first_record["paired"], second_record["paired"]
            )
            for field, passed in checks.items():
                if not passed:
                    errors.append(f"{name}: {field} failed")
            summary["runs"][name] = checks
        traces = [report.get("profile", {}).get("angular_search") for report in reports]
        if all(traces):
            equal = traces[0] == traces[1]
            summary["angular_search"] = {"status": "captured", "candidate_records_equal": equal,
                                         "scope": "captured candidate matrix ordering, costs and retained counts",
                                         "records": len(traces[0])}
            if not equal:
                errors.append("angular candidate ordering/pruning trace differs")
        summary["conditions_equal"] = not any(
            error.startswith(("input_sha256", "dof", "cost", "required", "unsupported"))
            or "reference schedule/definition" in error
            or "scientific condition differs" in error for error in errors
        )
        summary["status"] = "pass" if not errors else "fail"
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        errors.append(f"incomplete or invalid benchmark artifacts: {type(error).__name__}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--optimized-dir", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    summary = check_parity(args.baseline_dir, args.optimized_dir)
    text = json.dumps(summary, indent=2) + "\n"
    if args.output_json:
        args.output_json.write_text(text)
    print(text, end="")
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
