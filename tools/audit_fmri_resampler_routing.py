#!/usr/bin/env python3
"""Recheck frozen complete-run routing records without running FNIT or CUDA.

The executed validation tool stays frozen. Pass a separately saved final gate
tool, the four completed run folders, and both expected tool hashes. Reports
and comparison files are read only; an anonymous audit is written exclusively
to a new file. No source tree, image, checkpoint or private configuration is
modified, imported or executed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata_digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_bbr_transform(folder):
    """Read the unique published BBR matrix, including text and signed zero."""
    paths = list((Path(folder) / "outputs").rglob(
        "*_from-boldref_to-T1w_mode-image_xfm.txt"))
    if len(paths) != 1:
        raise ValueError("expected_exactly_one_published_bbr_transform")
    raw = paths[0].read_bytes()
    rows = []
    for line in raw.decode("utf-8").splitlines():
        tokens = line.split("#", 1)[0].split()
        if tokens:
            try:
                rows.append(tuple(float(token) for token in tokens))
            except ValueError as error:
                raise ValueError("published_bbr_transform_invalid_numeric_text") from error
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise ValueError("published_bbr_transform_is_not_4_by_4")
    values = tuple(value for row in rows for value in row)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("published_bbr_transform_contains_nonfinite_values")
    packed = struct.pack("<16d", *values)
    public = {"shape": [4, 4], "all_finite": True,
              "raw_size_bytes": len(raw),
              "raw_sha256": hashlib.sha256(raw).hexdigest(),
              "float64_bytes_sha256": hashlib.sha256(packed).hexdigest(),
              "float64_bytes_encoding": "struct.pack('<16d'), row-major little-endian"}
    return {"raw": raw, "values": values, "packed": packed, "public": public}


def normalized_final_routes(report, gate_tool):
    """Match the four final roles independently of public API wrapper names."""
    if report["variant"] == "baseline":
        api = "fnit.fmri.normalization.resample_world"
    elif report["backend"] == "fnirt":
        api = "fnit.applywarp.TorchApplyWarp.apply_world"
    else:
        api = "fnit.synthmorph.apply_transform"
    routes = {}
    for record in report["actual_public_routes"]:
        if record["api"] != api:
            continue
        role = gate_tool.route_role(record, report["expected_frames"])
        if role is None:
            continue
        if role in routes:
            raise ValueError("duplicate_normalized_final_route")
        chain = record["chain"]
        routes[role] = {
            "reference_geometry": chain["reference"],
            "reference_to_source_world": chain["reference_to_source_world"],
            "motion": chain["motion"], "pull": chain["pull"],
            "coordinate_precision": chain["coordinate_precision"],
            "interpolation": record["interpolation"],
            "boundary": record["boundary"],
            "output_mask_present": record["output_mask_present"],
        }
    if set(routes) != {"mask_mni", "clean_mni", "preproc_t1w", "preproc_mni"}:
        raise ValueError("missing_normalized_final_route")
    return routes


def audit_run(folder, gate_tool, execution_tool_sha256):
    """Audit captured witnesses and retain every existing completed-run gate."""
    folder = Path(folder)
    report_path = folder / "report.safe.json"
    report = json.loads(report_path.read_text())
    failures = []
    if report["validation_tool_sha256"] != execution_tool_sha256:
        failures.append("executed_validation_tool_sha256_mismatch")
    if report.get("real_data") is not True:
        failures.append("not_real_data")
    if report["backend"] not in ("fnirt", "synthmorph") or report["variant"] not in (
            "baseline", "candidate"):
        raise ValueError("unexpected backend or variant")
    frames = report["expected_frames"]
    if frames < 1:
        raise ValueError("positive expected_frames required")
    if not report["routing_gate"]["passed"]:
        failures.append("original_complete_run_gate_failed")
    if report["anatomical_cache_reused"]:
        failures.append("anatomical_cache_reused")
    expected_registrations = {"fnirt": int(report["backend"] == "fnirt"),
                              "synthmorph": int(report["backend"] == "synthmorph")}
    if report["actual_registration_counts"] != expected_registrations:
        failures.append("incorrect_actual_registration_counts")
    if report["input_summaries"]["bold"]["shape"][-1] != frames:
        failures.append("incomplete_raw_input_frames")
    for role in ("clean_native", "clean_mni", "preproc_t1w", "preproc_mni"):
        image = report["outputs"][role]
        if len(image["shape"]) != 4 or image["shape"][3] != frames:
            failures.append(f"{role}:incomplete_saved_output_frames")
        if not image["all_finite"]:
            failures.append(f"{role}:nonfinite_output")
    memory = report["memory"]
    limit = memory["allocator_limit_bytes"]
    if not (0 < memory["peak_cuda_allocated_bytes"] < limit <= 20_000_000_000):
        failures.append("allocated_peak_not_strictly_below_declared_limit")
    try:
        gate = gate_tool.routing_gate(
            report["actual_public_routes"], backend=report["backend"],
            variant=report["variant"], expected_frames=frames)
    except (KeyError, TypeError, ValueError) as error:
        gate = {"passed": False, "failures": [
            "incomplete_record_schema:" + type(error).__name__], "actual_call_counts": {}}
    if not gate["passed"]:
        failures.append("final_routing_gate_failed")
    witnessed_source_files = set()
    for route in report["actual_public_routes"]:
        for caller in route["caller_chain"]:
            source_file = caller["source_file"]
            # Source files are public repo-relative code paths, never data paths.
            if (Path(source_file).is_absolute() or ".." in Path(source_file).parts
                    or not source_file.startswith("src/fnit/")):
                failures.append("invalid_repo_relative_caller_source_file")
            if report["source_sha256"].get(source_file) != caller["source_sha256"]:
                failures.append("captured_caller_source_sha256_mismatch")
            witnessed_source_files.add(source_file)
    comparison = None
    comparison_sha256 = None
    if report["variant"] == "candidate":
        comparison_path = folder / "comparison.safe.json"
        comparison = json.loads(comparison_path.read_text())
        comparison_sha256 = sha256(comparison_path)
        if not (comparison["all_equal"]
                and report["strict_scientific_comparison_passed"] is True):
            failures.append("strict_scientific_comparison_failed")
    summary = {
        "backend": report["backend"], "variant": report["variant"],
        "expected_frames": frames, "passed": not failures, "failures": failures,
        "execution_report_sha256": sha256(report_path),
        "executed_validation_tool_sha256": report["validation_tool_sha256"],
        "executed_comparison_helper_sha256": report["comparison_helper_sha256"],
        "final_routing_gate": gate,
        "source_sha256": report["source_sha256"],
        "witnessed_caller_source_files": sorted(witnessed_source_files),
        "actual_registration_counts": report["actual_registration_counts"],
        "input_metadata_sha256": metadata_digest(report["input_summaries"]),
        "scientific_options_metadata_sha256": metadata_digest(report["scientific_options"]),
        "recorded_timing_seconds": report["timing_seconds"],
        "recorded_memory": memory,
        "strict_scientific_comparison_passed": None if comparison is None else comparison["all_equal"],
        "strict_comparison_file_sha256": comparison_sha256,
    }
    return report, summary


def audit_runs(folders, gate_tool, execution_tool_sha256):
    reports, summaries, bbrs, routes = {}, {}, {}, {}
    for folder in folders:
        report, summary = audit_run(folder, gate_tool, execution_tool_sha256)
        key = f"{report['variant']}_{report['backend']}"
        if key in reports:
            raise ValueError("duplicate backend/variant")
        try:
            bbrs[key] = read_bbr_transform(folder)
            summary["published_bbr_transform"] = bbrs[key]["public"]
        except (ValueError, UnicodeDecodeError) as error:
            bbrs[key] = None
            summary["published_bbr_transform"] = None
            summary["passed"] = False
            summary["failures"].append("published_bbr_matrix_invalid:" + (
                "invalid_utf8" if isinstance(error, UnicodeDecodeError) else str(error)))
        try:
            routes[key] = normalized_final_routes(report, gate_tool)
            summary["normalized_final_routes"] = routes[key]
        except (ValueError, KeyError, TypeError) as error:
            routes[key] = None
            summary["normalized_final_routes"] = None
            summary["passed"] = False
            summary["failures"].append("normalized_final_routes_invalid:" + type(error).__name__)
        reports[key], summaries[key] = report, summary
    expected = {f"{variant}_{backend}" for backend in ("fnirt", "synthmorph")
                for variant in ("baseline", "candidate")}
    if set(reports) != expected:
        raise ValueError("provide exactly baseline/candidate for both backends")
    pairs = {}
    for backend in ("fnirt", "synthmorph"):
        first_key, second_key = [f"{variant}_{backend}" for variant in ("baseline", "candidate")]
        first, second = reports[first_key], reports[second_key]
        first_bbr, second_bbr = bbrs[first_key], bbrs[second_key]
        first_routes, second_routes = routes[first_key], routes[second_key]
        valid_bbr = first_bbr is not None and second_bbr is not None
        valid_routes = first_routes is not None and second_routes is not None
        pairs[backend] = {
            "inputs_equal": first["input_summaries"] == second["input_summaries"],
            "scientific_options_equal": first["scientific_options"] == second["scientific_options"],
            "expected_frames_equal": first["expected_frames"] == second["expected_frames"],
            "execution_tool_equal": first["validation_tool_sha256"] == second["validation_tool_sha256"],
            "comparison_helper_equal": first["comparison_helper_sha256"] == second["comparison_helper_sha256"],
            "bbr_matrix_values_equal": valid_bbr and first_bbr["values"] == second_bbr["values"],
            "bbr_float64_bytes_equal": valid_bbr and first_bbr["packed"] == second_bbr["packed"],
            "bbr_raw_bytes_equal": valid_bbr and first_bbr["raw"] == second_bbr["raw"],
            "normalized_final_routes_equal": valid_routes and first_routes == second_routes,
            "final_route_fields_equal": {} if not valid_routes else {
                role: {field: first_routes[role][field] == second_routes[role][field]
                       for field in first_routes[role]}
                for role in sorted(first_routes)},
        }
    return {"all_passed": all(item["passed"] for item in summaries.values())
                            and all(all(value is True for name, value in item.items()
                                        if name != "final_route_fields_equal")
                                    for item in pairs.values()),
            "runs": summaries, "baseline_candidate_pairs": pairs}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--gate-tool", type=Path, required=True)
    parser.add_argument("--expected-gate-tool-sha256", required=True)
    parser.add_argument("--expected-execution-tool-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("use a new offline audit file")
    if sha256(args.gate_tool) != args.expected_gate_tool_sha256:
        raise AssertionError("final gate tool hash mismatch")
    spec = importlib.util.spec_from_file_location("_fmri_final_routing_gate", args.gate_tool)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    result = audit_runs(args.run_dir, tool, args.expected_execution_tool_sha256)
    result.update({
        "schema_version": 2,
        "scope": "CPU-only audit of saved actual routes, four-role composed transform contracts, the published 4x4 BBR text matrix and complete-run scientific comparisons; no pipeline execution or image recomputation",
        "audit_helper_sha256": sha256(__file__),
        "audit_gate_tool_sha256": args.expected_gate_tool_sha256,
        "executed_validation_tool_sha256": args.expected_execution_tool_sha256,
        "timing_note": "Original executed timings are retained. Execution tool a5a3ab90 also subtracts two peak-counter query durations outside its API timer; these tiny durations were not saved separately and are not reconstructed. All four paired runs used that same timing implementation. Numerical outputs, whole-run memory peaks and route records are unaffected.",
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"all_passed": result["all_passed"],
                      "audit_gate_tool_sha256": result["audit_gate_tool_sha256"]}))
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
