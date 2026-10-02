"""Read-only CPU comparison of actual completed raw connectome cohorts.

This reference tool never starts a pipeline, edits a source/report, imports FNIT
or torch, or substitutes a missing result. Ten-case completion describes
comparison coverage; MRtrix stochastic acceptance remains a separate decision.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import time

try:
    from . import compare_freesurfer_recon_outputs as anatomy
except ImportError:
    import compare_freesurfer_recon_outputs as anatomy

MATRICES = ("count", "sift2_fbc", "mean_length", "mean_fa")
_SHARED_IMAGES = ("five_tissue_dwi_world.nii.gz", "gmwmi_dwi_world.nii.gz", "fa_dwi.nii.gz", "brain_mask_dwi.nii.gz",
                  "preproc/eddy/data.nii.gz", "preproc/topup/fieldmap_out_fieldcoef.nii.gz", "preproc/topup/fieldmap_iout.nii.gz")
_FAILURE_PREFIXES = ("failed", "incomplete", "stopped")


class WaitingForActualResults(Exception):
    """No completed actual result exists yet; this is not scientific success."""


def check(condition, message):
    anatomy.check(condition, message)


def safe_json(path, touched=None):
    path = Path(path)
    check(path.is_file() and not path.is_symlink(), f"missing or linked JSON: {path}")
    digest = anatomy.sha(path)
    content = json.loads(path.read_bytes())
    if touched is not None:
        touched[str(path)] = digest
    return content, {"path": str(path), "sha256": digest}


def snapshot_json(path, snapshot):
    """Dynamic driver states are observations, never frozen back into source."""
    content, identity = safe_json(path)
    with Path(snapshot).open("xb") as stream:
        stream.write(Path(path).read_bytes())
    return content, identity


def manifest_cases(manifest):
    check(isinstance(manifest, dict) and isinstance(manifest.get("cases"), list), "canonical cases manifest required")
    check(all(manifest.get(key) for key in ("dataset", "snapshot", "license")), "dataset/snapshot/license provenance required")
    cases = manifest["cases"]
    check(len(cases) == 10, "formal comparator requires exactly ten raw cases; no pilot is called a ten-case result")
    ids, subjects = set(), set()
    for case in cases:
        case_id = case.get("case_id", "")
        check(isinstance(case_id, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", case_id), "unsafe or absent case_id")
        check(case_id not in ids and case.get("subject") not in subjects and case.get("subject"), "case/subject must be distinct")
        ids.add(case_id); subjects.add(case["subject"])
        check(Path(case.get("t1w", "")).is_absolute(), "raw T1 path must be absolute")
        files = case.get("input_files", [])
        check(files and all(Path(item["path"]).is_absolute() and re.fullmatch(r"[0-9a-fA-F]{64}", item.get("sha256", "")) for item in files), "raw input SHA/path missing")
        t1 = [item for item in files if item["kind"] == "raw_t1w"]
        check(len(t1) == 1 and t1[0]["path"] == case["t1w"], "canonical raw T1 identity missing")
    return cases


def load_origins(binding_path, cases):
    declarations, identity = safe_json(binding_path)
    check(set(declarations) == {"bindings"} and isinstance(declarations["bindings"], list) and declarations["bindings"], "explicit preparation binding list required")
    expected = {case["case_id"]: case for case in cases}
    origins, mapping = [], {}
    common = None
    for declaration in declarations["bindings"]:
        check(set(declaration) == {"prep_config", "prep_driver_report_dir", "case_ids"}, "unknown preparation mapping fields")
        config, config_identity = safe_json(declaration["prep_config"])
        check(config.get("scope") == "staged_anatomy_preparation_only" and config.get("candidate_source") == "unknown" and config.get("gpu_started") is False and
              config.get("cpu_threads") == 8 and "sources" not in config and "frozen_sources" not in config,
              "preparation was not source-independent fresh official eight-thread CPU work")
        root = Path(config["run_root"])
        manifest, manifest_identity = safe_json(root / "input_manifest.json")
        actual = {case["case_id"]: case for case in manifest_cases(manifest)}
        check(actual == expected, "preparation canonical raw cases differ")
        selected = declaration["case_ids"]
        check(isinstance(selected, list) and selected and len(set(selected)) == len(selected), "invalid preparation selection")
        plan = config.get("selected_cases", list(expected))
        check(all(case_id in plan and case_id in expected for case_id in selected), "mapped case was outside original preparation plan")
        official = config.get("official_origin", {}).get("identity")
        check(official, "original preparation official runtime identity absent")
        signature = {"official": official, "atlases": config["atlases"], "future_gpu_parameters": config["future_gpu_parameters"]}
        check(common is None or signature == common, "preparation batches differ in official identity or scientific settings")
        common = signature
        check(not root.is_symlink(), "preparation root must be a real directory")
        origin = {"config": config, "config_identity": config_identity, "manifest_identity": manifest_identity,
                  "root": str(root), "driver_dir": declaration["prep_driver_report_dir"], "case_ids": selected}
        for case_id in selected:
            check(case_id not in mapping, "case is bound to multiple original preparations")
            mapping[case_id] = len(origins)
        origins.append(origin)
    check(set(mapping) == set(expected), "preparation bindings do not cover all ten original cases")
    return origins, mapping, identity


def verify_source(identity):
    """Check the real frozen source ledger without importing FNIT or torch."""
    root = Path(identity["directory"])
    check(root.is_dir() and not root.is_symlink(), "frozen source absent or linked")
    files = sorted(path for path in (root / "src/fnit").rglob("*") if path.is_file() and
                   path.suffix not in {".pyc", ".pyo"} and "__pycache__" not in path.parts)
    files += [path for path in (root / "pyproject.toml", root / "environment.yml") if path.is_file()]
    check(all(not path.is_symlink() and path.resolve().is_relative_to(root.resolve()) for path in files), "source file links escape or alias the frozen tree")
    hashes = {str(path.relative_to(root)): anatomy.sha(path) for path in files}
    check({"src/fnit/cli.py", "src/fnit/__init__.py", "pyproject.toml", "environment.yml"}.issubset(hashes), "frozen source is incomplete")
    check(hashes == identity.get("source_sha256"), "actual frozen source files differ from the recorded ledger")
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    check(digest == identity.get("source_fingerprint"), "actual source fingerprint changed")
    return {"directory": str(root), "source_fingerprint": digest, "git_commit": identity.get("git_commit"), "file_count": len(hashes), "source_sha256": hashes}


def ready_case(state_path, version, case_id):
    path = Path(state_path)
    if not path.exists():
        raise WaitingForActualResults(f"actual {version} driver is absent")
    state, observation = safe_json(path)
    record = state.get("cases", {}).get(version + "/" + case_id)
    if record is None:
        if state.get("status", "").startswith(_FAILURE_PREFIXES):
            raise ValueError(f"actual {version} driver ended without this case: {state['status']}")
        raise WaitingForActualResults(f"actual {version}/{case_id} case has not been declared")
    status = record.get("status", "unknown")
    if status.startswith(_FAILURE_PREFIXES) or record.get("error") or record.get("validation_error"):
        raise ValueError(f"actual {version}/{case_id} failed: {status}; {record.get('error') or record.get('validation_error')}")
    if status != "completed":
        if state.get("status", "").startswith(_FAILURE_PREFIXES):
            raise ValueError(f"actual {version} driver ended with incomplete {case_id}: {status}")
        raise WaitingForActualResults(f"actual {version}/{case_id} is {status}")
    if record.get("timing_error"):
        raise ValueError(f"actual {version}/{case_id} timing validation failed")
    return record, state, observation



def validate_prepared_case(origin, case):
    record, state, observation = ready_case(Path(origin["driver_dir"]) / "status.json", "candidate", case["case_id"])
    check(state.get("config") == origin["config"], "actual preparation driver differs from original frozen config")
    for path_key, digest_key in (("worker_script", "worker_script_sha256"), ("anatomy_prep_script", "anatomy_prep_script_sha256")):
        check(anatomy.sha(origin["config"][path_key]) == origin["config"][digest_key], "original preparation helper changed")
    path = Path(origin["root"]) / "candidate" / case["case_id"] / "anatomy_prep_report.json"
    actual, identity = safe_json(path)
    check(record.get("preparation_report") == actual and actual.get("status") == "completed" and
          actual.get("case_id") == case["case_id"] and actual.get("candidate_source") == "unknown" and
          actual.get("gpu_started") is False and actual.get("recon_all_reused") is False,
          "preparation driver is not the actual completed fresh case report")
    validate_input_ledger(actual["input_verification_before"], case, "before independent reconstruction")
    validate_input_ledger(actual["input_verification_after"], case, "after independent reconstruction")
    return {"preparation_report": identity, "driver_observation": observation,
            "coordinator_timing": record.get("timing"), "preparation_worker_wall_seconds": record.get("preparation_worker_wall_seconds")}

def required_outputs(output, atlases):
    root = Path(output)
    paths = list(_SHARED_IMAGES) + ["dwi_to_t1_world.csv", "run_state.json", "dataset_description.json",
            "preproc/eddy/data.eddy_rotated_bvecs", "preproc/eddy/state.json", "preproc/eddy/data.eddy_qc.json",
            "preproc/topup/acqparams.txt", "preproc/topup/state.json", "preproc/raw/bids_selection.json"]
    for atlas in atlases:
        check(re.fullmatch(r"[A-Za-z0-9_.+-]+", atlas) is not None, "unsafe atlas name")
        paths += [f"atlases/{atlas}/{name}" for name in ("nodes.tsv", "region_labels.csv", "atlas_dwi.nii.gz", *(f"connectome_{kind}.csv" for kind in MATRICES))]
    for relative in paths:
        path = root / relative
        check(path.is_file() and path.stat().st_size > 0 and path.resolve().is_relative_to(root.resolve()),
              f"completed CLI output is missing, empty, or outside its namespace: {relative}")
    return paths


def validate_input_ledger(entries, case, description):
    expected = case["input_files"]
    check(len(entries) == len(expected), description + " input ledger is incomplete")
    for actual, wanted in zip(entries, expected):
        check(all(actual.get(key) == wanted[key] for key in ("path", "kind", "sha256")) and
              actual.get("actual_sha256") == wanted["sha256"], description + " input SHA provenance changed")


def validate_gpu_run(root, driver_path, version, case, subject_dir, atlases, touched):
    record, state, observation = ready_case(driver_path, version, case["case_id"])
    config = state["config"]
    check(isinstance(record.get("timing"), dict) and bool(record["timing"]), "completed actual GPU driver has no validated original wall/queue/gap timing")
    check(config["run_root"] == str(root) and config["atlases"] == atlases and version in config["frozen_sources"], "actual GPU config scope or atlas selection changed")
    job = Path(root) / version / case["case_id"]
    gpu, gpu_identity = safe_json(job / "gpu_report.json", touched)
    check(gpu.get("status") == "completed" and gpu.get("exit_code") == 0 and gpu.get("case_id") == case["case_id"] and gpu.get("version") == version,
          "actual GPU report is incomplete or wrong case/version")
    declared_result = record.get("gpu_report", record.get("gpu_result"))
    check(declared_result == gpu, "driver case does not match immutable actual GPU result")
    check(gpu.get("raw_input_provenance") == case["input_files"], "raw provenance differs from canonical case")
    validate_input_ledger(gpu["input_verification"], case, "before GPU")
    validate_input_ledger(gpu["input_verification_after"], case, "after GPU")
    source = config["frozen_sources"][version]
    check(gpu["source_before"] == source and gpu["source_after"] == source, "before/after actual source identity changed")
    verified_source = verify_source(source)
    wall_path = job / "raw_bids_wall.json"
    wall, wall_identity = safe_json(wall_path, touched)
    check(gpu["wall_report"] == str(wall_path) and wall.get("status") == "completed" and wall.get("exit_code") == 0 and
          wall.get("outputs", {}).get("status") == "complete", "actual raw CLI wall/output report is incomplete")
    check(gpu["wall_script_sha256"] == config["wall_script_sha256"] and anatomy.sha(config["wall_script"]) == config["wall_script_sha256"] and
          wall["provenance"]["benchmark_sha256"] == config["wall_script_sha256"], "actual common wall evaluator changed")
    initial = wall["initial_output_state"]
    check(initial.get("output_directory_existed") is False and initial.get("preexisting_run_state") is False and
          initial.get("preexisting_state_files") == [], "raw DWI run reused an output namespace")
    check(wall.get("preprocessing") == [{"topup": "completed", "eddy": "completed", "recon_all": "supplied"}] and
          wall.get("actual_eddy_gp_seeds") == [config["eddy_gp_seed"]], "raw DWI TOPUP/EDDY did not actually execute with the declared seed")
    selected = wall["selected_inputs"]
    output = job / "connectome"
    check(Path(selected["freesurfer_subject_dir"]).resolve() == Path(subject_dir).resolve() and
          selected["dwi"] == str(output / "preproc/eddy/data.nii.gz") and
          selected["bvecs"] == str(output / "preproc/eddy/data.eddy_rotated_bvecs"), "actual CLI selected the wrong anatomy or nonfresh corrected DWI")
    package = Path(wall["provenance"]["fnit_package"])
    check(package.resolve() == (Path(source["directory"]) / "src/fnit").resolve(), "loaded FNIT package is outside frozen source")
    for item in wall["provenance"]["loaded_fnit_files"].values():
        path = Path(item["path"])
        check(path.resolve().is_relative_to(package.resolve()) and anatomy.sha(path) == item["sha256"], "loaded FNIT module byte identity changed")
    files = required_outputs(output, atlases)
    for name, identity in wall["outputs"]["files"].items():
        path = output / name
        check(identity.get("path") == str(path) and identity.get("exists") is True and anatomy.sha(path) == identity["sha256"], "recorded CLI output hash changed")
    budget = gpu.get("memory_budget", {})
    measurements = budget.get("measurements", {})
    check(budget.get("status") == "observed_below_budget" and not budget.get("monitor_issues") and
          all(isinstance(measurements.get(name), (int, float)) and not isinstance(measurements.get(name), bool) and
              math.isfinite(measurements[name]) and 0 <= measurements[name] < 20_000_000_000
              for name in ("process_tree", "allocated_bytes", "reserved_bytes")), "GPU memory qualification is missing or outside the strict budget")
    for relative in files:
        touched[str(output / relative)] = anatomy.sha(output / relative)
    return output, {"gpu_report": gpu_identity, "wall_report": wall_identity, "driver_observation": observation,
                    "actual_source": verified_source, "memory_budget": budget, "driver_timing": record.get("timing"),
                    "worker_wall_seconds": gpu.get("worker_wall_seconds"), "gpu_lock_queue_seconds": gpu.get("gpu_lock_queue_seconds"),
                    "gpu_command_wall_seconds": gpu.get("gpu_command_wall_seconds"), "raw_dwi_cli_total_runtime_seconds": wall.get("total_runtime_seconds"),
                    "timing_scope": gpu.get("execution_scope"), "stages": wall.get("stages"), "stage_qc": wall.get("stage_qc"),
                    "precision_at_exit": wall["provenance"].get("precision_at_exit"), "cli_arguments": wall["cli_arguments"],
                    "stage_timing_note": "wall mode has no diagnostic stage hooks; only actually persisted stage-QC timers are available"}


def scalar_ulp_diagnostic(left, right, dtype):
    """Diagnostic distance after an explicit dtype parse; never an acceptance gate."""
    _, np = anatomy.scientific_modules()
    dtype = np.dtype(dtype)
    bits = np.dtype(f"u{dtype.itemsize}")
    sign = np.asarray(1 << (8 * dtype.itemsize - 1), dtype=bits)
    a = np.asarray(left, dtype=dtype).view(bits)
    b = np.asarray(right, dtype=dtype).view(bits)
    oa = np.where((a & sign) != 0, ~a, a ^ sign).astype(np.uint64)
    ob = np.where((b & sign) != 0, ~b, b ^ sign).astype(np.uint64)
    distance = np.where(oa >= ob, oa - ob, ob - oa)
    return {"diagnostic_dtype": dtype.name, "max_ulp": int(distance.max()) if distance.size else 0,
            "unequal_after_this_parse": int(np.count_nonzero(distance)),
            "ulp_le_1": int(np.count_nonzero(distance <= 1)), "ulp_le_4": int(np.count_nonzero(distance <= 4)),
            "interpretation": "small ULP is compatible with final-bit rounding/reduction differences; does not prove atomic operations caused the difference"}


def matrix_compare(left, right, node_count, kind):
    _, np = anatomy.scientific_modules()
    a, b = np.loadtxt(left, delimiter=",", ndmin=2), np.loadtxt(right, delimiter=",", ndmin=2)
    for matrix in (a, b):
        check(matrix.shape == (node_count, node_count) and np.isfinite(matrix).all() and np.array_equal(matrix, matrix.T), "matrix/node shape, finiteness or symmetry invalid")
        if kind == "count":
            check(np.all(matrix >= 0) and np.array_equal(matrix, np.rint(matrix)), "count matrix is not nonnegative exact integers")
    result = anatomy.compare_arrays(a, b)
    index = np.triu_indices(node_count, k=0)
    x, y = a[index], b[index]
    delta = y - x
    denominator = float(np.linalg.norm(x))
    common = (x != 0) & (y != 0)
    result.update(csv_decoding_dtype="numpy.loadtxt float64; original GPU tensor dtype is not stored in CSV",
                  edge_scope="upper triangle including retained self-connections", edge_neq=int(np.count_nonzero(x != y)),
                  support_neq=int(np.count_nonzero((x != 0) != (y != 0))), common_support_edges=int(common.sum()),
                  relative_l2=float(np.linalg.norm(delta) / denominator) if denominator else (0.0 if np.all(delta == 0) else None),
                  mean_absolute_error=float(np.abs(delta).mean()),
                  correlation=float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else None,
                  common_edge_correlation=float(np.corrcoef(x[common], y[common])[0, 1]) if common.sum() > 1 and np.std(x[common]) > 0 and np.std(y[common]) > 0 else None,
                  baseline_upper_sum=float(x.sum()), candidate_upper_sum=float(y.sum()),
                  strict_count_equal=bool(np.array_equal(a, b)) if kind == "count" else None)
    if kind != "count":
        result["ULP_diagnostics"] = [scalar_ulp_diagnostic(a, b, "float64"), scalar_ulp_diagnostic(a, b, "float32")]
        result["float32_diagnostic_scope"] = "explicit reparse of persisted CSV for last-bit diagnosis; exact/error comparisons above use original float64 text decoding; no inference of unstored tensor dtype"
    return result


def node_semantics(path):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fields = reader.fieldnames
        rows = list(reader)
    check(fields is not None and {"index", "original_label", "hemisphere", "name"}.issubset(fields) and rows and
          [int(row["index"]) for row in rows] == list(range(1, len(rows) + 1)), "invalid canonical nodes.tsv")
    return fields, rows


def image_compare(left, right):
    """Sequential 4D proxy frames bound CPU buffers; full voxels are compared."""
    nib, np = anatomy.scientific_modules()
    a, b = nib.load(left, keep_file_open=True), nib.load(right, keep_file_open=True)
    geometry = {"shape_equal": a.shape == b.shape, "stored_dtype_baseline": a.get_data_dtype().str,
                "stored_dtype_candidate": b.get_data_dtype().str, "stored_dtype_equal": a.get_data_dtype() == b.get_data_dtype(),
                "affine": anatomy.compare_arrays(a.affine, b.affine),
                "zooms": anatomy.compare_arrays(np.asarray(a.header.get_zooms()), np.asarray(b.header.get_zooms()))}
    check(len(a.shape) in (3, 4) and len(b.shape) in (3, 4), "only actual 3D/4D image outputs supported")
    check(geometry["shape_equal"], "actual image dimensions differ")
    ha, hb = hashlib.sha256(), hashlib.sha256()
    squared_error, maximum, unequal, bit_unequal, total, finite_total, dtype_equal = 0.0, 0.0, 0, 0, 0, 0, True
    nonfinite_frames = []
    frames = a.shape[3] if len(a.shape) == 4 else 1
    for frame in range(frames):
        selection = (..., frame) if len(a.shape) == 4 else (...,)
        x, y = np.asanyarray(a.dataobj[selection]), np.asanyarray(b.dataobj[selection])
        try:
            values = anatomy.compare_arrays(x, y, allow_matching_nonfinite=True)
        except ValueError as error:
            raise ValueError(f"actual image {left.name}, frame {frame}: {error}") from error
        ha.update(anatomy.canonical(x).tobytes()); hb.update(anatomy.canonical(y).tobytes())
        total += x.size; unequal += values["numeric_neq"]
        dtype_equal = dtype_equal and values["dtype_equal"]
        if values["raw_scalar_bits_neq"] is None: bit_unequal = None
        elif bit_unequal is not None: bit_unequal += values["raw_scalar_bits_neq"]
        finite_count = values["nonfinite"]["finite_intersection_elements"]
        finite_total += finite_count
        squared_error += values["rmse"] ** 2 * finite_count
        if finite_count != x.size:
            nonfinite_frames.append({"frame": frame, **values["nonfinite"]})
        maximum = max(maximum, values["max_abs_error"])
    forms = {}
    for form in ("qform", "sform"):
        if hasattr(a.header, "get_" + form) and hasattr(b.header, "get_" + form):
            aa, ca = getattr(a.header, "get_" + form)(coded=True)
            bb, cb = getattr(b.header, "get_" + form)(coded=True)
            forms[form] = {"baseline_code": int(ca), "candidate_code": int(cb), "code_equal": int(ca) == int(cb),
                           "matrix": anatomy.compare_arrays(aa, bb) if aa is not None and bb is not None else None,
                           "both_uncoded": aa is None and bb is None}
    geometry["NIfTI_forms"] = forms
    geometry["units_baseline"] = list(a.header.get_xyzt_units()) if hasattr(a.header, "get_xyzt_units") else None
    geometry["units_candidate"] = list(b.header.get_xyzt_units()) if hasattr(b.header, "get_xyzt_units") else None
    geometry.update(dataobj_scaling_baseline={"slope": float(getattr(a.dataobj, "slope", 1)), "inter": float(getattr(a.dataobj, "inter", 0))},
                    dataobj_scaling_candidate={"slope": float(getattr(b.dataobj, "slope", 1)), "inter": float(getattr(b.dataobj, "inter", 0))})
    data = {"numeric_neq": int(unequal), "raw_scalar_bits_neq": bit_unequal, "rmse": math.sqrt(squared_error / max(1, finite_total)),
            "max_abs_error": maximum, "elements": int(total), "effective_data_dtype_equal": dtype_equal,
            "baseline_payload_sha256": ha.hexdigest(), "candidate_payload_sha256": hb.hexdigest(),
            "payload_hash_order": "frames in order; each XYZ frame little-endian C order", "voxel_reduction_or_approximation": False,
            "finite_intersection_elements": int(finite_total), "nonfinite_frames": nonfinite_frames,
            "error_scope": "all actual finite elements; matching NaN payload and +/-Inf positions/signs checked exactly; no data changed"}
    strict = geometry["stored_dtype_equal"] and geometry["affine"]["exact_scientific_array_equal"] and geometry["zooms"]["exact_scientific_array_equal"] and \
             geometry["dataobj_scaling_baseline"] == geometry["dataobj_scaling_candidate"] and dtype_equal and unequal == 0 and bit_unequal == 0 and \
             geometry["units_baseline"] == geometry["units_candidate"] and all(entry["code_equal"] and
             (entry["both_uncoded"] or entry["matrix"] is not None and entry["matrix"]["exact_scientific_array_equal"]) for entry in forms.values())
    return {"geometry": geometry, "data": data, "strict_scientific_equal": bool(strict)}


def json_differences(left, right, prefix=""):
    """Describe actual JSON differences; paths/timers are not scientific gates."""
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(set(left) | set(right)):
            location = prefix + "/" + key
            if key not in left or key not in right:
                result.append({"path": location, "baseline": left.get(key), "candidate": right.get(key), "missing_on_one_side": True})
            else:
                result.extend(json_differences(left[key], right[key], location))
        return result
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return [item for index, (a, b) in enumerate(zip(left, right)) for item in json_differences(a, b, prefix + f"/{index}")]
    return [] if left == right else [{"path": prefix, "baseline": left, "candidate": right}]



def verify_cached_anatomy(result):
    """An earlier actual FS comparison never excuses later source mutations."""
    for version in ("baseline", "candidate"):
        record = result[version]
        for key in ("reconstruction_report", "validation_report"):
            identity = record[key]
            check(anatomy.sha(identity["path"]) == identity["sha256"], "original official report changed after FS comparison")
        t1 = record["raw_T1"]
        check(anatomy.sha(t1["path"]) == t1["sha256"], "raw T1 changed after independent FS comparison")
    for entry in result["files"].values():
        for key in ("baseline_file", "candidate_file"):
            identity = entry[key]
            check(anatomy.sha(identity["path"]) == identity["sha256"], "actual FS data changed after independent comparison")
            link = identity.get("internal_link")
            if link:
                path = Path(identity["path"])
                check(path.is_symlink() and os.readlink(path) == link["link_target"] and str(path.resolve()) == link["resolved"],
                      "official FS internal link changed after comparison")

def compare_gpu_case(case, baseline_root, candidate_root, baseline_driver, candidate_driver, fs_result, atlases):
    verify_cached_anatomy(fs_result)
    touched = {}
    a, baseline = validate_gpu_run(baseline_root, baseline_driver, "baseline", case, fs_result["baseline"]["subject_directory"], atlases, touched)
    b, candidate = validate_gpu_run(candidate_root, candidate_driver, "candidate", case, fs_result["candidate"]["subject_directory"], atlases, touched)
    check(baseline["precision_at_exit"] == candidate["precision_at_exit"], "actual precision settings differ")
    check(baseline["wall_report"] and candidate["wall_report"], "actual wall evidence absent")
    for entry in case["input_files"]:
        check(anatomy.sha(entry["path"]) == entry["sha256"], "actual raw input SHA changed before scientific comparison")
        touched[entry["path"]] = entry["sha256"]
    result = {"status": "completed", "baseline": baseline, "candidate": candidate, "atlases": {}, "images": {}, "numeric_files": {},
              "scientific_acceptance": "not_decided; exact count and numerical errors are reported separately from MRtrix repeat-envelope acceptance",
              "independent_official_anatomy_equal": fs_result["all_requested_scientific_data_equal"]}
    for relative in _SHARED_IMAGES:
        result["images"][relative] = image_compare(a / relative, b / relative)
    _, np = anatomy.scientific_modules()
    for relative, delimiter in (("dwi_to_t1_world.csv", ","), ("preproc/eddy/data.eddy_rotated_bvecs", None), ("preproc/topup/acqparams.txt", None)):
        result["numeric_files"][relative] = anatomy.compare_arrays(np.loadtxt(a / relative, delimiter=delimiter, ndmin=2), np.loadtxt(b / relative, delimiter=delimiter, ndmin=2))
    result["T1_to_MNI_transform"] = {"status": "not_serialized_by_current_CLI", "scope": "no invented transform comparison; derived atlas label volumes below are actual outputs"}
    for atlas in atlases:
        aa, bb = a / "atlases" / atlas, b / "atlases" / atlas
        fields_a, rows_a = node_semantics(aa / "nodes.tsv"); fields_b, rows_b = node_semantics(bb / "nodes.tsv")
        check(fields_a == fields_b and rows_a == rows_b, "atlas row/column node semantics differ: " + atlas)
        labels_a = np.loadtxt(aa / "region_labels.csv", delimiter=",", ndmin=1)
        labels_b = np.loadtxt(bb / "region_labels.csv", delimiter=",", ndmin=1)
        check(labels_a.size == len(rows_a) and np.array_equal(labels_a, labels_b), "atlas region label order differs: " + atlas)
        result["atlases"][atlas] = {"nodes_semantics_equal": True, "node_count": len(rows_a),
            "node_columns": fields_a, "node_rows": rows_a, "region_labels": anatomy.compare_arrays(labels_a, labels_b),
            "atlas_labels": image_compare(aa / "atlas_dwi.nii.gz", bb / "atlas_dwi.nii.gz"),
            "matrices": {kind: matrix_compare(aa / f"connectome_{kind}.csv", bb / f"connectome_{kind}.csv", len(rows_a), kind) for kind in MATRICES}}
    state_a, _ = safe_json(a / "run_state.json", touched); state_b, _ = safe_json(b / "run_state.json", touched)
    check(state_a["options"]["atlas"] == state_b["options"]["atlas"] == atlases and
          all(state_a["options"][key] == state_b["options"][key] for key in ("n_seeds", "seed", "shell_bvals", "device")), "run-state scientific settings differ")
    result["parameters"] = {"baseline": state_a["options"], "candidate": state_b["options"], "differences": json_differences(state_a["options"], state_b["options"])}
    qc_a, _ = safe_json(a / "preproc/eddy/data.eddy_qc.json", touched); qc_b, _ = safe_json(b / "preproc/eddy/data.eddy_qc.json", touched)
    result["persisted_EDDY_QC"] = {"baseline": qc_a, "candidate": qc_b, "differences": json_differences(qc_a, qc_b)}
    result["wall_stage_QC_differences"] = json_differences(baseline["stage_qc"], candidate["stage_qc"])
    result["count_exact_all_atlases"] = all(entry["matrices"]["count"]["strict_count_equal"] for entry in result["atlases"].values())
    result["matrix_numeric_exact_all_atlases"] = all(matrix["exact_scientific_array_equal"] for entry in result["atlases"].values() for matrix in entry["matrices"].values())
    result["no_FS_variance_attribution_to_FNIT"] = True
    anatomy.verify_unchanged(touched)
    return result


def preserve_configs(namespace, baseline_root, origins, binding_path, manifest_path):
    watched = {}
    paths = [Path(baseline_root) / "cohort_config.json", Path(binding_path), Path(manifest_path)]
    paths += [Path(origin["config_identity"]["path"]) for origin in origins]
    for index, path in enumerate(paths):
        check(path.is_file() and not path.is_symlink(), "immutable comparison input absent or linked")
        watched[str(path)] = anatomy.sha(path)
        with (Path(namespace) / f"original_config_snapshot.{index}.bytes.json").open("xb") as stream:
            stream.write(path.read_bytes())
    return watched


def compare_once(options, state, cases, origins, mapping):
    namespace = options.report_dir
    for case in cases:
        record = state["cases"][case["case_id"]]
        if record["status"] == "completed_comparison":
            continue
        try:
            origin = origins[mapping[case["case_id"]]]
            if not record.get("anatomy"):
                preparation_observation = validate_prepared_case(origin, case)
                raw_job = options.baseline_anatomy_root / "baseline" / case["case_id"]
                validation = raw_job / "recon_report.revalidated.json"
                if not validation.exists():
                    validation = options.baseline_root / "baseline" / case["case_id"] / "anatomy_origin.json"
                if not (raw_job / "recon_report.json").exists() or not validation.exists():
                    raise WaitingForActualResults("baseline official reconstruction/revalidation is not yet complete")
                validation_report, _ = safe_json(validation)
                if validation_report.get("status") == "running":
                    raise WaitingForActualResults("baseline actual revalidation is running")
                anatomy_result = anatomy.compare_fresh_case(case, options.baseline_anatomy_root, Path(origin["root"]), baseline_validation_path=validation)
                anatomy_result["candidate_preparation_observation"] = preparation_observation
                anatomy_result["original_preparation_config_identity"] = origin["config_identity"]
                anatomy_path = namespace / f"{case['case_id']}.official_anatomy.json"
                check(not anatomy_path.exists(), "existing anatomy comparison is never resumed or overwritten")
                anatomy.atomic_json(anatomy_path, anatomy_result)
                record["anatomy"] = {"path": str(anatomy_path), "sha256": anatomy.sha(anatomy_path),
                                      "all_requested_scientific_data_equal": anatomy_result["all_requested_scientific_data_equal"]}
            else:
                anatomy_result, identity = safe_json(record["anatomy"]["path"])
                check(identity == {"path": record["anatomy"]["path"], "sha256": record["anatomy"]["sha256"]}, "completed anatomy comparison changed")
            ready_case(options.baseline_driver, "baseline", case["case_id"])
            ready_case(options.candidate_driver, "candidate", case["case_id"])
            gpu_result = compare_gpu_case(case, options.baseline_root, options.candidate_root, options.baseline_driver,
                                          options.candidate_driver, anatomy_result, origins[0]["config"]["atlases"])
            output_path = namespace / f"{case['case_id']}.connectome.json"
            check(not output_path.exists(), "existing GPU comparison report is never overwritten")
            anatomy.atomic_json(output_path, gpu_result)
            fingerprints = {version: gpu_result[version]["actual_source"]["source_fingerprint"] for version in ("baseline", "candidate")}
            check(not state.get("actual_source_fingerprints") or state["actual_source_fingerprints"] == fingerprints,
                  "formal source versions changed across the compared cases")
            state.update(actual_source_fingerprints=fingerprints, candidate_source_status="actual completed candidate GPU source verified")
            record.update(status="completed_comparison", connectome={"path": str(output_path), "sha256": anatomy.sha(output_path)},
                          count_exact_all_atlases=gpu_result["count_exact_all_atlases"],
                          matrix_numeric_exact_all_atlases=gpu_result["matrix_numeric_exact_all_atlases"], completed_utc=anatomy.utc())
        except WaitingForActualResults as waiting:
            record.update(status="waiting_actual_outputs", waiting_reason=str(waiting), last_observation_utc=anatomy.utc())
        except Exception as error:
            record.update(status="failed_comparison", error={"type": type(error).__name__, "message": str(error)})
        anatomy.atomic_json(namespace / "status.json", state)
    completed = sum(record["status"] == "completed_comparison" for record in state["cases"].values())
    failed = sum(record["status"] == "failed_comparison" for record in state["cases"].values())
    state.update(completed_cases=completed, anatomy_compared_cases=sum(bool(record.get("anatomy")) for record in state["cases"].values()),
                 failed_cases=failed, status="completed_actual_ten_case_comparison" if completed == 10 else
                 "failed_actual_comparison" if failed else "waiting_actual_outputs",
                 scientific_acceptance="not_decided; comparison coverage is not MRtrix repeat-envelope acceptance")
    anatomy.atomic_json(namespace / "status.json", state)
    with (namespace / "cases.csv.partial").open("w", newline="") as stream:
        fields = ("case_id", "status", "waiting_reason", "count_exact_all_atlases", "matrix_numeric_exact_all_atlases")
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(state["cases"].values())
    os.replace(namespace / "cases.csv.partial", namespace / "cases.csv")
    return state



def check_report_namespace(report_dir, protected):
    destination = Path(report_dir).resolve()
    for path in protected:
        original = Path(path).resolve()
        check(not destination.is_relative_to(original) and not original.is_relative_to(destination),
              "comparison namespace overlaps original source/data/status/output")


def bind_prior_anatomy(prior_dir, options, state, watched):
    """Reuse only verified science from the immutable known v1 reader failure.

    This is a new CPU comparison namespace, never a resume of MRI computation.
    Original failures and their frozen helper identities remain visible.
    """
    prior_dir = Path(prior_dir)
    prior_path = prior_dir / "status.json"
    prior, identity = safe_json(prior_path)
    check(prior.get("status") == "failed_actual_comparison" and prior.get("end_utc"),
          "prior comparison must be a finished, preserved failure")
    check(prior.get("manifest") == state["manifest"] and prior.get("preparation_bindings") == state["preparation_bindings"] and
          prior.get("case_origin_index") == state["case_origin_index"], "prior comparison inputs/bindings changed")
    for key in ("manifest", "prep_bindings", "baseline_anatomy_root", "baseline_root", "candidate_root", "baseline_driver", "candidate_driver"):
        check(prior["configuration"].get(key) == str(getattr(options, key)), "prior actual comparison scope changed")
    errors = [record.get("error") for record in prior["cases"].values() if record.get("status") == "failed_comparison"]
    check(errors and all(error == {"type": "ValueError", "message": "nonfinite scientific array"} for error in errors),
          "prior failure is not the explicitly supported matching-undefined-image reader failure")
    watched[str(prior_path)] = identity["sha256"]
    reused = []
    for case_id, record in prior["cases"].items():
        check(case_id in state["cases"], "prior comparison has an unknown case")
        if not record.get("anatomy"):
            continue
        result_path = Path(record["anatomy"]["path"])
        check(result_path.parent.resolve() == prior_dir.resolve(), "prior anatomy report is outside its original namespace")
        result, report_identity = safe_json(result_path)
        check(report_identity["sha256"] == record["anatomy"]["sha256"] and result.get("case_id") == case_id and
              result.get("status") == "completed" and result.get("all_requested_scientific_data_equal") is True,
              "prior actual anatomy is incomplete, changed or has scientific differences")
        verify_cached_anatomy(result)
        state["cases"][case_id]["anatomy"] = dict(record["anatomy"])
        watched[str(result_path)] = report_identity["sha256"]
        reused.append(case_id)
    state["prior_comparison_binding"] = {"original_status": identity, "original_end_utc": prior["end_utc"],
        "original_tool_sha256": prior["tool_sha256"], "original_anatomy_tool_sha256": prior["anatomy_tool_sha256"],
        "original_errors": errors, "reused_actual_anatomy_cases": reused,
        "scope": "explicit binding of verified original anatomy reports only; original failed status and helper remain immutable; no MRI rerun/resume or preprocessing reuse"}

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prep-bindings", type=Path, required=True)
    parser.add_argument("--baseline-anatomy-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--baseline-driver", type=Path, required=True, help="actual baseline driver status.json")
    parser.add_argument("--candidate-driver", type=Path, required=True, help="actual candidate driver status.json, may be absent until formal run starts")
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--prior-comparison-dir", type=Path, help="explicit immutable v1 nonfinite-reader failure; reuse only its verified completed official anatomy reports in a NEW comparison namespace")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-hours", type=float, default=72)
    parser.add_argument("--once", action="store_true", help="one real observation; waiting remains waiting, never ten-case completion")
    options = parser.parse_args(argv)
    check(1 <= options.poll_seconds <= 60 and 0 < options.timeout_hours <= 168, "invalid bounded observation interval")
    for path in (options.manifest, options.prep_bindings, options.baseline_anatomy_root, options.baseline_root,
                 options.candidate_root, options.baseline_driver, options.candidate_driver, options.report_dir):
        check(path.is_absolute(), "explicit absolute paths are required")
    if options.prior_comparison_dir is not None:
        check(options.prior_comparison_dir.is_absolute(), "prior comparison must be an explicit absolute path")
    manifest, manifest_identity = safe_json(options.manifest)
    cases = manifest_cases(manifest)
    origins, mapping, binding_identity = load_origins(options.prep_bindings, cases)
    for root in (options.baseline_anatomy_root, options.baseline_root):
        original, _ = safe_json(root / "input_manifest.json")
        check(manifest_cases(original) == cases, "baseline original/raw-DWI canonical input cases changed")
    baseline_config, _ = safe_json(options.baseline_root / "cohort_config.json")
    check(baseline_config["atlases"] == origins[0]["config"]["atlases"] and baseline_config["cpu_threads"] == 8,
          "baseline scientific/official settings differ from candidate preparation")
    verify_source(baseline_config["frozen_sources"]["baseline"])
    protected = [options.baseline_anatomy_root, options.baseline_root, options.candidate_root]
    protected += [Path(origin["root"]) for origin in origins] + [Path(origin["driver_dir"]) for origin in origins]
    protected += [options.baseline_driver.parent, options.candidate_driver.parent, options.prep_bindings.parent]
    protected += [Path(identity["directory"]) for identity in baseline_config["frozen_sources"].values()]
    if options.prior_comparison_dir is not None:
        protected.append(options.prior_comparison_dir)
    check_report_namespace(options.report_dir, protected)
    options.report_dir.parent.mkdir(parents=True, exist_ok=True)
    options.report_dir.mkdir(exist_ok=False)
    watched = preserve_configs(options.report_dir, options.baseline_root, origins, options.prep_bindings, options.manifest)
    original_config = options.baseline_anatomy_root / "cohort_config.json"
    watched[str(original_config)] = anatomy.sha(original_config)
    with (options.report_dir / "original_baseline_anatomy_config.bytes.json").open("xb") as stream:
        stream.write(original_config.read_bytes())
    state = {"schema_version": 1, "status": "waiting_actual_outputs", "start_utc": anatomy.utc(),
             "controller_host": socket.gethostname(), "manifest": manifest_identity, "preparation_bindings": binding_identity,
             "case_origin_index": mapping, "requested_cases": 10, "candidate_source_status": "unknown until actual completed candidate GPU report/config is read",
             "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(options).items()},
             "tool_sha256": anatomy.sha(__file__), "anatomy_tool_sha256": anatomy.sha(anatomy.__file__),
             "cases": {case["case_id"]: {"case_id": case["case_id"], "status": "waiting_actual_outputs"} for case in cases},
             "scope": "read-only actual CPU scientific comparisons; no original-source/status/output mutations; unknown is not ready; staged wall and queue/gap observations are retained without stage-sum full-wall claims",
             "GPU_used": False, "original_namespaces_modified": False}
    if options.prior_comparison_dir is not None:
        bind_prior_anatomy(options.prior_comparison_dir, options, state, watched)
    started = time.perf_counter()
    try:
        while True:
            candidate_config = options.candidate_root / "staged_gpu_config.json"
            if candidate_config.exists() and str(candidate_config) not in watched:
                check(not candidate_config.is_symlink(), "candidate staged config must be an actual frozen file")
                candidate_config_value, _ = safe_json(candidate_config)
                check_report_namespace(options.report_dir, [Path(identity["directory"]) for identity in candidate_config_value["frozen_sources"].values()])
                with (options.report_dir / "candidate_staged_config.original_bytes.json").open("xb") as stream:
                    stream.write(candidate_config.read_bytes())
                watched[str(candidate_config)] = anatomy.sha(candidate_config)
                state["candidate_config_observation"] = {"path": str(candidate_config), "sha256": watched[str(candidate_config)]}
            for path, digest in watched.items():
                check(anatomy.sha(path) == digest, "original frozen comparison config/manifest changed")
            check(anatomy.sha(__file__) == state["tool_sha256"] and anatomy.sha(anatomy.__file__) == state["anatomy_tool_sha256"],
                  "frozen comparison helper changed while observing actual results")
            compare_once(options, state, cases, origins, mapping)
            print(json.dumps({"status": state["status"], "completed_cases": state["completed_cases"],
                              "anatomy_compared_cases": state["anatomy_compared_cases"], "failed_cases": state["failed_cases"]}), flush=True)
            if options.once or state["status"] != "waiting_actual_outputs":
                break
            if (options.report_dir / "STOP_OBSERVATION").exists():
                state["status"] = "stopped_observation_incomplete"
                break
            if time.perf_counter() - started >= options.timeout_hours * 3600:
                state["status"] = "timed_out_waiting_actual_outputs"
                break
            time.sleep(options.poll_seconds)
    except Exception as error:
        state.update(status="failed_actual_comparison", error={"type": type(error).__name__, "message": str(error)})
    state.update(end_utc=anatomy.utc(), observation_controller_wall_seconds=time.perf_counter() - started,
                 controller_timing_scope="actual CPU verification/comparison plus observation waits; not MRI pipeline runtime")
    anatomy.atomic_json(options.report_dir / "status.json", state)
    return 0 if state["status"] in ("waiting_actual_outputs", "completed_actual_ten_case_comparison") else 1


if __name__ == "__main__":
    raise SystemExit(main())
