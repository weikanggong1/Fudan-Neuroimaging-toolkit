#!/usr/bin/env python3
"""Two real-data TOPUP diagnostics with the exact saved reference b0 pair.

This independent component diagnostic deliberately reads original TOPUP inputs
and outputs. It never runs a selector, EDDY, FSL or a production pipeline, and
never replaces a cohort output. The private plan binds all read-only inputs;
the public report contains anonymous roles, values and hashes only. Admission
and GNU timing are supplied by the existing shared-lock cohort controller.
"""
from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
from dataclasses import asdict
import datetime
import functools
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import sys
import time

STARTED = time.perf_counter()
REVISION = "bf339a0368a7711d2c6ca3477c8d7dc1fc17e75a"
SOURCE_DIGEST = "f13a40989b96d9e3608a427a1fe10d1960b20f146c768a3dd101f84fe4deae1e"
CORE_DIGEST = "d6b9838ca62ffeaa32b608a860520fc3feb5e66582064303a6de47f199e2b8e8"
SAMPLING_DIGEST = "ee19a764849bda80137312ed3ab8f1bf0aaef5e13f852f49f435e511b40d2427"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def record(path):
    return {"available": True, "sha256": digest(path), "size_bytes": Path(path).stat().st_size}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def bound_record(path, expected):
    value = record(path)
    require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected), "expected_digest_invalid")
    require(value["sha256"] == expected, "input_or_resource_digest_mismatch")
    return value


def parse_configuration(text, config):
    """Reject any literal schedule/option outside the frozen implementation."""
    parsed = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"--([a-z]+)\s*=\s*(\S+)", line)
        require(match is not None, "configuration_line_unrecognized")
        key, value = match.groups()
        require(key not in parsed, "configuration_duplicate_option")
        parsed[key] = value
    schedules = {
        "warpres": config.warp_resolution_mm, "subsamp": config.subsampling,
        "fwhm": config.fwhm_mm, "miter": config.maximum_iterations,
        "lambda": config.regularization,
    }
    for key, values in schedules.items():
        require(key in parsed, "configuration_schedule_missing")
        actual = tuple(float(value) for value in parsed[key].split(","))
        require(actual == tuple(map(float, values)), "configuration_schedule_mismatch")
    options = {
        "ssqlambda": "1", "regmod": "bending_energy",
        "estmov": "1,1,1,1,1,0,0,0,0", "minmet": "0,0,0,0,0,1,1,1,1",
        "splineorder": "3", "numprec": "double", "interp": "spline", "scale": "1",
    }
    require(set(parsed) == set(schedules) | set(options), "configuration_option_set_mismatch")
    require(all(parsed[key] == value for key, value in options.items()), "configuration_option_mismatch")
    return {"literal_configuration_matches_frozen_defaults": True,
            "schedule": asdict(config), "other_options": options}


class CompilerObserver(AbstractContextManager):
    """Observe Triton's compiler entry without changing arguments or kernels.

    A fresh empty TRITON_CACHE_DIR is supplied by the launcher. Observed compiler
    time includes compiler/cache administration and is not kernel execution.
    Unrecognized runtime versions retain unknown compiler time.
    """
    def __init__(self):
        self.module = self.original = None
        self.calls = []
        self.available = False

    def __enter__(self):
        try:
            self.module = importlib.import_module("triton.compiler")
            self.original = getattr(self.module, "compile", None)
        except ImportError:
            return self
        if not callable(self.original):
            return self
        self.available = True

        @functools.wraps(self.original)
        def observed(*args, **kwargs):
            started = time.perf_counter()
            status = "failed"
            try:
                result = self.original(*args, **kwargs)
                status = "complete"
                return result
            finally:
                self.calls.append({"seconds": time.perf_counter() - started, "status": status})

        self.module.compile = observed
        return self

    def __exit__(self, *exception):
        if self.available:
            self.module.compile = self.original
        return False

    def report(self):
        observed = self.available and bool(self.calls)
        return {"status": "observed" if observed else "unknown",
                "compile_calls": len(self.calls) if self.available else None,
                "wall_seconds": sum(item["seconds"] for item in self.calls) if observed else None,
                "calls": self.calls,
                "scope": "Triton compiler entry wall time; includes compiler/cache administration; no warm-up inference"}


def movement_comparison(left, right, comparison):
    import numpy as np
    a, b = np.loadtxt(left, ndmin=2), np.loadtxt(right, ndmin=2)
    require(a.shape == b.shape == (2, 6), "movement_shape_mismatch")
    require(np.isfinite(a).all() and np.isfinite(b).all(), "movement_nonfinite")
    columns = []
    for index in range(6):
        accumulator = comparison.Statistics(2, sample_limit=2)
        accumulator.update(a[:, index], b[:, index])
        columns.append({"column": index, "unit": "mm" if index < 3 else "radians",
                        **accumulator.result()})
    return {"candidate": record(left), "reference": record(right), "shape": [2, 6],
            "columns": columns, "candidate_values": a.tolist(), "reference_values": b.tolist()}


def run(arguments, report):
    plan = json.loads(arguments.plan.read_text())
    require(plan.get("schema") == "fnit_matched_topup_private_plan_v1", "plan_schema_mismatch")
    require(plan.get("source_revision") == REVISION, "source_revision_mismatch")
    require({item.get("case_id") for item in plan["cases"]} == {"case02", "case08"}
            and len(plan["cases"]) == 2, "diagnostic_case_set_mismatch")
    row = next(item for item in plan["cases"] if item["case_id"] == arguments.case_id)
    source = Path(plan["frozen_src"]).resolve()
    package = source / "fnit"
    hashes = {str(path.relative_to(package)): digest(path) for path in sorted(package.rglob("*.py"))}
    source_digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    require(len(hashes) == 433 and source_digest == SOURCE_DIGEST, "frozen_source_manifest_mismatch")
    require(hashes["topup/core.py"] == CORE_DIGEST and hashes["topup/_sampling_cuda.py"] == SAMPLING_DIGEST,
            "frozen_topup_source_mismatch")
    sys.path.insert(0, str(source))
    from fnit.topup import core
    require(Path(core.__file__).resolve() == package / "topup/core.py", "frozen_core_import_mismatch")
    import torch
    import nibabel as nib
    import numpy as np

    index = json.loads(Path(plan["server_index"]).read_text())
    task = index.get("active_tasks", {}).get("dmri_public10_20261002", {})
    require(task.get("frozen_source_revision") == REVISION and
            Path(task["frozen_source_real_path"]).resolve() == source, "server_index_source_binding_mismatch")
    comparison_path = Path(plan["comparison_module"])
    comparison_record = bound_record(comparison_path, plan["comparison_module_sha256"])
    spec = importlib.util.spec_from_file_location("fixed_pair_comparison", comparison_path)
    comparison = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparison)
    report["source"] = {"revision": REVISION, "python_file_count": len(hashes),
        "python_manifest_sha256": source_digest, "topup_core_sha256": CORE_DIGEST,
        "topup_sampling_cuda_sha256": SAMPLING_DIGEST,
        "diagnostic_script": record(__file__), "comparison_module": comparison_record,
        "private_plan": record(arguments.plan), "index_at_launch": record(plan["server_index"]),
        "index_snapshot_used_for_preparation_sha256": plan["index_snapshot_sha256"]}
    reference_report_path = Path(row["reference_report"])
    reference_record = bound_record(reference_report_path, row["reference_report_sha256"])
    reference = json.loads(reference_report_path.read_text())
    require(reference.get("case_id") == arguments.case_id and reference.get("registration_backend") == "tbss"
            and reference.get("status") == "complete", "reference_report_case_or_status_mismatch")
    original = Path(row["reference_dir"])
    pair, acqp = original / "topup/B0_AP_PA.nii.gz", original / "topup/acqparams.txt"
    mask_path = original / "eddy/nodif_brain_mask.nii.gz"
    config_path, template_path = Path(plan["topup_config"]), Path(plan["reference_template"])
    checks = {
        "literal_reference_topup_pair": bound_record(pair, reference["preparation_outputs"]["topup_pair"]["sha256"]),
        "literal_reference_acquisition_parameters": bound_record(acqp, reference["preparation_outputs"]["acqp"]["sha256"]),
        "fixed_reference_native_brain_roi": bound_record(mask_path, reference["preparation_outputs"]["brain_mask"]["sha256"]),
        "literal_b02b0_configuration": bound_record(config_path, reference["configurations"]["b02b0.cnf"]["sha256"]),
        "reference_FA_template_provenance_only": bound_record(template_path, reference["templates"]["FA_reference"]["sha256"]),
    }
    outputs = {"field_hz": "fieldmap_fout.nii.gz", "corrected_b0": "fieldmap_iout.nii.gz",
               "movement": "fieldmap_out_movpar.txt"}
    for role, key in (("field_hz", "topup_fout"), ("corrected_b0", "topup_iout"), ("movement", "topup_movpar")):
        checks["saved_reference_" + role] = bound_record(original / "topup" / outputs[role], reference["intermediate_outputs"][key]["sha256"])
    for index in (1, 2):
        path = original / "topup" / f"fieldmap_jacout_{index:02d}.nii.gz"
        checks[f"saved_reference_jacobian_{index}"] = record(path) if path.is_file() else {"available": False}
    config = core.TOPUPConfig()
    report["configuration"] = parse_configuration(config_path.read_text(), config)
    pair_image = comparison.load_image(pair)
    require(pair_image["values"] is not None and pair_image["values"].shape[-1] == 2
            and pair_image["metadata"]["nonfinite_elements"] == 0, "literal_pair_format_invalid")
    mask = comparison.load_image(mask_path)
    roi = comparison.derived_mask(mask, mask["values"] > .5 if mask["values"] is not None else None)
    require(comparison.geometry_gate(pair_image, roi, spatial_only=True)["passed"], "fixed_native_roi_grid_mismatch")
    report["inputs"] = {"binding_status": "matched", "files": checks,
        "selected_pair_metadata": pair_image["metadata"], "acquisition_parameters": np.loadtxt(acqp, ndmin=2).tolist(),
        "roi": "saved original EDDY brain mask > 0.5; zeros retained inside this fixed ROI",
        "roi_voxels": int(roi["values"].sum()), "selector_bypassed": True}
    report["historical_reference"] = {"report": reference_record,
        "topup_stage_seconds": reference["stages_seconds"].get("topup"),
        "topup_executable": reference.get("FSL_binaries", {}).get("topup"),
        "timing_scope": "existing bound original whole-pipeline TOPUP stage, historical observation; original software not rerun under current load"}
    cache = Path(os.environ.get("TRITON_CACHE_DIR", ""))
    require(bool(os.environ.get("TRITON_CACHE_DIR")) and cache.is_dir() and not any(cache.iterdir()),
            "cold_jit_cache_must_be_fresh_empty_directory")
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda:0")
    require(torch.cuda.is_available(), "cuda_unavailable")
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    torch.cuda.set_per_process_memory_fraction(20_000_000_000 / properties.total_memory, device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats(device)
    runner = core.TorchTOPUP(device=device, config=config)
    print(json.dumps({"event": "matched_pair_topup_start", "case_id": arguments.case_id,
                      "pair_sha256": checks["literal_reference_topup_pair"]["sha256"]}), flush=True)
    with CompilerObserver() as compiler:
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        startup = started - STARTED
        result = runner(pair, acqp)
        torch.cuda.synchronize(device)
        api_wall = time.perf_counter() - started
    sampling = importlib.import_module("fnit.topup._sampling_cuda")
    require(Path(sampling.__file__).resolve() == package / "topup/_sampling_cuda.py"
            and digest(sampling.__file__) == SAMPLING_DIGEST and digest(core.__file__) == CORE_DIGEST,
            "frozen_topup_import_or_post_run_source_mismatch")
    compile_report = compiler.report()
    report["timing"] = {"startup_and_binding_before_api_seconds": startup,
        "synchronized_api_wall_seconds": api_wall, "model_qc_elapsed_seconds": result.qc["elapsed_seconds"],
        "cold_jit": {"cache_policy": "fresh empty per-case directory; no inference warmup", **compile_report},
        "api_minus_observed_compile_seconds": api_wall - compile_report["wall_seconds"] if compile_report["wall_seconds"] is not None else None,
        "scope": "API includes image decoding, CPU preparations, algorithm and transfers; subtracting observed compiler time does not yield pure GPU kernel time; GNU full process clock is separate; lock wait excluded"}
    report["memory"] = {"allocator_limit_bytes": 20_000_000_000,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device))}
    report["runtime"] = {"python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "nibabel": nib.__version__, "cuda_runtime": torch.version.cuda,
        "gpu_name": properties.name, "gpu_uuid": str(getattr(properties, "uuid", "unavailable")),
        "gpu_total_memory_bytes": properties.total_memory, "cuda_device": "cuda:0",
        "torch_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "tf32": torch.backends.cuda.matmul.allow_tf32, "autocast": torch.is_autocast_enabled(),
        "thread_environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "cpu_affinity": sorted(os.sched_getaffinity(0))}
    report["model_qc"] = result.qc
    saved = arguments.output_dir / "topup"
    saving = time.perf_counter()
    result.save(out=saved / "fieldmap_out", fout=saved / "fieldmap_fout.nii.gz",
                iout=saved / "fieldmap_iout.nii.gz", jacout=saved / "fieldmap_jacout.nii.gz")
    report["timing"]["output_save_seconds"] = time.perf_counter() - saving
    post = time.perf_counter()
    pairs = {role: comparison.image_pair(comparison.load_image(saved / outputs[role]),
                                        comparison.load_image(original / "topup" / outputs[role]), roi)
             for role in ("field_hz", "corrected_b0")}
    pairs["movement"] = movement_comparison(saved / outputs["movement"], original / "topup" / outputs["movement"], comparison)
    pairs["jacobians"] = []
    for index in (1, 2):
        name = f"fieldmap_jacout_{index:02d}.nii.gz"
        native = original / "topup" / name
        if not native.is_file():
            pairs["jacobians"].append({"volume": index, "available": False, "reason": "original_saved_jacobian_unavailable"})
            continue
        left, right = comparison.load_image(saved / name), comparison.load_image(native)
        require(right["metadata"].get("sha256") == checks[f"saved_reference_jacobian_{index}"]["sha256"],
                "read_only_jacobian_changed")
        require(comparison.geometry_gate(left, right)["passed"], "saved_jacobian_grid_mismatch")
        canonical_roi = roi["values"][::-1] if result.qc["canonical_x_flip"] else roi["values"]
        require(canonical_roi.shape == left["values"].shape and np.allclose(
            left["image"].header.get_zooms()[:3], pair_image["image"].header.get_zooms()[:3], rtol=0, atol=1e-5), "jacobian_canonical_roi_shape_mismatch")
        jac_roi = {"image": left["image"], "values": canonical_roi,
                   "metadata": {**left["metadata"], "nonfinite_elements": 0, "derived_boolean_mask": True}}
        pairs["jacobians"].append({"volume": index, "available": True,
            "roi_scope": "same fixed native ROI in canonical Analyze-style TOPUP storage; x reversal only when canonical_x_flip, no interpolation",
            **comparison.image_pair(left, right, jac_roi)})
    report["comparisons"] = pairs
    report["candidate_outputs"] = {path.name: record(path) for path in sorted(saved.iterdir()) if path.is_file()}
    report["timing"]["output_hash_and_comparison_seconds"] = time.perf_counter() - post
    require(all(pairs[role]["gate"]["passed"] for role in ("field_hz", "corrected_b0")), "required_output_format_failed")
    require(checks["literal_reference_topup_pair"]["sha256"] == digest(pair)
            and checks["literal_reference_acquisition_parameters"]["sha256"] == digest(acqp), "read_only_input_changed")
    report["status"] = "complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--case-id", choices=("case02", "case08"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()
    os.umask(0o077)
    require(not arguments.report.exists() and not arguments.output_dir.exists(), "diagnostic_output_already_exists")
    arguments.output_dir.mkdir(parents=True, mode=0o700)
    report = {"schema": "fnit_matched_topup_pair_diagnostic_v1", "case_id": arguments.case_id,
              "status": "failed", "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "scope": "independent fixed-reference-pair TOPUP component diagnostic; not a production pipeline run or replacement cohort result",
              "script_sha256": digest(__file__), "selector_bypassed": True,
              "numerical_equivalence_claimed": False, "accuracy_thresholds_applied": False,
              "whole_pipeline_rerun": False, "official_software_invoked": False}
    try:
        run(arguments, report)
    except Exception as error:
        report["failure_type"] = type(error).__name__
        message = str(error)
        if re.fullmatch(r"[a-z_]+", message):
            report["failure_code"] = message
    report["in_process_seconds_before_report_write"] = time.perf_counter() - STARTED
    arguments.report.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    arguments.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"event": "matched_pair_topup_end", "case_id": arguments.case_id,
                      "status": report["status"], "report_sha256": digest(arguments.report)}), flush=True)
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
