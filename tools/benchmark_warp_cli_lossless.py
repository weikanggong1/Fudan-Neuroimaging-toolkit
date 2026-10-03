#!/usr/bin/env python3
"""Time complete FNIT warp CLI processes and validate every output value.

Private configuration::

    {"cases": [{"backend": "applywarp", "input": "/private/dwi.nii.gz",
      "reference": "/private/reference.nii.gz", "warp": "/private/warp.nii.gz",
      "devices": ["cuda:0"],
      "expected_outputs_by_device": {"cuda:0": "/private/validated_baseline.nii"}},
      {"backend": "synthmorph", "input": "/private/bold.nii.gz",
       "warp": "/private/ras_warp.nii.gz", "devices": ["cpu", "cuda:0"],
       "expected_outputs_by_device": {"cpu": "/private/validated_cpu.nii",
                                      "cuda:0": "/private/all_channel_gpu_oracle.nii"}}]}

Use the same project Python and source tree as the component benchmark::

    PYTHONPATH=src python tools/benchmark_warp_cli_lossless.py \
        --case-json /private/cli_cases.json --output-dir /private/cli_benchmark

Each device executes once in a fresh Python process, with no warmup or frame
reduction. The child calls fnit.cli.main (the project has no fnit.__main__).
Only report.public.json is a publication candidate. Inputs, outputs, requests
and unfiltered CLI logs remain in server.private with restricted permissions.
Original software is never invoked. Existing validated same-device baselines
are supplied by the caller, including a full-channel oracle for SynthMorph
CUDA; this tool does not generate or silently substitute a baseline.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


_MARKER = "FNIT_WARP_CLI_METRICS "
_MEMORY_LIMIT_BYTES = 19_000_000_000
_REQUIRED_PEAK_LIMIT_BYTES = 20_000_000_000
_ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value, *, private=False):
    path = Path(path)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    if private:
        path.chmod(0o600)


def cli_arguments(case, device, output):
    """Build actual CLI arguments; automatic frame selection is left intact."""
    backend = case["backend"]
    if backend == "applywarp":
        interpolation = case.get("interpolation", "trilinear")
        if interpolation not in ("trilinear", "nearest", "nn"):
            raise ValueError("invalid ApplyWarp interpolation")
        convention = case.get("warp_convention", "auto")
        if convention not in ("auto", "absolute", "relative"):
            raise ValueError("invalid ApplyWarp warp convention")
        arguments = ["applywarp", "--in", str(case["input"]),
                     "--ref", str(case["reference"]), "--warp", str(case["warp"]),
                     "--out", str(output), "--interp", interpolation,
                     "--datatype", "float", "--device", device]
        if convention != "auto":
            arguments.append("--abs" if convention == "absolute" else "--rel")
        for name in ("premat", "postmat"):
            if case.get(name) is not None:
                arguments.extend([f"--{name}", str(case[name])])
        effective = {"command": "applywarp", "interpolation": interpolation,
                     "warp_convention": convention, "output_dtype": "float32",
                     "reference_argument": True,
                     "premat": case.get("premat") is not None,
                     "postmat": case.get("postmat") is not None}
    elif backend == "synthmorph":
        interpolation = case.get("interpolation", "linear")
        if interpolation not in ("linear", "nearest"):
            raise ValueError("invalid SynthMorph interpolation")
        fill = float(case.get("fill", 0))
        if not math.isfinite(fill):
            raise ValueError("fill must be finite")
        if case.get("premat") is not None or case.get("postmat") is not None:
            raise ValueError("SynthMorph apply CLI does not accept premat/postmat")
        arguments = ["apply", str(case["warp"]), str(case["input"]), str(output),
                     "--method", interpolation, "--fill", str(fill),
                     "--dtype", "float32", "--device", device]
        effective = {"command": "apply", "interpolation": interpolation,
                     "fill": fill, "output_dtype": "float32",
                     "reference_argument": False,
                     "output_grid": "RAS_warp_file_geometry"}
    else:
        raise ValueError("backend must be applywarp or synthmorph")
    effective.update(device=device, frame_chunk_size_argument=None,
                     frame_chunk_policy="automatic", header_only=False)
    return arguments, effective


def validate_case(case):
    """Reject ambiguous baseline selection before starting a timed child."""
    if case.get("backend") not in ("applywarp", "synthmorph"):
        raise ValueError("unsupported backend")
    devices = case.get("devices")
    if not isinstance(devices, list) or not devices or len(devices) != len(set(devices)):
        raise ValueError("devices must be a nonempty list without duplicates")
    expected = case.get("expected_outputs_by_device")
    if not isinstance(expected, dict) or any(device not in expected for device in devices):
        raise ValueError("every requested device needs its own validated expected output")
    for device in devices:
        if device != "cpu" and not (device.startswith("cuda:") and device[5:].isdigit()):
            raise ValueError("devices must use cpu or explicit cuda:N")
        cli_arguments(case, device, "unused.nii.gz")
    roles = ["input", "warp"]
    if case["backend"] == "applywarp" or case.get("reference") is not None:
        roles.append("reference")
    roles.extend(name for name in ("premat", "postmat") if case.get(name) is not None)
    for role in roles:
        if role not in case or not Path(case[role]).is_file():
            raise ValueError(f"missing input role: {role}")
    if any(not Path(expected[device]).is_file() for device in devices):
        raise ValueError("a validated expected output is missing")
    return roles


def run_child(request_path):
    """Fresh-process wrapper: invoke FNIT, report allocator peaks, then exit."""
    request = json.loads(Path(request_path).read_text())
    # Keep the measured interpreter tied to this checkout, without loading a
    # different installed FNIT or a user-site dependency in the child process.
    sys.path.insert(0, str(_ROOT / "src"))
    import torch
    import fnit.cli as cli

    if Path(cli.__file__).resolve() != (_ROOT / "src/fnit/cli.py").resolve():
        raise RuntimeError("child imported a different FNIT checkout")
    torch.set_num_threads(request["threads"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(request["device"])
    effective_cap = None
    if device.type == "cuda":
        torch.cuda.set_device(device)
        total = torch.cuda.get_device_properties(device).total_memory
        effective_cap = min(_MEMORY_LIMIT_BYTES, total)
        torch.cuda.set_per_process_memory_fraction(effective_cap / total, device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    cli.main(request["arguments"])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    main_seconds = time.perf_counter() - started
    metrics = {
        "cli_main_seconds": main_seconds, "device": str(device),
        "threads": torch.get_num_threads(), "tf32": True,
        "float16": False, "output_dtype": "float32",
        "allocator_cap_requested_bytes": _MEMORY_LIMIT_BYTES if device.type == "cuda" else None,
        "allocator_cap_effective_bytes": effective_cap,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
        "python": sys.version.split()[0], "torch": torch.__version__,
    }
    print(_MARKER + json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)


def marker_metrics(log_path):
    metrics = None
    with Path(log_path).open(errors="replace") as stream:
        for line in stream:
            if line.startswith(_MARKER):
                metrics = json.loads(line[len(_MARKER):])
    return metrics


def strict_comparison(expected, candidate, input_path):
    """Use the existing full-image science comparator, with stricter headers."""
    # Imports and complete-output validation are outside the subprocess wall
    # clock. compare_images inflates each saved image once and scans all slabs.
    comparator = _ROOT / "tools/validate_fnirt_pipeline_lossless.py"
    spec = importlib.util.spec_from_file_location("_fnit_cli_lossless_comparator", comparator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checks = module.compare_images(Path(expected), Path(candidate))
    import nibabel as nib
    import numpy as np

    source, output = [nib.load(str(path)) for path in (input_path, candidate)]
    input_frames = source.shape[3] if source.ndim == 4 else 1
    output_frames = output.shape[3] if output.ndim == 4 else 1
    checks.update({
        "all_frames_preserved": source.ndim == output.ndim and input_frames == output_frames,
        "float32_output": output.get_data_dtype().kind == "f" and output.get_data_dtype().itemsize == 4,
        "tr_preserved_from_input": source.header.get_zooms()[3:] == output.header.get_zooms()[3:],
        "maximum_error_finite": bool(np.isfinite(checks.get("maximum_absolute_error", float("nan")))),
    })
    required = ("shape_equal", "dtype_equal", "affine_equal", "scientific_header_equal",
                "full_header_bytes_equal_diagnostic", "extensions_equal", "bitwise_equal",
                "all_frames_preserved", "float32_output", "tr_preserved_from_input",
                "maximum_error_finite")
    passed = all(checks.get(key, False) for key in required)
    passed &= checks.get("changed_values_including_signed_zero") == 0
    passed &= checks.get("maximum_absolute_error") == 0
    return checks, bool(passed), {
        "shape": list(output.shape), "dtype": str(output.get_data_dtype()),
        "input_shape": list(source.shape), "input_frames": input_frames,
        "output_frames": output_frames,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("threads must be positive")
    configuration = json.loads(args.case_json.read_text())
    cases = configuration.get("cases")
    if not isinstance(cases, list) or not cases:
        parser.error("case-json needs a nonempty cases list")
    # Preflight only stats files. Hashing/decoding is deferred until after each
    # timed process, avoiding a deliberate full-image read before its CLI run.
    roles_by_case = [validate_case(case) for case in cases]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    private = args.output_dir / "server.private"
    private.mkdir(mode=0o700)
    write_json(private / "cases.json", configuration, private=True)
    source_files = ["src/fnit/__init__.py", "src/fnit/cli.py", "src/fnit/_nib.py",
                    "src/fnit/_sampling_plan.py", "src/fnit/_transforms.py",
                    "src/fnit/_world_resampling.py",
                    "src/fnit/applywarp/core.py", "src/fnit/synthmorph/pipeline.py",
                    "src/fnit/synthmorph/spatial.py", "tools/validate_fnirt_pipeline_lossless.py",
                    "tools/benchmark_warp_cli_lossless.py"]
    report = {
        "schema_version": 1,
        "source_sha256": {role: sha256(_ROOT / role) for role in source_files},
        "scope": "complete saved images; all frames and voxels; fixed transforms; no original software",
        "timing_boundary": "parent perf_counter immediately before child process creation until process exit; includes fresh interpreter/imports, CLI input decoding, preparation, computation, compressed output writing and process teardown",
        "validation_boundary": "complete saved-output comparison, hashing and metadata checks occur after the measured child exits and are separately timed",
        "cache_policy": "no cache flush; one fresh process per configured device, no warmup; storage cache state is uncontrolled and case/device run order is retained",
        "memory_policy": "CUDA allocator capped at min(19e9 bytes, device memory); report allocated/reserved peaks, not untracked driver/library allocations",
        "threads": args.threads, "runs_per_device": 1,
        "all_required_gates_passed": True, "cases": [],
    }
    env = os.environ.copy()
    env.update(PYTHONPATH=str(_ROOT / "src"), PYTHONNOUSERSITE="1",
               OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
               OPENBLAS_NUM_THREADS=str(args.threads), NUMEXPR_NUM_THREADS=str(args.threads))
    exit_code = 0
    for case_index, (case, roles) in enumerate(zip(cases, roles_by_case), 1):
        anonymous = f"case_{case_index:03d}"
        entry = {"case": anonymous, "backend": case["backend"], "runs": []}
        report["cases"].append(entry)
        for device_index, device in enumerate(case["devices"], 1):
            role = f"{anonymous}_device_{device_index:02d}"
            output = private / f"{role}.nii.gz"
            log = private / f"{role}.log"
            request_path = private / f"{role}.request.json"
            arguments, effective = cli_arguments(case, device, output)
            effective.update(threads=args.threads, tf32=True, float16=False,
                             output_compression="nii.gz")
            write_json(request_path, {"arguments": arguments, "device": device,
                                      "threads": args.threads}, private=True)
            record = {"device": device, "run_order": sum(len(c["runs"]) for c in report["cases"]) + 1,
                      "effective_parameters": effective, "lossless_gate_passed": False,
                      "memory_gate_passed": False,
                      "expected_baseline_role": "full_channel_cuda_oracle"
                      if case["backend"] == "synthmorph" and device.startswith("cuda:")
                      else "validated_same_device_baseline"}
            entry["runs"].append(record)
            try:
                with log.open("w") as stream:
                    log.chmod(0o600)
                    started = time.perf_counter()
                    completed = subprocess.run(
                        [sys.executable, str(Path(__file__).resolve()), "--_child", str(request_path)],
                        cwd=_ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False,
                    )
                    record["process_wall_seconds"] = time.perf_counter() - started
                record["process_returncode"] = completed.returncode
                metrics = marker_metrics(log)
                record["child_metrics_marker_found"] = metrics is not None
                if completed.returncode != 0 or metrics is None:
                    raise RuntimeError("CLI process failed or metrics marker missing")
                record["child_metrics"] = metrics
                record["memory_gate_passed"] = (
                    metrics["peak_cuda_allocated_bytes"] < _REQUIRED_PEAK_LIMIT_BYTES
                    and metrics["peak_cuda_reserved_bytes"] < _REQUIRED_PEAK_LIMIT_BYTES
                )
                if not output.is_file():
                    raise RuntimeError("CLI did not save its output")
                output.chmod(0o600)
                validation_started = time.perf_counter()
                expected = case["expected_outputs_by_device"][device]
                checks, passed, geometry = strict_comparison(expected, output, case["input"])
                record.update(geometry, strict_compare_images=checks, lossless_gate_passed=passed,
                              expected_output_sha256=sha256(expected), output_sha256=sha256(output),
                              output_size_bytes=output.stat().st_size)
                record["validation_seconds"] = time.perf_counter() - validation_started
                if "input_sha256" not in entry:
                    entry["input_sha256"] = {name: sha256(case[name]) for name in roles}
                if not passed or not record["memory_gate_passed"]:
                    report["all_required_gates_passed"] = False
                    exit_code = 1
            except Exception as error:
                record["error_type"] = type(error).__name__
                error_path = private / f"{role}.validation_error.log"
                error_path.write_text(traceback.format_exc())
                error_path.chmod(0o600)
                report["all_required_gates_passed"] = False
                exit_code = 1
            write_json(args.output_dir / "report.public.json", report)
            print(json.dumps({"case": anonymous, "device": device,
                              "process_wall_seconds": record.get("process_wall_seconds"),
                              "lossless_gate_passed": record["lossless_gate_passed"],
                              "memory_gate_passed": record["memory_gate_passed"]}), flush=True)
    return exit_code


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--_child":
        run_child(sys.argv[2])
    else:
        raise SystemExit(main())
