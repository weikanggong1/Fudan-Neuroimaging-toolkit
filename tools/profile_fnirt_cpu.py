#!/usr/bin/env python3
"""Instrument one complete real FNIRT API call, separately from benchmark clocks.

Timing scopes are inclusive and overlap. This profile must not be substituted
for a paired, uninstrumented benchmark. Input paths stay in private artifacts.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import json
import os
import resource
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--threads", type=int, choices=(1, 8), required=True)
    parser.add_argument("--cpus", required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    cpus = [int(value) for value in args.cpus.split(",")]
    if len(cpus) < args.threads:
        parser.error("cpus has fewer cores than the requested thread budget")
    manifest = json.loads(args.manifest.read_text())
    matching = [case for case in manifest["cases"] if case["id"] == args.case_id]
    if len(matching) != 1:
        parser.error("manifest must contain exactly one matching case")
    if args.output_dir.exists():
        parser.error("output-dir must be new; earlier profiles are retained")
    args.output_dir.mkdir(parents=True)
    status_file = args.output_dir / "profile.private.json"
    status_file.write_text(json.dumps({"status": "waiting", "benchmark": False}, indent=2) + "\n")
    # Set affinity before libraries initialize their worker pools. The same
    # dedicated group lock covers imports, compilation, execution and saves.
    lock = args.lock.open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    os.sched_setaffinity(0, cpus[:args.threads])
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = str(args.threads)
    sys.path[:0] = [str(source / "src"), str(source / "tools")]
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    adapter = importlib.import_module("benchmark_multimodal_cpu_fnirt")
    registration = importlib.import_module("fnit.fnirt.registration")
    spline = importlib.import_module("fnit.fnirt.spline")
    topology = importlib.import_module("fnit.fnirt.topology")
    counters = {}
    originals = []
    layout_copies = []
    solver_qc = []

    original_fnirt_call = registration.TorchFNIRT.__call__
    def capture_fnirt_result(*positional, **keyword):
        result = original_fnirt_call(*positional, **keyword)
        solver_qc.append(result.qc)
        return result
    registration.TorchFNIRT.__call__ = capture_fnirt_result
    originals.append((registration.TorchFNIRT, "__call__", original_fnirt_call))

    def instrument(owner, name, label):
        original = getattr(owner, name)
        def call(*positional, **keyword):
            start = time.perf_counter()
            try:
                return original(*positional, **keyword)
            finally:
                item = counters.setdefault(label, {"calls": 0, "inclusive_seconds": 0.0})
                item["calls"] += 1
                item["inclusive_seconds"] += time.perf_counter() - start
        setattr(owner, name, call)
        originals.append((owner, name, original))

    for name in ("expand_coefficients", "adjoint_field", "_trilinear_sample", "_fsl_masked_gaussian_blur", "design_diagonal", "_spline_jacobian", "_force_jacobian_range"):
        instrument(registration, name, "registration." + name)
    for name in ("expand_coefficients", "adjoint_field"):
        instrument(spline, name, "spline." + name)
    for name in ("energy", "normal", "diagonal"):
        instrument(spline.BendingOperator, name, "BendingOperator." + name)
    for name in ("evaluate", "linearize"):
        instrument(registration._LevelSystem, name, "LevelSystem." + name)
    for name in ("evaluate", "linearize"):
        instrument(registration._JointT1System, name, "JointT1System." + name)
    # The returned closure performs the repeated joint deformation/intensity
    # normal products. Timing only linearize would omit that CPU workload.
    joint_linearize = registration._JointT1System.linearize
    def linearize_with_matvec_profile(*positional, **keyword):
        state, gradient, matvec, diagonal = joint_linearize(*positional, **keyword)
        def joint_matvec(*args, **kwargs):
            before = time.perf_counter()
            try:
                return matvec(*args, **kwargs)
            finally:
                item = counters.setdefault("JointT1System.matvec", {"calls": 0, "inclusive_seconds": 0.0})
                item["calls"] += 1
                item["inclusive_seconds"] += time.perf_counter() - before
        return state, gradient, joint_matvec, diagonal
    registration._JointT1System.linearize = linearize_with_matvec_profile
    originals.append((registration._JointT1System, "linearize", joint_linearize))
    instrument(registration, "constrain_topology", "registration.constrain_topology")
    for name in ("gradient_field", "jacobian_check", "limit_gradient", "integrate_gradient_field", "_rounded_fft"):
        instrument(topology, name, "topology." + name)
    if (source / "src/fnit/fnirt/_normal_cpu.py").exists():
        normal_cpu = importlib.import_module("fnit.fnirt._normal_cpu")
        instrument(normal_cpu.SpatialNormalCPU, "__call__", "SpatialNormalCPU.__call__")
        for kernel in ("_serial", "_parallel", "_flat_serial", "_flat_parallel"):
            if hasattr(normal_cpu, kernel):
                instrument(normal_cpu, kernel, "normal_kernel." + kernel)
        if hasattr(normal_cpu.SpatialNormalCPU, "_prepare_layout"):
            prepare = normal_cpu.SpatialNormalCPU._prepare_layout
            def prepare_layout(operator, field):
                previous = operator._layout
                before = time.perf_counter()
                try:
                    return prepare(operator, field)
                finally:
                    if operator._layout != previous:
                        layout_copies.append({"geometry": list(field.shape[1:]),
                                              "copied_bytes": operator.layout_copy_bytes,
                                              "copy_seconds": time.perf_counter() - before})
            normal_cpu.SpatialNormalCPU._prepare_layout = prepare_layout
            originals.append((normal_cpu.SpatialNormalCPU, "_prepare_layout", prepare))
    if (source / "src/fnit/fnirt/_bending_cpu.py").exists():
        bending_cpu = importlib.import_module("fnit.fnirt._bending_cpu")
        instrument(bending_cpu, "multiply_square_", "bending_cpu.multiply_square_")

    metadata = {"status": "waiting", "scope": "one instrumented full API load + compute + save; imports and lock wait excluded", "benchmark": False,
                "inclusive_scopes_overlap": True, "threads": args.threads, "affinity": cpus[:args.threads], "case_id": args.case_id,
                "profile_tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "source_hashes": {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                                  for name in ("src/fnit/fnirt/registration.py", "src/fnit/fnirt/spline.py", "src/fnit/fnirt/topology.py", "src/fnit/fnirt/_normal_cpu.py", "src/fnit/fnirt/_normal_simd_cpu.py", "src/fnit/fnirt/_smoothing_cpu.py", "src/fnit/fnirt/_topology_cpu.py", "src/fnit/fnirt/_bending_cpu.py") if (source / name).exists()}}
    try:
        metadata["status"] = "running"
        status_file.write_text(json.dumps(metadata, indent=2) + "\n")
        start = time.perf_counter()
        outputs = adapter.run_case(matching[0], args.output_dir / "outputs", "cpu")
        metadata.update(status="completed", instrumented_api_seconds=time.perf_counter() - start,
                        counters=counters, outputs_private=outputs,
                        solver_qc=solver_qc,
                        solver_qc_sha256=hashlib.sha256(json.dumps(solver_qc, sort_keys=True, allow_nan=False).encode()).hexdigest(),
                        maximum_rss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                        layout_copies=layout_copies,
                        maximum_layout_copy_bytes_per_operator=max((item["copied_bytes"] for item in layout_copies), default=0),
                        output_sha256={key: hashlib.sha256(Path(path).read_bytes()).hexdigest() for key, path in outputs.items()})
    except BaseException as error:
        metadata.update(status="failed", error=str(error), counters=counters)
        raise
    finally:
        status_file.write_text(json.dumps(metadata, indent=2) + "\n")
        for owner, name, original in reversed(originals):
            setattr(owner, name, original)
        lock.close()


if __name__ == "__main__":
    main()
