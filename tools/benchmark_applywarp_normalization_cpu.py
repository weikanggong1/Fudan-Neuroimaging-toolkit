#!/usr/bin/env python3
"""Isolated complete-MRI normalization or complete InvWarp diagnostic gate.

The coordinate mode intercepts the first full real solver query and exits
before solving. It is a kernel diagnosis, never an end-to-end speed result.
The full mode retains all frames/grid, outputs and solver stopping settings.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import resource
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "manifest", "output-dir", "lock"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--threads", type=int, choices=(1, 8), required=True)
    parser.add_argument("--cpus", required=True)
    parser.add_argument("--mode", choices=("coordinate", "full"), required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    cases = [case for case in json.loads(args.manifest.read_text())["cases"] if case["id"] == args.case_id]
    if len(cases) != 1:
        parser.error("Exactly one matching real case is required")
    cpus = [int(value) for value in args.cpus.split(",")]
    if len(cpus) < args.threads or not set(cpus[:args.threads]).issubset(os.sched_getaffinity(0)):
        parser.error("Accessible CPUs must cover the thread budget")
    if args.output_dir.exists():
        parser.error("Preserve previous diagnostics; output-dir must be new")
    args.output_dir.mkdir(parents=True)
    status = args.output_dir / "report.private.json"
    report = {"status": "waiting", "mode": args.mode, "benchmark": False,
              "scope": "complete MRI coordinate kernel" if args.mode == "coordinate" else "full instrumented API read + compute + save",
              "entry_imports_excluded": True, "inside_call_lazy_import_and_JIT_included": True,
              "case_id": args.case_id, "threads": args.threads, "affinity": cpus[:args.threads],
              "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    status.write_text(json.dumps(report, indent=2) + "\n")
    with args.lock.open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        os.sched_setaffinity(0, cpus[:args.threads])
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
            os.environ[name] = str(args.threads)
        sys.path[:0] = [str(source / "src"), str(source / "tools")]
        import torch
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        adapter_name = Path(cases[0].get("adapter", "tools/benchmark_multimodal_cpu_invwarp.py")).stem
        adapter = importlib.import_module(adapter_name)
        apply = importlib.import_module("fnit.applywarp.core")
        inverse = importlib.import_module("fnit.invwarp.core")
        names = ("src/fnit/applywarp/core.py", "src/fnit/applywarp/_normalization_cpu.py", "src/fnit/convertwarp/core.py", "src/fnit/invwarp/core.py")
        report["source_hashes"] = {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                                   for name in names if (source / name).exists()}
        normalizers = {name: getattr(apply, name) for name in
                       ("_to_float32_grid_cpu", "_to_float64_grid_cpu", "_to_grid") if hasattr(apply, name)}
        original_call = inverse.TorchInvWarp.__call__
        original_apply = apply.ApplyWarpPlan.apply

        def legacy(coordinates, shape):
            grid = torch.empty((*coordinates.shape[1:], 3), dtype=torch.float32, device=coordinates.device)
            for axis, size in enumerate(shape):
                if size == 1:
                    grid[..., 2 - axis].zero_()
                else:
                    grid[..., 2 - axis].copy_(2 * coordinates[axis] / (size - 1) - 1)
            return grid[None]

        class Captured(Exception):
            pass

        counters = {"calls": 0, "inclusive_seconds": 0.0}
        def normalization(coordinates, shape, original, name):
            if args.mode == "coordinate" and coordinates.numel() >= 3_000_000:
                grid64 = name == "_to_float64_grid_cpu" or (name == "_to_grid" and coordinates.dtype == torch.float64)
                reference = normalizers["_to_grid"] if grid64 else legacy
                expected = reference(coordinates, shape)
                actual = original(coordinates, shape)
                bits = torch.int64 if grid64 else torch.int32
                mismatches = int((expected.view(bits) != actual.view(bits)).sum())
                report["coordinate_gate"] = {"shape": list(coordinates.shape), "source_shape": list(shape),
                                             "dtype": str(coordinates.dtype), "contiguous": coordinates.is_contiguous(),
                                             "stored_dtype": str(actual.dtype), "entry": name,
                                             "stored_values": actual.numel(), "different_stored_bits": mismatches}
                del expected, actual
                if mismatches:
                    raise AssertionError("Complete real normalization coordinate bits differ")
                timing = {"legacy": [], "fused": []}
                for iteration in range(3):
                    order = (("legacy", reference), ("fused", original)) if iteration % 2 == 0 else (("fused", original), ("legacy", reference))
                    for label, function in order:
                        before = time.perf_counter()
                        result = function(coordinates, shape)
                        timing[label].append(time.perf_counter() - before)
                        del result
                report["warmed_kernel_seconds"] = timing
                report["coordinate_scope"] = "first full real inverse-solver query; all target voxels; JIT warmed by the actual coarse affine fit"
                raise Captured()
            before = time.perf_counter()
            result = original(coordinates, shape)
            counters["calls"] += 1
            counters["inclusive_seconds"] += time.perf_counter() - before
            return result

        def capture_result(*positional, **keyword):
            result = original_call(*positional, **keyword)
            report["solver_qc"], report["valid_fraction"] = result.qc, result.valid_fraction
            return result

        def capture_apply(*positional, **keyword):
            result = original_apply(*positional, **keyword)
            report["apply_qc"] = result.qc
            report["valid_mask_sha256"] = hashlib.sha256(result.valid_mask.tobytes()).hexdigest()
            report["valid_mask_voxels"] = int(result.valid_mask.size)
            return result

        for name, original in normalizers.items():
            def wrapped(coordinates, shape, function=original, entry=name):
                return normalization(coordinates, shape, function, entry)
            setattr(apply, name, wrapped)
        inverse.TorchInvWarp.__call__ = capture_result
        apply.ApplyWarpPlan.apply = capture_apply
        report["status"] = "running"
        status.write_text(json.dumps(report, indent=2) + "\n")
        try:
            before = time.perf_counter()
            try:
                outputs = adapter.run_case(cases[0], args.output_dir / "outputs", "cpu")
            except Captured:
                outputs = {}
            if args.mode == "coordinate" and "coordinate_gate" not in report:
                raise AssertionError("No full real query was captured")
            report.update(status="completed", instrumented_api_seconds=time.perf_counter() - before,
                          normalization=counters, maximum_rss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                          outputs_private={key: str(value) for key, value in outputs.items()},
                          output_sha256={key: hashlib.sha256(Path(value).read_bytes()).hexdigest() for key, value in outputs.items()})
            report["input_sha256"] = {key: hashlib.sha256(Path(cases[0][key]).read_bytes()).hexdigest()
                                      for key in ("input", "reference", "warp", "accuracy_mask", "premat", "postmat")
                                      if isinstance(cases[0].get(key), str)}
        except BaseException as error:
            report.update(status="failed", error=str(error), normalization=counters)
            raise
        finally:
            for name, original in normalizers.items():
                setattr(apply, name, original)
            inverse.TorchInvWarp.__call__ = original_call
            apply.ApplyWarpPlan.apply = original_apply
            status.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
