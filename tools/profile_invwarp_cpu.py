#!/usr/bin/env python3
"""Profile one complete real InvWarp call, independently of native timings."""
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
    args = parser.parse_args()
    source = args.source_root.resolve()
    cases = [case for case in json.loads(args.manifest.read_text())["cases"] if case["id"] == args.case_id]
    if len(cases) != 1:
        parser.error("Exactly one matching real case is required")
    cpus = [int(value) for value in args.cpus.split(",")]
    if len(cpus) < args.threads or not set(cpus).issubset(os.sched_getaffinity(0)):
        parser.error("Accessible CPUs must cover the thread budget")
    if args.output_dir.exists():
        parser.error("Preserve existing profiles; output-dir must be new")
    args.output_dir.mkdir(parents=True)
    status = args.output_dir / "profile.private.json"
    report = {"status": "waiting", "benchmark": False, "inclusive_scopes_overlap": True,
              "scope": "One full instrumented API read + compute + save; imports and lock wait excluded",
              "case_id": args.case_id, "threads": args.threads, "affinity": cpus[:args.threads],
              "profile_tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    status.write_text(json.dumps(report, indent=2) + "\n")
    originals, counters = [], {}
    with args.lock.open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        os.sched_setaffinity(0, cpus[:args.threads])
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                     "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[name] = str(args.threads)
        sys.path[:0] = [str(source / "src"), str(source / "tools")]
        import torch
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        adapter = importlib.import_module("benchmark_multimodal_cpu_invwarp")
        apply = importlib.import_module("fnit.applywarp.core")
        convert = importlib.import_module("fnit.convertwarp.core")
        inverse = importlib.import_module("fnit.invwarp.core")

        def instrument(owner, name, label):
            original = getattr(owner, name)
            def wrapped(*positional, **keyword):
                before = time.perf_counter()
                try:
                    return original(*positional, **keyword)
                finally:
                    item = counters.setdefault(label, {"calls": 0, "inclusive_seconds": 0.0})
                    item["calls"] += 1
                    item["inclusive_seconds"] += time.perf_counter() - before
            setattr(owner, name, wrapped)
            originals.append((owner, name, original))

        for name in ("__init__", "sample"):
            instrument(convert._PullField, name, "PullField." + name)
        instrument(convert, "_sample_linear", "sample_linear")
        instrument(apply, "_sample_linear_cpu", "sample_linear_cpu")
        instrument(apply, "_inside", "bounds_mask")
        instrument(inverse, "_spatial_grid", "reference_grid")
        instrument(inverse, "_field_image", "output_image")
        instrument(torch, "mm", "correction_mm")
        call = inverse.TorchInvWarp.__call__
        def capture_result(*positional, **keyword):
            result = call(*positional, **keyword)
            report["solver_qc"] = result.qc
            report["valid_fraction"] = result.valid_fraction
            return result
        inverse.TorchInvWarp.__call__ = capture_result
        originals.append((inverse.TorchInvWarp, "__call__", call))
        report["source_hashes"] = {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                                    for name in ("src/fnit/applywarp/core.py", "src/fnit/convertwarp/core.py",
                                                 "src/fnit/invwarp/core.py")}
        try:
            report["status"] = "running"
            status.write_text(json.dumps(report, indent=2) + "\n")
            before = time.perf_counter()
            outputs = adapter.run_case(cases[0], args.output_dir / "outputs", "cpu")
            report.update(status="completed", instrumented_api_seconds=time.perf_counter() - before,
                          counters=counters, maximum_rss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                          outputs_private=outputs,
                          output_sha256={key: hashlib.sha256(Path(value).read_bytes()).hexdigest()
                                         for key, value in outputs.items()})
        except BaseException as error:
            report.update(status="failed", error=str(error), counters=counters)
            raise
        finally:
            status.write_text(json.dumps(report, indent=2) + "\n")
            for owner, name, original in reversed(originals):
                setattr(owner, name, original)


if __name__ == "__main__":
    main()
