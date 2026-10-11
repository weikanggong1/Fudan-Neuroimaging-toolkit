#!/usr/bin/env python3
"""在新空目录验证完整首缺陷候选，保留strict失败；不是原始T1整例。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--input-surface", type=Path, required=True)
    parser.add_argument("--input-qsphere", type=Path, required=True)
    parser.add_argument("--intensity-volume", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-source-commit", required=True)
    parser.add_argument("--tissue-scale-mm", type=float, default=1.)
    parser.add_argument("--edge-sample-spacing-mm", type=float, default=.5)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("new empty output-dir required; previous failures must be preserved")
    args.output_dir.mkdir(parents=True)
    source = args.source_root.resolve()
    module = source / "fnit/recon_all/topology_retessellation_python.py"
    inputs = {"orig_nofix": args.input_surface, "qsphere_nofix": args.input_qsphere,
              "MRI": args.intensity_volume}
    binding = {"source_python_sha256": {str(p.relative_to(source)): sha256(p)
                                        for p in sorted(source.rglob("*.py"))},
               "input_sha256": {k: sha256(p) for k, p in inputs.items()},
               "input_source_commit": args.input_source_commit,
               "candidate_module_sha256": sha256(module),
               "driver_sha256": sha256(Path(__file__))}
    (args.output_dir / "binding.private.json").write_text(json.dumps(binding, indent=2) + "\n")
    output = args.output_dir / "candidate.orig"
    report = args.output_dir / "retessellation.private.json"
    command = [sys.executable, "-m", "fnit.recon_all.topology_retessellation_python",
               "--input-surface", str(args.input_surface), "--input-qsphere", str(args.input_qsphere),
               "--intensity-volume", str(args.intensity_volume), "--output-surface", str(output),
               "--report-path", str(report), "--tissue-scale-mm", str(args.tissue_scale_mm),
               "--edge-sample-spacing-mm", str(args.edge_sample_spacing_mm)]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source)
    env["CUDA_VISIBLE_DEVICES"] = ""
    child_start = time.perf_counter()
    with (args.output_dir / "worker.log").open("w") as log:
        child = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    child_seconds = time.perf_counter() - child_start
    stage = json.loads(report.read_text()) if report.exists() else None
    runtime = {"hostname": platform.node(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
               "python": sys.version, "python_executable": sys.executable,
               "thread_environment": {k: env.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                        "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")},
               "versions": {name: importlib.metadata.version(name)
                            for name in ("numpy", "scipy", "numba", "nibabel")},
               "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
               "GPU_used": False, "GPU_peak_bytes": None,
               "child_max_rss_KiB": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}
    result = {"scope": "frozen same-input first discovered complete patch primitive; not full GA or recon-all",
              "child_returncode": child.returncode, "runtime": runtime, "binding": binding,
              "output_exists": output.exists(), "output_sha256": sha256(output) if output.exists() else None,
              "stage_report_exists": report.exists(), "stage_report_sha256": sha256(report) if report.exists() else None,
              "execution_status": None if stage is None else stage["execution_status"],
              "quality_status": None if stage is None else stage["mesh_quality_status"],
              "child_wall_seconds_including_interpreter_import_IO_JIT_and_QC": child_seconds,
              "harness_wall_seconds_from_main_including_preflight_source_hashes_and_child": time.perf_counter() - start,
              "harness_initial_import_seconds": "not_included",
              "native_program_called": False, "complete_ga_solver": False,
              "overall_metric_equivalence": "not_assessed"}
    (args.output_dir / "benchmark.private.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "driver.exitcode").write_text(str(child.returncode) + "\n")
    if child.returncode:
        raise SystemExit(child.returncode)


if __name__ == "__main__":
    main()
