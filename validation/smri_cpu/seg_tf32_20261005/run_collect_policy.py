"""Bounded sequential saved-result collection; no model or native execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    root = args.root
    workspace = root / "workspaces/smri_cpu_20261004/remaining_20261004/seg_tf32_v1"
    logroot = root / "logs/smri_cpu_20261004/remaining_20261004/seg_tf32_v1"
    state = logroot / "full_v2_collector_queue.private.json"
    report = {"schema": "fnit_parc_tf32_saved_collection_queue/v1", "status": "running", "jobs": [],
              "controller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with state.open("x") as stream:
        json.dump(report, stream, indent=2)
    for device in ("gpu", "cpu"):
        run = root / ("runs/smri_cpu_20261004/remaining_20261004/seg_tf32_full_" + device + "_v2")
        assert not (run / "FULL.public.json").exists()
        argv = ["taskset", "-c", "0,1,2,3,4,5,6,7", "timeout", "--kill-after=10", "600",
            str(root / "envs/default/bin/python"), str(workspace / "full_v2/collect_full_policy.py"),
            "--run", str(run), "--binding", str(workspace / "binding.private.json"),
            "--comparator", str(root / "workspaces/smri_cpu_20261004/t2_seg/compare_outputs.py"),
            "--official", str(root / "runs/smri_cpu_20261004/remaining_20261004/seg_memory_official_node7_v1"),
            "--output", str(run / "FULL.public.json")]
        start = time.perf_counter()
        with (logroot / (device + "_full_v2_collector.log")).open("x") as log:
            process = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, timeout=625,
                env={**os.environ, "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8"})
        report["jobs"].append({"device": device, "returncode": process.returncode,
                               "wall_seconds_not_inference": time.perf_counter() - start})
        if process.returncode:
            report["status"] = "failed"
            state.write_text(json.dumps(report, indent=2) + "\n")
            return process.returncode
        state.write_text(json.dumps(report, indent=2) + "\n")
    report["status"] = "complete"
    state.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
