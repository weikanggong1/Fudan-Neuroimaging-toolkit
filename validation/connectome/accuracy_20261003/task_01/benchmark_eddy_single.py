"""Frozen supplementary arm using the completed pair's exact worker/oracle.

This is a supplementary arm, not a fresh paired timing or ABBA experiment.
All scientific outputs are written to a new private directory.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from benchmark_eddy_pair import compare, monitor, sha


def source_binding(source):
    hashes = {str(path.relative_to(source)): sha(path)
              for path in sorted((source / "src").rglob("*.py"))}
    return {"files": hashes, "sha256": hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "reference", "paired-summary", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--subject", choices=("CON01", "CON03"), default="CON01")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    paired = json.loads(args.paired_summary.read_text())
    worker_script = Path(__file__).with_name("benchmark_eddy_pair.py")
    if sha(worker_script) != paired["script_SHA256"]:
        raise ValueError("supplementary arm must use the completed pair's exact worker")
    inputs = paired["cases"][args.subject]["input_sha256"]
    for path in inputs:
        # Frozen staging legitimately symlinks raw AP to the original BIDS.
        # Bind the staged location here; content SHA below follows symlinks.
        Path(path).absolute().relative_to(args.reference.absolute())
    if any(sha(path) != expected for path, expected in inputs.items()):
        raise ValueError("frozen official input hash changed")
    binding = source_binding(args.source)
    report = {"scope": "supplementary v2 EDDY including load/save; not new paired timing or ABBA",
              "subject": args.subject,
              "host": os.uname().nodename, "gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "shared_GPU_load": True, "stable_speedup_assessed": False,
              "script_SHA256": sha(__file__), "worker_script_SHA256": sha(worker_script),
              "paired_summary_SHA256": sha(args.paired_summary),
              "input_sha256": inputs, "source_binding": binding,
              "previous_runs": [{k: v for k, v in row.items() if k != "GPU_monitor"}
                                for row in paired["cases"][args.subject]["runs"]]}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with open("/tmp/fnit-recon-five-20261002-gongwk.gpu.lock", "a+") as lock:
        start = time.perf_counter()
        fcntl.flock(lock, fcntl.LOCK_EX)
        wait = time.perf_counter() - start
        env = dict(os.environ, PYTHONPATH=str(args.source / "src"), OMP_NUM_THREADS="8",
                   MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        command = [sys.executable, str(worker_script), "--worker", "--reference", str(args.reference),
                   "--output", str(args.output), "--ref-scan-no", str(76 if args.subject == "CON01" else 0)]
        with (args.output / "driver.log").open("w") as log:
            start = time.perf_counter()
            proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            stopped = threading.Event()
            samples = []
            thread = threading.Thread(target=monitor, args=(proc.pid, stopped, samples), daemon=True)
            thread.start()
            rc = proc.wait()
            elapsed = time.perf_counter() - start
            stopped.set()
            thread.join()
        fcntl.flock(lock, fcntl.LOCK_UN)
    report.update(process_wall_seconds=elapsed, lock_wait_seconds=wait, returncode=rc,
                  source_unchanged=source_binding(args.source) == binding,
                  GPU_monitor={"interval_seconds": .5, "samples": samples,
                      "process_tree_peak_bytes": max((s.get("process_tree_bytes", 0) for s in samples), default=None),
                      "max_gap_seconds": max((b["time"]-a["time"] for a, b in zip(samples, samples[1:])), default=None),
                      "failed_samples": sum("error" in s for s in samples)})
    if rc == 0:
        report["worker"] = json.loads((args.output / "worker_receipt.json").read_text())
        report["official_comparison"] = compare(args.output, args.reference / "eddy",
                                                 args.reference / "mask/nodif_brain_mask.nii.gz")
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    if rc or not report["source_unchanged"]:
        raise RuntimeError("supplementary stage failed or frozen source changed")
    print(json.dumps({k: report[k] for k in ("process_wall_seconds", "worker", "official_comparison")}), flush=True)


if __name__ == "__main__":
    main()
