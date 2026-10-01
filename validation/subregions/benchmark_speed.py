"""Run a frozen real-T1 benchmark with PID memory and shared-GPU monitoring.

The existing benchmark root contains the public subject, verified atlases,
weights, and saved official references. No official software is executed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--python", default="python")
    parser.add_argument("--mode", choices=("stage", "raw"), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--physical-gpu", type=int, default=1)
    parser.add_argument("--optimization", choices=("fast", "balanced"))
    parser.add_argument("--memory-fraction", type=float)
    parser.add_argument("--own-memory-limit-mib", type=int, default=19073)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.label):
        parser.error("label must contain only letters, numbers, underscores and hyphens")
    root, source = args.root.resolve(), args.source.resolve()
    output = root / args.label
    if output.exists() or (root / (args.label + "_status.json")).exists():
        parser.error("benchmark label already exists; use a fresh label")
    mri = root.parent / "reconall_reference_gpucw1/fs_sub01/mri"
    refs = root.parent / "fnit_subregions_plus_20260928"
    fraction = args.memory_fraction or (.14 if args.mode == "stage" else .23)
    command = [args.python, str(source / "validation/subregions/run_unified.py"),
               "--atlas-root", str(root / "atlases"), "--structures", "all",
               "--device", "cuda:0", "--gpu-memory-fraction", str(fraction),
               "--output-dir", str(output),
               "--reference-brainstem", str(refs / "official_brainstem/brainstemSsLabels.FSvoxelSpace.mgz"),
               "--reference-thalamus", str(refs / "official_thalamus_gpucw1_full_sub01_20260929/ThalamicNuclei.FSvoxelSpace.mgz"),
               "--reference-left", str(refs / "official_hippo_gpucw1_full_sub01_20260929/lh.hippoAmygLabels.FSvoxelSpace.mgz"),
               "--reference-right", str(refs / "official_hippo_gpucw1_full_sub01_20260929/rh.hippoAmygLabels.FSvoxelSpace.mgz")]
    if args.mode == "stage":
        command += ["--t1", str(mri / "norm.mgz"), "--aseg", str(mri / "aseg.mgz"),
                    "--wmparc", str(mri / "wmparc.mgz")]
    else:
        command += ["--t1", str(root.parent.parent / "examples/data/sub-01_T1w.nii.gz"),
                    "--weights", str(root / "weights")]
    if args.optimization:
        command += ["--optimization", args.optimization]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.physical_gpu),
               OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
               PYTHONPATH=str(source / "src"), PYTHONUNBUFFERED="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    status_path = root / (args.label + "_status.json")
    status = {"state": "running", "command": command, "source": str(source),
              "physical_gpu_index": args.physical_gpu, "allocator_fraction": fraction,
              "own_process_limit_mib": args.own_memory_limit_mib,
              "started_unix": time.time(), "max_own_process_memory_mib": 0}
    def save():
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(status_path)
    with (root / (args.label + ".log")).open("w") as log, \
         (root / (args.label + "_gpu_load.jsonl")).open("w") as load:
        process = subprocess.Popen(command, env=env, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        status["pid"] = process.pid
        save()
        while process.poll() is None:
            sample = {"unix_time": time.time(), "own_pid": process.pid}
            try:
                sample["gpus"] = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu",
                     "--format=csv,noheader,nounits"], text=True, timeout=3).strip().splitlines()
                apps = subprocess.check_output(
                    ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                     "--format=csv,noheader,nounits"], text=True, timeout=3).strip().splitlines()
                own = max([int(line.split(",")[1].strip()) for line in apps
                           if line.split(",")[0].strip() == str(process.pid)] or [0])
                sample["own_process_memory_mib"] = own
                status["max_own_process_memory_mib"] = max(status["max_own_process_memory_mib"], own)
                if own > args.own_memory_limit_mib:
                    status["memory_limit_exceeded_mib"] = own
                    process.terminate()
            except (subprocess.SubprocessError, ValueError) as error:
                sample["sampling_error"] = type(error).__name__
            load.write(json.dumps(sample) + "\n")
            load.flush()
            status["last_sample_unix"] = sample["unix_time"]
            save()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if "memory_limit_exceeded_mib" in status:
                    process.kill()
        status.update(exit_code=process.returncode, finished_unix=time.time(),
                      state="completed" if process.returncode == 0 else "failed")
        save()
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
