#!/usr/bin/env python3
"""Run the old/new complete 490-frame volume API on an authorized GPU.

The coordinator supplies an explicit GPU UUID and shared lock after admission.
This worker is never launched as part of preparation or CPU validation.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import threading
import time

from acceptance import compare_histories


def write(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def descendant_pids(parent):
    parents = {}
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        try:
            fields = (item / "stat").read_text().rpartition(") ")[2].split()
            parents[int(item.name)] = int(fields[1])
        except (OSError, ValueError, IndexError):
            continue
    result = {parent}
    while True:
        expanded = result | {pid for pid, ppid in parents.items() if ppid in result}
        if expanded == result:
            return result
        result = expanded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.output_dir.exists():
        raise FileExistsError("GPU regression requires a new output directory")
    os.umask(0o077)
    args.output_dir.mkdir(parents=True)
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    tool = Path(__file__).resolve().parent / "benchmark_fnit.py"
    records = []
    with args.lock.open("a") as resource_lock:
        fcntl.flock(resource_lock, fcntl.LOCK_EX)
        for name, version in (("baseline_1", "baseline"), ("candidate_1", "candidate"),
                              ("candidate_2", "candidate"), ("baseline_2", "baseline")):
            binding = dict(cfg["binding"])
            binding["source_root"] = cfg[version + "_source"]
            binding["frozen_baseline_commit"] = cfg["baseline_commit"]
            binding_path = args.output_dir / (name + ".binding.private.json")
            write(binding_path, binding)
            destination = args.output_dir / name
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
            for env_key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                         "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
                env[env_key] = str(cfg.get("threads", 8))
            command = ["/usr/bin/time", "-v", "-o",
                       str(args.output_dir / (name + ".clock.private.txt")),
                       cfg["python"], str(tool), "--binding", str(binding_path),
                       "--kind", "volume-api", "--threads", str(cfg.get("threads", 8)),
                       "--device", "cuda:0", "--output-dir", str(destination)]
            if cfg.get("cpu8"):
                command = ["taskset", "-c", ",".join(map(str, cfg["cpu8"]))] + command
            samples = []
            stopped = threading.Event()
            with (args.output_dir / (name + ".log")).open("wb") as log:
                start = time.perf_counter()
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)

                def sample():
                    while not stopped.is_set():
                        moment = time.perf_counter()
                        record = {"seconds": moment - start, "memory_bytes": None,
                                  "successful_query": False}
                        try:
                            owned = descendant_pids(child.pid)
                            query = subprocess.run(["nvidia-smi",
                                "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                                "--format=csv,noheader,nounits"], capture_output=True,
                                text=True, timeout=5)
                            if query.returncode == 0:
                                memory, matched, unavailable = 0, [], False
                                for line in query.stdout.splitlines():
                                    fields = [value.strip() for value in line.split(",")]
                                    if len(fields) != 3 or fields[0] != args.gpu_uuid:
                                        continue
                                    try:
                                        pid = int(fields[1])
                                    except ValueError:
                                        continue
                                    if pid in owned:
                                        matched.append(pid)
                                        try:
                                            memory += int(fields[2]) * 1024 * 1024
                                        except ValueError:
                                            unavailable = True
                                record.update({"successful_query": True,
                                               "owned_gpu_pids": matched,
                                               "memory_bytes": None if unavailable else memory})
                            activity = subprocess.run(["nvidia-smi", "-i", args.gpu_uuid,
                                "--query-gpu=uuid,utilization.gpu,memory.used",
                                "--format=csv,noheader,nounits"], capture_output=True,
                                text=True, timeout=5)
                            if activity.returncode == 0:
                                fields = [field.strip() for field in activity.stdout.strip().split(",")]
                                if len(fields) == 3 and fields[0] == args.gpu_uuid:
                                    record["gpu_utilization_percent"] = float(fields[1])
                                    record["total_gpu_memory_used_bytes"] = int(fields[2]) * 1024 * 1024
                        except (OSError, ValueError, subprocess.TimeoutExpired):
                            pass
                        samples.append(record)
                        stopped.wait(0.2)

                monitor = threading.Thread(target=sample, daemon=True)
                monitor.start()
                code = child.wait()
                child_seconds = time.perf_counter() - start
                stopped.set()
                monitor.join(timeout=6)
                process_seconds = time.perf_counter() - start
            write(args.output_dir / (name + ".memory_samples.private.json"), samples)
            measured = [s["memory_bytes"] for s in samples if s["successful_query"]
                        and s.get("owned_gpu_pids") and s["memory_bytes"] is not None]
            gaps = [right["seconds"] - left["seconds"] for left, right in zip(samples, samples[1:])]
            peak = max(measured) if measured else None
            record = {"name": name, "version": version, "returncode": code,
                      "child_seconds": child_seconds, "process_with_monitor_join_seconds": process_seconds,
                      "target_gpu_uuid": args.gpu_uuid, "sampling_interval_requested_seconds": 0.2,
                      "maximum_sampling_gap_seconds": max(gaps) if gaps else None,
                      "sample_count": len(samples),
                      "failed_sample_count": sum(not s["successful_query"] for s in samples),
                      "simultaneous_process_tree_peak_bytes": peak,
                      "peak_gb": peak / 1e9 if peak is not None else None,
                      "peak_gib": peak / 1024 ** 3 if peak is not None else None,
                      "within_20gb": peak <= 20_000_000_000 if peak is not None else None}
            utilization = [s["gpu_utilization_percent"] for s in samples
                           if "gpu_utilization_percent" in s]
            record["simultaneous_gpu_utilization_percent"] = {
                "minimum": min(utilization) if utilization else None,
                "maximum": max(utilization) if utilization else None,
                "mean": sum(utilization) / len(utilization) if utilization else None}
            records.append(record)
            write(args.output_dir / "gpu_execution.public.json", {"records": records})
            if code:
                raise RuntimeError("Complete GPU volume API failed; preserve its attempt")
            api_report = json.loads((destination / "report.public.json").read_text())
            if peak is not None and peak > 20_000_000_000:
                raise RuntimeError("Observed process tree GPU allocation exceeds 20 GB")
            if max(api_report["gpu"]["peak_allocated_bytes"],
                   api_report["gpu"]["peak_reserved_bytes"]) > 20_000_000_000:
                raise RuntimeError("Torch GPU allocation exceeds 20 GB")
    import nibabel as nib
    import numpy as np
    pair_controls = []
    old_report = json.loads((args.output_dir / "baseline_1/report.public.json").read_text())
    reports = {"baseline_1": old_report}
    for name in ("candidate_1", "candidate_2", "baseline_2"):
        new_report = json.loads((args.output_dir / name / "report.public.json").read_text())
        reports[name] = new_report
        controls = compare_outputs(args.output_dir / "baseline_1", args.output_dir / name)
        controls.update(compare_histories(old_report["history"], new_report["history"]))
        pair_controls.append({"baseline": "baseline_1", "comparison": name, "controls": controls})
    write(args.output_dir / "gpu_old_new_control.public.json", {
        "complete_frames": 490, "complete_fsLR_vertices": 64984,
        "order": [record["name"] for record in records],
        "pairs": pair_controls, "reports": reports,
        "timing_scope": "complete API and process clocks under contemporaneous shared GPU load",
        "shared_busy_gpu_observation": True})


def compare_outputs(left, right):
    import nibabel as nib
    import numpy as np
    controls = {}
    for name in ("labels_fslr32k_64984.npy", "lh_labels.npy", "rh_labels.npy"):
        old = np.load(left / name, allow_pickle=False)
        new = np.load(right / name, allow_pickle=False)
        controls[name] = {"different_values": int(np.count_nonzero(old != new)),
                          "shape_matches": old.shape == new.shape}
    for name in ("network_timeseries.tsv", "network_correlation.tsv"):
        old = np.loadtxt(left / name, skiprows=1)
        new = np.loadtxt(right / name, skiprows=1)
        controls[name] = {"equal_nan": bool(np.array_equal(old, new, equal_nan=True)),
                          "max_abs_error": float(np.nanmax(np.abs(old - new)))}
    for name in ("labels_fslr32k.dlabel.nii", "labels_mni.nii.gz"):
        old = nib.load(left / name)
        new = nib.load(right / name)
        controls[name] = {"different_values": int(np.count_nonzero(np.asarray(old.dataobj)
                                                                  != np.asarray(new.dataobj))),
                          "shape_matches": old.shape == new.shape,
                          "affine_max_abs_error": float(np.max(np.abs(old.affine - new.affine)))
                                              if name.endswith(".nii.gz") else None}
    return controls


if __name__ == "__main__":
    main()
