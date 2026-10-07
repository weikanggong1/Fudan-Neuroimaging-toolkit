"""Run original full node/spectra output branch under the feature CPU lock.

This is an output precision reference. Its function includes original FFT
and graphics, so its wall time is not divided by FNIT's maps/nodes wall time.
The native algorithms and dependencies are unchanged.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lock-file", type=Path, required=True)
    parser.add_argument("--cpuset", required=True)
    parser.add_argument("--threads", default="1,8")
    parser.add_argument("--frames", type=int, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    cpus = list(map(int, args.cpuset.split(",")))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    state = {"status": "waiting_for_same_cpu_lock", "scope": "Original complete nodes, spectra and maps output branch; precision reference, not a matched speed ratio",
             "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
             "records": []}
    result_path = args.output_dir / "run.private.json"
    def save():
        result_path.write_text(json.dumps(state, indent=2) + "\n")
    save()
    with args.lock_file.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        state["status"] = "running"
        save()
        for threads in map(int, args.threads.split(",")):
            if threads < 1 or threads > len(cpus):
                raise ValueError("Thread budget exceeds physical CPU affinity")
            env = dict(os.environ)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                         "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
                env[name] = str(threads)
            env["OMP_DYNAMIC"] = "FALSE"
            for case in manifest["cases"]:
                if case["operation"] != "regression":
                    continue
                original = case["reference"]["commands"][0]
                binding = Path(original[original.index("--binding")+1])
                cfg = json.loads(binding.read_text())
                cfg["nTPsForSpectra"] = args.frames
                output = args.output_dir / f"threads_{threads}" / case["id"]
                output.mkdir(parents=True, exist_ok=False)
                own_binding = output / "binding.private.json"
                own_binding.write_text(json.dumps(cfg, indent=2) + "\n")
                command = [value if value not in ("{output_dir}", "{threads}") else
                           (str(output) if value == "{output_dir}" else str(threads))
                           for value in original]
                command[command.index("--binding")+1] = str(own_binding)
                command.append("--jvm")
                command = ["taskset", "-c", ",".join(map(str, cpus[:threads])), *command]
                usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
                load_before = list(os.getloadavg())
                started = time.perf_counter()
                with (output / "stdout.private.log").open("w") as log:
                    completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                wall = time.perf_counter()-started
                usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
                node_file = output / "individual_maps_ts.txt"
                row = {"case_id": case["id"], "threads": threads, "affinity": cpus[:threads],
                       "returncode": completed.returncode, "process_seconds": wall,
                       "user_seconds": usage_after.ru_utime-usage_before.ru_utime,
                       "system_seconds": usage_after.ru_stime-usage_before.ru_stime,
                       "maximum_rss_kib": usage_after.ru_maxrss,
                       "load_before": load_before, "load_after": list(os.getloadavg()),
                       "original_nodes_written": node_file.is_file(),
                       "original_spectra_written": (output / "individual_maps_spectra.txt").is_file()}
                if (output / "reference_report.public.json").is_file():
                    row["original_function"] = json.loads((output / "reference_report.public.json").read_text())
                state["records"].append(row)
                save()
                if completed.returncode or not row["original_nodes_written"]:
                    state["status"] = "failed_original_branch"
                    save()
                    raise RuntimeError("Original complete nodes/spectra branch failed; retain its actual log")
        state["status"] = "complete"
        save()


if __name__ == "__main__":
    main()
