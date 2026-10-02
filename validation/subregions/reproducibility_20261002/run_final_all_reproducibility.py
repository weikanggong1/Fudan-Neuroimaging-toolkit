"""Six independent real-image production runs: stage/raw, three full runs each.

Every child invokes the frozen public run_unified.py with --structures all.
References are comparison-only arguments. A declared read-only context observer
hashes returned preprocessing arrays; no fitting policy wrapper is loaded.
Source/input/asset identities and the physical GPU UUID are fixed.
Preflight and free-memory waits are reported separately from process wall time.
Use --parallel-modes only after preceding experiments finish, with distinct GPUs.
Only derived reports/logs/identities are listed for transfer; images stay remote.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time


STRUCTURES = {"brainstem", "thalamus", "hippo-amygdala-left", "hippo-amygdala-right"}
MEMORY_LIMIT_MIB = 19073


def identity(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def verified_source(source: Path) -> tuple[dict, dict]:
    manifest_path = source / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    runtime = {}
    for entry in manifest["files"]:
        actual = identity(source / entry["path"])
        if actual["bytes"] != entry["bytes"] or actual["sha256"] != entry["sha256"]:
            raise ValueError(f"Frozen source changed: {entry['path']}")
        if entry["path"].startswith("src/fnit/") and entry["path"].endswith(".py"):
            runtime[entry["path"].removeprefix("src/fnit/")] = entry["sha256"]
    actual_paths = {str(p.relative_to(source / "src/fnit"))
                    for p in (source / "src/fnit").rglob("*.py")}
    if actual_paths != set(runtime):
        raise ValueError("Frozen manifest must cover all current runtime modules, including merged MSM")
    return identity(manifest_path), runtime


def gpu_identity(index: int) -> dict:
    rows = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.total",
        "--format=csv,noheader,nounits"], text=True, timeout=5).splitlines()
    for row in rows:
        fields = [value.strip() for value in row.split(",")]
        if int(fields[0]) == index:
            return {"index": index, "uuid": fields[1], "name": fields[2],
                    "total_memory_mib": int(fields[3])}
    raise ValueError(f"Physical GPU {index} does not exist")


def gpu_sample(gpu: dict, own_pid: int | None = None) -> dict:
    rows = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True, timeout=5).splitlines()
    row = next(row for row in rows if row.split(",")[1].strip() == gpu["uuid"])
    fields = [value.strip() for value in row.split(",")]
    if int(fields[0]) != gpu["index"]:
        raise ValueError("Physical GPU index/UUID changed during the experiment")
    processes = []
    rows = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
        "--format=csv,noheader,nounits"], text=True, timeout=5).splitlines()
    for row in rows:
        values = [value.strip() for value in row.split(",")]
        if values[0] == gpu["uuid"]:
            processes.append({"pid": int(values[1]), "used_memory_mib": int(values[2])})
    return {"unix_time": time.time(), "physical_gpu": gpu,
            "memory_used_mib": int(fields[2]), "memory_free_mib": int(fields[3]),
            "utilization_percent": int(fields[4]), "processes": processes,
            "own_memory_mib": max([p["used_memory_mib"] for p in processes
                                   if p["pid"] == own_pid] or [0]),
            "other_processes": [p for p in processes if p["pid"] != own_pid]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--context-observer", type=Path, required=True)
    parser.add_argument("--stage-gpu", type=int, default=1)
    parser.add_argument("--raw-gpu", type=int, default=1)
    parser.add_argument("--parallel-modes", action="store_true")
    parser.add_argument("--preceding-queue", type=Path,
                        help="wait for an existing experiment to complete before any child starts")
    parser.add_argument("--optimization", choices=("fast", "balanced"), default="fast")
    parser.add_argument("--output-prefix", default="final_production_all")
    args = parser.parse_args()
    if Path(args.output_prefix).name != args.output_prefix or args.output_prefix in (".", ".."):
        raise ValueError("output-prefix must be one filename component")
    if args.parallel_modes and args.stage_gpu == args.raw_gpu:
        raise ValueError("Parallel stage/raw modes require distinct physical GPUs")
    root, source = args.root.resolve(), args.source.resolve()
    observer = args.context_observer.resolve()
    if not observer.is_file():
        raise ValueError("The declared read-only context observer is unavailable")
    folder = root / "reproducibility_20261002"
    state_path = folder / (args.output_prefix + "_queue.json")
    if state_path.exists():
        raise ValueError("Preserve the existing final queue; use a newly declared output prefix")
    source_manifest, runtime = verified_source(source)
    driver = source / "validation/subregions/run_unified.py"
    mri = root.parent / "reconall_reference_gpucw1/fs_sub01/mri"
    raw_t1 = root.parent.parent / "examples/data/sub-01_T1w.nii.gz"
    official = folder / "official_r1"
    references = {"brainstem": official / "brainstem/brainstemSsLabels.FSvoxelSpace.mgz",
                  "thalamus": official / "thalamus/ThalamicNuclei.FSvoxelSpace.mgz",
                  "left": official / "hippo-amygdala/lh.hippoAmygLabels.FSvoxelSpace.mgz",
                  "right": official / "hippo-amygdala/rh.hippoAmygLabels.FSvoxelSpace.mgz"}
    input_files = [mri / name for name in ("norm.mgz", "aseg.mgz", "wmparc.mgz")] + [raw_t1]
    assets = [p for name in ("atlases", "weights") for p in sorted((root / name).rglob("*"))
              if p.is_file()]
    fixed = {str(p): identity(p) for p in input_files + list(references.values()) + assets + [observer]}
    gpus = {"stage": gpu_identity(args.stage_gpu), "raw": gpu_identity(args.raw_gpu)}
    state = {"state": "preflight", "scope": "six independent structures=all production runs",
             "planned_successful_cases": 6, "repeats_per_mode": 3,
             "source": str(source), "source_manifest": source_manifest,
             "verified_runtime_python_files": len(runtime), "run_driver": identity(driver),
             "queue_script": identity(Path(__file__)), "fixed_inputs_references_assets": fixed,
             "physical_gpu_by_mode": gpus, "optimization": args.optimization,
             "reference_used_for_fitting": False, "policy_wrapper": None,
             "context_observer": identity(observer), "observer_changes_solver_options": False,
             "parallel_modes": args.parallel_modes, "threads_per_process": 4,
             "cublas_workspace_config": None, "global_deterministic_algorithms_changed": False,
             "started_unix": time.time(), "runs": []}
    lock = threading.RLock()

    def save():
        with lock:
            temporary = state_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(state, indent=2) + "\n")
            temporary.replace(state_path)

    def update(record, **values):
        with lock:
            record.update(values)
            save()

    def verify_fixed():
        current_manifest, current_runtime = verified_source(source)
        if current_manifest != source_manifest or current_runtime != runtime:
            raise ValueError("Frozen production source identity changed")
        for name, expected in fixed.items():
            if identity(Path(name)) != expected:
                raise ValueError(f"Input/reference/asset identity changed: {name}")

    save()
    if args.preceding_queue:
        wait_start = time.monotonic()
        while True:
            previous = json.loads(args.preceding_queue.read_text())
            if previous["state"] == "completed":
                state["preceding_queue"] = identity(args.preceding_queue)
                break
            if previous["state"] == "failed":
                state.update(state="failed", failure="preceding experiment failed; no final fit started")
                save()
                raise SystemExit(state["failure"])
            time.sleep(5)
        state["preceding_queue_wait_seconds"] = time.monotonic() - wait_start
        save()

    def run_attempt(mode, repeat, attempt):
        label = f"{args.output_prefix}_{mode}_r{repeat}"
        output = folder / (label if attempt == 1 else label + f"_attempt{attempt}")
        if output.exists():
            raise ValueError(f"Preserve existing output: {output}")
        gpu = gpus[mode]
        fraction, free_guard = (.14, 16000) if mode == "stage" else (.23, 22000)
        if fraction * gpu["total_memory_mib"] > MEMORY_LIMIT_MIB:
            raise ValueError("Declared memory fraction exceeds the 20 GB per-process limit")
        command = [args.python, str(observer), "--run-driver", str(driver),
                   "--atlas-root", str(root / "atlases"),
                   "--structures", "all", "--optimization", args.optimization,
                   "--device", "cuda:0", "--gpu-memory-fraction", str(fraction),
                   "--output-dir", str(output)]
        for name, path in references.items():
            command += ["--reference-" + name, str(path)]
        inputs = [mri / name for name in ("norm.mgz", "aseg.mgz", "wmparc.mgz")] if mode == "stage" else [raw_t1]
        if mode == "stage":
            command += ["--t1", str(inputs[0]), "--aseg", str(inputs[1]), "--wmparc", str(inputs[2])]
        else:
            command += ["--t1", str(raw_t1), "--weights", str(root / "weights")]
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu["uuid"],
                           OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                           NUMEXPR_NUM_THREADS="4", PYTHONPATH=str(source / "src"),
                           PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        environment.pop("CUBLAS_WORKSPACE_CONFIG", None)
        record = {"mode": mode, "repeat": repeat, "structure": "all", "attempt": attempt,
                  "phase": "preflight", "output": str(output), "command": command,
                  "physical_gpu": gpu, "cuda_visible_devices": gpu["uuid"],
                  "gpu_memory_fraction": fraction, "free_memory_guard_mib": free_guard,
                  "own_memory_limit_mib": MEMORY_LIMIT_MIB, "threads": 4,
                  "source_manifest_sha256": source_manifest["sha256"],
                  "input_sha256": {str(p): fixed[str(p)]["sha256"] for p in inputs},
                  "preflight_started_unix": time.time(), "sampled_peak_own_memory_mib": 0}
        with lock:
            state["runs"].append(record)
            state["state"] = "running"
            save()
        preflight_started = time.monotonic()
        verify_fixed()
        update(record, preflight_identity_seconds=time.monotonic() - preflight_started)
        wait_started = time.monotonic()
        while True:
            sample = gpu_sample(gpu)
            update(record, phase="waiting_for_free_budget", gpu_before_launch=sample)
            if sample["memory_free_mib"] >= free_guard:
                break
            time.sleep(5)
        update(record, gpu_budget_wait_seconds=time.monotonic() - wait_started)
        log_path = folder / (output.name + ".log")
        monitor_path = folder / (output.name + "_gpu_load.jsonl")
        with log_path.open("x") as log, monitor_path.open("x") as monitor:
            # This timestamp is immediately before Popen. Identity checking and
            # GPU-budget waiting above are excluded from process wall time.
            process_started = time.monotonic()
            process_started_unix = time.time()
            process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
            finished = {}

            def observe_exit():
                process.wait()
                finished.update(monotonic=time.monotonic(), unix=time.time())

            exit_watcher = threading.Thread(target=observe_exit, daemon=True)
            exit_watcher.start()
            update(record, phase="fitting", pid=process.pid, started_unix=process_started_unix)
            while process.poll() is None:
                try:
                    sample = gpu_sample(gpu, process.pid)
                    with lock:
                        record["sampled_peak_own_memory_mib"] = max(
                            record["sampled_peak_own_memory_mib"], sample["own_memory_mib"])
                    if sample["own_memory_mib"] > MEMORY_LIMIT_MIB:
                        update(record, memory_limit_exceeded=True)
                        process.terminate()
                except (ValueError, subprocess.SubprocessError) as error:
                    sample = {"unix_time": time.time(), "sampling_error": str(error)}
                monitor.write(json.dumps(sample) + "\n")
                monitor.flush()
                save()
                exit_watcher.join(timeout=5)
                if exit_watcher.is_alive() and record.get("memory_limit_exceeded"):
                    process.kill()
            exit_watcher.join()
            update(record, exit_code=process.returncode, finished_unix=finished["unix"],
                   process_wall_seconds=finished["monotonic"] - process_started, phase="exited")
        if process.returncode:
            text = log_path.read_text()
            initialization_only = ("set_per_process_memory_fraction" in text and
                                   "CUDA error: out of memory" in text and
                                   not (output / "api_report.json").exists() and
                                   not (output / "subregions_native.nii.gz").exists() and
                                   not record.get("memory_limit_exceeded"))
            update(record, failure_scope="CUDA initialization before fitting" if initialization_only
                   else "fit_or_driver_failure")
            return record
        report = json.loads((output / "report.json").read_text())
        api_report = json.loads((output / "api_report.json").read_text())
        context_report = json.loads((output / "context_identity.json").read_text())
        if set(api_report["structures"]) != STRUCTURES or set(report["comparisons"]) != STRUCTURES:
            raise ValueError("Final acceptance requires every run to include all four recipes")
        if report["source_sha256"] != runtime or report["input_sha256"] != record["input_sha256"]:
            raise ValueError("Final run did not match declared source/input identities")
        phases = [record["phase"] for record in context_report["contexts"]]
        expected_phases = ["prepared_context"] + (["raw_processing_context"] if mode == "raw" else [])
        if (phases != expected_phases or not context_report["runtime_source_unchanged"] or
                context_report["runtime_source_sha256_before"] != runtime or
                context_report["observer_sha256"] != fixed[str(observer)]["sha256"]):
            raise ValueError("Final preprocessing observation/source identity incomplete")
        update(record, compute_seconds_including_context_observer=report["wall_seconds"],
               context_observer_seconds=context_report["observer_seconds"],
               context_identity=identity(output / "context_identity.json"),
               api_total_seconds=report["api_total_seconds"],
               output_save_seconds=report["output_save_seconds"],
               report=identity(output / "report.json"), api_report=identity(output / "api_report.json"),
               phase="verified_full_run")
        return record

    def run_mode(mode):
        for repeat in (1, 2, 3):
            for attempt in (1, 2, 3):
                record = run_attempt(mode, repeat, attempt)
                if record["exit_code"] == 0:
                    break
                if record["failure_scope"] != "CUDA initialization before fitting" or attempt == 3:
                    raise RuntimeError(f"Final {mode} repeat {repeat} failed; preserve every attempt")
                update(record, retry_reason="pre-fit CUDA initialization OOM only; failed output preserved")
                time.sleep(5)

    try:
        if args.parallel_modes:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(run_mode, mode) for mode in ("stage", "raw")]
                for future in futures:
                    future.result()
        else:
            for mode in ("stage", "raw"):
                run_mode(mode)
        successful = [r for r in state["runs"] if r.get("phase") == "verified_full_run"]
        if {(r["mode"], r["repeat"], r["structure"]) for r in successful} != {
                (mode, repeat, "all") for mode in ("stage", "raw") for repeat in (1, 2, 3)}:
            raise ValueError("Final evidence must contain exactly six successful full runs")
        verify_fixed()
        state.update(state="completed", finished_unix=time.time(), successful_full_runs=len(successful))
        save()
    except BaseException as error:
        state.update(state="failed", finished_unix=time.time(), failure=str(error))
        save()
        raise
    artifacts = []
    for record in state["runs"]:
        output = Path(record["output"])
        for filename in ("report.json", "api_report.json", "comparison.tsv", "context_identity.json"):
            path = output / filename
            if path.is_file():
                artifacts.append(identity(path))
        for path in (folder / (output.name + ".log"), folder / (output.name + "_gpu_load.jsonl")):
            artifacts.append(identity(path))
    artifacts.append(identity(state_path))
    (folder / (args.output_prefix + "_artifacts.json")).write_text(json.dumps({
        "scope": "derived reports, logs and identities only; no image/atlas/weight/license transfer",
        "artifacts": artifacts}, indent=2) + "\n")


if __name__ == "__main__":
    main()
