"""Run a fixed real T1 cohort through frozen FNIT, with bounded shared GPU use.

Two queues are served by distinct physical GPUs. Raw input is independent of
FreeSurfer; stage input waits for that subject's successful fresh recon-all.
Official label files are never passed into the production fit. All failures and
waits are retained. The read-only observer hashes actual preprocessing arrays.
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


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def verified_source(source):
    manifest = json.loads((source / "source_manifest.json").read_text())
    runtime = {}
    for entry in manifest["files"]:
        actual = identity(source / entry["path"])
        if any(actual[key] != entry[key] for key in ("bytes", "sha256")):
            raise ValueError("Frozen source changed: " + entry["path"])
        if entry["path"].startswith("src/fnit/") and entry["path"].endswith(".py"):
            runtime[entry["path"].removeprefix("src/fnit/")] = entry["sha256"]
    actual_paths = {str(p.relative_to(source / "src/fnit"))
                    for p in (source / "src/fnit").rglob("*.py")}
    if actual_paths != set(runtime):
        raise ValueError("Unmanifested native module")
    return identity(source / "source_manifest.json"), runtime


def gpu_identity(index):
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total",
                                   "--format=csv,noheader,nounits"], text=True, timeout=10).splitlines()
    for row in rows:
        fields = [v.strip() for v in row.split(",")]
        if int(fields[0]) == index:
            return dict(index=index, uuid=fields[1], name=fields[2], total_memory_mib=int(fields[3]))
    raise ValueError("Physical GPU unavailable")


def gpu_sample(gpu, own_pid=None):
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu",
                                   "--format=csv,noheader,nounits"], text=True, timeout=10).splitlines()
    fields = next([v.strip() for v in r.split(",")] for r in rows if r.split(",")[1].strip() == gpu["uuid"])
    if int(fields[0]) != gpu["index"]:
        raise ValueError("Physical GPU index changed")
    processes = []
    rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                                   "--format=csv,noheader,nounits"], text=True, timeout=10).splitlines()
    for row in rows:
        values = [v.strip() for v in row.split(",")]
        if values[0] == gpu["uuid"]:
            processes.append({"pid": int(values[1]), "used_memory_mib": int(values[2])})
    return dict(unix_time=time.time(), physical_gpu=gpu, memory_used_mib=int(fields[2]),
                memory_free_mib=int(fields[3]), utilization_percent=int(fields[4]), processes=processes,
                own_memory_mib=max([p["used_memory_mib"] for p in processes if p["pid"] == own_pid] or [0]),
                other_processes=[p for p in processes if p["pid"] != own_pid])


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--observer", type=Path, required=True)
    parser.add_argument("--raw-gpu", type=int, default=1)
    parser.add_argument("--stage-gpu", type=int, default=0)
    parser.add_argument("--modes", nargs="+", choices=("raw", "stage"), default=["raw", "stage"])
    args = parser.parse_args()
    if len(set(args.modes)) != len(args.modes):
        raise ValueError("Duplicate mode")
    if len(args.modes) == 2 and args.raw_gpu == args.stage_gpu:
        raise ValueError("Concurrent queues require separate physical GPUs")
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    state_path = root / ("fnit_queue.json" if len(args.modes) == 2 else f"fnit_{args.modes[0]}_queue.json")
    if state_path.exists():
        raise ValueError("Preserve existing cohort queue")
    source, assets_root, observer = args.source.resolve(), args.assets_root.resolve(), args.observer.resolve()
    source_identity, runtime = verified_source(source)
    driver = source / "validation/subregions/run_unified.py"
    assets = [identity(p) for directory in ("atlases", "weights")
              for p in sorted((assets_root / directory).rglob("*")) if p.is_file()]
    gpus = {m: gpu_identity(args.raw_gpu if m == "raw" else args.stage_gpu) for m in args.modes}
    state = dict(state="running", planned_cases=manifest["planned_subjects"], modes=args.modes,
                 source=source_identity, source_base_commit=json.loads((source / "source_manifest.json").read_text())["base_commit"],
                 runtime_python_files=len(runtime), manifest=identity(manifest_path), assets=assets,
                 run_driver=identity(driver), observer=identity(observer), queue_script=identity(Path(__file__)),
                 observer_changes_solver_options=False, reference_used_for_fitting=False,
                 optimization="fast", threads_per_process=4, own_memory_limit_mib=MEMORY_LIMIT_MIB,
                 physical_gpu_by_mode=gpus, started_unix=time.time(), runs=[])
    lock = threading.RLock()

    def save():
        with lock:
            write_json(state_path, state)

    def update(record, **values):
        with lock:
            record.update(values)
            save()

    def run_case(case, mode):
        case_root = Path(case["case_root"])
        case_root.mkdir(parents=True, exist_ok=True)
        output = case_root / f"fnit_{mode}"
        gpu = gpus[mode]
        fraction, free_guard = (.23, 22000) if mode == "raw" else (.14, 16000)
        record = dict(case_id=case["id"], component=f"fnit_{mode}", mode=mode, status="pending", state="pending",
                      phase="waiting_for_input", output=str(output), physical_gpu=gpu,
                      gpu_memory_fraction=fraction, own_memory_limit_mib=MEMORY_LIMIT_MIB,
                      sampled_peak_own_memory_mib=0, source_manifest_sha256=source_identity["sha256"])
        with lock:
            state["runs"].append(record)
            save()
        try:
            input_wait = time.monotonic()
            if mode == "stage":
                official_status = Path(case["official"]["status_file"])
                while True:
                    if official_status.exists():
                        official = json.loads(official_status.read_text())
                        recon = official.get("components", {}).get("reconall", {})
                        if recon.get("state") == "completed":
                            break
                        if official.get("state") == "failed" or recon.get("state") == "failed":
                            update(record, status="blocked", phase="failed_official_preprocessing",
                                   input_wait_seconds=time.monotonic() - input_wait)
                            return
                    time.sleep(10)
                inputs = [Path(case["official"][key]) for key in ("norm", "aseg", "wmparc")]
            else:
                inputs = [Path(case["raw_t1"]["path"])]
            update(record, input_wait_seconds=time.monotonic() - input_wait, phase="preflight")
            preflight = time.monotonic()
            current_source, current_runtime = verified_source(source)
            if current_source != source_identity or current_runtime != runtime:
                raise ValueError("Frozen source identity drift")
            actual_inputs = [identity(p) for p in inputs]
            if mode == "raw" and any(actual_inputs[0][key] != case["raw_t1"][key] for key in ("bytes", "sha256")):
                raise ValueError("Raw T1 differs from pinned cohort input")
            if output.exists():
                raise ValueError("Existing fit output must be preserved")
            if fraction * gpu["total_memory_mib"] > MEMORY_LIMIT_MIB:
                raise ValueError("Allocator budget exceeds declared 20 GB")
            command = [args.python, str(observer), "--run-driver", str(driver), "--t1", str(inputs[0]),
                       "--atlas-root", str(assets_root / "atlases"), "--structures", "all",
                       "--optimization", "fast", "--device", "cuda:0", "--gpu-memory-fraction", str(fraction),
                       "--output-dir", str(output)]
            if mode == "stage":
                command += ["--aseg", str(inputs[1]), "--wmparc", str(inputs[2])]
            else:
                command += ["--weights", str(assets_root / "weights")]
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu["uuid"], OMP_NUM_THREADS="4",
                               MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", NUMEXPR_NUM_THREADS="4",
                               PYTHONPATH=str(source / "src"), PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
                               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
            environment.pop("CUBLAS_WORKSPACE_CONFIG", None)
            update(record, command=command, inputs=actual_inputs, input_sha256={p["path"]:p["sha256"] for p in actual_inputs},
                   preflight_identity_seconds=time.monotonic()-preflight, phase="waiting_for_gpu_budget")
            budget_start = time.monotonic()
            while True:
                sample = gpu_sample(gpu)
                update(record, gpu_before_launch=sample)
                if sample["memory_free_mib"] >= free_guard:
                    break
                time.sleep(10)
            update(record, gpu_budget_wait_seconds=time.monotonic()-budget_start)
            with (case_root / f"fnit_{mode}.log").open("x") as log, (case_root / f"fnit_{mode}_gpu_load.jsonl").open("x") as monitor:
                start, start_unix = time.monotonic(), time.time()
                process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
                finished = {}

                def watch_exit():
                    process.wait()
                    finished.update(monotonic=time.monotonic(), unix=time.time())

                watcher = threading.Thread(target=watch_exit, daemon=True)
                watcher.start()
                update(record, pid=process.pid, started_unix=start_unix, status="running", state="running", phase="fitting")
                while watcher.is_alive():
                    try:
                        sample = gpu_sample(gpu, process.pid)
                        with lock:
                            record["sampled_peak_own_memory_mib"] = max(record["sampled_peak_own_memory_mib"],sample["own_memory_mib"])
                        if sample["own_memory_mib"] > MEMORY_LIMIT_MIB:
                            update(record, memory_limit_exceeded=True)
                            process.terminate()
                    except (ValueError, subprocess.SubprocessError) as error:
                        sample = dict(unix_time=time.time(), sampling_error=str(error))
                    monitor.write(json.dumps(sample)+"\n")
                    monitor.flush()
                    save()
                    watcher.join(timeout=5)
                    if watcher.is_alive() and record.get("memory_limit_exceeded"):
                        process.kill()
                update(record, exit_code=process.returncode, process_wall_seconds=finished["monotonic"]-start,
                       finished_unix=finished["unix"], phase="exited",
                       process_wall_scope="Immediately before Popen through independent wait() watcher; input/GPU-budget waits excluded.",
                       gpu_monitor=identity(case_root/f"fnit_{mode}_gpu_load.jsonl"))
            if process.returncode:
                raise RuntimeError("FNIT subprocess failed; retained log and partial output")
            report = json.loads((output / "report.json").read_text())
            api = json.loads((output / "api_report.json").read_text())
            context = json.loads((output / "context_identity.json").read_text())
            if set(api["structures"]) != STRUCTURES or report["source_sha256"] != runtime:
                raise ValueError("Incomplete structures or wrong source")
            if report["input_sha256"] != record["input_sha256"] or not context["runtime_source_unchanged"]:
                raise ValueError("Input or observer/source identity mismatch")
            if context["runtime_source_sha256_before"] != runtime or context["observer_sha256"] != state["observer"]["sha256"]:
                raise ValueError("Observer runtime identity mismatch")
            if len(api["labels"]) != 110 or api["labels"] != manifest["canonical_label_metadata"]:
                raise ValueError("Cohort label contract changed")
            update(record, status="success", state="completed", phase="verified_full_run", compute_seconds=report["wall_seconds"],
                   api_total_seconds=report["api_total_seconds"], output_save_seconds=report["output_save_seconds"],
                   context_observer_seconds=context["observer_seconds"], report=identity(output/"report.json"),
                   api_report=identity(output/"api_report.json"), context_identity=identity(output/"context_identity.json"))
        except Exception as error:
            update(record, status="failed", state="failed", phase="failed", failure=repr(error), failure_recorded_unix=time.time())
        print(json.dumps({key:record.get(key) for key in ("case_id","component","status","process_wall_seconds","failure")}),flush=True)

    def run_mode(mode):
        for case in manifest["cases"]:
            run_case(case, mode)

    save()
    try:
        with ThreadPoolExecutor(max_workers=len(args.modes)) as pool:
            futures = [pool.submit(run_mode, mode) for mode in args.modes]
            for future in futures:
                future.result()
        for asset in assets:
            if identity(asset["path"]) != asset:
                raise ValueError("Benchmark assets changed during cohort")
        if verified_source(source)[0] != source_identity:
            raise ValueError("Benchmark source changed during cohort")
        state["state"] = "completed"
        state["successful_runs"] = sum(r["status"] == "success" for r in state["runs"])
        state["failed_or_blocked_runs"] = sum(r["status"] in ("failed","blocked") for r in state["runs"])
    except Exception as error:
        state.update(state="failed", failure=repr(error))
        raise
    finally:
        state["finished_unix"] = time.time()
        save()


if __name__ == "__main__":
    main()
