"""Schedule ten fresh official CPU pipelines; preserve per-subject outcomes.

The audited subject runner performs recon-all and three sequential subregion
commands. This manager retains exact child exit times, independent component
records and host load samples. No image or license is moved by this helper.
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


def identity(path):
    return dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--python", required=True)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--jobs", type=int, choices=(1,2,3,4), default=2)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    queue_path = root / "official_queue.json"
    if queue_path.exists():
        raise ValueError("Preserve the existing official cohort queue")
    state = dict(state="running",planned_cases=10,jobs=args.jobs,threads_per_component=4,
                 manager=identity(Path(__file__)),runner=identity(args.runner.resolve()),
                 manifest=identity(manifest_path),started_unix=time.time(),runs=[],cases={},
                 official_cpu_only=True,previous_timings_reused=False)
    lock = threading.RLock()
    startup_ok = threading.Event()
    startup_failed = threading.Event()
    first_case = manifest["cases"][0]["id"]

    def save():
        with lock:
            temporary = queue_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(state,indent=2)+"\n")
            temporary.replace(queue_path)

    def collect(case):
        path = Path(case["official"]["status_file"])
        if path.exists():
            report = json.loads(path.read_text())
            with lock:
                state["runs"] = [run for run in state["runs"] if run["case_id"] != case["id"]]
                state["runs"] += list(report.get("components",{}).values())
                entry = state["cases"][case["id"]]
                entry.update(official_status=report["state"],
                    official_end_to_end_seconds=report.get("complete_pipeline_wall_seconds"),
                    official_end_to_end_scope=report.get("complete_wall_scope"),
                    official_report=str(path),software=report.get("software"))
                for run in state["runs"]:
                    if run["case_id"] == case["id"]:
                        run["process_wall_scope"] = run.get("process_wall_scope", "Before Popen through independent wait() watcher")
                save()

    def run_case(case):
        case_root = Path(case["case_root"])
        case_root.mkdir(parents=True,exist_ok=True)
        command = [args.python,str(args.runner.resolve()),"--case-id",case["id"],"--case-root",str(case_root),
                   "--input-t1",case["raw_t1"]["path"],"--expected-input-sha256",case["raw_t1"]["sha256"]]
        entry = dict(case_id=case["id"],state="running",command=command)
        with lock:
            state["cases"][case["id"]] = entry
            save()
        try:
            if case["id"] != first_case:
                entry["state"] = "waiting_for_first_real_startup"
                save()
                while not startup_ok.wait(timeout=1):
                    if startup_failed.is_set():
                        raise RuntimeError("First official startup failed; remaining subjects were not launched")
            if identity(args.runner.resolve()) != state["runner"]:
                raise ValueError("Official subject runner changed after cohort launch")
            with (case_root/"official_runner.log").open("x") as log, (case_root/"official_resources.jsonl").open("x") as resource_log:
                started,started_unix = time.monotonic(),time.time()
                process = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                ended = {}
                def watch():
                    process.wait()
                    ended.update(monotonic=time.monotonic(),unix=time.time())
                watcher = threading.Thread(target=watch,daemon=True)
                watcher.start()
                with lock:
                    entry.update(pid=process.pid,started_unix=started_unix)
                    save()
                while watcher.is_alive():
                    collect(case)
                    if case["id"] == first_case and not startup_ok.is_set():
                        startup_record = case_root / "official/reconall_record.json"
                        if startup_record.exists():
                            recon = json.loads(startup_record.read_text())
                            if recon.get("pid") and recon.get("exit_code") is None and time.time()-recon.get("started_unix",time.time()) >= 15:
                                state["first_real_reconall_startup"] = dict(case_id=case["id"],pid=recon["pid"],observed_unix=time.time(),running_seconds=time.time()-recon["started_unix"])
                                save()
                                startup_ok.set()
                    sample = dict(unix_time=time.time(),loadavg=list(os.getloadavg()),logical_cpus=os.cpu_count())
                    resource_log.write(json.dumps(sample)+"\n");resource_log.flush()
                    watcher.join(timeout=10)
                collect(case)
                with lock:
                    entry.update(exit_code=process.returncode,finished_unix=ended["unix"],
                        runner_process_wall_seconds=ended["monotonic"]-started,
                        state="completed" if process.returncode==0 else "failed",log=identity(case_root/"official_runner.log"),
                        resources_log=identity(case_root/"official_resources.jsonl"))
                    save()
                if process.returncode:
                    raise RuntimeError("Official child failed; preserved complete log and subject directory")
                if entry["official_status"] != "completed" or len([r for r in state["runs"] if r["case_id"]==case["id"]])!=4:
                    raise ValueError("Full official pipeline did not produce four completed component records")
        except Exception as error:
            if case["id"] == first_case and not startup_ok.is_set():
                startup_failed.set()
            with lock:
                entry.update(state="failed",failure=repr(error),failure_recorded_unix=time.time())
                save()
        print(json.dumps({key:entry.get(key) for key in ("case_id","state","official_end_to_end_seconds","failure")}),flush=True)

    save()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_case,case) for case in manifest["cases"]]
        for future in futures:
            future.result()
    with lock:
        state.update(state="completed" if all(c["state"]=="completed" for c in state["cases"].values()) else "completed_with_failures",
                     finished_unix=time.time(),successful_cases=sum(c["state"]=="completed" for c in state["cases"].values()))
        save()


if __name__ == "__main__":
    main()
