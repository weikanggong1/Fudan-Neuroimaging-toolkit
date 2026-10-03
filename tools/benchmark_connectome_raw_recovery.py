"""Private same-run recovery for one known anatomy-report serialization bug.

This benchmark tool never reruns recon-all and never resumes a DWI calculation.
It revalidates this round's successful official anatomy, then starts a fresh
raw-DWI GPU job with the original frozen source and wall runner. Original
reports, workers, and the CPU scheduler remain immutable.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import sys
import time

if __package__:
    from . import benchmark_connectome_raw_cohort as cohort
else:
    import benchmark_connectome_raw_cohort as cohort

RECOVERY_MODE = "known_anatomy_int32_serialization"
ERROR_PREFIX = "actual nibabel anatomy validation failed: Traceback (most recent call last):"
ERROR_SUFFIX = "TypeError: Object of type int32 is not JSON serializable"
SCHEMA_VERSION = 1


def timestamp(value):
    date = datetime.fromisoformat(value)
    if date.tzinfo is None:
        raise ValueError("recovery timestamps must include a UTC offset")
    return date.timestamp()


def assert_namespace(config, case, version, job):
    root = Path(config["run_root"])
    expected = root / version / case["case_id"]
    if Path(job) != expected or not root.is_absolute():
        raise ValueError("recovery job does not match the exact original namespace")
    if any(path.is_symlink() for path in (root, root / version, expected)):
        raise ValueError("recovery namespace must not be a symbolic link")
    _, _, subject = cohort.recon_command(config, case, expected)
    if subject.is_symlink() or subject.parent.is_symlink() or not subject.resolve().is_relative_to(expected.resolve()):
        raise ValueError("official anatomy escapes the fresh original case namespace")
    return subject


def validate_original_recon(config, case, version, job, report):
    """Reject every failure except the exact post-reconstruction int32 bug."""
    subject = assert_namespace(config, case, version, job)
    error = report.get("error", {})
    message = error.get("message", "")
    if (report.get("action") != "recon" or report.get("status") != "failed"
            or report.get("exit_code") != 0 or error.get("type") != "RuntimeError"
            or not isinstance(message, str) or not message.startswith(ERROR_PREFIX)
            or not message.rstrip().endswith(ERROR_SUFFIX)):
        raise ValueError("only the exact anatomy-child int32 JSON serialization failure is recoverable")
    for key, expected in (("case_id", case["case_id"]), ("subject", case["subject"]), ("version", version)):
        if report.get(key) != expected:
            raise ValueError(f"original reconstruction {key} differs from the declared case")
    command, launch, _ = cohort.recon_command(config, case, job)
    if report.get("command") != command or report.get("launch_arguments") != launch:
        raise ValueError("original recon-all command must be the exact raw -i -all command in this namespace")
    if report.get("cpu_threads") != config["cpu_threads"]:
        raise ValueError("original reconstruction thread budget changed")
    fs_version = report.get("freesurfer_version", {})
    if fs_version.get("returncode") != 0 or "freesurfer" not in fs_version.get("stdout", "").lower():
        raise ValueError("original official FreeSurfer version was not verified")
    seconds = report.get("recon_command_seconds")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("original reconstruction duration is invalid")
    if timestamp(report["end_utc"]) < timestamp(report["start_utc"]):
        raise ValueError("original reconstruction timestamp order is invalid")
    if report.get("raw_input_provenance") != case["input_files"]:
        raise ValueError("original raw input provenance differs from this case")
    verified = report.get("input_verification", [])
    if len(verified) != len(case["input_files"]):
        raise ValueError("original raw input byte verification is incomplete")
    for actual, declared in zip(verified, case["input_files"]):
        for key in ("path", "kind", "sha256"):
            if actual.get(key) != declared[key]:
                raise ValueError("original verified input declaration changed")
        if actual.get("actual_sha256", "").lower() != declared["sha256"].lower():
            raise ValueError("original raw input hash verification failed")
    if "scripts/recon-all.done" not in report.get("anatomy", {}):
        raise ValueError("original report lacks the completed official anatomy hashes")
    for name, record in report["anatomy"].items():
        expected_path = subject / name
        if (Path(name).is_absolute() or ".." in Path(name).parts
                or record.get("path") != str(expected_path)
                or not expected_path.resolve().is_relative_to(subject.resolve())):
            raise ValueError("original anatomy file escapes this run's subject namespace")


def assert_fresh_gpu(job):
    for name in ("connectome", "gpu_report.json", "raw_bids_wall.json", "raw_bids_wall.log"):
        if (Path(job) / name).exists():
            raise FileExistsError(f"recovery cannot reuse or overwrite any GPU output: {Path(job) / name}")


def verify_origin(config):
    """Bind the repair to an immutable snapshot of the original driver."""
    origin = config.get("recovery_origin", {})
    if config.get("recovery_mode") != RECOVERY_MODE:
        raise ValueError("missing explicit private recovery mode")
    snapshot = Path(origin["snapshot_path"])
    if cohort.sha256(snapshot) != origin["snapshot_sha256"]:
        raise ValueError("original driver snapshot changed")
    state = json.loads(snapshot.read_text())
    original = state["config"]
    for key, value in original.items():
        if key not in ("worker_script", "worker_script_sha256") and config.get(key) != value:
            raise ValueError(f"recovery changed original frozen configuration: {key}")
    if config["run_root"] != state.get("fresh_namespace", {}).get("path"):
        raise ValueError("original driver did not claim this fresh namespace")
    if state.get("fresh_namespace", {}).get("status") != "claimed_fresh_namespace":
        raise ValueError("original fresh-namespace claim was not successful")
    if cohort.sha256(original["worker_script"]) != original["worker_script_sha256"]:
        raise ValueError("original frozen worker changed")
    if cohort.sha256(config["wall_script"]) != original["wall_script_sha256"]:
        raise ValueError("original frozen raw-DWI wall runner changed")
    manifest_path = Path(config["run_root"]) / "input_manifest.json"
    if cohort.sha256(manifest_path) != origin["input_manifest_sha256"]:
        raise ValueError("original run's input manifest changed")
    return state


def immutable_original(path, expected_sha):
    if cohort.sha256(path) != expected_sha:
        raise ValueError("original failure report changed during recovery")


def revalidate_worker(payload):
    """Perform real CPU-host nibabel reads and write one separate repair."""
    config, case, version = payload["config"], payload["case"], payload["version"]
    if cohort.sha256(__file__) != config["recovery_worker_sha256"]:
        raise ValueError("recovery worker differs from the declared new tool")
    verify_origin(config)
    job = Path(config["run_root"]) / version / case["case_id"]
    subject = assert_namespace(config, case, version, job)
    original_path = job / "recon_report.json"
    repair_path = job / "recon_report.revalidated.json"
    if repair_path.exists() or (job / "recovery_claim.json").exists():
        raise FileExistsError("revalidation is performed once; existing repair/claim is not resumed")
    original_sha = cohort.sha256(original_path)
    original = json.loads(original_path.read_text())
    validate_original_recon(config, case, version, job, original)
    assert_fresh_gpu(job)
    claim = {"status": "claimed_once", "start_utc": cohort.utc(), "original_report_sha256": original_sha,
             "recovery_worker_sha256": config["recovery_worker_sha256"]}
    # Exclusive creation prevents two controllers from repairing this case.
    with (job / "recovery_claim.json").open("x") as stream:
        stream.write(json.dumps(claim, indent=2) + "\n")
    started = time.perf_counter()
    report = {"schema_version": SCHEMA_VERSION, "action": "revalidate_official_anatomy",
              "status": "running", "case_id": case["case_id"], "subject": case["subject"], "version": version,
              "start_utc": cohort.utc(), "identity": cohort.host_identity(), "recovery_mode": RECOVERY_MODE,
              "original_report": {"path": str(original_path), "sha256": original_sha},
              "original_recon_command_seconds": original["recon_command_seconds"],
              "original_recon_start_utc": original["start_utc"], "original_recon_end_utc": original["end_utc"],
              "original_failure": original["error"], "recon_all_rerun": False, "raw_dwi_preprocessing_resumed": False,
              "origin_snapshot_sha256": config["recovery_origin"]["snapshot_sha256"],
              "recovery_worker_sha256": config["recovery_worker_sha256"],
              "cohort_worker_sha256": config["worker_script_sha256"]}
    try:
        if cohort.sha256(config["recon_all"]) != original["executable_sha256"]:
            raise ValueError("official recon-all executable changed")
        if cohort.sha256(Path(config["freesurfer_home"]) / "SetUpFreeSurfer.sh") != original["setup_script_sha256"]:
            raise ValueError("official setup script changed")
        report["input_verification"] = cohort.verify_inputs(case)
        anatomy = cohort.check_anatomy(subject, config["atlases"])
        if anatomy != original["anatomy"]:
            raise ValueError("official anatomy changed after the completed reconstruction")
        report["anatomy"] = anatomy
        thread_keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        previous_threads = {key: cohort.os.environ.get(key) for key in thread_keys}
        report["validation_environment"] = {key: str(config["cpu_threads"]) for key in thread_keys}
        try:
            cohort.os.environ.update(report["validation_environment"])
            report["anatomy_geometry"] = cohort.validate_anatomy_child(subject, anatomy, config["anatomy_validation_python"])
        finally:
            for key, value in previous_threads.items():
                if value is None:
                    cohort.os.environ.pop(key, None)
                else:
                    cohort.os.environ[key] = value
        if cohort.check_anatomy(subject, config["atlases"]) != anatomy:
            raise ValueError("anatomy changed while real geometry was being revalidated")
        report["input_verification_after"] = cohort.verify_inputs(case)
        report["raw_t1_sha256_after"] = cohort.sha256(case["t1w"])
        immutable_original(original_path, original_sha)
        assert_fresh_gpu(job)
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
    report.update(end_utc=cohort.utc(), revalidation_wall_seconds=time.perf_counter() - started)
    cohort.atomic_json(repair_path, report)
    immutable_original(original_path, original_sha)
    return report


def load_revalidated_recon(config, case, version, job, original):
    """GPU-worker gate: verify a declared repair, never reinterpret old status."""
    verify_origin(config)
    job = Path(job)
    subject = assert_namespace(config, case, version, job)
    validate_original_recon(config, case, version, job, original)
    key = f"{version}/{case['case_id']}"
    path = Path(config["revalidated_recon_reports"][key])
    if path != job / "recon_report.revalidated.json" or path.is_symlink():
        raise ValueError("revalidated report must be the declared separate file in the original case namespace")
    repair = json.loads(path.read_text())
    if (repair.get("action") != "revalidate_official_anatomy" or repair.get("status") != "completed"
            or repair.get("recovery_mode") != RECOVERY_MODE or repair.get("recon_all_rerun") is not False
            or repair.get("raw_dwi_preprocessing_resumed") is not False):
        raise ValueError("separate anatomy revalidation did not successfully complete")
    for key, value in (("case_id", case["case_id"]), ("subject", case["subject"]), ("version", version),
                       ("origin_snapshot_sha256", config["recovery_origin"]["snapshot_sha256"]),
                       ("recovery_worker_sha256", config["recovery_worker_sha256"]),
                       ("cohort_worker_sha256", config["worker_script_sha256"])):
        if repair.get(key) != value:
            raise ValueError(f"revalidated report binding changed: {key}")
    original_path = job / "recon_report.json"
    if repair.get("original_report") != {"path": str(original_path), "sha256": cohort.sha256(original_path)}:
        raise ValueError("original failure report hash does not match the separate repair")
    if (repair.get("original_failure") != original["error"]
            or repair.get("original_recon_command_seconds") != original["recon_command_seconds"]
            or repair.get("original_recon_start_utc") != original["start_utc"]
            or repair.get("original_recon_end_utc") != original["end_utc"]):
        raise ValueError("repair changed original failure or official reconstruction timing")
    if repair.get("anatomy_geometry", {}).get("status") != "actual_images_surfaces_annotations_read":
        raise ValueError("repair did not read actual MRI arrays/surfaces/annotations")
    actual = cohort.check_anatomy(subject, config["atlases"])
    if actual != original["anatomy"] or repair.get("anatomy") != actual:
        raise ValueError("completed official anatomy hashes changed")
    inputs = cohort.verify_inputs(case)
    if repair.get("input_verification") != inputs or repair.get("input_verification_after") != inputs:
        raise ValueError("revalidated raw inputs changed")
    t1_sha = next(item["sha256"] for item in case["input_files"] if item["kind"] == "raw_t1w")
    if repair.get("raw_t1_sha256_after") != t1_sha:
        raise ValueError("revalidation did not verify the original raw T1 hash")
    copied = copy.deepcopy(original)
    copied.update(status="completed", anatomy_geometry=repair["anatomy_geometry"],
                  recovery={"mode": RECOVERY_MODE, "original_report": repair["original_report"],
                            "revalidated_report": {"path": str(path), "sha256": cohort.sha256(path)},
                            "original_status": "failed", "original_failure": original["error"],
                            "revalidation_wall_seconds": repair["revalidation_wall_seconds"],
                            "execution_scope": "same fresh reconstruction; benchmark validation repaired; not pristine cold timing"})
    return copied


def recovery_timing(original_start_utc, original_end_utc, repaired_start_utc, repaired_end_utc,
                    gpu_end_utc, recon_seconds, gpu_driver_queue, gpu_lock_queue,
                    *, original_recon_start_utc=None, original_recon_worker_wall_seconds=None):
    times = [timestamp(value) for value in (original_start_utc, original_end_utc, repaired_start_utc, repaired_end_utc, gpu_end_utc)]
    if any(right < left for left, right in zip(times, times[1:])):
        raise ValueError("recovery timestamp order is invalid")
    durations = (recon_seconds, gpu_driver_queue, gpu_lock_queue)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in durations):
        raise ValueError("recovery durations must be finite and nonnegative")
    full = times[-1] - times[0]
    queue = gpu_driver_queue + gpu_lock_queue
    # Command and worker timers belong to the CPU host. A head-driver UTC
    # start is not comparable with a CPU-host UTC end when clocks differ.
    cpu_start = timestamp(original_recon_start_utc) if original_recon_start_utc else times[0]
    cpu_wall = times[1] - cpu_start
    if recon_seconds > cpu_wall + .001:
        raise ValueError("official reconstruction timer exceeds its original wall interval")
    if original_recon_worker_wall_seconds is not None and recon_seconds > original_recon_worker_wall_seconds + .001:
        raise ValueError("official reconstruction timer exceeds its same-host worker monotonic interval")
    if queue > times[4] - times[3] + .001:
        raise ValueError("GPU queue exceeds the post-revalidation GPU elapsed interval")
    return {"recovered_full_elapsed_utc_seconds": full,
            "recovered_full_elapsed_utc_excluding_gpu_queue_seconds": max(0., full - queue),
            "original_recon_command_seconds": recon_seconds,
            "original_recon_same_host_wall_utc_seconds": cpu_wall,
            "cpu_worker_start_minus_head_driver_start_utc_seconds": cpu_start - times[0],
            "original_failure_to_revalidation_gap_utc_seconds": times[2] - times[1],
            "revalidation_elapsed_utc_seconds": times[3] - times[2],
            "post_revalidation_through_gpu_completion_utc_seconds": times[4] - times[3],
            "gpu_driver_queue_seconds": gpu_driver_queue, "gpu_lock_queue_seconds": gpu_lock_queue,
            "scope": "actual UTC interval from original driver case start through recovered GPU completion, including the known tool failure, repair gap, reads and queues; not a continuous monotonic cold-run timer and not a sum of stage medians"}


def recovery_remote(config, host_kind, action, case, version, log_path):
    command = [config[host_kind + "_python"], config["recovery_worker_script"], "_worker"]
    payload = {"action": action, "config": config, "case": case, "version": version}
    result = subprocess.run(cohort.ssh_command(config[host_kind + "_host"], config.get(host_kind + "_port"),
                                              config.get(host_kind + "_control_path"), command),
                            input=json.dumps(payload), capture_output=True, text=True)
    Path(log_path).write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f"{host_kind} recovery worker failed: see {log_path}")
    return json.loads(result.stdout)


def atomic_cases_csv(path, cases):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{cohort.uuid.uuid4().hex}.tmp")
    columns = ("version", "case_id", "subject", "status", "original_driver_start_utc",
               "original_recon_start_utc", "original_recon_end_utc", "end_utc",
               "recovered_full_elapsed_utc_seconds", "recovered_full_elapsed_utc_excluding_gpu_queue_seconds",
               "original_recon_command_seconds", "original_failure_to_revalidation_gap_utc_seconds",
               "revalidation_elapsed_utc_seconds", "post_revalidation_through_gpu_completion_utc_seconds",
               "gpu_driver_queue_seconds", "gpu_lock_queue_seconds")
    try:
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for record in cases.values():
                writer.writerow({key: record.get(key, record.get("timing", {}).get(key)) for key in columns})
        cohort.os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_recovery(options):
    report_dir = cohort.require_fresh(options.report_dir)
    original_status = options.original_driver_report_dir / "status.json"
    snapshot_bytes = original_status.read_bytes()
    snapshot = json.loads(snapshot_bytes)
    snapshot_path = report_dir / "origin_driver_snapshot.json"
    snapshot_path.write_bytes(snapshot_bytes)
    config = copy.deepcopy(snapshot["config"])
    if config.get("sources", {}).keys() != {"baseline"} or config.get("pilot"):
        raise ValueError("private recovery is restricted to the original formal baseline cohort")
    manifest_path = Path(config["run_root"]) / "input_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    cases = cohort.validate_manifest(manifest)
    if set(snapshot.get("cases", {})) != {f"baseline/{case['case_id']}" for case in cases}:
        raise ValueError("original driver and manifest do not describe the same ten cases")
    config.update(worker_script=str(options.worker_script), worker_script_sha256=cohort.sha256(options.worker_script),
                  recovery_worker_script=str(Path(__file__).resolve()), recovery_worker_sha256=cohort.sha256(__file__),
                  recovery_mode=RECOVERY_MODE,
                  recovery_origin={"snapshot_path": str(snapshot_path), "snapshot_sha256": cohort.sha256(snapshot_path),
                                   "original_driver_status": str(original_status),
                                   "input_manifest_sha256": cohort.sha256(manifest_path)},
                  revalidated_recon_reports={f"baseline/{case['case_id']}": str(Path(config["run_root"]) / "baseline" / case["case_id"] / "recon_report.revalidated.json") for case in cases})
    verify_origin(config)
    if cohort.sha256(config["worker_script"]) != cohort.sha256(cohort.__file__):
        raise ValueError("new GPU worker path does not contain this recovery harness's cohort implementation")
    state = {"schema_version": SCHEMA_VERSION, "status": "running", "start_utc": cohort.utc(),
             "scope": "formal baseline same-run recovery after known tool validation failure; not pristine cold benchmark",
             "config": config, "requested_cases": len(cases), "cases": {}, "scientific_parity": "not_assessed", "speedup": "not_assessed"}
    for case in cases:
        key = f"baseline/{case['case_id']}"
        state["cases"][key] = {"version": "baseline", "case_id": case["case_id"], "subject": case["subject"], "status": "waiting_original_cpu_report"}
    def save():
        cohort.atomic_json(report_dir / "status.json", state)
        atomic_cases_csv(report_dir / "cases.csv", state["cases"])
    save()
    started = time.perf_counter()
    submitted, gpu_futures = set(), {}
    with ThreadPoolExecutor(max_workers=1) as pool:
        while True:
            current_driver = json.loads(original_status.read_text())
            for case in cases:
                key = f"baseline/{case['case_id']}"
                if key in submitted:
                    continue
                job = Path(config["run_root"]) / "baseline" / case["case_id"]
                path = job / "recon_report.json"
                if not path.is_file():
                    continue
                original = json.loads(path.read_text())
                if original.get("status") not in ("failed", "completed"):
                    continue
                submitted.add(key)
                record = state["cases"][key]
                original_driver_case = current_driver.get("cases", {}).get(key, {})
                record.update(original_driver_start_utc=original_driver_case.get("start_utc", original["start_utc"]),
                              original_recon_start_utc=original["start_utc"], original_recon_end_utc=original.get("end_utc"),
                              original_recon_command_seconds=original.get("recon_command_seconds"),
                              original_recon_worker_wall_seconds=original.get("worker_wall_seconds"),
                              original_report={"path": str(path), "sha256": cohort.sha256(path)}, original_failure=original.get("error"))
                try:
                    validate_original_recon(config, case, "baseline", job, original)
                    assert_fresh_gpu(job)
                    record["status"] = "revalidating_original_anatomy"
                    save()
                    repair = recovery_remote(config, "cpu", "revalidate", case, "baseline", report_dir / f"{case['case_id']}-revalidate.stderr.log")
                    record["revalidation_report"] = repair
                    if repair.get("status") != "completed":
                        raise RuntimeError(f"actual anatomy revalidation failed: {repair.get('error')}")
                    ready = time.perf_counter()
                    record["status"] = "gpu_queued"
                    def downstream(case=case, ready=ready):
                        begun = time.perf_counter()
                        result = cohort.remote(config, "gpu", {"action": "gpu", "config": config, "case": case, "version": "baseline"},
                                               report_dir / f"{case['case_id']}-gpu.stderr.log")
                        return result, begun - ready, cohort.utc()
                    gpu_futures[pool.submit(downstream)] = key
                except Exception as error:
                    record.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
                save()
            for future, key in list(gpu_futures.items()):
                if not future.done():
                    continue
                record = state["cases"][key]
                try:
                    gpu, queue, finished_utc = future.result()
                    repair = record["revalidation_report"]
                    # Save the scientific execution result before validating
                    # reporting timers; a timer error must never mask it.
                    record.update(status=gpu["status"], gpu_report=gpu, end_utc=finished_utc)
                    if gpu.get("error"):
                        record["error"] = gpu["error"]
                    try:
                        record["timing"] = recovery_timing(record["original_driver_start_utc"], record["original_recon_end_utc"],
                                                         repair["start_utc"], repair["end_utc"], finished_utc,
                                                         record["original_recon_command_seconds"], queue, gpu.get("gpu_lock_queue_seconds", 0.),
                                                         original_recon_start_utc=record["original_recon_start_utc"],
                                                         original_recon_worker_wall_seconds=record.get("original_recon_worker_wall_seconds"))
                    except Exception as timer_error:
                        record["timing_error"] = {"type": type(timer_error).__name__, "message": str(timer_error)}
                    print(json.dumps({"case": key, "status": record["status"], "timing": record.get("timing"), "timing_error": record.get("timing_error")}), flush=True)
                except Exception as error:
                    record.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
                del gpu_futures[future]
                save()
            if len(submitted) == len(cases) and not gpu_futures:
                break
            if time.perf_counter() - started > options.timeout_hours * 3600:
                # Already submitted GPU jobs continue to their real terminal
                # result; pending CPU cases remain explicitly incomplete.
                state["timeout_reached"] = True
                if not gpu_futures:
                    break
            time.sleep(min(options.poll_seconds, 60.))
    completed = sum(record["status"] == "completed" for record in state["cases"].values())
    state.update(completed_cases=completed, status="completed_recovered_execution" if completed == len(cases) else "failed_or_incomplete",
                 comparison_ready=False, end_utc=cohort.utc(), recovery_controller_wall_seconds=time.perf_counter() - started)
    save()
    return 0 if completed == len(cases) else 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_worker":
        payload = json.load(sys.stdin)
        if payload["action"] != "revalidate":
            raise ValueError("private recovery worker only revalidates official anatomy")
        print(json.dumps(revalidate_worker(payload), allow_nan=False), flush=True)
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-driver-report-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--worker-script", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.)
    parser.add_argument("--timeout-hours", type=float, default=36.)
    options = parser.parse_args(argv)
    for path in (options.original_driver_report_dir, options.report_dir, options.worker_script):
        cohort.absolute_path(str(path), "recovery path")
    if not 1 <= options.poll_seconds <= 60 or not 0 < options.timeout_hours <= 168:
        parser.error("poll must be 1..60 seconds and timeout positive up to 168 hours")
    return run_recovery(options)


if __name__ == "__main__":
    raise SystemExit(main())
