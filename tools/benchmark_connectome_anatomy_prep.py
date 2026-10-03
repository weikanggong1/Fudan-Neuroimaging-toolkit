"""Prepare ten fresh official recon-all subjects before candidate source freeze.

This stdlib benchmark harness has no FNIT source, GPU stage or resume path.
It reuses the cohort's official reconstruction worker, then binds its actual
reports, complete anatomy reads and unchanged raw inputs in a separate ledger.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import copy
import csv
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

SCHEMA_VERSION = 1
VERSION = "candidate"
SCOPE = "staged_anatomy_preparation_only"
CPU_KEYS = ("cpu_host", "cpu_port", "cpu_control_path", "cpu_python", "anatomy_validation_python",
            "freesurfer_home", "recon_all", "fs_license", "atlases")
FUTURE_KEYS = ("gpu_host", "gpu_port", "gpu_control_path", "gpu_python", "gpu_uuid", "device",
               "n_seeds", "seed", "eddy_gp_seed", "cuda_visible_devices", "gpu_cpu_threads",
               "gpu_path_prefix", "gpu_lock", "atlas_options")


def finite_nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def overlaps(left, right):
    left, right = Path(left).resolve(), Path(right).resolve()
    return left.is_relative_to(right) or right.is_relative_to(left)


def reject_source_fields(value):
    """Candidate calculation code remains unknown during official preparation."""
    if isinstance(value, dict):
        if any(key in value for key in ("sources", "frozen_sources")):
            raise ValueError("anatomy preparation must not contain FNIT source configuration")
        for child in value.values():
            reject_source_fields(child)
    elif isinstance(value, list):
        for child in value:
            reject_source_fields(child)


def validate_config(config):
    reject_source_fields(config)
    if config.get("scope") != SCOPE or config.get("candidate_source") != "unknown":
        raise ValueError("official preparation must declare its staged scope and unknown candidate")
    if config.get("gpu_status") != "GPU_not_started" or config.get("gpu_started") is not False:
        raise ValueError("official anatomy preparation cannot declare a GPU execution")
    if config.get("cpu_threads") != 8 or isinstance(config.get("cpu_threads"), bool):
        raise ValueError("official reconstruction must use exactly eight CPU threads")
    maximum_jobs = 8 if config.get("selected_cases") else 2
    if config.get("cpu_jobs") not in range(1, maximum_jobs + 1) or isinstance(config.get("cpu_jobs"), bool):
        raise ValueError("preparation supports up to eight jobs only with an explicit case selection; ordinary preparation allows two")
    if "selected_cases" in config and (not isinstance(config["selected_cases"], list) or not config["selected_cases"]
            or any(not isinstance(value, str) for value in config["selected_cases"])
            or len(set(config["selected_cases"])) != len(config["selected_cases"])):
        raise ValueError("explicit preparation selection must be a nonempty unique case list")
    for name in ("run_root", "worker_script", "anatomy_prep_script", "cpu_python", "anatomy_validation_python",
                 "freesurfer_home", "recon_all"):
        cohort.absolute_path(config.get(name), name)
    if not config.get("cpu_host") or not config.get("atlases") or any(name not in cohort.ATLAS_NAMES for name in config["atlases"]):
        raise ValueError("official preparation needs the original CPU host and supported atlas declarations")
    transport = config.get("cpu_transport", "ssh")
    if transport not in ("ssh", "local"):
        raise ValueError("unknown official preparation CPU transport")
    if transport == "local":
        if config["cpu_host"] != "nodecw10":
            raise ValueError("local CPU transport requires the frozen nodecw10 host")
    else:
        cohort.ssh_command(config["cpu_host"], config.get("cpu_port"), config.get("cpu_control_path"), [config["cpu_python"]])


def official_origin_identity(original_config, cases):
    """Bind official identities already recorded by this round's raw-T1 jobs."""
    identities = []
    for case in cases:
        path = Path(original_config["run_root"]) / "baseline" / case["case_id"] / "recon_report.json"
        if not path.is_file():
            continue
        report = json.loads(path.read_text())
        version = report.get("freesurfer_version", {})
        if (report.get("action") != "recon" or report.get("case_id") != case["case_id"]
                or report.get("version") != "baseline" or report.get("exit_code") != 0
                or version.get("returncode") != 0 or "freesurfer" not in version.get("stdout", "").lower()):
            continue
        command, launch, _ = cohort.recon_command(original_config, case, path.parent)
        if report.get("command") != command or report.get("launch_arguments") != launch:
            raise ValueError("original official probe does not bind this round's raw -i -all command")
        identity = {"executable_sha256": report["executable_sha256"],
                    "setup_script_sha256": report["setup_script_sha256"], "version": version["stdout"].strip()}
        if identities and identity != identities[0]["identity"]:
            raise ValueError("this round recorded inconsistent official FreeSurfer identities")
        identities.append({"identity": identity, "report": {"path": str(path), "sha256": cohort.sha256(path)}})
    if not identities:
        raise ValueError("this round has no verified exit-zero official raw-T1 reconstruction probe")
    return identities[0]


def freeze_runtime_files(config):
    paths = [("cohort_worker", config["worker_script"]), ("preparation_worker", config["anatomy_prep_script"]),
             ("recon_all", config["recon_all"]),
             ("official_setup", str(Path(config["freesurfer_home"]) / "SetUpFreeSurfer.sh")),
             ("cpu_python", config["cpu_python"]), ("anatomy_validation_python", config["anatomy_validation_python"])]
    records = []
    for role, value in paths:
        path = Path(value)
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"required official preparation file missing/empty: {path}")
        if role in ("recon_all", "cpu_python", "anatomy_validation_python") and not cohort.os.access(path, cohort.os.X_OK):
            raise PermissionError(f"required preparation executable is not executable: {path}")
        records.append({"role": role, "path": str(path), "size_bytes": path.stat().st_size,
                        "sha256": cohort.sha256(path)})
    return records


def verify_runtime_files(config):
    validate_config(config)
    records = config["frozen_runtime_files"]
    current = freeze_runtime_files(config)
    if current != records:
        raise ValueError("official preparation executable, Python or benchmark worker bytes changed")
    actual = {item["role"]: item["sha256"] for item in current}
    origin = config["official_origin"]["identity"]
    if actual["recon_all"] != origin["executable_sha256"] or actual["official_setup"] != origin["setup_script_sha256"]:
        raise ValueError("official preparation installation differs from this round's original identity")
    if actual["cohort_worker"] != config["worker_script_sha256"] or actual["preparation_worker"] != config["anatomy_prep_script_sha256"]:
        raise ValueError("official preparation worker differs from the frozen benchmark identity")
    if cohort.sha256(cohort.__file__) != config["worker_script_sha256"] or cohort.sha256(__file__) != config["anatomy_prep_script_sha256"]:
        raise ValueError("imported preparation harness differs from its declared shared worker")
    binding = config["origin_binding"]
    if cohort.sha256(binding["input_manifest_path"]) != binding["input_manifest_sha256"]:
        raise ValueError("original raw manifest changed after anatomy preparation started")
    report = config["official_origin"]["report"]
    if cohort.sha256(report["path"]) != report["sha256"]:
        raise ValueError("recorded original official identity report changed")
    if config.get("parent_connection"):
        connection = config["parent_connection"]
        if cohort.sha256(connection["path"]) != connection["sha256"]:
            raise ValueError("original head connection-start record changed")
    verify_prior_preparations(config)


def select_cases(cases, selected):
    if selected is None:
        return cases
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("explicit preparation selection must be nonempty and unique")
    available = {case["case_id"]: case for case in cases}
    if any(value not in available for value in selected):
        raise ValueError("selected preparation case is absent from the full canonical manifest")
    return [available[value] for value in selected]


def verify_prior_preparations(config):
    """Prevent a new subset from duplicating any already started earlier job."""
    selected = config.get("selected_cases", [])
    for binding in config.get("prior_preparations", []):
        path = Path(binding["path"])
        if path.is_symlink() or cohort.sha256(path) != binding["sha256"]:
            raise ValueError("prior preparation config bytes changed")
        prior = json.loads(path.read_bytes())
        if (prior.get("scope") != SCOPE or prior.get("candidate_source") != "unknown"
                or prior.get("gpu_started") is not False or prior.get("cpu_threads") != 8):
            raise ValueError("prior preparation does not describe this round's official staged anatomy")
        if (prior.get("official_origin", {}).get("identity") != config["official_origin"]["identity"]
                or prior.get("atlases") != config["atlases"]
                or prior.get("future_gpu_parameters") != config["future_gpu_parameters"]):
            raise ValueError("prior preparation differs in official identity or shared future parameters")
        manifest = Path(prior["run_root"]) / "input_manifest.json"
        if cohort.sha256(manifest) != config["origin_binding"]["input_manifest_sha256"]:
            raise ValueError("prior preparation full raw manifest differs from this round")
        stop = Path(binding["stop_dispatch_path"])
        if not stop.is_file() or stop.is_symlink() or cohort.sha256(stop) != binding["stop_dispatch_sha256"]:
            raise ValueError("prior preparation must have an unchanged explicit STOP_DISPATCH")
        for case_id in selected:
            job = Path(prior["run_root"]) / VERSION / case_id
            if job.exists() or job.is_symlink():
                raise FileExistsError("new preparation selection overlaps an already started official subject")


def derive_config(options):
    origin_path = options.origin_driver_report_dir / "status.json"
    origin_bytes = origin_path.read_bytes()
    origin = json.loads(origin_bytes)
    original = origin["config"]
    if original.get("pilot") or original.get("cpu_threads") != 8:
        raise ValueError("preparation must derive from the formal eight-thread original cohort")
    manifest_path = Path(original["run_root"]) / "input_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    full_cases = cohort.validate_manifest(manifest)
    if len(full_cases) != 10:
        raise ValueError("this formal preparation requires exactly ten distinct raw subjects")
    selected = getattr(options, "selected_cases", None)
    if getattr(options, "avoid_preparations", None) and selected is None:
        raise ValueError("avoiding earlier preparations requires an explicit new case subset")
    cases = select_cases(full_cases, selected)
    claim = origin.get("fresh_namespace", {})
    if claim.get("status") != "claimed_fresh_namespace" or claim.get("path") != original["run_root"]:
        raise ValueError("original raw round did not record a fresh namespace claim")
    for path in (options.run_root, options.report_dir):
        if overlaps(path, original["run_root"]):
            raise ValueError("new anatomy/report namespace must not overlap original reconstruction output")
    if overlaps(options.run_root, options.report_dir):
        raise ValueError("preparation run and driver reports need separate fresh namespaces")
    config = {key: copy.deepcopy(original[key]) for key in CPU_KEYS if key in original}
    config.update(schema_version=SCHEMA_VERSION, scope=SCOPE, candidate_source="unknown", gpu_status="GPU_not_started",
                  gpu_started=False, run_root=str(options.run_root), cpu_jobs=options.cpu_jobs, cpu_threads=options.cpu_threads,
                  worker_script=str(options.worker_script), worker_script_sha256=cohort.sha256(options.worker_script),
                  anatomy_prep_script=str(Path(__file__).resolve()), anatomy_prep_script_sha256=cohort.sha256(__file__),
                  future_gpu_parameters={key: copy.deepcopy(original[key]) for key in FUTURE_KEYS if key in original},
                  origin_binding={"driver_status_path": str(origin_path),
                                  "driver_status_sha256_at_launch": cohort.hashlib.sha256(origin_bytes).hexdigest(),
                                  "input_manifest_path": str(manifest_path), "input_manifest_sha256": cohort.sha256(manifest_path)},
                  official_origin=official_origin_identity(original, full_cases))
    if getattr(options, "cpu_transport", "ssh") == "local":
        config["cpu_transport"] = "local"
    if getattr(options, "parent_connection_record", None):
        if config.get("cpu_transport") != "local":
            raise ValueError("head connection-start record only belongs to an explicit CPU-local coordinator")
        path = options.parent_connection_record
        config["parent_connection"] = {"path": str(path), "sha256": cohort.sha256(path)}
    if selected is not None:
        config["selected_cases"] = list(selected)
        config["full_manifest_case_count"] = len(full_cases)
    config["prior_preparations"] = []
    for declaration in getattr(options, "avoid_preparations", []) or []:
        path, driver = map(Path, declaration)
        prior = json.loads(path.read_bytes())
        for destination in (options.run_root, options.report_dir):
            if overlaps(destination, prior["run_root"]) or overlaps(destination, driver):
                raise ValueError("new preparation/report namespace overlaps an earlier preparation or driver")
        stop = driver / "STOP_DISPATCH"
        config["prior_preparations"].append({"path": str(path), "sha256": cohort.sha256(path),
            "stop_dispatch_path": str(stop), "stop_dispatch_sha256": cohort.sha256(stop)})
    validate_config(config)
    config["frozen_runtime_files"] = freeze_runtime_files(config)
    verify_runtime_files(config)
    return config, manifest, cases


def worker(payload):
    config, action = payload["config"], payload["action"]
    if config.get("cpu_transport") == "local" and cohort.socket.gethostname().split(".")[0] != config["cpu_host"]:
        raise ValueError("local official CPU execution requires the actual frozen nodecw10 hostname")
    verify_runtime_files(config)
    if action == "preflight":
        case = payload["case"]
        _, launch, subject = cohort.recon_command(config, case, Path(config["run_root"]) / VERSION / case["case_id"])
        environment = cohort.recon_environment(config, subject)
        result = subprocess.run([*launch[:6], "-version"], env=environment, capture_output=True, text=True)
        version = {"returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
        if result.returncode or version["stdout"] != config["official_origin"]["identity"]["version"]:
            raise ValueError("CPU official FreeSurfer version differs from this round's bound version")
        return {"status": "verified_official_cpu_environment", "identity": cohort.host_identity(),
                "freesurfer_version": version, "gpu_started": False}
    if action != "recon" or payload.get("version") != VERSION:
        raise ValueError("anatomy preparation worker only accepts official candidate reconstruction")
    case = payload["case"]
    if config.get("selected_cases") and case["case_id"] not in config["selected_cases"]:
        raise ValueError("official worker refuses a case outside its explicit preparation selection")
    manifest = json.loads(Path(config["origin_binding"]["input_manifest_path"]).read_text())
    if [item for item in manifest["cases"] if item["case_id"] == case["case_id"]] != [case]:
        raise ValueError("prepared raw subject differs from the bound original manifest")
    job = Path(config["run_root"]) / VERSION / case["case_id"]
    if job.exists() or any(path.is_symlink() for path in (Path(config["run_root"]), job.parent, job)):
        raise FileExistsError("anatomy preparation never resumes or overwrites an existing case")
    started = time.perf_counter()
    report = {"schema_version": SCHEMA_VERSION, "action": "prepare_official_anatomy", "status": "running",
              "scope": SCOPE, "case_id": case["case_id"], "subject": case["subject"], "version": VERSION,
              "start_utc": cohort.utc(), "identity": cohort.host_identity(), "candidate_source": "unknown",
              "gpu_status": "GPU_not_started", "gpu_started": False, "recon_all_reused": False}
    try:
        report["input_verification_before"] = cohort.verify_inputs(case)
        result = cohort.worker({"action": "recon", "config": config, "case": case, "version": VERSION})
        report["reconstruction_result"] = result
        validate_reconstruction(config, case, result)
        report["input_verification_after"] = cohort.verify_inputs(case)
        if report["input_verification_after"] != report["input_verification_before"]:
            raise ValueError("raw input records changed during official reconstruction")
        _, _, subject = cohort.recon_command(config, case, job)
        if cohort.check_anatomy(subject, config["atlases"]) != result["anatomy"]:
            raise ValueError("prepared anatomy changed after actual array/surface validation")
        verify_runtime_files(config)
        report.update(status="completed", recon_command_seconds=result["recon_command_seconds"],
                      reconstruction_report={"path": str(job / "recon_report.json"), "sha256": cohort.sha256(job / "recon_report.json")})
    except Exception as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
    report.update(end_utc=cohort.utc(), preparation_worker_wall_seconds=time.perf_counter() - started)
    cohort.atomic_json(job / "anatomy_prep_report.json", report)
    return report


def validate_reconstruction(config, case, report):
    job = Path(config["run_root"]) / VERSION / case["case_id"]
    command, launch, subject = cohort.recon_command(config, case, job)
    if (report.get("status") != "completed" or report.get("action") != "recon" or report.get("exit_code") != 0
            or report.get("case_id") != case["case_id"] or report.get("subject") != case["subject"]
            or report.get("version") != VERSION or report.get("cpu_threads") != 8
            or report.get("command") != command or report.get("launch_arguments") != launch):
        raise ValueError("official raw-T1 reconstruction did not complete with the exact fresh -i -all command")
    identity = config["official_origin"]["identity"]
    if (report.get("executable_sha256") != identity["executable_sha256"]
            or report.get("setup_script_sha256") != identity["setup_script_sha256"]
            or report.get("freesurfer_version", {}).get("returncode") != 0
            or report.get("freesurfer_version", {}).get("stdout", "").strip() != identity["version"]):
        raise ValueError("prepared official FreeSurfer identity differs from its frozen original probe")
    if report.get("raw_input_provenance") != case["input_files"]:
        raise ValueError("prepared reconstruction provenance differs from the selected raw subject")
    if report.get("anatomy_geometry", {}).get("status") != "actual_images_surfaces_annotations_read":
        raise ValueError("prepared anatomy has not passed actual MRI/surface/annotation reads")
    if finite_nonnegative(report.get("recon_command_seconds"), "official recon timer") <= 0:
        raise ValueError("official reconstruction needs a positive measured command duration")
    if not report.get("anatomy") or "scripts/recon-all.done" not in report["anatomy"]:
        raise ValueError("prepared official anatomy or recon-all.done is incomplete")
    if subject.is_symlink() or not subject.resolve().is_relative_to(job.resolve()):
        raise ValueError("prepared official subject escaped its new candidate namespace")


def validate_preparation_result(config, case, result):
    if config.get("selected_cases") and case["case_id"] not in config["selected_cases"]:
        raise ValueError("case is outside this preparation's declared selection")
    job = Path(config["run_root"]) / VERSION / case["case_id"]
    actual_path = job / "anatomy_prep_report.json"
    if not actual_path.is_file() or actual_path.is_symlink() or json.loads(actual_path.read_text()) != result:
        raise ValueError("CPU preparation result does not match its actual new report")
    if (result.get("status") != "completed" or result.get("scope") != SCOPE
            or result.get("action") != "prepare_official_anatomy" or result.get("case_id") != case["case_id"]
            or result.get("subject") != case["subject"] or result.get("version") != VERSION
            or result.get("candidate_source") != "unknown" or result.get("gpu_status") != "GPU_not_started"
            or result.get("gpu_started") is not False or result.get("recon_all_reused") is not False):
        raise ValueError("actual fresh official anatomy preparation did not complete")
    recon_path = job / "recon_report.json"
    if result.get("reconstruction_report") != {"path": str(recon_path), "sha256": cohort.sha256(recon_path)}:
        raise ValueError("prepared official reconstruction report byte identity changed")
    actual_recon = json.loads(recon_path.read_text())
    if result.get("reconstruction_result") != actual_recon:
        raise ValueError("prepared reconstruction ledger differs from the actual official report")
    validate_reconstruction(config, case, actual_recon)
    inputs = cohort.verify_inputs(case)
    if result.get("input_verification_before") != inputs or result.get("input_verification_after") != inputs:
        raise ValueError("prepared raw inputs changed after the CPU job")
    subject = cohort.recon_command(config, case, job)[2]
    if cohort.check_anatomy(subject, config["atlases"]) != actual_recon["anatomy"]:
        raise ValueError("prepared official anatomy bytes changed after the CPU job")
    verify_runtime_files(config)


def remote(config, action, case, log_path):
    if config.get("cpu_transport") == "local":
        validate_config(config)
        if cohort.socket.gethostname().split(".")[0] != config["cpu_host"]:
            raise ValueError("local official CPU execution requires the actual frozen nodecw10 hostname")
        payload = {"action": action, "config": config, "case": case, "version": VERSION}
        result = subprocess.run([config["cpu_python"], config["anatomy_prep_script"], "_worker"],
                                input=json.dumps(payload), capture_output=True, text=True)
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(f"local official CPU worker failed with code {result.returncode}; see {log_path}")
        return json.loads(result.stdout)
    transport = {**config, "worker_script": config["anatomy_prep_script"]}
    return cohort.remote(transport, "cpu", {"action": action, "config": config, "case": case, "version": VERSION}, log_path)


def case_timing(started, ended, queued, recon_seconds, *, cpu_local=False):
    full = finite_nonnegative(ended - started, "head reconstruction/report wall")
    queue = finite_nonnegative(started - queued, "CPU scheduler queue")
    recon = finite_nonnegative(recon_seconds, "official reconstruction command timer")
    if recon > full + .001:
        raise ValueError("official command timer exceeds its enclosing head monotonic task wall")
    result = {"preparation_coordinator_case_wall_seconds": full, "cpu_driver_queue_seconds": queue, "recon_command_seconds": recon,
              "scope": "one preparation coordinator monotonic timer before local CPU worker through actual report/input/anatomy checks; GPU not started" if cpu_local else
                       "one head-process monotonic timer before CPU SSH through actual report/input/anatomy checks; CPU scheduler queue before dispatch recorded separately; GPU not started"}
    if not cpu_local:
        result["head_case_wall_seconds"] = full
    return result


def collect_preparation_result(record, context):
    """Keep actual official execution independent from validation/timing errors."""
    result = context["result"]
    record.update(status=result["status"], preparation_report=result,
                  start_utc=context["start_utc"], end_utc=context["end_utc"],
                  preparation_worker_wall_seconds=result.get("preparation_worker_wall_seconds"))
    if result.get("error"):
        record["error"] = result["error"]
    if context.get("validation_error"):
        record["validation_error"] = context["validation_error"]
        record["status"] = "failed_preparation_validation"
    if context.get("timing"):
        record["timing"] = context["timing"]
    if context.get("timing_error"):
        record["timing_error"] = context["timing_error"]
    return record


def atomic_cases_csv(path, records):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{cohort.uuid.uuid4().hex}.tmp")
    columns = ("version", "case_id", "subject", "status", "start_utc", "end_utc", "head_case_wall_seconds",
               "preparation_coordinator_case_wall_seconds", "cpu_driver_queue_seconds", "recon_command_seconds", "preparation_worker_wall_seconds")
    try:
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for record in records.values():
                writer.writerow({key: record.get(key, record.get("timing", {}).get(key)) for key in columns})
        cohort.os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(options):
    started, start_utc = time.perf_counter(), cohort.utc()
    # Reject an original/nested namespace before creating any report files.
    config, manifest, cases = derive_config(options)
    report_dir = cohort.require_fresh(options.report_dir)
    state = {"schema_version": SCHEMA_VERSION, "status": "preparing_configuration", "start_utc": start_utc,
             "scope": SCOPE, "candidate_source": "unknown", "gpu_status": "GPU_not_started", "gpu_started": False,
             "full_pipeline_benchmark": False, "comparison_ready": False, "cases": {},
             "scientific_parity": "not_assessed", "speedup": "not_assessed"}
    state["coordinator_identity"] = cohort.host_identity()
    state_lock = cohort.threading.RLock()
    def save():
        with state_lock:
            cohort.atomic_json(report_dir / "status.json", state)
            atomic_cases_csv(report_dir / "cases.csv", state["cases"])
    def update_case(key, **values):
        with state_lock:
            state["cases"][key].update(values)
            save()
    save()
    try:
        state.update(config=config, requested_cases=len(cases),
                     dataset={key: manifest[key] for key in ("dataset", "snapshot", "license", "source_url", "download_completed_utc")})
        state["official_environment"] = remote(config, "preflight", cases[0], report_dir / "preflight.stderr.log")
        if state["official_environment"].get("status") != "verified_official_cpu_environment":
            raise ValueError("official CPU environment preflight was not successful")
        root = cohort.require_fresh(config["run_root"])
        cohort.atomic_json(root / "input_manifest.json", manifest)
        cohort.atomic_json(root / "anatomy_prep_config.json", config)
        state.update(status="running", fresh_namespace={"status": "claimed_fresh_anatomy_namespace", "path": str(root)})
        pending = list(cases)
        queued = time.perf_counter()
        for case in cases:
            state["cases"][VERSION + "/" + case["case_id"]] = {"version": VERSION, "case_id": case["case_id"],
                "subject": case["subject"], "status": "cpu_queued", "coordinator_identity": state["coordinator_identity"],
                "coordinator_transport": config.get("cpu_transport", "ssh"), "parent_connection": config.get("parent_connection")}
        save()
        def reconstruct(case):
            if (report_dir / "STOP_DISPATCH").exists():
                return {"result": {"status": "dispatch_paused", "gpu_started": False},
                        "start_utc": None, "end_utc": None}
            begun, begun_utc = time.perf_counter(), cohort.utc()
            update_case(VERSION + "/" + case["case_id"], status="cpu_running", start_utc=begun_utc,
                        cpu_driver_queue_seconds=begun-queued)
            result = remote(config, "recon", case, report_dir / f"candidate-{case['case_id']}-recon.stderr.log")
            context = {"result": result, "start_utc": begun_utc}
            if result.get("status") == "completed":
                try:
                    validate_preparation_result(config, case, result)
                except Exception as error:
                    context["validation_error"] = {"type": type(error).__name__, "message": str(error)}
            ended, ended_utc = time.perf_counter(), cohort.utc()
            context["end_utc"] = ended_utc
            try:
                context["timing"] = case_timing(begun, ended, queued, result.get("recon_command_seconds", 0.), cpu_local=config.get("cpu_transport") == "local")
            except Exception as error:
                context["timing_error"] = {"type": type(error).__name__, "message": str(error)}
            return context
        with ThreadPoolExecutor(max_workers=config["cpu_jobs"]) as pool:
            active = {}
            while pending or active:
                if (report_dir / "STOP_DISPATCH").exists():
                    with state_lock:
                        state["dispatch_paused"] = True
                while pending and len(active) < config["cpu_jobs"] and not state.get("dispatch_paused"):
                    case = pending.pop(0)
                    update_case(VERSION + "/" + case["case_id"], status="cpu_dispatched")
                    active[pool.submit(reconstruct, case)] = case
                save()
                if not active:
                    break
                done, _ = wait(active, timeout=options.poll_seconds, return_when=FIRST_COMPLETED)
                for future in done:
                    case = active.pop(future)
                    key = VERSION + "/" + case["case_id"]
                    record = state["cases"][key]
                    try:
                        context = future.result()
                        with state_lock:
                            collect_preparation_result(record, context)
                    except Exception as error:
                        update_case(key, status="failed", end_utc=cohort.utc(), error={"type": type(error).__name__, "message": str(error)})
                    print(json.dumps({"case": key, "status": record["status"], "error": record.get("error")}), flush=True)
                    save()
        completed = sum(record["status"] == "completed" for record in state["cases"].values())
        timing_complete = all(record.get("timing") and not record.get("timing_error") for record in state["cases"].values())
        status = "failed_or_incomplete_anatomy_preparation"
        if completed == len(cases):
            status = "completed_anatomy_preparation" if timing_complete else "completed_anatomy_preparation_with_timing_errors"
        state.update(completed_cases=completed, timing_complete=timing_complete, status=status)
    except Exception as error:
        state.update(status="failed_anatomy_preparation", error={"type": type(error).__name__, "message": str(error)})
    state.update(end_utc=cohort.utc(), preparation_coordinator_wall_seconds=time.perf_counter() - started)
    if config.get("cpu_transport") != "local":
        state["head_preparation_wall_seconds"] = state["preparation_coordinator_wall_seconds"]
    save()
    return 0 if state["status"] == "completed_anatomy_preparation" else 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_worker":
        if sys.version_info < (3, 10):
            raise RuntimeError("official preparation workers require Python 3.10 or newer")
        print(json.dumps(worker(json.load(sys.stdin)), allow_nan=False), flush=True)
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("origin-driver-report-dir", "run-root", "report-dir", "worker-script"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cpu-jobs", type=int, choices=range(1, 9), default=2)
    parser.add_argument("--selected-cases", nargs="+", help="explicit subset of the full ten-case canonical manifest; allows up to eight concurrent CPU jobs")
    parser.add_argument("--avoid-preparations", nargs=2, action="append", metavar=("PREP_CONFIG", "DRIVER_DIR"),
                        help="prior original preparation config plus its driver STOP_DISPATCH; selected subjects must never have started there")
    parser.add_argument("--cpu-threads", type=int, choices=(8,), default=8)
    parser.add_argument("--cpu-transport", choices=("ssh", "local"), default="ssh",
                        help="local only on the actual frozen nodecw10 coordinator; does not use ssh localhost")
    parser.add_argument("--parent-connection-record", type=Path, help="immutable actual head SSH-start record for a CPU-local coordinator")
    parser.add_argument("--poll-seconds", type=float, default=5.)
    options = parser.parse_args(argv)
    for value in vars(options).values():
        if isinstance(value, Path):
            cohort.absolute_path(str(value), "official anatomy preparation path")
    if not 1 <= options.poll_seconds <= 60:
        parser.error("poll interval must be 1..60 seconds")
    return run(options)


if __name__ == "__main__":
    raise SystemExit(main())
