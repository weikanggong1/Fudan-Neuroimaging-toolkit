"""Private fresh anatomy -> later frozen candidate -> fresh raw-DWI benchmark.

Official reconstruction and raw-DWI processing remain separate real stages.
This tool never resumes preprocessing or claims a continuous cold pipeline.
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

if __package__:
    from . import benchmark_connectome_raw_cohort as cohort
    from . import benchmark_connectome_raw_rerun as rerun
else:
    import benchmark_connectome_raw_cohort as cohort
    import benchmark_connectome_raw_rerun as rerun

MODE = "fresh_anatomy_later_frozen_candidate_raw_dwi"
VERSION = "candidate"
ALLOCATOR = "expandable_segments:True"
SHARED_KEYS = ("atlases", "atlas_options", "n_seeds", "seed", "eddy_gp_seed", "device",
               "cuda_visible_devices", "gpu_uuid", "gpu_lock", "gpu_cpu_threads", "gpu_path_prefix", "gpu_python")

# The clean child imports the original frozen helper AND its original cohort.
# Current helpers must not impersonate the original preparation tool bytes.
PREPARATION_CHECK_CODE = r'''
import hashlib, importlib.util, json, pathlib, sys
p = json.load(sys.stdin)
b = p["binding"]
original, snapshot = pathlib.Path(b["original_path"]), pathlib.Path(b["snapshot_path"])
raw = original.read_bytes()
if original.is_symlink() or snapshot.is_symlink() or snapshot.read_bytes() != raw:
    raise ValueError("original preparation config and snapshot differ")
if hashlib.sha256(raw).hexdigest() != b["sha256"]:
    raise ValueError("original preparation config changed")
c = json.loads(raw)
script = pathlib.Path(c["anatomy_prep_script"])
sys.path.insert(0, str(script.parent))
spec = importlib.util.spec_from_file_location("benchmark_connectome_anatomy_prep", script)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)
prep.verify_runtime_files(c)
r = {"status":"original_preparation_runtime_verified", "prep_config_sha256":b["sha256"]}
if p.get("case") is not None:
    case = p["case"]
    manifest = json.loads((pathlib.Path(c["run_root"]) / "input_manifest.json").read_text())
    selected = [v for v in prep.cohort.validate_manifest(manifest) if v["case_id"] == case["case_id"]]
    if selected != [case]:
        raise ValueError("staged case differs from original preparation manifest")
    job = pathlib.Path(c["run_root"]) / "candidate" / case["case_id"]
    report_path = job / "anatomy_prep_report.json"
    report_bytes = report_path.read_bytes()
    prepared = json.loads(report_bytes)
    prep.validate_preparation_result(c, case, prepared)
    subject = prep.cohort.recon_command(c, case, job)[2]
    recon = prepared["reconstruction_result"]
    geometry = prep.cohort.validate_anatomy_child(subject, recon["anatomy"], c["anatomy_validation_python"])
    if geometry.get("status") != "actual_images_surfaces_annotations_read":
        raise ValueError("actual anatomy image/surface/annotation re-read failed")
    prep.validate_preparation_result(c, case, prepared)
    if report_path.read_bytes() != report_bytes:
        raise ValueError("preparation report changed during verification")
    r.update(status="actual_fresh_preparation_revalidated", preparation_report=prepared,
        preparation_report_binding={"path":str(report_path), "sha256":hashlib.sha256(report_bytes).hexdigest()},
        anatomy_subject_dir=str(subject), anatomy_geometry=geometry)
if original.read_bytes() != raw or snapshot.read_bytes() != raw:
    raise ValueError("original preparation config changed during verification")
print(json.dumps(r, allow_nan=False))
'''


def overlaps(left, right):
    left, right = Path(left).resolve(), Path(right).resolve()
    return left.is_relative_to(right) or right.is_relative_to(left)


def for_case(config, case=None):
    """Select an explicit original prep binding without altering that config."""
    origins = config.get("preparation_origins")
    if origins is None:
        return config
    if case is None and "active_preparation_origin_index" in config:
        return config
    index = 0 if case is None else config["case_origin_index"].get(case["case_id"])
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(origins):
        raise ValueError("case has no valid declared preparation origin")
    origin = origins[index]
    if case is not None and case["case_id"] not in origin["case_ids"]:
        raise ValueError("case-to-preparation origin mapping changed")
    return {**config, "preparation_config":origin["preparation_config"],
            "preparation_driver_status_path":origin["preparation_driver_status_path"], "active_preparation_origin_index":index}


def original_prep_config(config, case=None):
    config = for_case(config, case)
    b = config["preparation_config"]
    original, snapshot = Path(b["original_path"]), Path(b["snapshot_path"])
    raw = original.read_bytes()
    if original.is_symlink() or snapshot.is_symlink() or snapshot.read_bytes() != raw or hashlib.sha256(raw).hexdigest() != b["sha256"]:
        raise ValueError("original preparation config bytes changed")
    prep = json.loads(raw)
    if (prep.get("scope") != "staged_anatomy_preparation_only" or prep.get("candidate_source") != "unknown"
            or prep.get("gpu_started") is not False or "sources" in prep or "frozen_sources" in prep):
        raise ValueError("preparation falsely claims a candidate source or GPU computation")
    return prep


def verify_preparation(config, case=None):
    """Use the ORIGINAL config with the ORIGINAL helper in a clean process."""
    config = for_case(config, case)
    prep = original_prep_config(config)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run([prep["anatomy_validation_python"], "-c", PREPARATION_CHECK_CODE],
                            input=json.dumps({"binding":config["preparation_config"], "case":case}),
                            env=env, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("original preparation validation failed: " + result.stderr[-2500:].strip())
    checked = json.loads(result.stdout)
    expected = "original_preparation_runtime_verified" if case is None else "actual_fresh_preparation_revalidated"
    if checked.get("status") != expected or checked.get("prep_config_sha256") != config["preparation_config"]["sha256"]:
        raise ValueError("original preparation child returned invalid identity/status")
    if case is not None:
        prepared = checked.get("preparation_report", {})
        if (prepared.get("status") != "completed" or prepared.get("case_id") != case["case_id"]
                or prepared.get("subject") != case["subject"] or prepared.get("candidate_source") != "unknown"
                or prepared.get("gpu_started") is not False or prepared.get("recon_all_reused") is not False
                or checked.get("anatomy_geometry", {}).get("status") != "actual_images_surfaces_annotations_read"):
            raise ValueError("staged preparation is incomplete or identifies another source/subject")
        job = Path(prep["run_root"]) / VERSION / case["case_id"]
        subject = cohort.recon_command(prep, case, job)[2]
        report = job / "anatomy_prep_report.json"
        if (Path(checked.get("anatomy_subject_dir", "")).resolve() != subject.resolve()
                or checked.get("preparation_report_binding") != {"path":str(report), "sha256":cohort.sha256(report)}):
            raise ValueError("staged anatomy/report differs from its actual original namespace")
    original_prep_config(config)
    return checked


def extra_arguments(arguments):
    """Allow exact scheduling knobs, never inputs, precision or skipping stages."""
    if not isinstance(arguments, list) or any(not isinstance(v, str) for v in arguments):
        raise ValueError("candidate_cli_arguments must be a string list")
    seen, index = set(), 0
    while index < len(arguments):
        flag = arguments[index]
        if flag in seen:
            raise ValueError("candidate scheduling flag repeated")
        seen.add(flag)
        if flag == "--compile-arc":
            index += 1
        elif flag == "--tracking-batch-size" and index + 1 < len(arguments):
            if not arguments[index+1].isdigit() or int(arguments[index+1]) <= 0:
                raise ValueError("candidate batch size must be a positive integer")
            index += 2
        else:
            raise ValueError(f"unapproved candidate argument: {flag}")
    return arguments


def verify_source(config):
    if config.get("staged_mode") != MODE or config.get("candidate_ready") is not True:
        raise ValueError("binding requires explicit staged mode and a ready frozen candidate")
    if set(config.get("sources", {})) != {VERSION} or set(config.get("frozen_sources", {})) != {VERSION}:
        raise ValueError("GPU source must be honestly labeled candidate only")
    current = cohort.source_manifest(config["sources"][VERSION])
    if current != config["frozen_sources"][VERSION]:
        raise ValueError("actual candidate source differs from its full frozen ledger")
    required = {"src/fnit/flirt/core.py", "src/fnit/connectome/pipeline.py", "src/fnit/weights.py",
                "src/fnit/connectome/atlas_manifest.json", "pyproject.toml", "environment.yml"}
    if not required.issubset(current["source_sha256"]):
        raise ValueError("candidate freeze is missing required source/install files")
    if config.get("cuda_alloc_conf") != ALLOCATOR or config.get("gpu_lock") != "/tmp/fnit-recon-five-20261002-gongwk.gpu.lock":
        raise ValueError("candidate needs the common expandable allocator and shared GPU lock")
    if not config.get("fnit_weights") or not config.get("gpu_uuid") or config.get("cpu_threads") != 8 or config.get("pilot") is not False:
        raise ValueError("formal candidate needs weights, verified GPU UUID, eight CPU threads and non-pilot protocol")
    extra_arguments(config.get("candidate_cli_arguments", []))
    cli_tree = ast.parse((Path(config["sources"][VERSION]) / "src/fnit/cli.py").read_text())
    supported = {arg.value for node in ast.walk(cli_tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
                 for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)}
    if any(flag not in supported for flag in config.get("candidate_cli_arguments", []) if flag.startswith("--")):
        raise ValueError("declared scheduling option is absent from the actual frozen candidate CLI")
    for field in ("gpu_configuration", "input_manifest"):
        binding = config[field]
        path = Path(binding["path"])
        if path.is_symlink() or cohort.sha256(path) != binding["sha256"]:
            raise ValueError(f"original binding byte identity changed: {field}")
    if config.get("origins_declaration"):
        binding = config["origins_declaration"]
        declaration_path = Path(binding["path"])
        if declaration_path.is_symlink() or cohort.sha256(declaration_path) != binding["sha256"]:
            raise ValueError("declared preparation origin mapping bytes changed")
        declarations = json.loads(declaration_path.read_bytes())["bindings"]
        report_dir = Path(config["stop_dispatch_path"]).parent
        origins,mapping,_,_,_ = read_preparation_origins(declarations,report_dir)
        if origins != config["preparation_origins"] or mapping != config["case_origin_index"]:
            raise ValueError("in-memory case-to-origin mapping differs from the actual declared bytes")
    for path_key, sha_key in (("worker_script","worker_script_sha256"), ("staged_worker_script","staged_worker_sha256"),
                              ("resource_helper_script","resource_helper_sha256"), ("wall_script","wall_script_sha256")):
        if cohort.sha256(config[path_key]) != config[sha_key]:
            raise ValueError(f"staged helper byte identity changed: {path_key}")
    if (cohort.sha256(cohort.__file__) != config["worker_script_sha256"]
            or cohort.sha256(__file__) != config["staged_worker_sha256"]
            or cohort.sha256(rerun.__file__) != config["resource_helper_sha256"]):
        raise ValueError("imported benchmark helper differs from declared bytes")
    return current


def validate_gpu_configuration(prep, supplied):
    if supplied.get("candidate_ready") is not True:
        raise ValueError("candidate_ready must be true after actual candidate validation")
    if set(supplied.get("sources", {})) != {VERSION} or set(supplied.get("frozen_sources", {})) != {VERSION}:
        raise ValueError("GPU configuration requires the actual candidate source and freeze")
    forbidden = {"recovery_mode", "rerun_origin", "revalidated_recon_reports", "preprocessed_root", "resume",
                 "skip_topup", "skip_eddy", "run_root", "preparation_config", "staged_mode",
                 "preparation_origins", "case_origin_index", "origins_declaration", "active_preparation_origin_index",
                 "preparation_driver_status_path", "gpu_configuration", "input_manifest", "resources_manifest",
                 "stop_dispatch_path", "worker_script", "worker_script_sha256", "staged_worker_script", "staged_worker_sha256"}
    if forbidden.intersection(supplied):
        raise ValueError("candidate cannot supply recovery, reused output or binding metadata")
    future = {**prep["future_gpu_parameters"], "atlases":prep["atlases"]}
    for key in SHARED_KEYS:
        if key not in future or supplied.get(key) != future[key]:
            raise ValueError(f"candidate changed declared shared scientific/runtime setting: {key}")
    for key in ("gpu_host", "gpu_python", "wall_script", "wall_script_sha256", "fnit_weights"):
        if not supplied.get(key):
            raise ValueError(f"missing actual GPU configuration: {key}")
    if supplied.get("cuda_alloc_conf") != ALLOCATOR:
        raise ValueError("candidate and common baseline require expandable_segments:True")
    extra_arguments(supplied.get("candidate_cli_arguments", []))


def assert_new_job(config, case, version, job):
    if version != VERSION or Path(job) != Path(config["run_root"]) / VERSION / case["case_id"]:
        raise ValueError("GPU job must use the exact new candidate namespace")
    path = Path(job)
    if path.is_symlink() or not path.resolve().is_relative_to(Path(config["run_root"]).resolve()):
        raise ValueError("GPU job escaped its new namespace")
    allowed = {"staged_anatomy_binding.json", "staged_eligibility.json", "gpu_report.json", "raw_bids_wall.json", "raw_bids_wall.log", "connectome"}
    if path.exists() and any(v.name not in allowed or v.is_symlink() for v in path.iterdir()):
        raise ValueError("GPU output contains foreign/preprocessed/reconstruction files")


def preparation_driver_case(config, case):
    config = for_case(config, case)
    path = Path(config["preparation_driver_status_path"])
    if path.is_symlink():
        raise ValueError("preparation driver cannot be a symlink")
    raw = path.read_bytes()
    state = json.loads(raw)
    prep = original_prep_config(config)
    if state.get("config") != prep or state.get("fresh_namespace") != {"status":"claimed_fresh_anatomy_namespace", "path":prep["run_root"]}:
        raise ValueError("preparation driver no longer identifies the original fresh config/namespace")
    record = state.get("cases", {}).get(VERSION + "/" + case["case_id"], {})
    if (record.get("status") != "completed" or record.get("validation_error") or record.get("timing_error")
            or not record.get("start_utc") or not record.get("end_utc") or not record.get("timing")):
        raise ValueError("preparation driver case is incomplete or has validation/timing errors")
    return copy.deepcopy(record), {"path":str(path), "sha256_observed":hashlib.sha256(raw).hexdigest()}


def bind_anatomy(config, case, version, job):
    assert_new_job(config, case, version, job)
    verify_source(config)
    if Path(job).exists():
        raise FileExistsError("anatomy binding requires an absent GPU job")
    record, observation = preparation_driver_case(config, case)
    checked = verify_preparation(config, case)
    if record.get("preparation_report") != checked["preparation_report"]:
        raise ValueError("driver preparation ledger differs from the actual CPU report")
    selected = for_case(config, case)
    result = {"status":"completed", "mode":MODE, "case_id":case["case_id"], "subject":case["subject"], "version":VERSION,
              "prep_config_sha256":selected["preparation_config"]["sha256"], "original_preparation":checked,
              "preparation_driver_record":record, "driver_status_observation":observation, "bound_utc":cohort.utc(),
              "candidate_source_fingerprint":config["frozen_sources"][VERSION]["source_fingerprint"],
              "raw_dwi_preprocessing_resumed":False, "raw_dwi_outputs_copied":False,
              "scope":"fresh official raw-T1 anatomy before candidate freeze; new complete raw-DWI stage; not continuous cold pipeline"}
    cohort.require_fresh(job)
    cohort.atomic_json(Path(job) / "staged_anatomy_binding.json", result)
    return result


def load_anatomy(config, case, version, job):
    assert_new_job(config, case, version, job)
    verify_source(config)
    path = Path(job) / "staged_anatomy_binding.json"
    binding = json.loads(path.read_bytes())
    selected = for_case(config, case)
    expected = {"status":"completed", "mode":MODE, "case_id":case["case_id"], "subject":case["subject"], "version":VERSION,
                "prep_config_sha256":selected["preparation_config"]["sha256"],
                "candidate_source_fingerprint":config["frozen_sources"][VERSION]["source_fingerprint"],
                "raw_dwi_preprocessing_resumed":False, "raw_dwi_outputs_copied":False}
    if any(binding.get(k) != v for k, v in expected.items()):
        raise ValueError("staged binding metadata differs from its actual source/round")
    checked = verify_preparation(config, case)
    for key in ("preparation_report", "preparation_report_binding", "anatomy_subject_dir", "prep_config_sha256"):
        if checked.get(key) != binding["original_preparation"].get(key):
            raise ValueError("original anatomy/raw provenance changed after binding")
    record, _ = preparation_driver_case(config, case)
    if record != binding["preparation_driver_record"]:
        raise ValueError("original preparation driver case changed after binding")
    recon = copy.deepcopy(checked["preparation_report"]["reconstruction_result"])
    recon["staged_anatomy"] = {"mode":MODE, "binding_path":str(path), "binding_sha256":cohort.sha256(path),
                              "prep_config_sha256":selected["preparation_config"]["sha256"],
                              "prepared_anatomy_subject_dir":checked["anatomy_subject_dir"],
                              "candidate_source_fingerprint":config["frozen_sources"][VERSION]["source_fingerprint"],
                              "recon_all_reused_from_other_round":False, "raw_dwi_preprocessing_resumed":False}
    return recon


def anatomy_subject(config, case, job):
    # This callback executes after the shared lock is actually acquired. A
    # stopped lock waiter must not start a new scientific subprocess later.
    if Path(config["stop_dispatch_path"]).exists():
        raise RuntimeError("STOP_DISPATCH prevents starting the queued raw-DWI computation")
    verify_source(config)
    rerun.verify_resources(config)
    prep = original_prep_config(config, case)
    return cohort.recon_command(prep, case, Path(prep["run_root"]) / VERSION / case["case_id"])[2]


def memory_eligible(budget):
    measurements = budget.get("measurements", {})
    groups = (("process_tree",), ("allocated_bytes", "peak_allocated_bytes", "max_memory_allocated_bytes"),
              ("reserved_bytes", "peak_reserved_bytes", "max_memory_reserved_bytes"))
    def valid(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value < 20_000_000_000
    return (budget.get("status") == "observed_below_budget"
            and not budget.get("monitor_issues")
            and all(valid(value) for value in measurements.values() if value is not None)
            and all(any(valid(measurements.get(key)) for key in names) for names in groups))


def worker(payload):
    config, action = payload["config"], payload["action"]
    verify_source(config)
    if action == "preflight":
        return {"resources":rerun.build_resources(config, source_version=VERSION),
                "original_preparation":[verify_preparation(for_case(config, {"case_id":origin["case_ids"][0]}))
                    for origin in config["preparation_origins"]] if config.get("preparation_origins") else verify_preparation(config),
                "identity":cohort.host_identity()}
    rerun.verify_resources(config)
    if action == "bind":
        case = payload["case"]
        return bind_anatomy(config, case, VERSION, Path(config["run_root"]) / VERSION / case["case_id"])
    if action != "gpu" or payload.get("version") != VERSION:
        raise ValueError("staged worker only preflights, binds or executes candidate GPU")
    if Path(config["stop_dispatch_path"]).exists():
        return {"status":"dispatch_paused", "gpu_started":False}
    job = Path(config["run_root"]) / VERSION / payload["case"]["case_id"]
    assert_new_job(config, payload["case"], VERSION, job)
    for name in ("gpu_report.json", "staged_eligibility.json", "raw_bids_wall.json", "raw_bids_wall.log", "connectome"):
        if (job / name).exists() or (job / name).is_symlink():
            raise FileExistsError("staged GPU worker never overwrites a previous/partial GPU attempt")
    result = cohort.worker(payload, anatomy_loader=load_anatomy, anatomy_subject=anatomy_subject,
                           extra_cli_arguments=extra_arguments(config.get("candidate_cli_arguments", [])))
    eligibility = {"status":"not_eligible", "execution_status":result.get("status"), "memory_budget":result.get("memory_budget"), "staged_mode":MODE}
    if result.get("status") == "completed" and memory_eligible(result.get("memory_budget", {})):
        try:
            verify_source(config)
            rerun.verify_resources(config)
            load_anatomy(config, payload["case"], VERSION, Path(config["run_root"]) / VERSION / payload["case"]["case_id"])
            eligibility["status"] = "execution_complete_memory_observed_below_budget"
        except Exception as error:
            eligibility["validation_error"] = {"type":type(error).__name__, "message":str(error)}
    cohort.atomic_json(Path(config["run_root"]) / VERSION / payload["case"]["case_id"] / "staged_eligibility.json", eligibility)
    return {"gpu_result":result, "eligibility":eligibility}


def remote(config, action, case, log_path):
    transport = {**config, "worker_script":config["staged_worker_script"]}
    return cohort.remote(transport, "gpu", {"action":action, "config":config, "case":case, "version":VERSION}, log_path)


def parse_utc(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("UTC times require timezone information")
    return parsed


def staged_timing(prepared, gpu, gpu_head_start, gpu_head_end, gpu_head_wall):
    positive = rerun.nonnegative
    recon = positive(prepared["preparation_report"]["recon_command_seconds"], "official reconstruction duration")
    worker_wall = positive(prepared["preparation_worker_wall_seconds"], "original preparation worker wall")
    prep_wall = positive(prepared["timing"]["head_case_wall_seconds"], "original head preparation wall")
    if recon > worker_wall + .001 or worker_wall > prep_wall + .001:
        raise ValueError("preparation timers exceed their same-process intervals")
    wall, queue = positive(gpu_head_wall, "head raw-DWI monotonic wall"), positive(gpu.get("gpu_lock_queue_seconds", 0.), "GPU lock queue")
    if queue > wall + .001 or parse_utc(gpu_head_end) < parse_utc(gpu_head_start):
        raise ValueError("invalid head GPU interval or lock queue")
    elapsed = positive((parse_utc(gpu_head_end)-parse_utc(prepared["start_utc"])).total_seconds(), "observed staged UTC elapsed")
    gap = positive((parse_utc(gpu_head_start)-parse_utc(prepared["end_utc"])).total_seconds(), "observed preparation/GPU gap")
    return {"official_recon_command_seconds":recon, "preparation_worker_wall_seconds":worker_wall, "preparation_head_wall_seconds":prep_wall,
            "cpu_driver_queue_seconds":positive(prepared["timing"]["cpu_driver_queue_seconds"], "CPU scheduler queue"),
            "raw_dwi_head_wall_seconds":wall, "raw_dwi_head_wall_excluding_gpu_queue_seconds":max(0.,wall-queue),
            "raw_dwi_cli_seconds":gpu.get("raw_dwi_cli_total_runtime_seconds"), "gpu_lock_queue_seconds":queue,
            "prep_to_gpu_observed_utc_gap_seconds":gap, "staged_observed_utc_elapsed_seconds":elapsed,
            "staged_observed_utc_elapsed_excluding_gpu_queue_seconds":max(0.,elapsed-queue), "continuous_cold_pipeline":False,
            "scope":"original head preparation UTC start -> head GPU/report UTC end, including freeze/wait gap; stage walls are separate same-process monotonic timers; no cross-host monotonic reconstruction"}


def collect_gpu_result(record, response, head_start, head_end, head_wall):
    gpu, eligibility = response["gpu_result"], response["eligibility"]
    record.update(gpu_result=gpu, eligibility=eligibility, gpu_head_start_utc=head_start, gpu_head_end_utc=head_end, status=gpu.get("status","failed"))
    if gpu.get("status") != "completed":
        record["status"] = "failed_gpu_execution"
        if gpu.get("error"):
            record["error"] = gpu["error"]
    elif eligibility.get("status") != "execution_complete_memory_observed_below_budget":
        record["status"] = "failed_gpu_eligibility"
        record["error"] = eligibility.get("validation_error") or {
            "type":"GPUEligibilityError", "message":"actual execution completed, but memory measurements or immutable binding validation did not meet the formal gate"}
    try:
        record["timing"] = staged_timing(record["anatomy_binding"]["preparation_driver_record"], gpu, head_start, head_end, head_wall)
    except Exception as error:
        record["timing_error"] = {"type":type(error).__name__, "message":str(error)}


def atomic_cases_csv(path, records):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{cohort.uuid.uuid4().hex}.tmp")
    columns = ("case_id","subject","status","official_recon_command_seconds","preparation_head_wall_seconds","raw_dwi_cli_seconds",
               "raw_dwi_head_wall_seconds","gpu_lock_queue_seconds","prep_to_gpu_observed_utc_gap_seconds","staged_observed_utc_elapsed_seconds","timing_error")
    try:
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for record in records.values():
                writer.writerow({k:record.get(k,record.get("timing",{}).get(k)) for k in columns})
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def read_preparation_origins(declarations, report_dir):
    """One full raw manifest; every formal case assigned exactly one real origin."""
    if not isinstance(declarations, list) or not declarations:
        raise ValueError("preparation bindings must be a nonempty list")
    origins, mapping, raw_records = [], {}, []
    manifest_hash = None
    reference = None
    seen_paths = set()
    for index, declared in enumerate(declarations):
        if not isinstance(declared, dict) or set(declared) != {"prep_config","prep_driver_report_dir","case_ids"}:
            raise ValueError("preparation origin requires exactly config, driver and case_ids")
        path, driver = Path(declared["prep_config"]), Path(declared["prep_driver_report_dir"])
        cohort.absolute_path(str(path), "preparation config")
        cohort.absolute_path(str(driver), "preparation driver")
        if path.is_symlink() or str(path.resolve()) in seen_paths:
            raise ValueError("duplicate/symlink original preparation config")
        seen_paths.add(str(path.resolve()))
        raw = path.read_bytes()
        prep = json.loads(raw)
        if path != Path(prep["run_root"]) / "anatomy_prep_config.json":
            raise ValueError("origin config escaped its actual original namespace")
        if (prep.get("scope") != "staged_anatomy_preparation_only" or prep.get("candidate_source") != "unknown"
                or prep.get("gpu_started") is not False or prep.get("cpu_threads") != 8
                or "sources" in prep or "frozen_sources" in prep):
            raise ValueError("origin was not fresh source-independent eight-thread official preparation")
        manifest_path = Path(prep["run_root"]) / "input_manifest.json"
        actual_hash = cohort.sha256(manifest_path)
        manifest = json.loads(manifest_path.read_bytes())
        cases = cohort.validate_manifest(manifest)
        if len(cases) != 10 or (manifest_hash is not None and manifest_hash != actual_hash):
            raise ValueError("preparation origins do not share the exact full ten-case canonical raw manifest")
        identity = {"official":prep["official_origin"]["identity"], "atlases":prep["atlases"], "future":prep["future_gpu_parameters"]}
        if reference is not None and identity != reference:
            raise ValueError("preparation origins differ in official identity or shared parameters")
        reference, manifest_hash = identity, actual_hash
        selected = declared["case_ids"]
        available = {case["case_id"] for case in cases}
        if (not isinstance(selected,list) or not selected or any(not isinstance(value,str) for value in selected)
                or len(set(selected)) != len(selected) or set(selected)-available):
            raise ValueError("origin case selection is empty, duplicated or outside the manifest")
        if prep.get("selected_cases") and not set(selected).issubset(prep["selected_cases"]):
            raise ValueError("origin case selection includes a subject that its official prep never planned")
        for case_id in selected:
            if case_id in mapping:
                raise ValueError("a raw case was assigned multiple preparation origins")
            mapping[case_id] = index
        snapshot = Path(report_dir) / f"original_prep_config.{index}.bytes.json"
        origin = {"preparation_config":{"original_path":str(path),"snapshot_path":str(snapshot),"sha256":hashlib.sha256(raw).hexdigest()},
                  "preparation_driver_status_path":str(driver/"status.json"), "case_ids":list(selected)}
        origins.append(origin)
        raw_records.append({"path":str(snapshot), "bytes":raw})
    if set(mapping) != {case["case_id"] for case in cases}:
        raise ValueError("formal origin mapping must cover all ten raw subjects exactly once")
    return origins,mapping,manifest,cases,raw_records


def derive_config(options):
    if getattr(options,"prep_bindings",None):
        declarations = json.loads(options.prep_bindings.read_bytes())["bindings"]
        origins,mapping,manifest,cases,raw_records = read_preparation_origins(declarations,options.report_dir)
        single = copy.copy(options)
        single.prep_bindings = None
        single.prep_config = Path(origins[0]["preparation_config"]["original_path"])
        single.prep_driver_report_dir = Path(origins[0]["preparation_driver_status_path"]).parent
        config,_,_,_ = derive_config(single)
        for origin in origins:
            prior = json.loads(Path(origin["preparation_config"]["original_path"]).read_bytes())
            for path in (options.run_root,options.report_dir):
                if overlaps(path,prior["run_root"]) or overlaps(path,Path(origin["preparation_driver_status_path"]).parent):
                    raise ValueError("new GPU/report namespace overlaps an original preparation origin")
        config.update(preparation_origins=origins,case_origin_index=mapping,
                      preparation_config=origins[0]["preparation_config"],
                      origins_declaration={"path":str(options.prep_bindings),"sha256":cohort.sha256(options.prep_bindings)})
        return config,manifest,cases,raw_records
    raw = options.prep_config.read_bytes()
    if options.prep_config.is_symlink():
        raise ValueError("original preparation config cannot be a symlink")
    prep, supplied = json.loads(raw), json.loads(options.gpu_config.read_bytes())
    if options.prep_config != Path(prep["run_root"]) / "anatomy_prep_config.json":
        raise ValueError("preparation config must be its actual original file")
    validate_gpu_configuration(prep,supplied)
    manifest_path = Path(prep["run_root"]) / "input_manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    cases = cohort.validate_manifest(manifest)
    if len(cases) != 10:
        raise ValueError("formal staged benchmark requires ten distinct raw subjects")
    for path in (options.run_root,options.report_dir):
        for existing in (prep["run_root"],options.prep_driver_report_dir,supplied["sources"][VERSION]):
            if overlaps(path,existing):
                raise ValueError("new output/report overlaps original preparation/source")
        if Path(path).exists() or Path(path).is_symlink():
            raise FileExistsError("binding requires fresh run and report namespaces")
    if overlaps(options.run_root,options.report_dir):
        raise ValueError("GPU outputs and driver reports need separate namespaces")
    config = copy.deepcopy(supplied)
    config.update(staged_mode=MODE,run_root=str(options.run_root),pilot=False,cpu_threads=8,
        cpu_python=prep["cpu_python"],anatomy_validation_python=prep["anatomy_validation_python"],
        worker_script=str(Path(cohort.__file__).resolve()),worker_script_sha256=cohort.sha256(cohort.__file__),
        staged_worker_script=str(Path(__file__).resolve()),staged_worker_sha256=cohort.sha256(__file__),
        resource_helper_script=str(Path(rerun.__file__).resolve()),resource_helper_sha256=cohort.sha256(rerun.__file__),
        preparation_config={"original_path":str(options.prep_config),"snapshot_path":str(options.report_dir/"original_prep_config.bytes.json"),"sha256":hashlib.sha256(raw).hexdigest()},
        preparation_driver_status_path=str(options.prep_driver_report_dir/"status.json"),
        gpu_configuration={"path":str(options.gpu_config),"sha256":cohort.sha256(options.gpu_config)},
        input_manifest={"path":str(manifest_path),"sha256":cohort.sha256(manifest_path)},stop_dispatch_path=str(options.report_dir/"STOP_DISPATCH"))
    verify_source(config)
    return config,manifest,cases,raw


def run(options):
    start,start_utc = time.perf_counter(),cohort.utc()
    config,manifest,cases,raw = derive_config(options)
    report_dir = cohort.require_fresh(options.report_dir)
    if isinstance(raw,bytes):
        Path(config["preparation_config"]["snapshot_path"]).write_bytes(raw)
    else:
        for record in raw:
            Path(record["path"]).write_bytes(record["bytes"])
    (report_dir/"original_gpu_config.bytes.json").write_bytes(options.gpu_config.read_bytes())
    original_driver = Path(config["preparation_driver_status_path"])
    (report_dir/"prep_driver_observation_at_bind.bytes.json").write_bytes(original_driver.read_bytes())
    if config.get("preparation_origins"):
        (report_dir/"original_prep_bindings.bytes.json").write_bytes(Path(config["origins_declaration"]["path"]).read_bytes())
        for index,origin in enumerate(config["preparation_origins"]):
            (report_dir/f"prep_driver_observation_at_bind.{index}.bytes.json").write_bytes(Path(origin["preparation_driver_status_path"]).read_bytes())
    state = {"schema_version":1,"status":"preflighting","staged_mode":MODE,"start_utc":start_utc,"config":config,"requested_cases":len(cases),
             "cases":{},"comparison_ready":False,"scientific_parity":"not_assessed","speedup":"not_assessed","continuous_cold_pipeline":False}
    for case in cases:
        state["cases"][VERSION+"/"+case["case_id"]] = {"case_id":case["case_id"],"subject":case["subject"],"status":"waiting_original_preparation_report"}
    def save():
        cohort.atomic_json(report_dir/"status.json",state)
        atomic_cases_csv(report_dir/"cases.csv",state["cases"])
    save()
    try:
        checked = remote(config,"preflight",None,report_dir/"preflight.stderr.log")
        resource_path = report_dir/"resources.json"
        cohort.atomic_json(resource_path,checked["resources"])
        config["resources_manifest"] = {"path":str(resource_path),"sha256":cohort.sha256(resource_path)}
        state["preflight"] = checked
        save()
        if options.preflight_only:
            state["status"] = "preflight_only_GPU_not_started"
        else:
            root = cohort.require_fresh(config["run_root"])
            cohort.atomic_json(root/"staged_gpu_config.json",config)
            cohort.atomic_json(root/"input_manifest.json",manifest)
            state.update(status="running",fresh_namespace={"status":"claimed_fresh_raw_DWI_namespace","path":str(root)})
            pending = list(cases)
            while pending:
                if Path(config["stop_dispatch_path"]).exists() or time.perf_counter()-start > options.timeout_hours*3600:
                    state["dispatch_paused"] = True
                    break
                progressed = False
                prep_states = [json.loads(Path(origin["preparation_driver_status_path"]).read_bytes())
                    for origin in config["preparation_origins"]] if config.get("preparation_origins") else [json.loads(original_driver.read_bytes())]
                for case in list(pending):
                    key = VERSION+"/"+case["case_id"]
                    prep_state = prep_states[config["case_origin_index"][case["case_id"]]] if config.get("preparation_origins") else prep_states[0]
                    if prep_state.get("cases",{}).get(key,{}).get("status") not in ("completed","failed","failed_preparation_validation"):
                        continue
                    pending.remove(case)
                    progressed = True
                    record = state["cases"][key]
                    try:
                        record["status"] = "binding_actual_fresh_anatomy"
                        save()
                        binding = remote(config,"bind",case,report_dir/f"{case['case_id']}-bind.stderr.log")
                        record.update(anatomy_binding=binding,status="gpu_queued")
                        save()
                        if Path(config["stop_dispatch_path"]).exists():
                            record["status"] = "bound_GPU_not_dispatched"
                        else:
                            begun,begun_utc = time.perf_counter(),cohort.utc()
                            response = remote(config,"gpu",case,report_dir/f"{case['case_id']}-gpu.stderr.log")
                            ended,ended_utc = time.perf_counter(),cohort.utc()
                            if response.get("status") == "dispatch_paused":
                                record["status"] = "bound_GPU_not_dispatched"
                            else:
                                collect_gpu_result(record,response,begun_utc,ended_utc,ended-begun)
                    except Exception as error:
                        record.update(status="failed",error={"type":type(error).__name__,"message":str(error)})
                    print(json.dumps({"case":key,"status":record["status"],"error":record.get("error"),"timing_error":record.get("timing_error")}),flush=True)
                    save()
                    break  # One GPU stage; observe the current preparation state next.
                if not progressed:
                    if all(prep_state.get("status","").startswith(("failed","completed_anatomy")) for prep_state in prep_states):
                        for case in pending:
                            state["cases"][VERSION+"/"+case["case_id"]]["status"] = "incomplete_original_preparation"
                        pending.clear()
                    else:
                        time.sleep(options.poll_seconds)
            completed = sum(v["status"] == "completed" for v in state["cases"].values())
            timed = all(v.get("timing") and not v.get("timing_error") for v in state["cases"].values())
            state.update(completed_cases=completed,timing_complete=timed,status="completed_staged_raw_cohort" if completed==10 and timed else "failed_or_incomplete_staged_raw_cohort")
    except Exception as error:
        state.update(status="failed_staged_binding",error={"type":type(error).__name__,"message":str(error)})
    state.update(end_utc=cohort.utc(),gpu_controller_monotonic_seconds=time.perf_counter()-start)
    save()
    return 0 if state["status"] in ("preflight_only_GPU_not_started","completed_staged_raw_cohort") else 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_worker":
        print(json.dumps(worker(json.load(sys.stdin)),allow_nan=False),flush=True)
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("gpu-config","run-root","report-dir"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--prep-config",type=Path)
    parser.add_argument("--prep-driver-report-dir",type=Path)
    parser.add_argument("--prep-bindings",type=Path,help="explicit full-ten-case mapping to original prep configurations and driver reports")
    parser.add_argument("--preflight-only",action="store_true",help="actual source/resource/original-prep checks in a fresh report namespace; no GPU compute")
    parser.add_argument("--poll-seconds",type=float,default=30.)
    parser.add_argument("--timeout-hours",type=float,default=72.)
    options = parser.parse_args(argv)
    if bool(options.prep_bindings) == bool(options.prep_config or options.prep_driver_report_dir):
        parser.error("use either --prep-bindings or both single-origin preparation parameters")
    if not options.prep_bindings and not (options.prep_config and options.prep_driver_report_dir):
        parser.error("single origin requires both --prep-config and --prep-driver-report-dir")
    for value in vars(options).values():
        if isinstance(value,Path):
            cohort.absolute_path(str(value),"staged benchmark path")
    if not 1 <= options.poll_seconds <= 60 or not 0 < options.timeout_hours <= 168:
        parser.error("poll must be 1..60 seconds, timeout positive up to 168 hours")
    return run(options)


if __name__ == "__main__":
    raise SystemExit(main())
