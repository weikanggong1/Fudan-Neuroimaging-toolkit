#!/usr/bin/env python3
"""Strict CPU metadata scheduler for actual CON11 official modeling/anatomy/repeats.

Numerical workers are the existing immutable cb06/0c6191 files. This scheduler
never creates upstream contracts, changes old queues, or reruns preparation.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

CASE = "sub-CON11"
OTHER = ["sub-CON01", "sub-CON03", *[f"sub-CON{i:02d}" for i in range(4, 11)]]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def record(path):
    p = Path(path)
    return {"path": str(p), "size_bytes": p.stat().st_size, "sha256": sha(p)}


def check(row):
    require(isinstance(row, dict) and "path" in row and "sha256" in row, "bound file required")
    p = Path(row["path"])
    require(p.is_file() and sha(p) == row["sha256"], "bound source/input changed: " + str(p))
    require("size_bytes" not in row or p.stat().st_size == row["size_bytes"], "bound size changed")
    return p


def load(path):
    return json.loads(Path(path).read_text())


def save(path, data, exclusive=False):
    p = Path(path)
    if exclusive:
        with p.open("x") as f:
            f.write(json.dumps(data, indent=2, allow_nan=False) + "\n")
    else:
        tmp = p.with_name(p.name + ".partial")
        tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        tmp.replace(p)


def utc():
    return datetime.now(timezone.utc).isoformat()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def process_identity(pid):
    p = Path("/proc") / str(pid)
    require(p.exists(), "expected process absent: " + str(pid))
    fields = (p / "stat").read_text().rsplit(")", 1)[1].split()
    return {"PID": pid, "UID": p.stat().st_uid, "start_ticks": int(fields[19]),
            "argv": [x.decode() for x in (p / "cmdline").read_bytes().split(b"\0") if x],
            "children": (p / "task" / str(pid) / "children").read_text().split()}


def check_old_queue(config, anatomy, full=False):
    old = config["old_anatomy"]
    check(old["launch"])
    launch = load(old["launch"]["path"])
    identity = process_identity(old["PID"])
    require(identity["UID"] == os.getuid() and identity["start_ticks"] == old["start_ticks"]
            and identity["argv"] == launch["argv"] and not identity["children"],
            "old anatomy identity or CPU child guard differs")
    body = load(old["state_path"])
    require(set(body["cases"]) == set(OTHER + [CASE]), "old anatomy case list differs")
    require(all(body["cases"][case]["state"] == "completed" for case in OTHER), "first nine anatomy pending")
    require(body["cases"][CASE]["state"] == "waiting_official_dwi", "old CON11 must remain waiting")
    require(not Path(old["old_CON11_dwi_contract"]).exists()
            and not Path(old["old_CON11_dwi_contract"]).is_symlink(), "old CON11 alias forbidden")
    consumers = {}
    if full:
        for case in OTHER:
            bound = body["cases"][case]["consumer_contract"]
            consumer = anatomy.read_bound_json(bound)
            require(consumer["case_id"] == case and consumer["state"] == "completed", "old completed case differs")
            report = anatomy.read_bound_json(consumer["official_anatomy_report"])
            require(report["execution_completed"] and report["case_id"] == case, "old anatomy report incomplete")
            require(all(x["returncode"] == 0 for x in report["commands"]), "old official command failed")
            for value in report["outputs"].values():
                anatomy.verify_file(value)
            consumers[case] = bound
    return {"actual_state": record(old["state_path"]), "process": identity, "nine_completed_consumers": consumers,
            "old_CON11_waiting": True, "old_namespace_unmodified": True}


def verify_baseline_sources(config):
    origin = load(check(config["known_inputs"]["root_origin"]))
    receipt = load(check(config["known_inputs"]["baseline_source_receipt"]))
    require(origin["case_id"] == CASE and origin["source_files"] == receipt["source_files"], "actual baseline origin differs")
    require(len(receipt["source_files"]) == 39 and len({x["path"] for x in receipt["source_files"]}) == 39,
            "actual frozen 39-source fingerprint required")
    for entry in receipt["source_files"]:
        check(entry)
    lineage = load(check(config["known_inputs"]["expected_baseline_lineage"]))
    for entry in lineage.values():
        check(entry)
    check(origin["packing"])
    require(not Path(origin["packing"]["path"]).is_symlink(), "actual packing cannot be an alias")
    return {"root_origin": config["known_inputs"]["root_origin"], "baseline_source_receipt": config["known_inputs"]["baseline_source_receipt"],
            "source_count": 39, "source_fingerprint": receipt["actual_source_fingerprint"],
            "actual_packing": origin["packing"], "actual_raw_frame_indices": origin["actual_raw_frame_indices"],
            "source_bytes_verified": True}


def model_ready(config, anatomy):
    root = Path(config["shared_root"])
    model = Path(config["modeling_root"])
    cp, rp = model / CASE / "consumer_contract.json", model / CASE / "report.json"
    status = load(root / "task_02/CON11_subset_dispatcher_v1/status.json")
    require(status.get("state") != "CON11_subset_modeling_failed_preserved", "actual model failed; preserve source")
    if not cp.is_file() or not rp.is_file() or not load(cp).get("execution_completed"):
        return None
    require(not model.is_symlink() and not cp.is_symlink(), "fresh model namespace/contract alias forbidden")
    if status.get("state") != "CON11_subset_modeling_completed":
        return None  # Consumer publication precedes the dispatcher's final write.
    check(status["consumer_contract"]); check(status["report"])
    require(status["consumer_contract"]["path"] == str(cp) and status["report"]["path"] == str(rp), "actual model completed path differs")
    consumer, report = load(cp), load(rp)
    require(consumer["case_id"] == CASE and consumer["scope"] == "official_self_produced_raw_dwi_chain"
            and consumer["state"] == "completed", "independent same-case model contract required")
    require(report["subject"] == "CON11" and report["state"] == "completed" and report["execution_completed"], "actual model report incomplete")
    require(report["harness_sha256"] == config["frozen_sources"]["model_worker"]["sha256"]
            and consumer["harness_sha256"] == report["harness_sha256"], "model worker changed")
    require(consumer["modeling_report"] == str(rp) and consumer["modeling_report_sha256"] == sha(rp), "modeling report hash differs")
    freeze_path = model / "freeze.json"
    freeze = load(freeze_path)
    check(freeze["worker"]); check(freeze["wrapper"]); check(freeze["base_config"]); check(freeze["configuration"])
    require(freeze["worker"]["sha256"] == config["frozen_sources"]["model_worker"]["sha256"]
            and freeze["wrapper"]["sha256"] == config["frozen_sources"]["model_wrapper"]["sha256"]
            and freeze["CPU_threads"] == 8 and freeze["CPU_concurrency"] == 1 and freeze["GPU"] is False, "model actual freeze differs")
    require(report["configuration"] == load(freeze["configuration"]["path"]), "model report/config differs")
    actual_config = load(freeze["configuration"]["path"])
    base_config = load(freeze["base_config"]["path"])
    require(actual_config["output_root"] == str(model)
            and actual_config["subjects"] == [{"subject": "CON11", "ready_contract": str(root / "task_01/official_CON11_CPU_fresh_origin_v1/sub-CON11/report.json"),
                "binding_ready": True, "expected_upstream_solver": "cpu", "expected_selection": {"ap_index": 0, "pa_index": 0}}],
            "actual single-case model route or scientific selection differs")
    require(actual_config["explicit_fresh_CON11_origin_bindings"] == freeze["actual_input_bindings"], "model configuration input bindings differ")
    for name in ("output_root", "subjects", "explicit_fresh_CON11_origin_bindings"):
        actual_config.pop(name, None); base_config.pop(name, None)
    actual_config["profile"].pop("upstream_EDDY_solver", None)
    base_config["profile"].pop("upstream_EDDY_solver", None)
    require(actual_config == base_config, "actual model scientific configuration differs from frozen original")
    require(report["threads"] == 8 and report["environment"]["CUDA_VISIBLE_DEVICES"] == "", "model CPU budget differs")
    require(report["commands"] and all(x["returncode"] == 0 for x in report["commands"]), "actual model command incomplete")
    for command in report["commands"]:
        require(sha(command["command"][0]) == command["binary_sha256"], "official model program changed")
        for path, digest in command["input_sha256"].items():
            check({"path": path, "sha256": digest})
        for path, digest in command["output_sha256"].items():
            check({"path": path, "sha256": digest})
    for entry in consumer["files"].values():
        check(entry)
    for entry in freeze["actual_input_bindings"].values():
        if isinstance(entry, dict) and "path" in entry:
            check(entry)
    # Reuse the actual immutable Task02 origin guard only as a read-only function.
    sys.path.insert(0, str(root / "task_02"))
    wrapper = module(check(config["frozen_sources"]["model_wrapper"]), "CON11_actual_origin_verifier")
    worker, original_config, _, _ = wrapper.load_worker(root)
    upstream, upstream_state = wrapper.ready(root, worker, original_config)
    require(upstream is not None and upstream_state == "ready_actual_verified_CPU", "actual fresh CPU origin guard not ready")
    require(consumer["upstream_official_rawprep_report"] == upstream["upstream_report"]
            and consumer["upstream_official_rawprep_report_sha256"] == sha(upstream["upstream_report"]), "new model rawprep origin differs")
    for name, bound in freeze["actual_input_bindings"].items():
        if isinstance(bound, dict):
            require(bound == upstream["bindings"][name], "actual frozen producer binding differs: " + name)
    anatomy.verify_dwi_contract(record(cp), CASE)
    return {"consumer_contract": record(cp), "modeling_report": record(rp), "model_freeze": record(freeze_path),
            "model_configuration": freeze["configuration"], "actual_upstream_bindings": freeze["actual_input_bindings"],
            "actual_model_commands_exit_zero": len(report["commands"]), "files_verified": len(consumer["files"]),
            "timing_scope": "fresh official CPU rawprep and own official model; anatomy prepare separately reused"}


def run_child(state_root, name, argv, config, state, seal):
    check(state["configuration"])
    require(sha(__file__) == config["orchestrator_sha256"], "orchestration source changed before dispatch")
    for value in list(config["known_inputs"].values()) + list(config["frozen_sources"].values()):
        check(value)
    for key in ("actual_anatomy_contract", "actual_DWI_contract", "actual_preflight"):
        if key in seal:
            check(seal[key])
    if "model_inputs" in seal:
        for key in ("consumer_contract", "modeling_report", "model_freeze", "model_configuration"):
            check(seal["model_inputs"][key])
    seal_path = state_root / (name + "_execution_config.json")
    save(seal_path, {"argv": argv, "CPU_threads": 8, "GPU": False, **seal}, exclusive=True)
    env = {**os.environ, **config["environment"]}
    started = time.perf_counter()
    with (state_root / (name + ".log")).open("x") as log:
        process = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        identity = process_identity(process.pid)
        launched = {"state": "actual_CPU_dispatched", "UTC": utc(), "host": socket.gethostname(),
                    "process": identity, "argv": argv, "configuration": record(seal_path), "GPU": False,
                    "orchestrator": record(__file__), "environment": config["environment"]}
        save(state_root / (name + "_launch.json"), launched, exclusive=True)
        state.update(state="running_" + name, active_launch=record(state_root / (name + "_launch.json")))
        save(state_root / "status.json", state)
        code = process.wait()
    complete = {**launched, "returncode": code, "completed_UTC": utc(), "child_wall_seconds": time.perf_counter() - started}
    save(state_root / (name + "_exit.json"), complete, exclusive=True)
    require(code == 0, "actual " + name + " failed; logs and source namespace preserved")
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--inspect-only", action="store_true", help="actual frozen input/state checks; never dispatch")
    args = parser.parse_args()
    require(sha(args.config) == args.config_sha256, "orchestrator config bytes changed")
    config = load(args.config)
    require(config["case_id"] == CASE and config["GPU"] is False and config["CPU_threads"] == 8, "CON11 CPU-only config required")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "GPU must be hidden")
    require(sha(__file__) == config["orchestrator_sha256"], "orchestration source changed")
    for value in list(config["known_inputs"].values()) + list(config["frozen_sources"].values()):
        check(value)
    anatomy = module(check(config["frozen_sources"]["anatomy_worker"]), "CON11_frozen_anatomy")
    prepared_record = config["known_inputs"]["prepared_report"]
    prepared = anatomy.read_bound_json(prepared_record)
    require(prepared["case_id"] == CASE and prepared["execution_completed"] and prepared["state"] == "official_structural_reference_completed", "actual completed CON11 prepare required")
    require(len(prepared["outputs"]) == 27, "actual original 27-output prepare required")
    for value in prepared["outputs"].values():
        anatomy.verify_file(value)
    anatomy.verify_anatomy(prepared["config"])
    old = check_old_queue(config, anatomy, full=not args.inspect_only)
    baseline = verify_baseline_sources(config)
    if args.inspect_only:
        print(json.dumps({"state": "actual_readonly_preflight_completed", "old_queue": old, "baseline": baseline,
                          "prepared_outputs_verified": 27, "model_completion_not_checked": True, "new_MRI_launched": False}, indent=2))
        return
    directory = Path(config["state_root"])
    for path in [directory, Path(config["anatomy_output"]), Path(config["reference_output_root"]), Path(config["reference_dry_run"])] :
        require(not path.exists() and not path.is_symlink(), "fresh namespace required: " + str(path))
    directory.mkdir(parents=True, exist_ok=False)
    lock = (directory / "owner.lock").open("x")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {"schema_version": 1, "case_id": CASE, "state": "waiting_actual_completed_model_contract",
             "started_UTC": utc(), "host": socket.gethostname(), "process": process_identity(os.getpid()),
             "configuration": record(args.config), "script": record(__file__), "GPU": False,
             "prepare_rerun": False, "old_queue_modified": False, "anatomy_completed": False,
             "reference_completed": False, "scientific_parity": "not_assessed"}
    save(directory / "freeze.json", {"configuration": record(args.config), "script": record(__file__),
                                     "source_files": config["frozen_sources"], "known_inputs": config["known_inputs"],
                                     "initial_old_queue": old, "baseline_lineage": baseline, "GPU": False}, exclusive=True)
    started = time.monotonic()
    try:
        while True:
            ready = model_ready(config, anatomy)
            state["last_update_UTC"] = utc()
            state["wait_seconds"] = time.monotonic() - started
            save(directory / "status.json", state)
            if ready is not None:
                break
            require(state["wait_seconds"] < config["wait_timeout_seconds"], "actual model wait deadline reached")
            time.sleep(config["poll_seconds"])
        save(directory / "completed_model_inputs.json", ready, exclusive=True)
        old = check_old_queue(config, anatomy, full=True)
        baseline = verify_baseline_sources(config)
        argv = [config["anatomy_python"], "-u", config["frozen_sources"]["anatomy_cohort"]["path"],
                "--config-template", config["known_inputs"]["config_template"]["path"],
                "--baseline-root", config["baseline_root"], "--raw-root", config["raw_root"],
                "--official-dwi-root", config["modeling_root"], "--validation-script", config["known_inputs"]["validation_script"]["path"],
                "--tool-commit", config["anatomy_source_commit"], "--case-ids", CASE,
                "--prepared-case-report", CASE + "=" + prepared_record["path"], "--workers", "1",
                "--poll-seconds", "30", "--wait-timeout-seconds", "21600", "--output", config["anatomy_output"]]
        run_child(directory, "anatomy", argv, config, state,
                  {"model_inputs": ready, "prepared_report": prepared_record, "old_queue_guard": old,
                   "baseline_lineage": baseline, "frozen_sources": config["frozen_sources"], "prepare_rerun": False})
        anatomy_consumer = Path(config["anatomy_output"]) / CASE / "consumer_contract.json"
        body = anatomy.read_bound_json(anatomy.file_record(anatomy_consumer))
        complete = anatomy.read_bound_json(body["official_anatomy_report"])
        require(body["state"] == "completed" and body["case_id"] == CASE and body["prepared_report"] == prepared_record,
                "actual anatomy completed source binding differs")
        require(body["official_dwi_contract"] == ready["consumer_contract"] and complete["execution_completed"]
                and len(complete["outputs"]) == 20 and all(x["returncode"] == 0 for x in complete["commands"]), "actual anatomy output incomplete")
        for entry in complete["outputs"].values():
            anatomy.verify_file(entry)
        state.update(anatomy_completed=True, anatomy_contract=record(anatomy_consumer))
        reference = config["reference"]
        argv = [config["reference_python"], "-u", config["frozen_sources"]["reference_worker"]["path"],
                "--anatomy-contract", str(anatomy_consumer), "--dwi-contract", ready["consumer_contract"]["path"],
                "--raw-manifest", config["known_inputs"]["raw_reference_manifest"]["path"],
                "--raw-manifest-sha256", config["known_inputs"]["raw_reference_manifest"]["sha256"],
                "--case-id", CASE, "--verified-reference-manifest", config["known_inputs"]["verified_reference_manifest"]["path"],
                "--verified-reference-manifest-sha256", config["known_inputs"]["verified_reference_manifest"]["sha256"],
                "--mrtrix-bin", reference["mrtrix_bin"], "--output-dir", config["reference_output"],
                "--seeds", *map(str, reference["seeds"]), "--n-seeds", str(reference["n_seeds"]),
                "--downstream-threads", "8"]
        require(reference["seeds"] == [0, 1, 2, 3, 4] and reference["n_seeds"] == 100000, "same official repeat flags required")
        env = {**os.environ, **config["environment"]}
        with Path(config["reference_dry_run"]).open("x") as output:
            dry = subprocess.run([*argv, "--dry-run"], env=env, stdout=output, stderr=subprocess.PIPE, text=True)
        (directory / "reference_dry_run.stderr.log").write_text(dry.stderr)
        require(dry.returncode == 0, "actual official repeat preflight failed; no tracking dispatched")
        plan = load(config["reference_dry_run"])
        require(plan["raw_case_binding"]["rawprep_canonical_dwi_coverage_verified"] is True
                and len(plan["source"]["profiles"]) == 8, "actual repeat preflight raw/atlas proof incomplete")
        # Keep the original controller schema/source, so the existing strict
        # per-case origin mapper can accept this real single-case dispatch.
        controller_argv = [config["reference_python"], "-u", config["frozen_sources"]["reference_controller"]["path"],
                "--raw-manifest", config["known_inputs"]["raw_reference_manifest"]["path"],
                "--raw-manifest-sha256", config["known_inputs"]["raw_reference_manifest"]["sha256"],
                "--official-dwi-root", config["modeling_root"], "--official-anatomy-root", config["anatomy_output"],
                "--verified-reference-manifest", config["known_inputs"]["verified_reference_manifest"]["path"],
                "--verified-reference-manifest-sha256", config["known_inputs"]["verified_reference_manifest"]["sha256"],
                "--mrtrix-bin", reference["mrtrix_bin"], "--python", config["reference_python"],
                "--output-root", config["reference_output_root"], "--case-ids", CASE,
                "--n-seeds", "100000", "--seeds", "0", "1", "2", "3", "4", "--poll-seconds", "15"]
        controller_config = {"raw_manifest": config["known_inputs"]["raw_reference_manifest"]["path"],
            "raw_manifest_sha256": config["known_inputs"]["raw_reference_manifest"]["sha256"],
            "official_dwi_root": config["modeling_root"], "official_anatomy_root": config["anatomy_output"],
            "verified_reference_manifest": config["known_inputs"]["verified_reference_manifest"]["path"],
            "verified_reference_manifest_sha256": config["known_inputs"]["verified_reference_manifest"]["sha256"],
            "mrtrix_bin": reference["mrtrix_bin"], "python": config["reference_python"],
            "source_files": {x["path"]: x["sha256"] for x in config["frozen_sources"].values()},
            "n_seeds": 100000, "seeds": [0, 1, 2, 3, 4], "case_ids": [CASE], "poll_seconds": 15,
            "controller_script": config["frozen_sources"]["reference_controller"]["path"],
            "output_root": config["reference_output_root"],
            "actual_anatomy_contract": record(anatomy_consumer), "actual_DWI_contract": ready["consumer_contract"],
            "actual_preflight": record(config["reference_dry_run"]), "frozen_sources": config["frozen_sources"],
            "reference_parameters": reference}
        run_child(directory, "reference", controller_argv, config, state, controller_config)
        manifest = Path(config["reference_output"]) / "reference_manifest.json"
        final = load(manifest)
        require(final["execution_completed"] is True and final["case_id"] == CASE, "actual reference execution incomplete")
        require(len(final["completed_commands"]) == len(final["commands"])
                and all(x["returncode"] == 0 for x in final["completed_commands"]), "actual reference command failed")
        state.update(state="completed", reference_completed=True, reference_manifest=record(manifest),
                     completed_UTC=utc(), total_followon_wall_including_wait=time.monotonic() - started,
                     timing_scope="separate fresh rawprep/modeling, reused structural prepare, new anatomy completion and repeats; not continuous cold raw chain")
        save(directory / "completed_handoff.json", {"schema_version": 1, "case_id": CASE, "state": "completed",
             "model_consumer": ready["consumer_contract"], "anatomy_consumer": record(anatomy_consumer),
             "reference_manifest": record(manifest), "anatomy_launch": record(directory / "anatomy_launch.json"),
             "reference_launch": record(directory / "reference_launch.json"),
             "reference_controller_configuration": record(directory / "reference_execution_config.json"),
             "reference_controller_status": record(Path(config["reference_output_root"]) / "cohort_status.json"),
             "configuration": record(args.config),
             "source": record(__file__), "scientific_workers": config["frozen_sources"], "scientific_parity": "not_assessed"}, exclusive=True)
    except Exception as error:
        state.update(state="failed_preserved", error={"type": type(error).__name__, "message": str(error)}, observed_UTC=utc())
        raise
    finally:
        save(directory / "status.json", state)


if __name__ == "__main__":
    main()
