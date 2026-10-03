"""十例独立官方解剖参照调度；最多两例CPU8，无GPU、无额外recon-all。

从baseline本轮真实fresh FS开始，等待每例官方raw-DWI contract，再运行FLIRT。
输出仅解剖/atlas consumer contract，不宣称tractography或connectome完成。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

SOURCE = Path(__file__).with_name("benchmark_connectome_anatomy_official.py")
SPEC = importlib.util.spec_from_file_location("official_anatomy", SOURCE)
anatomy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(anatomy)
CASES = ("sub-CON01", "sub-CON03", *[f"sub-CON{i:02d}" for i in range(4, 12)])


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def completed_baseline_origin(directory, case_id):
    original_path = directory / "recon_report.json"
    if not original_path.is_file():
        return None
    original = json.loads(original_path.read_text())
    if original.get("status") == "running":
        return None
    if original.get("case_id") != case_id or original.get("exit_code") != 0:
        raise ValueError("baseline fresh official recon-all did not complete with exit0")
    if original.get("status") != "completed":
        error = original.get("error", {})
        if error.get("type") != "RuntimeError" or "Object of type int32 is not JSON serializable" not in error.get("message", ""):
            raise ValueError("only the measured post-reconstruction JSON int32 failure may be revalidated")
    command = original["command"]
    subject = (Path(command[command.index("-sd") + 1]) / command[command.index("-s") + 1]).resolve()
    expected = directory / "freesurfer" / f"{case_id}_ses-preop"
    if subject != expected.resolve() or not (subject / "scripts/recon-all.done").is_file():
        raise ValueError("fresh anatomy must be the exact baseline subject with actual recon-all.done")
    return original, original_path, subject


def case_config(template, case_id, baseline_root, raw_root, directory, validation_source, tool_commit):
    ready = completed_baseline_origin(baseline_root / case_id, case_id)
    if ready is None:
        return None
    original, original_path, subject = ready
    raw = raw_root / case_id / "ses-preop/anat" / f"{case_id}_ses-preop_T1w.nii.gz"
    config = {**template, "case_id": case_id, "tool_commit": tool_commit,
              "subject_dir": str(subject), "raw_t1w": anatomy.file_record(raw)}
    existing = original_path.with_name("recon_report.revalidated.json")
    if existing.is_file():
        # Existing same-round sidecar is still validated by verify_anatomy.
        config["anatomy_report"] = anatomy.file_record(existing)
        anatomy.verify_anatomy(config)
        return config
    original_record = anatomy.file_record(original_path)
    sidecar = directory / "fresh_fs_readonly_revalidation.json"
    bound = {"case_id": case_id, "status": "running", "original_report": original_record,
             "recon_all_rerun": False, "anatomy": original["anatomy"],
             "validation_source": anatomy.file_record(validation_source), "start_utc": utc(),
             "scope": "read actual arrays of this round's already successful baseline official reconstruction"}
    # Reuse the project's existing actual-array reader in an isolated CPU child.
    # It imports nibabel/numpy only in this mode; does not run FNIT algorithms.
    payload = {"subject_dir": str(subject), "files": list(original["anatomy"])}
    argv = [config["python"], str(validation_source), "_anatomy"]
    started = time.perf_counter()
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    environment.update({name: "8" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})
    result = subprocess.run(argv, input=json.dumps(payload), capture_output=True, text=True, env=environment)
    (directory / "fresh_fs_readonly_revalidation.stderr.log").write_text(result.stderr)
    bound.update(argv=argv, returncode=result.returncode, seconds=time.perf_counter() - started)
    try:
        if result.returncode:
            raise RuntimeError("actual-array anatomy reader failed; see the preserved sidecar log")
        geometry = json.loads(result.stdout)
        if geometry.get("status") != "actual_images_surfaces_annotations_read":
            raise ValueError("fresh FS actual-array geometry validation is incomplete")
        anatomy.verify_file(original_record)
        bound.update(status="completed", anatomy_geometry=geometry, end_utc=utc())
        atomic_json(sidecar, bound)
        config["anatomy_report"] = anatomy.file_record(sidecar)
        anatomy.verify_anatomy(config)
    except Exception as error:
        bound.update(status="failed", end_utc=utc(), error={"type": type(error).__name__, "message": str(error)})
        atomic_json(sidecar, bound)
        raise
    return config


def verify_prepared(path, case_id):
    record = anatomy.file_record(path)
    prepared = anatomy.read_bound_json(record)
    if (prepared.get("case_id") != case_id or prepared.get("state") != "official_structural_reference_completed"
            or prepared.get("execution_completed") is not True):
        raise ValueError("same-case completed official structural reference required")
    for output in prepared["outputs"].values():
        anatomy.verify_file(output)
    return record, prepared


def consumer_contract(case_id, prepared_record, complete_path, dwi_record):
    complete_record = anatomy.file_record(complete_path)
    report = anatomy.read_bound_json(complete_record)
    if (report.get("case_id") != case_id or report.get("state") != "official_anatomy_and_dwi_atlas_completed"
            or report.get("execution_completed") is not True):
        raise ValueError("completed same-case official anatomy/atlas report required")
    for output in report["outputs"].values():
        anatomy.verify_file(output)
    if report["prepared_origin"] != prepared_record or report["official_dwi_origin"]["contract"] != dwi_record:
        raise ValueError("complete report input contract identity differs")
    prepared = anatomy.read_bound_json(prepared_record)
    atlases = {}
    for name in ("fs-aparc", *anatomy.PROFILES):
        nodes = report["outputs"][f"nodes:{name}"]
        atlases[name] = {"atlas_dwi": report["outputs"][f"atlas:{name}"], "nodes": nodes,
                        "n_nodes": len(anatomy.read_nodes(nodes["path"]))}
    return {"schema_version": 1, "case_id": case_id, "state": "completed",
            "scope": "official_self_produced_fresh_fs_anatomy_and_raw_dwi_atlases",
            "prepared_report": prepared_record, "official_anatomy_report": complete_record,
            "official_dwi_contract": dwi_record, "raw_t1w": prepared["preflight"]["anatomy"]["raw_t1w"],
            "fresh_fs_origin": prepared["preflight"]["anatomy"],
            "files": {name: report["outputs"][source] for name, source in
                      (("five_tissue_dwi_world", "five_tissue"), ("gmwmi_dwi_world", "gmwmi"),
                       ("dwi_to_t1_fsl", "dwi_to_t1_fsl"), ("dwi_to_t1_mrtrix", "dwi_to_t1_mrtrix"))},
            "atlases": atlases, "tractography_completed": False, "connectome_completed": False,
            "timing_scope": "official staged anatomy and atlas only; original reconstruction, preparation, completion and waits are distinct"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-template", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--official-dwi-root", type=Path, required=True)
    parser.add_argument("--validation-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tool-commit", required=True)
    parser.add_argument("--case-ids", nargs="+", default=CASES)
    parser.add_argument("--prepared-case-report", action="append", default=[], metavar="CASE=REPORT")
    parser.add_argument("--recover-complete-case-proof", action="append", default=[], metavar="CASE=PROOF")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--wait-timeout-seconds", type=float, default=21600)
    args = parser.parse_args(argv)
    if args.output.exists() or len(set(args.case_ids)) != len(args.case_ids) or any(not re.fullmatch(r"sub-CON\d{2}", x) for x in args.case_ids):
        parser.error("fresh output and unique exact case IDs required")
    if args.poll_seconds < 1 or args.wait_timeout_seconds < 1:
        parser.error("positive bounded poll and timeout required")
    template = json.loads(args.config_template.read_text())
    validation_identity = anatomy.file_record(args.validation_script)
    seeded = {}
    for item in args.prepared_case_report:
        case, path = item.split("=", 1)
        if case not in args.case_ids or case in seeded:
            parser.error("one prepared origin per selected case required")
        seeded[case] = verify_prepared(Path(path), case)
    completion_recoveries = {}
    for item in args.recover_complete_case_proof:
        case, path = item.split("=", 1)
        if case not in seeded or case in completion_recoveries:
            parser.error("one completion recovery proof per explicitly prepared selected case required")
        record = anatomy.file_record(path)
        proof = anatomy.read_bound_json(record)
        if proof.get("state") != "completed" or case not in proof.get("cases", {}):
            parser.error("completed same-case layout proof required")
        completion_recoveries[case] = record
    args.output.mkdir(parents=True, exist_ok=False)
    state = {"schema_version": 1, "state": "running", "execution_completed": False,
             "scope": "independent official fresh FS and raw-DWI anatomy/atlas cohort",
             "connectome_completed": False, "start_utc": utc(), "source_commit": args.tool_commit,
             "script": anatomy.file_record(__file__), "anatomy_script": anatomy.file_record(SOURCE),
             "validation_script": validation_identity, "max_cpu_cases": args.workers, "threads_per_case": 8,
             "gpu": {"used": False, "allocated": None, "reserved": None}, "cases": {}}
    for case in args.case_ids:
        (args.output / case).mkdir()
        state["cases"][case] = {"state": "waiting_fresh_fs", "config": None, "prepared_report": None}
        if case in seeded:
            record, prepared = seeded[case]
            config = {**prepared["config"], "tool_commit": args.tool_commit}
            expected = args.baseline_root / case / "freesurfer" / f"{case}_ses-preop"
            if Path(config["subject_dir"]).resolve() != expected.resolve():
                raise ValueError("seeded pilot is not the selected baseline fresh FS subject")
            anatomy.verify_anatomy(config)
            state["cases"][case].update(state="waiting_official_dwi", config=config,
                prepared_report=record, prepared_origin="bound separately timed pilot; no preparation rerun")
            if case in completion_recoveries:
                state["cases"][case]["complete_recovery_proof"] = completion_recoveries[case]
    running = {}
    started = time.perf_counter()
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONUNBUFFERED": "1"}
    try:
        while True:
            for case, (process, stream, mode) in list(running.items()):
                if process.poll() is None:
                    continue
                stream.close()
                row = state["cases"][case]
                row[f"{mode}_returncode"] = process.returncode
                del running[case]
                if process.returncode:
                    row.update(state="failed", error=f"official {mode} subprocess exited {process.returncode}; original logs preserved")
                    continue
                if mode == "prepare":
                    record, _ = verify_prepared(args.output / case / "prepare/reference_anatomy.json", case)
                    row.update(state="waiting_official_dwi", prepared_report=record)
                else:
                    contract = consumer_contract(case, row["prepared_report"],
                        args.output / case / "complete/reference_anatomy.json", row["official_dwi_contract"])
                    path = args.output / case / "consumer_contract.json"
                    atomic_json(path, contract)
                    row.update(state="completed", consumer_contract=anatomy.file_record(path))
            for case in args.case_ids:
                if len(running) >= args.workers:
                    break
                row = state["cases"][case]
                directory = args.output / case
                if row["state"] == "waiting_fresh_fs":
                    config = case_config(template, case, args.baseline_root, args.raw_root, directory,
                                         args.validation_script, args.tool_commit)
                    if config is None:
                        continue
                    row["config"] = config
                    mode = "prepare"
                elif row["state"] == "waiting_official_dwi":
                    path = args.official_dwi_root / case / "consumer_contract.json"
                    if not path.is_file():
                        continue
                    record = anatomy.file_record(path)
                    body = anatomy.read_bound_json(record)
                    if body.get("state") != "completed":
                        continue
                    anatomy.verify_dwi_contract(record, case)
                    row["official_dwi_contract"] = record
                    mode = "complete"
                else:
                    continue
                config_path = directory / "config.json"
                if not config_path.exists():
                    atomic_json(config_path, row["config"])
                    row["config_record"] = anatomy.file_record(config_path)
                else:
                    anatomy.verify_file(row["config_record"])
                anatomy.verify_file(validation_identity)
                anatomy.verify_file(state["anatomy_script"])
                command_mode = "recover-complete" if mode == "complete" and row.get("complete_recovery_proof") else mode
                command = [row["config"]["python"], str(SOURCE), command_mode, "--config", str(config_path),
                           "--output", str(directory / mode)]
                if mode == "complete":
                    command += ["--prepared-report", row["prepared_report"]["path"],
                                "--official-dwi-contract", row["official_dwi_contract"]["path"]]
                    row["complete_mode"] = command_mode
                    if command_mode == "recover-complete":
                        anatomy.verify_file(row["complete_recovery_proof"])
                        command += ["--complete-prefix-proof", row["complete_recovery_proof"]["path"]]
                log = directory / f"{mode}.log"
                stream = log.open("x")
                process = subprocess.Popen(command, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                running[case] = (process, stream, mode)
                row.update(state=f"running_{mode}", **{f"{mode}_argv": command, f"{mode}_pid": process.pid,
                                                       f"{mode}_started_utc": utc(), f"{mode}_log": str(log)})
            state["last_update_utc"] = utc()
            state["total_wall_seconds_including_waits"] = time.perf_counter() - started
            atomic_json(args.output / "cohort_reference.json", state)
            if all(row["state"] in ("completed", "failed") for row in state["cases"].values()):
                break
            if time.perf_counter() - started > args.wait_timeout_seconds:
                raise TimeoutError("cohort wait deadline reached; preserve partial reports and actual states")
            time.sleep(args.poll_seconds)
        state["execution_completed"] = all(row["state"] == "completed" for row in state["cases"].values())
        state["state"] = "completed" if state["execution_completed"] else "failed"
    except Exception as error:
        state.update(state="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        # A driver exception must not silently leave two unmonitored children.
        for process, stream, _ in running.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            stream.close()
        state["end_utc"] = utc()
        state["total_wall_seconds_including_waits"] = time.perf_counter() - started
        atomic_json(args.output / "cohort_reference.json", state)
    return 0 if state["execution_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
