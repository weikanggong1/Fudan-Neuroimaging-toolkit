#!/usr/bin/env python3
"""等待具名十例 posthoc 保存绑定，串行执行独立完整自交补充检查。

不调用 MRI workflow；复用已有补测须逐文件核验实际 mesh、报告及源码。
输入私有配置，输出公开数值和 SHA；所有原 MRI 与历史报告只读。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CASES = ("CON01", "CON03", "CON04", "CON05", "CON06", "CON07", "CON08", "CON09", "CON10", "CON11")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def legacy_quality_running(root):
    # 只读检查同一独立 posthoc 的实际 worker；不发送信号。
    for process_path in root.glob("*/process.private.json"):
        process = json.loads(process_path.read_text())
        command_path = Path("/proc") / str(process.get("pid", "absent")) / "cmdline"
        if command_path.exists():
            command = command_path.read_bytes()
            if b"compare_reconstruction.py" in command and str(process_path.parent).encode() in command:
                return True
    return False


def verify_saved_worker(report_path, digest, entry, source_sha, scanner_sha):
    if sha(report_path) != digest:
        raise ValueError("saved self-scan report changed")
    report = json.loads(report_path.read_text())
    if (report.get("status") != "measured" or report.get("inputs_and_sources_unchanged") is not True
            or any(report.get(key) != entry[key] for key in
                   ("cohort_id", "case_id", "role", "hemisphere", "surface_kind", "candidate_source_revision"))
            or report["input_sha256"].get("surface") != entry["surface"]["sha256"]
            or report["input_sha256"].get("run_report") != entry["run_report"]["sha256"]
            or report["input_sha256"].get("predicate_module") != source_sha
            or report["input_sha256"].get("scanner") != scanner_sha):
        raise ValueError("saved self-scan does not bind the same complete actual case")
    return {"status": report["status"], "report_sha256": digest,
            "mesh_sha256": entry["surface"]["sha256"],
            "ordered_faces_sha256": report["ordered_faces_sha256"],
            "intersecting_faces": report["intersecting_faces"], "marked_vertices": report["marked_vertices"],
            "inputs_and_sources_unchanged": True, "posthoc_wall_seconds": report["posthoc_wall_seconds"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("positive poll interval required")
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    if config.get("cohort_id") != "formal-v4" or tuple(config.get("cases", [])) != CASES:
        raise ValueError("fixed formal-v4 ten-case identity required")
    scanner, source_root, posthoc = (Path(config[key]).resolve() for key in ("scanner", "source_root", "posthoc_root"))
    predicate = source_root / "src/fnit/recon_all/mris_remove_intersection_python.py"
    if sha(scanner) != config["scanner_sha256"] or sha(predicate) != config["predicate_module_sha256"]:
        raise ValueError("declared frozen scanner or predicate differs")
    output = args.output_root.resolve()
    protected = [source_root, posthoc, *(Path(path).resolve() for path in config["protected_roots"])]
    if any(output.is_relative_to(root) or root.is_relative_to(output) for root in protected):
        raise ValueError("new self-scan cohort output overlaps protected inputs")
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.environ.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4",
                      MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    sys.dont_write_bytecode = True
    state = {"cohort_id": config["cohort_id"], "status": "waiting", "cases": {},
             "candidate_source_revision": config["candidate_source_revision"],
             "scanner_sha256": sha(scanner), "predicate_module_sha256": sha(predicate),
             "launcher_sha256": sha(__file__), "configuration_sha256": hashlib.sha256(config_bytes).hexdigest(),
             "requested_threads_per_worker": 4, "device": "cpu",
             "scope": "independent saved-mesh self-intersection QC, excluded from MRI clocks",
             "new_worker_process_seconds_sum": 0., "wait_seconds": 0.}
    started = time.perf_counter()
    pending = list(CASES)
    while pending:
        if legacy_quality_running(posthoc):
            wait_start = time.perf_counter();time.sleep(args.poll_seconds)
            state["wait_seconds"] += time.perf_counter() - wait_start
            continue
        progress = False
        for case in list(pending):
            binding_path, posthoc_report = posthoc / case / "files.private.json", posthoc / case / "report.public.json"
            if not binding_path.exists() or not posthoc_report.exists():
                continue
            evidence = json.loads(posthoc_report.read_text())
            if evidence.get("status") == "running":
                continue
            target = output / case;target.mkdir(mode=0o700)
            row = {"status": "running", "results": {}, "posthoc_report_sha256": sha(posthoc_report),
                   "posthoc_binding_sha256": sha(binding_path)}
            state["cases"][case] = row;state["status"] = "running"
            save(output / "cohort.public.json", state)
            try:
                if (evidence.get("candidate_source_revision") != config["candidate_source_revision"]
                        or evidence.get("inputs_unchanged") is not True or evidence.get("code_unchanged") is not True):
                    raise ValueError("exact guarded formal reconstruction comparison required")
                binding = json.loads(binding_path.read_text())
                case_started = time.perf_counter()
                for role in ("reference", "candidate"):
                    run_report = Path(binding[role + "_report"])
                    if sha(run_report) != binding["input_sha256"][role + "_report"]:
                        raise ValueError("completed MRI report changed after posthoc binding")
                    for hemisphere in ("lh", "rh"):
                        for kind in ("white", "pial"):
                            key = role + "/" + hemisphere + "." + kind
                            mesh = Path(binding[role + "_subject"]) / "surf" / (hemisphere + "." + kind)
                            entry = {"cohort_id": config["cohort_id"], "case_id": case, "role": role,
                                "hemisphere": hemisphere, "surface_kind": kind,
                                "candidate_source_revision": config["candidate_source_revision"],
                                "source_root": str(source_root), "predicate_module_sha256": config["predicate_module_sha256"],
                                "surface": {"path": str(mesh), "sha256": sha(mesh)},
                                "run_report": {"path": str(run_report), "sha256": sha(run_report)},
                                "protected_roots": config["protected_roots"]}
                            previous = config.get("reuse", {}).get(case, {}).get(key)
                            if previous:
                                result = verify_saved_worker(Path(previous["path"]), previous["sha256"], entry,
                                    config["predicate_module_sha256"], config["scanner_sha256"])
                                result.update(reused_guarded_complete_measurement=True, new_worker_process_seconds=None)
                            else:
                                while legacy_quality_running(posthoc):
                                    wait_start = time.perf_counter();time.sleep(args.poll_seconds)
                                    state["wait_seconds"] += time.perf_counter() - wait_start
                                private_manifest = target / (role + "-" + hemisphere + "." + kind + ".private.json")
                                save(private_manifest, entry)
                                destination = target / (role + "-" + hemisphere + "." + kind)
                                worker_started = time.perf_counter()
                                with (target / (destination.name + ".private.log")).open("w") as log:
                                    worker = subprocess.run([sys.executable, str(scanner), "--manifest", str(private_manifest),
                                        "--output-root", str(destination), "--threads", "4", "--timeout-seconds", "900"],
                                        stdout=log, stderr=subprocess.STDOUT, timeout=960)
                                elapsed = time.perf_counter() - worker_started
                                state["new_worker_process_seconds_sum"] += elapsed
                                report_path = destination / "report.public.json"
                                result = verify_saved_worker(report_path, sha(report_path), entry,
                                    config["predicate_module_sha256"], config["scanner_sha256"])
                                result.update(reused_guarded_complete_measurement=False, new_worker_process_seconds=elapsed,
                                              worker_exit_code=worker.returncode)
                            row["results"][key] = result
                            save(output / "cohort.public.json", state)
                row.update(status="measured", all_zero_under_native_predicate=all(result["intersecting_faces"] == 0
                    for result in row["results"].values()), case_elapsed_seconds_including_intervening_wait=time.perf_counter()-case_started)
            except Exception as error:
                row.update(status="failed", error_type=type(error).__name__,
                           error_sha256=hashlib.sha256(str(error).encode()).hexdigest())
                (target / "error.private.txt").write_text(str(error) + "\n")
            save(target / "cohort.public.json", row)
            pending.remove(case);progress = True
            save(output / "cohort.public.json", state)
        if not progress and pending:
            wait_start = time.perf_counter();time.sleep(args.poll_seconds)
            state["wait_seconds"] += time.perf_counter() - wait_start
    state.update(status="measured" if all(row["status"] == "measured" for row in state["cases"].values()) else "partially_measured",
        elapsed_seconds_including_wait=time.perf_counter() - started,
        configuration_unchanged=sha(args.config) == state["configuration_sha256"],
        launcher_unchanged=sha(__file__) == state["launcher_sha256"],
        scanner_unchanged=sha(scanner) == state["scanner_sha256"],
        predicate_unchanged=sha(predicate) == state["predicate_module_sha256"])
    if any(state[key] is not True for key in ("configuration_unchanged", "launcher_unchanged", "scanner_unchanged", "predicate_unchanged")):
        state["status"] = "failed_source_guard"
    save(output / "cohort.public.json", state)


if __name__ == "__main__":
    main()
