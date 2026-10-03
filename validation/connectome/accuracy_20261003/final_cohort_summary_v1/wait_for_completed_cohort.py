"""CPU deployment waiter; invoke unchanged final tools only on actual completion."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path("/cwStorage/home/gongwk/Notebook_code/fnit_connectome_accuracy_20261003_v1")
CONTROL = ROOT / "final_cohort_wait_v1"
CONFIG = ROOT / "formal_frozen_v1/accuracy_configuration.json"
CONFIG_SHA = "f8eba1cbf3eab2baa549172b32be2c9d3b1014ad478f6e3702d7987378f52f8c"
FULL_REPORT = ROOT / "root_matrix_analysis_v3/full_cohort/report.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def atomic(value):
    temporary = CONTROL / "status.tmp"
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(CONTROL / "status.json")


def source_hashes():
    paths = [CONFIG, Path(__file__), ROOT / "final_source_acceptance_v1/verify_source_acceptance.py",
             ROOT / "final_cohort_summary_v1/extract_completed_cohort.py",
             ROOT / "private_cpu_tools_v3/tools/analyze_connectome_accuracy_cohort.py"]
    names = ["benchmark_connectome_raw_cohort", "benchmark_connectome_accuracy_cohort", "compare_connectome_matrices",
             "connectome_repeat_common", "benchmark_connectome_raw_cohort_envelope",
             "benchmark_connectome_tracking_population", "analyze_connectome_accuracy_cohort"]
    paths += [ROOT / "formal_frozen_v1/tools_source/tools" / (name + ".py") for name in names]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def execute():
    state = {"status": "waiting", "pid": os.getpid(), "scope": "CPU deployment waiter only; no producer/science edits or GPU work",
             "poll_seconds": 55, "commands": [], "source_sha256_before": source_hashes()}
    atomic(state)
    try:
        require(state["source_sha256_before"][str(CONFIG)] == CONFIG_SHA, "original frozen configuration changed")
        config = json.loads(CONFIG.read_bytes())
        candidate_cases = {item["case_id"] for item in config["execution_order"] if item["version"] == "candidate"}
        while True:
            phase = json.loads((Path(config["run_root"]) / "status.json").read_bytes())
            report = json.loads(FULL_REPORT.read_bytes()) if FULL_REPORT.is_file() else None
            state["observation"] = {"phase_status": phase["status"], "producer_status": {
                f"{item['version']}/{item['case_id']}": phase.get("cases", {}).get(
                    f"{item['version']}/{item['case_id']}", {}).get("status", "not_started")
                for item in config["execution_order"]}, "full_cohort_status": report.get("status") if report else "not_created"}
            atomic(state)
            require(phase["status"] != "failed", "actual raw phase failed; final tools not started")
            require(report is None or report.get("status") != "failed", "actual full_cohort analysis failed; final tools not started")
            if phase["status"] == "execution_completed" and report and report["status"] == "analysis_completed":
                require(len(candidate_cases) == 10 and set(report["cases"]) == candidate_cases and
                        report["coverage"]["full_ten_case_cohort"] is True and
                        all(value == "completed" for value in state["observation"]["producer_status"].values()),
                        "terminal reports lack complete planned coverage")
                break
            time.sleep(55)
        require(source_hashes() == state["source_sha256_before"], "tool/config changed while waiting")
        state["status"] = "final_tools_running"
        atomic(state)
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
        commands = [
            [sys.executable, str(ROOT / "final_source_acceptance_v1/verify_source_acceptance.py"),
             "--source", str(ROOT / "formal_frozen_v1/candidate"), "--configuration", str(CONFIG),
             "--test-directory", str(ROOT / "root_integrated_tests_v3"), "--check-producers", "--require-complete",
             "--output", str(ROOT / "final_source_acceptance_v1/final_completed_source_receipt.json")],
            [sys.executable, str(ROOT / "final_cohort_summary_v1/extract_completed_cohort.py"),
             "--configuration", str(CONFIG), "--analysis-report", str(FULL_REPORT),
             "--output-dir", str(ROOT / "final_cohort_summary_v1/completed_excerpt"),
             "--receipt", str(ROOT / "final_cohort_summary_v1/final_completed_extraction_receipt.json")],
        ]
        for index, command in enumerate(commands):
            started = time.perf_counter()
            with (CONTROL / f"final_tool_{index}.log").open("x") as log:
                completed = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
            state["commands"].append({"command": command, "exit_code": completed.returncode,
                                      "wall_seconds": time.perf_counter() - started})
            atomic(state)
            require(completed.returncode == 0, f"actual final tool {index} failed")
        state["memory"] = {}
        for item in config["execution_order"]:
            key = f"{item['version']}/{item['case_id']}"
            gpu = json.loads((Path(config["run_root"]) / key / "gpu_report.json").read_bytes())
            budget = gpu["memory_budget"]
            measurement = budget["measurements"]
            passed = all(type(measurement.get(name)) in (int, float) and math.isfinite(measurement[name]) and
                         0 <= measurement[name] < 20_000_000_000 for name in ("process_tree", "allocated_bytes", "reserved_bytes"))
            state["memory"][key] = {"original_memory_budget": budget,
                                    "three_fields_below_20e9": passed, "monitor_issues_empty": budget["monitor_issues"] == []}
        state["memory_gate"] = "passed" if all(entry["three_fields_below_20e9"] and entry["monitor_issues_empty"]
                                                for entry in state["memory"].values()) else "failed"
        state["source_sha256_after"] = source_hashes()
        require(state["source_sha256_after"] == state["source_sha256_before"], "tool/config changed during final checks")
        state["status"] = "final_tools_completed"
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", source_sha256_after=source_hashes())
    atomic(state)
    return 0 if state["status"] == "final_tools_completed" else 1


if __name__ == "__main__":
    raise SystemExit(execute())
