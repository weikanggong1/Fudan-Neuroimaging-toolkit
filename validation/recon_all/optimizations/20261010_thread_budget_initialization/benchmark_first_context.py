"""首次Numba初始化的真实线程状态回归；旧/新各使用一个全新Python进程。

本工具没有MRI输入、GPU计算或原软件调用。它冻结Git参考和工作树模块，
固定OMP_NUM_THREADS=4并移除NUMBA_NUM_THREADS，比较同一个首次作用域。
真实getter/setter观察器只记录操作，不伪造线程数或预先启动Numba线程池。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


THREAD_ENVIRONMENT = ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                      "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_DYNAMIC")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def child(module_path: Path, output: Path) -> None:
    started = time.perf_counter()
    import numba
    import numpy
    import torch

    environment_before = {name: os.environ.get(name) for name in THREAD_ENVIRONMENT}
    original_torch_get = torch.get_num_threads
    original_torch_set = torch.set_num_threads
    original_numba_get = numba.get_num_threads
    original_numba_set = numba.set_num_threads
    events = []

    def layer():
        try:
            return numba.threading_layer()
        except ValueError:
            return "not_initialized"

    def observe_torch_get():
        value = original_torch_get()
        events.append({"operation": "torch.get_num_threads", "value": value})
        return value

    def observe_torch_set(value):
        before = original_torch_get()
        original_torch_set(value)
        events.append({"operation": "torch.set_num_threads", "requested": value,
                       "before": before, "after": original_torch_get()})

    def observe_numba_get():
        before = original_torch_get()
        layer_before = layer()
        value = original_numba_get()
        events.append({"operation": "numba.get_num_threads", "value": value,
                       "torch_before": before, "torch_after": original_torch_get(),
                       "numba_threading_layer_before": layer_before,
                       "numba_threading_layer_after": layer()})
        return value

    def observe_numba_set(value):
        before = original_torch_get()
        original_numba_set(value)
        events.append({"operation": "numba.set_num_threads", "requested": value,
                       "torch_before": before, "torch_after": original_torch_get()})

    before = {"torch_intraop": original_torch_get(),
              "torch_interop": torch.get_num_interop_threads(),
              "numba_mask": "not_read_before_first_context",
              "numba_initial_capacity": int(numba.config.NUMBA_NUM_THREADS),
              "numba_threading_layer": layer(), "cuda_initialized": torch.cuda.is_initialized(),
              "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
              "cudnn_tf32": torch.backends.cudnn.allow_tf32}
    # Direct import avoids FNIT package imports initializing unrelated components.
    specification = importlib.util.spec_from_file_location("frozen_thread_budget", module_path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    torch.get_num_threads, torch.set_num_threads = observe_torch_get, observe_torch_set
    numba.get_num_threads, numba.set_num_threads = observe_numba_get, observe_numba_set
    report = {"scope": "first actual Torch/Numba context; no MRI or CUDA work",
              "module_sha256": digest(module_path), "before": before, "events": events,
              "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                          "numba": numba.__version__, "numpy": numpy.__version__,
                          "system": platform.system(), "release": platform.release(),
                          "machine": platform.machine(), "logical_cpus": os.cpu_count(),
                          "affinity": sorted(os.sched_getaffinity(0))},
              "requested_threads": 4, "environment_before": environment_before,
              "inside": None, "context_report": None, "exception": None}
    context_started = time.perf_counter()
    try:
        with module.thread_budget(threads=4) as context_report:
            report["context_report"] = context_report
            report["inside"] = {"torch_intraop": original_torch_get(),
                                "numba_mask": original_numba_get(),
                                "torch_interop": torch.get_num_interop_threads(),
                                "numba_threading_layer": layer()}
        report["status"] = "context_completed"
    except Exception as error:
        report["status"] = "context_failed"
        report["exception"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        report["context_seconds"] = time.perf_counter() - context_started
        torch.get_num_threads, torch.set_num_threads = original_torch_get, original_torch_set
        numba.get_num_threads, numba.set_num_threads = original_numba_get, original_numba_set
    report["after"] = {"torch_intraop": original_torch_get(), "numba_mask": original_numba_get(),
                       "torch_interop": torch.get_num_interop_threads(),
                       "numba_threading_layer": layer(), "cuda_initialized": torch.cuda.is_initialized(),
                       "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                       "cudnn_tf32": torch.backends.cudnn.allow_tf32}
    report["environment_after"] = {name: os.environ.get(name) for name in THREAD_ENVIRONMENT}
    report["environment_unchanged"] = report["environment_after"] == environment_before
    report["child_seconds_including_imports"] = time.perf_counter() - started
    write_json(output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--baseline-ref", default="HEAD")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child-module", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child_module is not None:
        child(args.child_module, args.output)
        return
    if args.repository is None:
        parser.error("--repository is required for the parent driver")
    started_utc = datetime.now(timezone.utc).isoformat()
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    repository = args.repository.resolve()
    relative = "src/fnit/recon_all/thread_budget.py"
    baseline_commit = subprocess.check_output(
        ["git", "rev-parse", args.baseline_ref], cwd=repository, text=True).strip()
    baseline = subprocess.check_output(["git", "show", f"{baseline_commit}:{relative}"], cwd=repository)
    candidate = (repository / relative).read_bytes()
    write_json(args.output / "frozen_modules.json", {
        "baseline_commit": baseline_commit,
        "baseline_sha256": hashlib.sha256(baseline).hexdigest(),
        "candidate_sha256": hashlib.sha256(candidate).hexdigest(),
        "candidate_identity": "uncommitted working tree snapshot; not relabelled as baseline commit",
        "module_path": relative})
    child_environment = os.environ.copy()
    child_environment["OMP_NUM_THREADS"] = "4"
    child_environment.pop("NUMBA_NUM_THREADS", None)
    child_environment["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        child_environment[name] = "4"
    children = []
    for label, source in (("baseline", baseline), ("candidate", candidate)):
        module_path = args.output / f"{label}_module.py"
        module_path.write_bytes(source)
        output = args.output / f"{label}.json"
        started = time.perf_counter()
        process = subprocess.run([str(args.python), str(Path(__file__).resolve()),
                                  "--child-module", str(module_path.resolve()),
                                  "--output", str(output.resolve())], env=child_environment,
                                 capture_output=True, text=True)
        (args.output / f"{label}.stdout.txt").write_text(process.stdout)
        (args.output / f"{label}.stderr.txt").write_text(process.stderr)
        children.append({"label": label, "returncode": process.returncode,
                         "fresh_process_wall_seconds": time.perf_counter() - started,
                         "report": output.name if output.exists() else None})
    complete = all(row["returncode"] == 0 and row["report"] is not None for row in children)
    comparison = {"scope": "local fresh-process thread initialization only; no MRI benchmark",
                  "started_UTC": started_utc,
                  "completed_UTC": datetime.now(timezone.utc).isoformat(),
                  "script_sha256": digest(Path(__file__)), "baseline_commit": baseline_commit,
                  "python_program_sha256": digest(args.python.resolve()), "children": children,
                  "overall_metric_equivalence": "not_applicable_no_image_input",
                  "whole_pipeline_speedup": "not_measured"}
    if complete:
        old = json.loads((args.output / "baseline.json").read_text())
        new = json.loads((args.output / "candidate.json").read_text())
        initialization = [item for item in new["events"]
                          if item["operation"] == "numba.get_num_threads"
                          and item["numba_threading_layer_before"] == "not_initialized"]
        baseline_initialization = [item for item in old["events"]
                                   if item["operation"] == "numba.get_num_threads"
                                   and item["numba_threading_layer_before"] == "not_initialized"]
        comparison.update({"baseline_status": old["status"], "candidate_status": new["status"],
                           "actual_numba_initialization_observed": bool(initialization),
                           "baseline_initialization_changes_torch": bool(baseline_initialization)
                           and baseline_initialization[0]["torch_before"] != baseline_initialization[0]["torch_after"],
                           "baseline_bug_reproduced": old["status"] == "context_failed"
                           and old["exception"]["type"] == "RuntimeError"
                           and old["before"]["torch_intraop"] == 4
                           and old["after"]["torch_intraop"] != 4,
                           "candidate_requested_budget_applied": new["inside"] is not None
                           and new["inside"]["torch_intraop"] == 4
                           and new["inside"]["numba_mask"] == 4,
                           "candidate_restored": new["context_report"] is not None
                           and new["context_report"]["restoration_complete"],
                           "environment_unchanged": old["environment_unchanged"] and new["environment_unchanged"],
                           "cuda_not_initialized": not old["after"]["cuda_initialized"]
                           and not new["after"]["cuda_initialized"]})
        comparison["status"] = ("passed" if comparison["candidate_requested_budget_applied"]
                                and comparison["candidate_restored"] and comparison["environment_unchanged"]
                                and comparison["cuda_not_initialized"] else "failed")
    else:
        comparison["status"] = "driver_or_child_failed"
    write_json(args.output / "comparison.json", comparison)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
