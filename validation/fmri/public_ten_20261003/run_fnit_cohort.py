"""串行启动真实完整整例，每例独立进程和新输出，失败保留后停止队列。"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--physical-gpu", required=True)
    parser.add_argument("--continue-on-failure", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, PYTHONPATH=str(args.source_root / "src"),
                       CUDA_VISIBLE_DEVICES=args.physical_gpu,
                       OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8")
    # All native commands for FNIT are independently built in its Conda env.
    environment["PATH"] = str(Path(args.python).parent) + os.pathsep + environment.get("PATH", "")
    summary = {"source_revision": args.source_revision, "pid": os.getpid(), "cases": {},
               "status": "running", "queue_seconds_excluded_from_case_wall": True}
    state = args.output_root / ("queue_" + args.subjects[0] + ".private.json")
    def save():
        temporary = state.with_suffix(".tmp")
        temporary.write_text(json.dumps(summary, indent=2) + "\n")
        temporary.replace(state)
    save()
    for subject in args.subjects:
        case = args.output_root / subject
        report = case / "report"
        log = case / "command.private.log"
        case.mkdir(parents=True, exist_ok=True)
        if log.exists() or report.exists():
            raise FileExistsError("use a new candidate attempt directory")
        argv = [args.python, str(args.source_root / "validation/fmri/public_ten_20261003/run_fnit_subject.py"),
                "--config", str(args.configs / (subject + ".json")), "--output", str(report),
                "--source-root", str(args.source_root), "--source-revision", args.source_revision]
        started = time.perf_counter()
        with log.open("x") as stream:
            process = subprocess.Popen(argv, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT)
            summary["cases"][subject] = {"status": "running", "pid": process.pid, "argv": argv}
            save()
            code = process.wait()
        summary["cases"][subject].update(status="complete" if code == 0 else "failed", exit_code=code,
                                         process_wall_seconds=time.perf_counter() - started)
        save()
        if code and not args.continue_on_failure:
            summary["status"] = "stopped_on_failure"
            save()
            return code
    summary["status"] = "complete" if all(item["exit_code"] == 0 for item in summary["cases"].values()) else "failed"
    save()
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
