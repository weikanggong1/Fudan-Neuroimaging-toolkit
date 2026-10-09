"""复现固定四网格 remesh 标量存储配对，全部使用独立 CPU 子进程。

--source 指冻结 src，--input-dir 须含 sub06/sub07 的双侧 orig.premesh，
--output 必须不存在。固定 --threads=4 默认预算；调用者另外用 taskset 固定
亲和性。--numba-cache 指同源码已预热缓存，否则第一次的 JIT 明确计入。
GC 恒为 inherit。执行 sub07 LH 的 ABBA、独立逐边 trace，随后其余三网格
各 A/B。A 是原 NumPy 标量，B 是显式 Python 双精度；不调用 GPU、官方命令
或读取官方结果。保存包含启动/退出、读写的进程墙钟及 user/system CPU 时间。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--numba-cache", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.threads <= 0:
        raise ValueError("new output and positive thread budget required")
    names = ("sub07_lh", "sub07_rh", "sub06_lh", "sub06_rh")
    if any(not (args.input_dir / (name + ".orig.premesh")).is_file() for name in names):
        raise ValueError("four real public self-produced input meshes required")
    worker = Path(__file__).with_name("profile_remesh_current.py")
    if not worker.is_file():
        raise ValueError("existing complete remesh diagnostic driver missing")
    if not (args.source / "fnit/recon_all/mris_remesh_python.py").is_file():
        raise ValueError("frozen source module missing")
    args.output.mkdir(parents=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        environment[name] = str(args.threads)
    environment["NUMBA_CACHE_DIR"] = str(args.numba_cache)
    tools = Path(__file__).resolve().parents[2] / "python_gpu_port"
    environment["PYTHONPATH"] = os.pathsep.join((str(args.source.resolve()), str(tools),
                                                environment.get("PYTHONPATH", "")))
    jobs = [("abba_A1", "sub07_lh", "numpy", False), ("abba_B1", "sub07_lh", "python", False),
            ("abba_B2", "sub07_lh", "python", False), ("abba_A2", "sub07_lh", "numpy", False),
            ("python_decision_trace", "sub07_lh", "python", True)]
    for mesh in names[1:]:
        jobs.extend(((mesh + "_A", mesh, "numpy", False), (mesh + "_B", mesh, "python", False)))
    rows = []
    for name, mesh, storage, trace in jobs:
        command = [sys.executable, str(worker), "--source", str(args.source),
                   "--input", str(args.input_dir / (mesh + ".orig.premesh")),
                   "--output", str(args.output / name), "--version", args.version,
                   "--threads", str(args.threads), "--scalar-storage", storage, "--gc-policy", "inherit"]
        if trace:
            command.append("--decision-trace")
        started = time.perf_counter()
        previous = resource.getrusage(resource.RUSAGE_CHILDREN)
        with (args.output / (name + ".stdout.log")).open("wb") as stream:
            status = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT).returncode
        current = resource.getrusage(resource.RUSAGE_CHILDREN)
        rows.append(dict(name=name, mesh=mesh, storage=storage, trace=trace, status=status,
                         outer_seconds_including_child_startup_exit=time.perf_counter()-started,
                         user_seconds=current.ru_utime-previous.ru_utime,
                         system_seconds=current.ru_stime-previous.ru_stime))
        (args.output / "process_receipts.json").write_text(json.dumps(rows, indent=2) + "\n")
        print("DONE", name, status, rows[-1]["outer_seconds_including_child_startup_exit"], flush=True)
        if status:
            return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
