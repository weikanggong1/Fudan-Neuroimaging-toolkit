"""用已有分析 Python 绘制已保存结果，独立记录事后绘图耗时和源码 SHA。

不安装包，不调用原软件的推理算法。reference-env 仅启用已安装分析
Python 的库；影像与比较结果均只读，输出目录须不存在。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-env", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--plot-script", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--source-label", required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or args.record.exists():
        raise ValueError("render output and receipt must be new")
    command = ["/bin/bash", "-l", str(args.reference_env), str(args.python), str(args.plot_script),
               "--reference", str(args.reference), "--candidate", str(args.candidate),
               "--region-report", str(args.comparison_dir / "regions.private.json"),
               "--dice-report", str(args.comparison_dir / "dice.private.json"),
               "--output-dir", str(args.output_dir), "--code-commit", args.source_label]
    environment = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMBA_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
                "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
        environment[key] = "4"
    environment["CUDA_VISIBLE_DEVICES"] = ""
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    process = subprocess.run(command, env=environment, timeout=300)
    receipt = {"scope": "posthoc rendering only; excluded from reconstruction benchmark",
               "status": "complete" if process.returncode == 0 else "failed",
               "returncode": process.returncode, "wall_seconds": time.monotonic() - started,
               "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(),
               "threads_environment_budget": 4, "hostname": os.uname().nodename,
               "cuda_visible_devices": "", "candidate_source_label": args.source_label,
               "plot_script_sha256": hashlib.sha256(args.plot_script.read_bytes()).hexdigest(),
               "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "command": command}
    args.record.write_text(json.dumps(receipt, indent=2) + "\n")
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
