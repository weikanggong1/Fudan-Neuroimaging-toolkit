"""顺序运行空目录整例，每例单独取得共用锁并保留完整监测记录。

--configs 为 execute_whole_case.py 的完整 JSON 配置路径，按给定顺序执行；
--lock 为服务器本地共用 flock 文件；--output 为新队列报告目录。
不修改输入、算法或精度。每例锁等待单列，算法墙钟由 run_monitored.py
记录。失败保留非零退出码，继续独立的下一例，最终队列非零退出。
这是验证调度器，无独立 FreeSurfer 等价命令；字段及具名示例见 RUN_VALIDATION.md。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, nargs="+", required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--monitor", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, path in enumerate(args.configs):
        config = json.loads(path.read_text())
        monitor_root = Path(config["output"] + "_monitor")
        argv = ["flock", str(args.lock), config["python"], str(args.monitor),
                "--gpu-uuid", config["gpu_uuid"], "--output", str(monitor_root),
                "--interval", "2", "--query-timeout", "5", "--",
                config["python"], str(args.launcher), "--config", str(path)]
        row = {"index": index, "config": str(path), "command": argv,
               "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "queued_utc": datetime.now(timezone.utc).isoformat(),
               "status": "queued_or_running", "monitor_root": str(monitor_root)}
        rows.append(row)
        report = {"rows": rows, "status": "running",
                  "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (args.output / "queue.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        started = time.perf_counter()
        with (args.output / f"case_{index}.log").open("w") as log:
            row["exit_code"] = subprocess.call(argv, stdout=log, stderr=subprocess.STDOUT)
        row.update(status="complete" if row["exit_code"] == 0 else "failed",
                   queue_wait_and_command_seconds=time.perf_counter() - started,
                   finished_utc=datetime.now(timezone.utc).isoformat())
        try:
            measurement = json.loads((monitor_root / "monitor.json").read_text())
            row["measured_command_seconds"] = measurement["command_wall_seconds"]
            row["queue_wait_and_wrapper_seconds"] = (
                row["queue_wait_and_command_seconds"] - row["measured_command_seconds"])
        except (OSError, ValueError, KeyError) as error:
            row["measurement_error"] = repr(error)
        (args.output / "queue.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    report["status"] = "complete" if all(row["exit_code"] == 0 for row in rows) else "failed"
    (args.output / "queue.json").write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["status"] == "complete" else 1)


if __name__ == "__main__":
    main()
