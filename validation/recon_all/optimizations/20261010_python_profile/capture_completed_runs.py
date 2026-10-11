"""只读冻结已退出的 Python 整例收据，保留原字节和实际 SHA。

run_root 是本轮 benchmark 目录；archive 为新 tar 文件，或 - 写到标准输出。
只含 JSON 收据、执行审计与哈希清单，不含影像、坐标文件、模型或许可证。
候选必须已有 exit 收据及非 running 运行状态；失败原样保留，不补跑。
输出 CAPTURE.json 和所有所选原字节；收据在采集时变化则报错。无官方命令。
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import time


def capture(*, run_root: Path, archive: str, collector_sha256: str | None = None) -> dict:
    """stdin执行时显式给出采集器SHA；普通文件执行则从自身读取。"""
    script = Path(__file__)
    if script.is_file():
        actual_sha = hashlib.sha256(script.read_bytes()).hexdigest()
        if collector_sha256 is not None and collector_sha256 != actual_sha:
            raise ValueError("collector source SHA differs")
        collector_sha256 = actual_sha
    if collector_sha256 is None or len(collector_sha256) != 64 \
            or any(char not in '0123456789abcdef' for char in collector_sha256):
        raise ValueError("stdin execution requires the actual collector SHA-256")
    run_root = run_root.resolve(strict=True)
    selected = set()
    cases = {}
    for case in ("sub06", "sub07"):
        exit_path = run_root/f"{case}-exit.json"
        benchmark = run_root/f"{case}-candidate/benchmark.json"
        runtime = run_root/f"{case}-candidate/subject/fnit-native-free-run.json"
        for path in (exit_path, benchmark, runtime):
            if not path.is_file():
                raise FileNotFoundError(path)
        exit_report = json.loads(exit_path.read_bytes())
        bench = json.loads(benchmark.read_bytes())
        run = json.loads(runtime.read_bytes())
        if run.get("status") not in ("complete", "failed"):
            raise RuntimeError(f"{case} has not reached a final state")
        if bench.get("exit_code") != exit_report.get("exit_code"):
            raise RuntimeError(f"{case} exit receipts disagree")
        if bench.get("runtime_status") != run["status"]:
            raise RuntimeError(f"{case} runtime receipts disagree")
        cases[case] = {
            "execution_status": run["status"],
            "exit_code": bench["exit_code"],
            "failed_stage": run.get("failed_stage"),
            "full_workflow_complete": run["status"] == "complete" and bench["exit_code"] == 0,
            "api_wall_seconds": bench.get("api_wall_seconds"),
            "cli_wall_seconds": bench.get("cli_wall_seconds"),
            "full_harness_seconds": bench.get("full_harness_seconds"),
            "code_version": bench.get("code_version"),
            "input_sha256": bench.get("input_sha256"),
            "original_stage_program_sha256": bench.get("native_program_sha256"),
            "overall_metric_equivalence": bench.get("overall_metric_equivalence"),
        }
        selected.update((exit_path, benchmark, runtime))
        for path in run_root.glob(f"{case}-*.json"):
            if "prefix" not in path.name:
                selected.add(path)
        scripts = runtime.parent/"scripts"
        for pattern in ("*.hemisphere-group.json", "*.startup-*.report.json"):
            selected.update(scripts.glob(pattern))
        for path in (run_root/f"{case}-candidate").rglob("*.jsonl"):
            if "audit" in str(path.relative_to(run_root)):
                selected.add(path)
    for name in ("source_receipt.json", "launcher_receipt.json", "import_preflight.json"):
        path = run_root/name
        if path.is_file():
            selected.add(path)
    captured = {}
    bindings = {}
    for path in sorted(selected):
        data = path.read_bytes()
        relative = str(path.relative_to(run_root))
        captured[relative] = data
        bindings[relative] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    # 结束后重新核对原件，避免把运行中的不同检查点拼成同一版本。
    for path in selected:
        relative = str(path.relative_to(run_root))
        if hashlib.sha256(path.read_bytes()).hexdigest() != bindings[relative]["sha256"]:
            raise RuntimeError(f"receipt changed during capture: {relative}")
    manifest = {"scope": "completed raw T1 execution receipts; failure retained",
                "captured_unix": time.time(), "collector_sha256": collector_sha256,
                "producer_reports_unmodified": True, "cases": cases, "files": bindings}
    captured["CAPTURE.json"] = (json.dumps(manifest, indent=2)+"\n").encode()
    if archive == "-":
        stream = tarfile.open(fileobj=sys.stdout.buffer, mode="w|")
    else:
        path = Path(archive)
        if path.exists():
            raise FileExistsError(path)
        stream = tarfile.open(path, mode="w")
    with stream:
        for name, data in captured.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o600
            stream.addfile(info, io.BytesIO(data))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--collector-sha256", help="stdin方式执行时显式传实际脚本SHA-256")
    args = parser.parse_args()
    capture(run_root=args.run_root, archive=args.archive, collector_sha256=args.collector_sha256)
