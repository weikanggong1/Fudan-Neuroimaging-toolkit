"""原始单T1空目录整例包装：含CLI启动、加载/传输/读写及同期进程树显存。

仅运行FNIT候选，不产生官方参考。全部输入路径由具名参数显式提供。
输出为原始recon-all目录、CLI日志、源码/输入/资源/程序哈希和benchmark.json。
失败保留部分报告并返回非零，不重试、不补跑、不读取官方产物。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

from fnit.recon_all.profiling import ProcessTreeDeviceSampler


def save_report(path: Path, report: dict) -> None:
    """原子写机器报告，保留最近完整检查点，避免外部中断留下半个JSON。"""
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(path)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t1", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--native-bin-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--hemisphere-workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--defects-backend", choices=("native", "torch"), default="torch")
    parser.add_argument("--wm-edit-backend", choices=("native", "torch-hybrid"), default="native")
    parser.add_argument("--wm-backend", choices=("native", "torch", "torch-optimized"), default="native")
    parser.add_argument("--gca-inverse-backend", choices=("cpu", "torch"), default="cpu")
    parser.add_argument("--gca-candidate-chunk", type=int, default=64)
    parser.add_argument("--gca-execution", choices=("in-process", "isolated"), default="in-process")
    parser.add_argument("--fill-backend", choices=("python", "numba", "torch-numba"), default="python")
    parser.add_argument("--sphere-normals-backend", choices=("numba", "torch"), default="numba")
    parser.add_argument("--native-optimizations", choices=("auto", "original", "torch"), default="auto")
    parser.add_argument("--code-version", required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError("output-root must not exist: " + str(args.output_root))
    if not args.t1.is_file():
        raise FileNotFoundError(args.t1)
    args.output_root.mkdir(parents=True)
    import fnit
    source = Path(fnit.__file__).parent
    setup_tick = time.perf_counter()
    report = {
        "scope": "raw T1 + empty subject directory, CLI end-to-end including process startup",
        "code_version": args.code_version, "source_sha256": {
            str(path.relative_to(source)): sha(path) for path in source.rglob("*.py")},
        "script_sha256": sha(Path(__file__)), "input_sha256": sha(args.t1),
        "resource_sha256": {
            kind: {str(path.relative_to(root)): sha(path) for path in root.rglob("*")
                   if path.is_file() and "license" not in path.name.lower()}
            for kind, root in (("weights", args.weights_dir), ("assets", args.assets_dir))},
        "native_program_sha256": {path.name: sha(path) for path in args.native_bin_dir.iterdir()
                                  if path.is_file() and os.access(path, os.X_OK)},
        "host": platform.node(), "cpu": next((row.split(":", 1)[1].strip()
            for row in Path("/proc/cpuinfo").read_text().splitlines() if row.startswith("model name")), None),
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "threads": args.threads,
        "device": args.device, "defects_backend": args.defects_backend,
        "sphere_normals_backend": args.sphere_normals_backend,
        "wm_edit_backend": args.wm_edit_backend,
        "wm_backend": args.wm_backend,
        "gca_inverse_backend": args.gca_inverse_backend,
        "gca_candidate_chunk": args.gca_candidate_chunk,
        "gca_execution": args.gca_execution,
        "fill_backend": args.fill_backend,
        "environment": {key: os.environ.get(key) for key in (
            "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "PYTORCH_NO_CUDA_MEMORY_CACHING")},
        "execution": "pending", "output_completeness": "not_assessed",
        "strict_reproduction": "not_assessed", "optimization_regression": "not_assessed",
        "overall_metric_equivalence": "not_assessed", "isolation_validation": "not_verified",
    }
    report["harness_hash_validation_seconds"] = time.perf_counter() - setup_tick
    report_path = args.output_root / "benchmark.json"
    save_report(report_path, report)
    subject = args.output_root / "subject"
    # 仅benchmark子进程启用故障栈与无缓冲日志，硬信号失败仍保留定位信息。
    command = [sys.executable, "-X", "faulthandler", "-u", "-m", "fnit.recon_all.native_free", str(args.t1), str(subject),
               "--weights-dir", str(args.weights_dir), "--assets-dir", str(args.assets_dir),
               "--native-bin-dir", str(args.native_bin_dir), "--device", args.device,
               "--threads", str(args.threads), "--hemisphere-workers", str(args.hemisphere_workers),
               "--native-optimizations", args.native_optimizations,
               "--defects-backend", args.defects_backend, "--profile-stages"]
    command += ["--sphere-normals-backend", args.sphere_normals_backend]
    command += ["--wm-edit-backend", args.wm_edit_backend]
    command += ["--wm-backend", args.wm_backend, "--gca-inverse-backend", args.gca_inverse_backend,
                "--gca-candidate-chunk", str(args.gca_candidate_chunk), "--gca-execution", args.gca_execution,
                "--fill-backend", args.fill_backend]
    report["cli_command"] = command
    tick = time.perf_counter()
    with (args.output_root / "run.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        sampler = ProcessTreeDeviceSampler(device=args.device, parent_pid=process.pid, interval=.5)
        report.update(execution="running", cli_pid=process.pid, harness_pid=os.getpid(),
                      periodic_checkpoint_seconds=30)

        def interrupted(signum, frame):
            # 记录自身收到的信号；不重跑、不更改算法、不终止其他任务。
            report.update(execution="interrupted", interruption_signal=signum,
                          cli_elapsed_seconds_partial=time.perf_counter() - tick,
                          exit_code=process.poll(), process_memory=sampler.report())
            save_report(report_path, report)
            raise SystemExit(128 + signum)

        previous_handlers = {signum: signal.signal(signum, interrupted)
                             for signum in (signal.SIGHUP, signal.SIGTERM)}
        checkpoint = tick
        save_report(report_path, report)
        try:
            while process.poll() is None:
                sampler.sample_if_due()
                now = time.perf_counter()
                if now - checkpoint >= 30:
                    report.update(cli_elapsed_seconds_partial=now - tick,
                                  process_memory=sampler.report())
                    save_report(report_path, report)
                    checkpoint = now
                time.sleep(.05)
            report["cli_wall_seconds"] = time.perf_counter() - tick
            report["exit_code"] = process.returncode
            report["termination_signal"] = (signal.Signals(-process.returncode).name
                                             if process.returncode < 0 else None)
            report["process_memory"] = sampler.report()
        except BaseException as error:
            report.update(execution="interrupted", harness_error=repr(error),
                          cli_elapsed_seconds_partial=time.perf_counter() - tick,
                          process_memory=sampler.report())
            save_report(report_path, report)
            raise
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
    runtime_path = subject / "fnit-native-free-run.json"
    if runtime_path.exists():
        runtime = json.loads(runtime_path.read_text())
        report["runtime_status"] = runtime.get("status")
        report["api_wall_seconds"] = runtime.get("total_seconds")
        report["output_completeness"] = runtime.get("output_validation", "not_assessed")
        report["grid_quality"] = runtime.get("mesh_validation", "not_assessed")
        report["stages"] = [{key: value for key, value in row.items() if key != "gpu_memory"}
                            for row in runtime.get("stages", [])]
    report["execution"] = "complete" if process.returncode == 0 else "failed"
    report["full_harness_seconds"] = time.perf_counter() - setup_tick
    save_report(report_path, report)
    return process.returncode


if __name__ == "__main__":
    sys.exit(main())
