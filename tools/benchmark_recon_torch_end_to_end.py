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
    setup_tick = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t1", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--native-bin-dir", type=Path)
    parser.add_argument("--backend", choices=("native", "python"), default="native")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--n4-backend", choices=("native", "torch", "simpleitk"), default="native")
    parser.add_argument("--n4-execution", choices=("in-process", "isolated"), default="in-process")
    parser.add_argument("--input-execution", choices=("serial", "parallel"), default="serial")
    parser.add_argument("--normalization-controls-backend", choices=("cpu", "torch"), default="cpu")
    parser.add_argument("--normalization-initial-bias-backend", choices=("cpu", "torch"), default="cpu")
    parser.add_argument("--inflate-backend", choices=("native", "torch"), default="native")
    parser.add_argument("--sphere-finish-backend", choices=("cpu", "torch"), default="cpu")
    parser.add_argument("--surface-placement-backend", choices=("native", "python"), default="native")
    parser.add_argument("--topology-backend", choices=("native", "python", "python-ga"), default="native")
    parser.add_argument("--annotation-gibbs-backend", choices=("python", "numba"), default="python")
    parser.add_argument("--remesh-scalar-storage", choices=("numpy", "python"), default="numpy")
    parser.add_argument("--mni-execution", choices=("in-process", "parallel-late"), default="in-process")
    parser.add_argument("--hemisphere-workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--hemisphere-execution", choices=("groups", "persistent"), default="groups")
    parser.add_argument("--defects-backend", choices=("native", "torch"), default="torch")
    parser.add_argument("--wm-edit-backend", choices=("native", "python", "torch-hybrid"), default="native")
    parser.add_argument("--wm-backend", choices=("native", "torch", "torch-optimized"), default="native")
    parser.add_argument("--wm-execution", choices=("in-process", "isolated"), default="in-process")
    parser.add_argument("--gca-inverse-backend", choices=("cpu", "torch"), default="cpu")
    parser.add_argument("--gca-candidate-chunk", type=int)
    parser.add_argument("--gca-reuse-atlas-context", action="store_true")
    parser.add_argument("--gca-execution", choices=("in-process", "isolated"), default="in-process")
    parser.add_argument("--fill-backend", choices=("python", "numba", "torch-numba"), default="python")
    parser.add_argument("--sphere-normals-backend", choices=("numba", "torch"), default="numba")
    parser.add_argument("--topology-distance-backend", choices=("serial", "parallel", "gpu_defect_signed_distance"), default="serial")
    parser.add_argument("--topology-state-backend", choices=("legacy", "packed"), default="legacy")
    parser.add_argument("--native-optimizations", choices=("auto", "original", "torch"), default="auto")
    parser.add_argument("--code-version", required=True)
    args = parser.parse_args()
    if args.gca_candidate_chunk is None:
        args.gca_candidate_chunk = 1024 if args.backend == "python" else 64
    if args.backend == "python":
        from fnit.recon_all.python_profile import python_profile_options
        for key, value in python_profile_options(device=args.device, threads=args.threads,
                n4_backend="simpleitk" if args.n4_backend == "native" else args.n4_backend,
                n4_execution=args.n4_execution,
                input_execution=args.input_execution,
                topology_backend="python" if args.topology_backend == "native" else args.topology_backend,
                hemisphere_execution=args.hemisphere_execution,
                gca_reuse_atlas_context=args.gca_reuse_atlas_context,
                gca_candidate_chunk=args.gca_candidate_chunk,
                topology_distance_backend=args.topology_distance_backend,
                topology_state_backend=args.topology_state_backend).items():
            setattr(args, key, value)
    elif args.native_bin_dir is None:
        parser.error("--native-bin-dir is required for the mixed benchmark control")
    from fnit.recon_all.native_free import (_normalization_controls_options, _normalization_initial_bias_options,
                                          _validate_inflate_backend, _validate_mni_execution,
                                          _validate_sphere_finish_backend, _validate_annotation_gibbs_backend,
                                          _validate_remesh_scalar_storage, _validate_surface_placement_backend)
    _validate_annotation_gibbs_backend(args.annotation_gibbs_backend)
    _validate_remesh_scalar_storage(args.remesh_scalar_storage)
    _validate_surface_placement_backend(args.surface_placement_backend)
    _normalization_controls_options(args.normalization_controls_backend, args.device)
    _normalization_initial_bias_options(args.normalization_initial_bias_backend, args.device)
    _validate_inflate_backend(args.inflate_backend, args.device, args.hemisphere_workers)
    _validate_sphere_finish_backend(args.sphere_finish_backend, args.device, args.hemisphere_workers)
    _validate_mni_execution(args.mni_execution, args.device, args.threads)
    from fnit.recon_all.input_n4_chain import validate_n4_execution, require_n4_runtime_dependencies
    validate_n4_execution(n4_backend=args.n4_backend, n4_execution=args.n4_execution, device=args.device)
    from fnit.recon_all.native_free import _validate_input_execution
    _validate_input_execution(args.input_execution, n4_backend=args.n4_backend,
                             n4_execution=args.n4_execution, device=args.device, threads=args.threads)
    n4_dependencies = require_n4_runtime_dependencies(n4_backend=args.n4_backend)
    if args.output_root.exists():
        raise FileExistsError("output-root must not exist: " + str(args.output_root))
    if not args.t1.is_file():
        raise FileNotFoundError(args.t1)
    args.output_root.mkdir(parents=True)
    import fnit
    source = Path(fnit.__file__).parent
    hash_tick = time.perf_counter()
    report = {
        "scope": "raw T1 + empty subject directory, CLI end-to-end including process startup",
        "full_harness_timing_version": "main-entry-v2",
        "full_harness_scope": "main entry including argument parsing, input/preflight validation, source/resource hashing, CLI startup/execution/output IO and prior report writes; excludes parent interpreter initial imports and current final metadata publication",
        "cli_wall_scope": "child interpreter startup, validation, model loading, transfers, compute, output IO and process exit",
        "code_version": args.code_version, "source_sha256": {
            str(path.relative_to(source)): sha(path) for path in source.rglob("*.py")},
        "script_sha256": sha(Path(__file__)), "input_sha256": sha(args.t1),
        "resource_sha256": {
            kind: {str(path.relative_to(root)): sha(path) for path in root.rglob("*")
                   if path.is_file() and "license" not in path.name.lower()}
            for kind, root in (("weights", args.weights_dir), ("assets", args.assets_dir))},
        "native_program_sha256": ({path.name: sha(path) for path in args.native_bin_dir.iterdir()
                                  if path.is_file() and os.access(path, os.X_OK)} if args.backend != "python" else {}),
        "host": platform.node(), "cpu": next((row.split(":", 1)[1].strip()
            for row in Path("/proc/cpuinfo").read_text().splitlines() if row.startswith("model name")), None),
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "threads": args.threads,
        "backend": args.backend, "topology_backend": args.topology_backend,
        "topology_distance_backend": args.topology_distance_backend,
        "topology_state_backend": args.topology_state_backend,
        "device": args.device, "defects_backend": args.defects_backend,
        "sphere_normals_backend": args.sphere_normals_backend,
        "wm_edit_backend": args.wm_edit_backend,
        "wm_backend": args.wm_backend,
        "n4_backend": args.n4_backend, "n4_execution": args.n4_execution,
        "n4_runtime_dependencies": n4_dependencies,
        "input_execution": args.input_execution,
        "normalization_controls_backend": args.normalization_controls_backend,
        "normalization_initial_bias_backend": args.normalization_initial_bias_backend,
        "inflate_backend": args.inflate_backend,
        "sphere_finish_backend": args.sphere_finish_backend,
        "surface_placement_backend": args.surface_placement_backend,
        "annotation_gibbs_backend": args.annotation_gibbs_backend,
        "remesh_scalar_storage": args.remesh_scalar_storage,
        "mni_execution": args.mni_execution,
        "wm_execution": args.wm_execution,
        "gca_inverse_backend": args.gca_inverse_backend,
        "gca_candidate_chunk": args.gca_candidate_chunk,
        "gca_reuse_atlas_context": args.gca_reuse_atlas_context,
        "gca_execution": args.gca_execution,
        "hemisphere_execution": args.hemisphere_execution,
        "fill_backend": args.fill_backend,
        "environment": {key: os.environ.get(key) for key in (
            "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "PYTORCH_NO_CUDA_MEMORY_CACHING")},
        "execution": "pending", "output_completeness": "not_assessed",
        "strict_reproduction": "not_assessed", "optimization_regression": "not_assessed",
        "overall_metric_equivalence": "not_assessed", "isolation_validation": "not_verified",
    }
    report["harness_hash_validation_seconds"] = time.perf_counter() - hash_tick
    report_path = args.output_root / "benchmark.json"
    save_report(report_path, report)
    subject = args.output_root / "subject"
    # 仅benchmark子进程启用故障栈与无缓冲日志，硬信号失败仍保留定位信息。
    command = [sys.executable, "-X", "faulthandler", "-u", "-m", "fnit.recon_all.native_free", str(args.t1), str(subject),
               "--weights-dir", str(args.weights_dir), "--assets-dir", str(args.assets_dir),
               "--backend", args.backend, "--device", args.device,
               "--threads", str(args.threads), "--hemisphere-workers", str(args.hemisphere_workers),
               "--hemisphere-execution", args.hemisphere_execution,
               "--native-optimizations", args.native_optimizations,
               "--defects-backend", args.defects_backend, "--profile-stages"]
    if args.backend != "python":
        command += ["--native-bin-dir", str(args.native_bin_dir)]
    command += ["--sphere-normals-backend", args.sphere_normals_backend]
    command += ["--wm-edit-backend", args.wm_edit_backend]
    command += ["--wm-backend", args.wm_backend, "--gca-inverse-backend", args.gca_inverse_backend,
                "--gca-candidate-chunk", str(args.gca_candidate_chunk), "--gca-execution", args.gca_execution,
                "--fill-backend", args.fill_backend]
    if args.gca_reuse_atlas_context:
        command += ["--gca-reuse-atlas-context"]
    command += ["--n4-backend", args.n4_backend, "--n4-execution", args.n4_execution]
    command += ["--input-execution", args.input_execution]
    command += ["--normalization-controls-backend", args.normalization_controls_backend]
    command += ["--normalization-initial-bias-backend", args.normalization_initial_bias_backend]
    command += ["--inflate-backend", args.inflate_backend]
    command += ["--sphere-finish-backend", args.sphere_finish_backend]
    command += ["--surface-placement-backend", args.surface_placement_backend]
    command += ["--topology-backend", args.topology_backend]
    command += ["--topology-distance-backend", args.topology_distance_backend,
                "--topology-state-backend", args.topology_state_backend]
    command += ["--annotation-gibbs-backend", args.annotation_gibbs_backend]
    command += ["--remesh-scalar-storage", args.remesh_scalar_storage]
    command += ["--mni-execution", args.mni_execution]
    command += ["--wm-execution", args.wm_execution]
    report["cli_command"] = command
    child_environment = os.environ.copy()
    if args.backend == "python":
        audit_directory = args.output_root / "execution-audit"
        child_environment["FNIT_RECON_AUDIT_DIR"] = str(audit_directory)
        audit_module = Path(__file__).parent / "recon_python_execution_audit"
        child_environment["PYTHONPATH"] = os.pathsep.join(
            [str(audit_module), child_environment.get("PYTHONPATH", "")])
        report["execution_audit"] = {"directory": str(audit_directory),
            "scope": "Python audited process-launch events in CLI and inherited child interpreters; not OS file/library isolation",
            "source_sha256": sha(audit_module / "sitecustomize.py"), "status": "running"}
    tick = time.perf_counter()
    with (args.output_root / "run.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=child_environment)
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
    if args.backend == "python":
        rows = [json.loads(line) for path in audit_directory.glob("*.jsonl")
                for line in path.read_text().splitlines() if line.strip()]
        report["execution_audit"].update(status="complete", processes=sorted({row["pid"] for row in rows}),
            events=len(rows), original_program_attempts=[row for row in rows if row.get("blocked")],
            passed=bool(rows) and not any(row.get("blocked") for row in rows))
    report["full_harness_seconds"] = time.perf_counter() - setup_tick
    save_report(report_path, report)
    return process.returncode


if __name__ == "__main__":
    sys.exit(main())
