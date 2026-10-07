"""完整真实输入的 MCFLIRT/运动包装/BBR 单进程 adapter。

本文件仅属于独立 benchmark，不进入生产依赖。协调者用相同 CPU 亲和性、
1/8 线程预算与计时锁顺序运行官方/FNIT。输入清单保留在服务器私有目录；
report.safe.json 只记录函数、哈希、硬件配置与聚合计时，不写原始路径。
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


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def parse_cpus(value):
    result = set()
    for token in value.split(","):
        if "-" in token:
            first, last = map(int, token.split("-", 1))
            result.update(range(first, last + 1))
        else:
            result.add(int(token))
    return result


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--case-json", type=Path, required=True)
    parser.add_argument("--case", required=True, help="清单中的通用案例标签，不含被试 ID")
    parser.add_argument("--function", choices=("mcflirt", "motion_wrapper", "bbr"), required=True)
    parser.add_argument("--backend", choices=("fnit", "official"), required=True)
    parser.add_argument("--source", type=Path, required=True, help="冻结工作树 src 目录")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, choices=(1, 8), required=True)
    parser.add_argument("--cpu-list", required=True)
    parser.add_argument("--cpu-lock", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warm-repeats", type=int, default=0)
    parser.add_argument("--native-trace", action="store_true")
    parser.add_argument("--reference-mode", choices=("external", "middle"), default="external")
    parser.add_argument("--stages", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--stage-iterations", type=int, nargs=3, default=(1, 1, 1))
    parser.add_argument("--interpolation", choices=("linear", "spline"), default="spline")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--rms", action="store_true")
    parser.add_argument("--execution", choices=("batched", "reference"), default="batched")
    parser.add_argument("--candidate-batch-size", type=int, default=128)
    parser.add_argument("--no-grid-search", action="store_true")
    parser.add_argument("--initialization", choices=("path", "array", "automatic"), default="path")
    parser.add_argument("--input-mode", choices=("path", "image"), default="path")
    parser.add_argument("--wrapper-mask", action="store_true")
    return parser.parse_args(argv)


def run(args):
    # 设置发生在 NumPy/PyTorch/Numba 导入之前，不改变共享 Conda prefix。
    cpu_set = parse_cpus(args.cpu_list)
    os.sched_setaffinity(0, cpu_set)
    if os.sched_getaffinity(0) != cpu_set:
        raise RuntimeError("CPU affinity differs from declared budget")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.output_dir.exists():
        raise FileExistsError("Choose a new output directory")
    configuration = json.loads(args.case_json.read_text())
    case = configuration[args.case]
    args.output_dir.mkdir(parents=True)
    prefix = args.output_dir / "result"
    # 私有计时锁由协调者统一分配；禁止抢占其他任务的锁或 CPU 组。
    import fcntl
    args.cpu_lock.parent.mkdir(parents=True, exist_ok=True)
    with args.cpu_lock.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        records = []
        if args.warm_repeats < 0:
            raise ValueError("warm_repeats must be nonnegative")
        for repeat in range(args.warm_repeats + 1):
            run_dir = args.output_dir / f"repeat_{repeat}"
            run_dir.mkdir()
            record = _execute(args, case, configuration, run_dir / "result")
            record["repeat"] = repeat
            record["call_type"] = ("first_fnit_call" if repeat == 0 else "warm_fnit_api_call") if args.backend == "fnit" else "fresh_native_process"
            records.append(record)
        report = dict(records[0])
        report["repeat_records"] = records
    report.update(schema_version=1, source_revision=args.source_revision,
                  function=args.function, backend=args.backend, case=args.case,
                  cpu_threads=args.threads, cpu_affinity=sorted(cpu_set),
                  full_input_required=True, slice_timing=False,
                  adapter_sha256=digest(__file__))
    (args.output_dir / "report.safe.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def _execute(args, case, configuration, prefix):
    import_start = time.perf_counter()
    import nibabel as nib
    import numpy as np
    import torch
    torch.set_num_threads(args.threads)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    sys.path.insert(0, str(args.source.resolve()))
    # 官方模式也只在私有验证子进程里调用外部软件。
    if args.function in ("mcflirt", "motion_wrapper"):
        from fnit.mcflirt import TorchMCFLIRT
        from fnit.fmri.motion import estimate_motion
    else:
        from fnit.fmri.bbr import register_bbr
    import_seconds = time.perf_counter() - import_start
    if args.function == "motion_wrapper" and args.rms:
        raise ValueError("The compatibility wrapper has no RMS save option")
    if args.estimate_only and args.rms:
        raise ValueError("RMS file outputs require normal MCFLIRT output")
    if args.function == "bbr" and args.input_mode == "image" and args.initialization == "automatic":
        raise ValueError("Automatic BBR initialization requires paths")
    inputs = {key: digest(case[key]) for key in
              (("bold", "reference") if args.function != "bbr" else ("epi", "t1", "wmseg", "init"))
              if key in case}
    source_hashes = {str(p.relative_to(args.source)): digest(p)
                     for folder in ("fnit/mcflirt", "fnit/flirt", "fnit/fmri")
                     for p in (args.source / folder).glob("*.py")}
    raw = nib.load(case["bold"]) if args.function != "bbr" else None
    shape = list(raw.shape) if raw is not None else list(nib.load(case["epi"]).shape)
    if raw is not None and (raw.ndim != 4 or raw.shape[3] not in (180, 490)):
        raise ValueError("Use the complete verified 180/490-frame BOLD")
    metadata = {"input_sha256": inputs, "source_sha256": source_hashes,
                "input_shape": shape, "import_seconds": import_seconds,
                "software": {"python": sys.version.split()[0], "torch": torch.__version__,
                             "numpy": np.__version__, "nibabel": nib.__version__}}
    if args.backend == "official":
        return _official(args, case, configuration, prefix, metadata)
    loading_start = time.perf_counter()
    if args.function != "bbr":
        reference = case["reference"] if args.reference_mode == "external" else None
        bold = nib.load(case["bold"]) if args.input_mode == "image" else case["bold"]
        if args.input_mode == "image" and reference is not None:
            reference = nib.load(reference)
    else:
        epi, t1, wm = (case[name] for name in ("epi", "t1", "wmseg"))
        if args.input_mode == "image":
            epi, t1, wm = map(nib.load, (epi, t1, wm))
        init = (None if args.initialization == "automatic" else
                np.loadtxt(case["init"]) if args.initialization == "array" else case["init"])
    loading_seconds = time.perf_counter() - loading_start
    api_start = time.perf_counter()
    if args.function == "mcflirt":
        result = TorchMCFLIRT(device=args.device).run(
            bold, reference, stages=args.stages, stage_iterations=tuple(args.stage_iterations),
            resample=not args.estimate_only, interpolation=args.interpolation,
            output=None if args.estimate_only else str(prefix),
            mats=not args.estimate_only, plots=not args.estimate_only,
            rmsrel=args.rms, rmsabs=args.rms,
        )
    elif args.function == "motion_wrapper":
        if reference is None:
            raise ValueError("The compatibility wrapper requires an explicit reference")
        result = estimate_motion(bold, reference, device=args.device,
            iterations=tuple(args.stage_iterations), resample=not args.estimate_only,
            mask=case.get("brain_mask") if args.wrapper_mask else None)
    else:
        result = register_bbr(epi, t1, wm, init=init, device=args.device,
            grid_search=not args.no_grid_search, execution=args.execution,
            candidate_batch_size=args.candidate_batch_size)
    if str(args.device).startswith("cuda"):
        torch.cuda.synchronize(torch.device(args.device))
    api_seconds = time.perf_counter() - api_start
    save_start = time.perf_counter()
    if args.function != "bbr":
        matrices = result.matrices if args.function == "mcflirt" else result.fsl_matrices
        if args.function == "motion_wrapper":
            matrix_dir = Path(str(prefix) + ".mat")
            matrix_dir.mkdir()
            for frame, matrix in enumerate(matrices):
                np.savetxt(matrix_dir / f"MAT_{frame:04d}", matrix, fmt="%.12g")
            np.savetxt(str(prefix) + ".par", result.parameters, fmt="%.9g")
            if result.corrected is not None:
                nib.save(result.corrected, str(prefix) + ".nii.gz")
        metadata.update(cost_evaluations=getattr(result, "cost_evaluations", None),
                        parameter_convention="mcflirt" if args.function == "mcflirt" else "legacy_pull",
                        full_frame_count=len(matrices))
        # MCFLIRT 原接口直接保存含 RMS 的正常输出，包装器另存其返回结果。
    else:
        result.save(output=str(prefix) + ".nii.gz", omat=str(prefix) + ".mat")
        metadata.update(cost_evaluations=result.cost_evaluations,
                        phase_timings=result.phase_timings,
                        phase_cost_evaluations=result.phase_cost_evaluations,
                        boundary_points=result.boundary_points,
                        initial_cost=result.initial_cost, final_cost=result.final_cost)
    save_seconds = time.perf_counter() - save_start
    # 未舍入证据在正式 API/文件输出钟之外保存。
    evidence_start = time.perf_counter()
    np.save(str(prefix) + ".matrices.npy", matrices if args.function != "bbr" else result.matrix)
    if args.function != "bbr":
        np.save(str(prefix) + ".parameters.npy", result.parameters)
    evidence_seconds = time.perf_counter() - evidence_start
    metadata.update(unrounded_evidence_save_seconds=evidence_seconds,
                    api_includes_output_writes=args.function == "mcflirt" and not args.estimate_only, status="complete", api_seconds=api_seconds, save_seconds=save_seconds,
                    input_object_load_seconds=loading_seconds,
                    measured_application_seconds=import_seconds + loading_seconds + api_seconds + save_seconds,
                    timing_scope="Import + header loading + complete API + normal output writes. Python interpreter startup, input/source hashes, .npy evidence writes and comparison excluded. MCFLIRT output writes are included in API time.",
                    source_unchanged=all(digest(args.source / p) == h for p, h in source_hashes.items()))
    return metadata


def _official(args, case, configuration, prefix, metadata):
    fsl = Path(configuration["fsl_root"])
    executable = fsl / "bin" / ("flirt" if args.function == "bbr" else "mcflirt")
    if args.function != "bbr":
        if args.estimate_only or tuple(args.stage_iterations) != (1, 1, 1):
            raise ValueError("Official adapter requires full native outputs and default iteration schedule")
        command = [str(executable), "-in", case["bold"], "-out", str(prefix),
                   "-mats", "-plots", "-stages", str(args.stages)]
        if args.reference_mode == "external":
            command += ["-reffile", case["reference"]]
        if args.interpolation == "spline":
            command += ["-spline_final"]
        if args.rms:
            command += ["-rmsrel", "-rmsabs"]
    else:
        if args.no_grid_search or args.initialization != "path":
            raise ValueError("Official BBR default pair requires fixed init and complete bbr.sch")
        command = [str(executable), "-in", case["epi"], "-ref", case["t1"],
                   "-wmseg", case["wmseg"], "-init", case["init"], "-dof", "6", "-cost", "bbr",
                   "-schedule", str(fsl / "etc/flirtsch/bbr.sch"), "-out", str(prefix),
                   "-omat", str(prefix) + ".mat"]
    environment = dict(os.environ, FSLDIR=str(fsl), FSLOUTPUTTYPE="NIFTI_GZ")
    environment["PATH"] = str(fsl / "bin") + ":" + environment.get("PATH", "")
    environment["LD_LIBRARY_PATH"] = str(fsl / "lib") + ":" + environment.get("LD_LIBRARY_PATH", "")
    binary_sha = digest(executable)
    if args.native_trace:
        command = ["strace", "-f", "-e", "trace=execve,exit_group", "-o", str(prefix.parent / "native_exec.private.trace")] + command
    with (prefix.parent / "native.private.log").open("wb") as stream:
        start = time.perf_counter()
        process = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        seconds = time.perf_counter() - start
    metadata.update(status="complete" if process.returncode == 0 else "native_exit_requires_diagnostic",
                    native_exit_code=process.returncode, measured_application_seconds=seconds,
                    native_trace_enabled=args.native_trace,
                    native_executable_sha256=binary_sha, native_executable_unchanged=digest(executable) == binary_sha,
                    fsl_version=(fsl / "etc/fslversion").read_text().strip(),
                    timing_scope="Native subprocess startup, full read/decompression, full algorithm, native artifacts and exit; hashes/comparison excluded.")
    # 包装器异常退出不可自动转成通过；协调者检查真实 C++ child exit 和完整产物。
    return metadata


if __name__ == "__main__":
    report = run(arguments())
    print(json.dumps({key: report.get(key) for key in
                     ("status", "function", "backend", "case", "measured_application_seconds")}))
