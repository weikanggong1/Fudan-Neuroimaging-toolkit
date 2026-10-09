"""末尾 MNI 非线性与 CPU 网格验证并行；复用现有 fresh-exec worker。"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, result: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(path)


def run_mni_stage(*, subject: str, weights: str, assets: str,
                  warp_binaries: list[str], device: str, threads: int) -> dict:
    """复用完整 MNI GPU 链，返回输入/资源/源码和输出 SHA，不改变算法。

    subject 含 conform orig、MNI affine/crop；weights/assets 为已声明资源；
    三个 warp_binaries 为兼容接口（GPU 后端不执行）。device 显式 cuda:N，
    threads 控制已有实现。输出原有毫米前向/逆向位移与检查图，空间和 dtype
    不变。输入或计算失败抛异常；所有哈希及写出包含在阶段墙钟内。
    """
    from . import mni_aux_chain, mni_nonlinear_chain
    import torch
    root, weights_path, assets_path = Path(subject), Path(weights), Path(assets)
    transforms = root / "mri/transforms/synthmorph.1.0mm.1.0mm"
    inputs = {"orig": root / "mri/orig.mgz", "crop": transforms / "invol.crop.nii.gz",
              "affine": transforms / "aff.lta", "deform_weight": weights_path / "synthmorph.deform.3.h5",
              "cropped_template": assets_path / mni_aux_chain.TEMPLATE_DIR / "mni152.1.0mm.cropped.nii.gz",
              "full_template": assets_path / mni_aux_chain.TEMPLATE_DIR / "mni152.1.0mm.nii.gz"}
    input_hashes = {name: _sha(path) for name, path in inputs.items()}
    programs = {str(i): {"sha256": _sha(Path(path)), "executed": False}
                for i, path in enumerate(warp_binaries)}
    # Triton launch 的 current device 必须与 tensor target 相同。局部作用域
    # 也覆盖既有逆场 kernel，退出恢复调用者设备，不使用全局 set_device。
    with torch.cuda.device(device):
        result = mni_nonlinear_chain.run_mni_nonlinear_chain(
            subject_dir=root, weights_dir=weights_path, assets_dir=assets_path,
            warp_convert=warp_binaries[0], ca_register=warp_binaries[1], mri_convert=warp_binaries[2],
            device=device, threads=threads, postprocess_backend="gpu")
    source_modules = ("mni_mesh_parallel", "mni_nonlinear_chain", "mni_aux_chain",
                      "mni_warp_sampling", "mni_warp_inverse", "hemisphere_worker", "profiling", "thread_budget")
    result.update(inputs_sha256=input_hashes, declared_programs=programs,
                  outputs_sha256={name: _sha(Path(result[name])) for name in ("forward", "inverse", "check")},
                  source_sha256={name: _sha(Path(importlib.import_module("fnit.recon_all." + name).__file__))
                                 for name in source_modules})
    return result


def run_mni_and_validate(*, subject: str | Path, weights: str | Path, assets: str | Path,
                         warp_binaries: tuple[str | Path, str | Path, str | Path],
                         report_path: str | Path, device: str, threads: int = 4,
                         execution: str = "parallel", profile_stages: bool = False,
                         code_version: str = "FNIT-source-hashes") -> dict:
    """完整 MNI GPU 链与现有 CPU 网格验证，同步后返回两个原始结果及报告。

    subject 必须已完成所有半球目录复制/写出，含八个 orig/white/pial/sphere.reg
    表面与 MNI 前置文件。weights/assets/warp_binaries 同 run_mni_stage。
    report_path 必须是新 JSON，旁边保留 request/worker/log。device 必须显式
    cuda:N。threads 默认4且至少2；parallel 分成父 floor(n/2)、子余数，serial
    两步各用 n 作诊断。profile_stages 默认False；True仅计时前后同步目标GPU。
    code_version 记录冻结源码。并行复用 hemisphere_worker 新exec，继承缓存
    环境，不改父 allocator/CUDA设备/TF32；父线程掩码返回时恢复。
    成功返回 status/mesh_validation/mni_nonlinear/墙钟/线程/同期采样。
    两侧join前不标complete；网格不通过或子失败抛异常，保留失败JSON/log，
    只取消本helper新建子进程树。输入参考结果不被读取；不会复制被试目录。
    """
    import torch
    from .native_free import _validate_meshes
    from .thread_budget import thread_budget, native_thread_environment
    from .profiling import ProcessTreeDeviceSampler, parallel_intervals
    from .hemisphere_parallel import _cancel
    started = time.monotonic()
    target = torch.device(device)
    if target.type != "cuda" or target.index is None:
        raise ValueError("MNI parallel helper requires explicit cuda:N")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 2:
        raise ValueError("threads must be an integer >=2")
    if execution not in ("parallel", "serial") or len(warp_binaries) != 3:
        raise ValueError("execution must be parallel/serial and exactly three declared programs are required")
    root, report_path = Path(subject).resolve(), Path(report_path)
    if report_path.exists():
        raise FileExistsError(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    mesh_hashes = {f"{hemi}.{name}": _sha(root / "surf" / f"{hemi}.{name}")
                   for hemi in ("lh", "rh") for name in ("orig", "white", "pial", "sphere.reg")}
    parent_threads = threads if execution == "serial" else threads // 2
    child_threads = threads if execution == "serial" else threads - parent_threads
    precision_before = {"matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32)}
    allocator_environment = os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING")
    parent_cuda_initialized = torch.cuda.is_initialized()
    if profile_stages and parent_cuda_initialized:
        torch.cuda.synchronize(target)
    sampler = ProcessTreeDeviceSampler(device=device, parent_pid=os.getpid(), interval=.5)
    stop = threading.Event()
    def sample():
        while not stop.is_set():
            sampler.sample_if_due(force=True)
            stop.wait(.5)
    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    report = {"status": "running", "execution": execution, "code_version": code_version,
              "device": device, "thread_budget_total": threads, "parent_threads": parent_threads,
              "child_threads": child_threads, "mesh_inputs_sha256": mesh_hashes,
              "parent_cuda_initialized_at_entry": parent_cuda_initialized,
              "parent_current_device_at_entry": torch.cuda.current_device() if parent_cuda_initialized else None,
              "allocator_environment_at_entry": allocator_environment,
              "parent_allocator_actual": "unchanged; not inferred from initialized environment",
              "precision_at_entry": precision_before, "runtime_helper_sha256": _sha(Path(__file__)),
              "scope": "complete two-stage group including hashing/exec/import/loading/transfers/writes/join; not raw T1 whole"}
    process, log = None, None
    intervals = {}
    try:
        kwargs = dict(subject=str(root), weights=str(Path(weights).resolve()), assets=str(Path(assets).resolve()),
                      warp_binaries=[str(Path(p).resolve()) for p in warp_binaries], device=device, threads=child_threads)
        if execution == "parallel":
            request_path = report_path.with_suffix(".request.json")
            worker_path = report_path.with_suffix(".worker.json")
            log_path = report_path.with_suffix(".worker.log")
            for path in (request_path, worker_path, log_path):
                if path.exists():
                    raise FileExistsError(path)
            request = {"callable": "fnit.recon_all.mni_mesh_parallel:run_mni_stage", "operation": "MNI_nonlinear",
                       "kwargs": kwargs, "device": device, "threads": child_threads,
                       "precision": precision_before, "profile_stages": profile_stages,
                       "allocator_policy": "auto"}
            request_path.write_text(json.dumps(request, indent=2) + "\n")
            environment, report["child_environment"] = native_thread_environment(threads=child_threads)
            bootstrap = ("import sys,runpy,fnit.recon_all; fnit.recon_all.__path__.insert(0,sys.argv[1]); "
                         "sys.argv=sys.argv[2:]; runpy.run_module('fnit.recon_all.hemisphere_worker',run_name='__main__')")
            log = log_path.open("w")
            process = subprocess.Popen([sys.executable, "-X", "faulthandler", "-c", bootstrap,
                                        str(Path(__file__).parent), "fnit.recon_all.hemisphere_worker",
                                        str(request_path), str(worker_path)], env=environment,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            sampler.add_worker(process.pid)
            report["worker_pid"] = process.pid
        else:
            tick = time.monotonic()
            with thread_budget(threads=threads) as budget:
                report["mni_nonlinear"] = run_mni_stage(**kwargs)
                torch.cuda.synchronize(target)
            report["serial_mni_thread_budget"] = budget
            intervals["mni"] = {"started_monotonic": tick, "finished_monotonic": time.monotonic()}
        tick = time.monotonic()
        with thread_budget(threads=parent_threads) as budget:
            report["mesh_validation"] = _validate_meshes(root)
        report["mesh_thread_budget"] = budget
        intervals["mesh"] = {"started_monotonic": tick, "finished_monotonic": time.monotonic()}
        if process is not None:
            process.wait()
            report["worker_returncode"] = process.returncode
            if process.returncode or not worker_path.is_file():
                raise RuntimeError(f"MNI worker failed: code={process.returncode}; inspect {log_path}")
            worker = json.loads(worker_path.read_text())
            report["worker"] = worker
            if worker["status"] != "complete":
                raise RuntimeError("MNI worker report did not complete")
            report["mni_nonlinear"] = worker["value"]
            intervals["mni"] = {"started_monotonic": worker["operation_started_monotonic"],
                                "finished_monotonic": worker["operation_finished_monotonic"]}
        if report["mesh_validation"]["status"] != "passed":
            raise RuntimeError("mesh validation failed; inspect preserved report")
        if profile_stages and torch.cuda.is_initialized():
            torch.cuda.synchronize(target)
        report["status"] = "complete"
    except BaseException as error:
        if process is not None:
            _cancel([process])
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        stop.set()
        monitor.join()
        if log is not None:
            log.close()
        report.update(intervals=intervals, **parallel_intervals(intervals),
                      device_process_tree=sampler.report(), group_wall_seconds=time.monotonic() - started,
                      precision_after={"matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                                       "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32)},
                      allocator_environment_after=os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING"))
        report["parent_current_device_after"] = torch.cuda.current_device() if torch.cuda.is_initialized() else None
        _write(report_path, report)
    return report
