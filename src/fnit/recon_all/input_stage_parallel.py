"""独立 N4/SynthSeg 组：完整既有算子、fresh exec、显式 GPU 与四线程预算。"""
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


def _write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _precision() -> dict:
    import torch
    return {"matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic)}


def _sources(names: tuple[str, ...]) -> dict:
    return {name: _sha(Path(importlib.import_module(name).__file__)) for name in names}


def run_n4_task(*, input_path: str, output_path: str, threads: int,
                report_path: str) -> dict:
    """完整 SimpleITK N4 文件 API；拟合固定1，重建为 threads，不读参考。

    input_path/output_path 为同一 conformed MRI 网格的单帧 MGZ，输出 uint8；
    threads 控制确定性重建，不更改拟合、参数、量化或几何。report_path 写
    既有 N4 剖析 JSON。失败抛异常，成功返回既有结果、输入/输出/源码 SHA。
    """
    from .input_n4_chain import run_n4_stage
    input_hash = _sha(Path(input_path))
    value = run_n4_stage(input_path=input_path, output_path=output_path,
                         n4_backend="simpleitk", n4_execution="in-process",
                         native_binary=None, device="cpu", threads=threads,
                         profile=False, report_path=report_path)
    value.update(input_sha256=input_hash, output_sha256=_sha(Path(output_path)),
                 source_sha256=_sources(("fnit.recon_all.input_stage_parallel",
                    "fnit.recon_all.input_n4_chain", "fnit.recon_all.n4_itk",
                    "fnit.recon_all.n4_simpleitk")))
    if input_hash != _sha(Path(input_path)):
        raise RuntimeError("N4 source image changed during execution")
    return value


def run_synthseg_task(*, input_path: str, segmentation_path: str, volumes_path: str,
                     weights_dir: str, color_lut: str, device: str, threads: int,
                     cudnn_benchmark: bool, cudnn_deterministic: bool) -> dict:
    """复用完整 FNIT GPU SynthSeg，构造后 cuDNN FP32，保留 matmul TF32。

    原始及翻转前向、原网格恢复和软组织体积全部执行。input_path 是 conform
    MGZ；segmentation_path 复用原保存器写原网格 MGZ，存储为 float32、
    数值为原整数标签（未改标签语义）；volumes_path 为原 CSV
    格式（mm³）。四个权重文件及 LUT 只读取并记 SHA，不下载。device 必须
    显式 cuda:N。threads 约束 CPU 后处理。返回真实前向 precision、输出 SHA
    和体积；不启用半精度。CUDA/模型/写出失败抛异常，不回退 CPU。
    """
    import torch
    from fnit.synthseg_parc import SynthSeg
    if torch.device(device).type != "cuda" or torch.device(device).index is None:
        raise ValueError("SynthSeg task requires explicit cuda:N")
    before = _precision()
    input_hash = _sha(Path(input_path))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.deterministic = cudnn_deterministic
    try:
        with torch.cuda.device(device):
            model = SynthSeg(weights=weights_dir, device=device, threads=threads,
                             cudnn_tf32=False)
            # 在完整模型构造后显式施加 FP32 例外；segmenter 前向作用域
            # 记录 actual precision 并保留 matmul TF32。这里不全局关闭 TF32。
            model.segmenter.cudnn_tf32 = False
            result = model(input_path, keep_geometry=True, color_lut=color_lut)
            result.segmentation.save(segmentation_path)
            result.write_volumes_csv(input_path, volumes_path)
            torch.cuda.synchronize(torch.device(device))
            if not all(parameter.dtype == torch.float32 and parameter.device == torch.device(device)
                       for parameter in model.segmenter.model.parameters()):
                raise RuntimeError("SynthSeg model is not float32 on the requested GPU")
            value = {"segmentation": segmentation_path, "volumes_csv": volumes_path,
                     "total_intracranial_mm3": result.total_intracranial_mm3,
                     "precision": result.precision, "input_sha256": input_hash,
                     "outputs_sha256": {"segmentation": _sha(Path(segmentation_path)),
                                        "volumes_csv": _sha(Path(volumes_path))},
                     "actual_model_device": str(next(model.segmenter.model.parameters()).device),
                     "actual_model_dtype": str(next(model.segmenter.model.parameters()).dtype),
                     "source_sha256": _sources(("fnit.recon_all.input_stage_parallel",
                        "fnit.synthseg_parc.synthseg", "fnit.synthseg_parc.segment",
                        "fnit.synthseg_parc.preprocess", "fnit.synthseg_parc.postprocess",
                        "fnit.synthseg_parc.precision"))}
    finally:
        torch.backends.cuda.matmul.allow_tf32 = before["matmul_tf32"]
        torch.backends.cudnn.allow_tf32 = before["cudnn_tf32"]
        torch.backends.cudnn.benchmark = before["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = before["cudnn_deterministic"]
    if input_hash != _sha(Path(input_path)):
        raise RuntimeError("SynthSeg source image changed during execution")
    return value


def run_input_stage_group(*, orig_path: str | Path, n4_output: str | Path,
                          segmentation_output: str | Path, volumes_output: str | Path,
                          weights_dir: str | Path, color_lut: str | Path,
                          report_path: str | Path, device: str, threads: int = 4,
                          execution: str = "serial", profile_stages: bool = False,
                          code_version: str = "FNIT-source-hashes") -> dict:
    """完整 N4/SynthSeg 文件组；串行默认，parallel 以1+3计算线程并行。

    orig_path 是已完成 conform 的单帧 MGZ。N4/SynthSeg 只读同一个 orig，
    n4_output 与 segmentation_output 保持其原网格及各自 dtype/标签语义；
    volumes_output 为 SynthSeg mm³ CSV。weights_dir 为已有四文件资源目录，
    color_lut 为声明 LUT。三个输出、report_path 及旁边 request/log/worker
    文件必须是新路径，不能覆盖输入。device 显式 cuda:N，threads 固定4，
    serial 两个 fresh exec 各4线程，parallel N4=1、SynthSeg=3；拟合都1。
    profile_stages 默认False，仅True增加 worker阶段同步。code_version 记录
    冻结版本，仍记实际源码/输入/资源 SHA。成功 join 后返回两完整原结果、
    输出存在性、进程同期采样和含校验/exec/导入/读写的 group_wall_seconds。
    不改变父线程、设备、精度、环境或 CUDA allocator；父已初始化也可用。
    失败保存 JSON/log，取消本组新建进程树后抛异常。不是原始 T1 整例接口。
    """
    import torch
    from .thread_budget import native_thread_environment
    from .profiling import ProcessTreeDeviceSampler, parallel_intervals
    from .hemisphere_parallel import _cancel
    started = time.monotonic()
    target = torch.device(device)
    if target.type != "cuda" or target.index is None:
        raise ValueError("input group requires explicit cuda:N")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads != 4:
        raise ValueError("input group currently requires total threads=4")
    if execution not in ("serial", "parallel"):
        raise ValueError("execution must be serial or parallel")
    orig, weights, lut = (Path(path).resolve() for path in (orig_path, weights_dir, color_lut))
    outputs = {"n4": Path(n4_output).resolve(), "segmentation": Path(segmentation_output).resolve(),
               "volumes": Path(volumes_output).resolve()}
    report_path = Path(report_path).resolve()
    generated = [*outputs.values(), report_path, report_path.with_suffix(".N4.json")]
    artifacts = {name: {suffix: report_path.with_suffix(f".{name}.{suffix}")
                       for suffix in ("request.json", "worker.json", "log")} for name in ("n4", "synthseg")}
    generated.extend(path for files in artifacts.values() for path in files.values())
    if len(set(generated)) != len(generated) or orig in generated:
        raise ValueError("outputs/report/worker paths must be distinct from each other and input")
    for path in generated:
        if path.exists():
            raise FileExistsError(path)
    inputs = {"orig": orig, "color_lut": lut}
    inputs.update({name: weights / name for name in ("synthseg_2.0.h5",
        "synthseg_segmentation_labels_2.0.npy", "synthseg_segmentation_names_2.0.npy",
        "synthseg_topological_classes_2.0.npy")})
    hashes = {name: _sha(path) for name, path in inputs.items()}
    # 所有路径/资源校验完成后创建文件，父进程只做协调和采样。
    for path in generated:
        path.parent.mkdir(parents=True, exist_ok=True)
    initialized = torch.cuda.is_initialized()
    before = _precision()
    state = {"cuda_initialized": initialized,
             "current_device": torch.cuda.current_device() if initialized else None,
             "torch_threads": torch.get_num_threads(),
             "allocator_environment": os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING")}
    report = {"status": "running", "code_version": code_version, "execution": execution,
              "device": device, "thread_budget_total": 4,
              "algorithm_threads": {"n4": 1 if execution == "parallel" else 4,
                                    "synthseg": 3 if execution == "parallel" else 4},
              "inputs_sha256": hashes, "precision_at_entry": before, "parent_state_at_entry": state,
              "runtime_helper_sha256": _sha(Path(__file__)),
              "scope": "complete N4 + SynthSeg group including validation, exec, imports, loading, transfers, writes, sync and join; not whole recon-all"}
    sampler = ProcessTreeDeviceSampler(device=device, parent_pid=os.getpid(), interval=.5)
    stop = threading.Event()
    def sample():
        while not stop.is_set():
            sampler.sample_if_due(force=True)
            stop.wait(.5)
    monitor = threading.Thread(target=sample, daemon=True)
    processes, logs, intervals = {}, [], {}
    bootstrap = ("import sys,runpy,fnit.recon_all; fnit.recon_all.__path__.insert(0,sys.argv[1]); "
                 "sys.argv=sys.argv[2:]; runpy.run_module('fnit.recon_all.hemisphere_worker',run_name='__main__')")
    def launch(name):
        budget = report["algorithm_threads"][name]
        kwargs = dict(input_path=str(orig), threads=budget)
        if name == "n4":
            kwargs.update(output_path=str(outputs["n4"]), report_path=str(report_path.with_suffix(".N4.json")))
            callable_name, child_device = "run_n4_task", "cpu"
        else:
            kwargs.update(segmentation_path=str(outputs["segmentation"]), volumes_path=str(outputs["volumes"]),
                weights_dir=str(weights), color_lut=str(lut), device=device,
                cudnn_benchmark=before["cudnn_benchmark"], cudnn_deterministic=before["cudnn_deterministic"])
            callable_name, child_device = "run_synthseg_task", device
        request = {"callable": "fnit.recon_all.input_stage_parallel:" + callable_name,
                   "operation": name, "kwargs": kwargs, "threads": budget,
                   "device": child_device, "precision": before,
                   "profile_stages": profile_stages, "allocator_policy": "auto"}
        _write(artifacts[name]["request.json"], request)
        environment, description = native_thread_environment(threads=budget)
        report.setdefault("child_environments", {})[name] = description
        log = artifacts[name]["log"].open("w")
        logs.append(log)
        process = subprocess.Popen([sys.executable, "-X", "faulthandler", "-c", bootstrap,
            str(Path(__file__).parent), "fnit.recon_all.hemisphere_worker",
            str(artifacts[name]["request.json"]), str(artifacts[name]["worker.json"])],
            env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes[name] = process
        sampler.add_worker(process.pid)
    def collect(name):
        process = processes[name]
        worker_path = artifacts[name]["worker.json"]
        report.setdefault("returncodes", {})[name] = process.returncode
        if worker_path.is_file():
            worker = json.loads(worker_path.read_text())
            report.setdefault("workers", {})[name] = worker
        else:
            worker = {}
        if process.returncode or worker.get("status") != "complete":
            raise RuntimeError(f"{name} worker failed: code={process.returncode}; inspect {artifacts[name]['log']}")
        intervals[name] = {"started_monotonic": worker["operation_started_monotonic"],
                           "finished_monotonic": worker["operation_finished_monotonic"]}
    monitor.start()
    try:
        if execution == "serial":
            for name in ("n4", "synthseg"):
                launch(name)
                processes[name].wait()
                collect(name)
        else:
            for name in ("n4", "synthseg"):
                launch(name)
            pending = set(processes)
            while pending:
                for name in tuple(pending):
                    if processes[name].poll() is not None:
                        collect(name)
                        pending.remove(name)
                if pending:
                    time.sleep(.05)
        report["outputs_sha256"] = {name: _sha(path) for name, path in outputs.items()}
        report["final_inputs_sha256"] = {name: _sha(path) for name, path in inputs.items()}
        if report["final_inputs_sha256"] != hashes:
            raise RuntimeError("input image or resources changed during execution")
        report["status"] = "complete"
    except BaseException as error:
        _cancel(list(processes.values()))
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        stop.set()
        monitor.join()
        for log in logs:
            log.close()
        report.update(intervals=intervals, **parallel_intervals(intervals),
            device_process_tree=sampler.report(), precision_after=_precision(),
            parent_state_after={"cuda_initialized": torch.cuda.is_initialized(),
                "current_device": torch.cuda.current_device() if torch.cuda.is_initialized() else None,
                "torch_threads": torch.get_num_threads(),
                "allocator_environment": os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING")},
            group_wall_seconds=time.monotonic() - started)
        report["parent_state_preserved"] = (report["parent_state_after"] == state and report["precision_after"] == before)
        _write(report_path, report)
    return report
