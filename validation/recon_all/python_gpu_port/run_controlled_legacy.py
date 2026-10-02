"""只用于 benchmark 的受控 e036f57 整例入口，不改变旧算法或缓冲区。

输入：t1 为原始单幅 T1，subject 为不存在/为空的 FNIT 输出目录；
--legacy-source 必须是冻结 e036f57 的 src；--weights-dir/--assets-dir 是
已声明的资源；--native-bin-dir 默认为运行环境 bin；--device 默认 cuda:0；
--threads 固定为 4。GPU 在调用旧 API 前初始化，分配缓存关闭，TF32
矩阵乘法保持开启；仅 SynthSegSegmenter.posterior 的 CUDA 作用域关闭
cuDNN TF32，返回/异常时恢复。CPU posterior 不改变精度开关。

输出：旧标准被试目录及原 fnit-native-free-run.json，并添加 benchmark_control
版本说明；run-controlled-legacy.json 记录原始输入/源码/脚本 SHA-256、
真实模型前向精度与线程、完整调用墙钟及失败状态。表面使用 surface RAS/mm，
体积/顶点图约定均沿用旧入口。缺失输入、非空输出、来源哈希不符或前向设置
不符直接失败；发生异常不写 complete。此通用精度控制不读取官方结果，
不修改张量、权重、后处理阈值、候选缓冲区或旧生产源文件。

具名参数的示例（只准备命令，不自动运行）：

    # 原始 T1；空输出；冻结旧源码；已校验资源；明确 GPU；固定线程预算。
    python run_controlled_legacy.py /data/raw/sub-01_T1w.nii.gz /data/benchmark/sub-01 \\
      --legacy-source /data/frozen_e036f57/src \\
      --weights-dir /data/fnit/weights --assets-dir /data/fnit/assets \\
      --native-bin-dir /opt/conda/envs/fnit/bin --device cuda:0 --threads 4

它验证的旧标准流程对应 recon-all 单 T1 -all profile；精度作用域控制与
前向记录没有独立的官方命令。外层 NVML 监测须另统计父子同时占用、
采样间隔和包括解释器启动/最终日志刷新的整个命令墙钟。
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import sys
import time


LEGACY_COMMIT = "e036f57b62b99d2af4cd8853ab2f1e6d2a9f8c68"
# 从该实际提交读取；禁止把当前生产文件误当冻结旧实现。
EXPECTED_SOURCE_SHA256 = {
    "fnit/recon_all/native_free.py": "0a6e950122bd5d1840b56134a225a09337143085f55e239cbd57d475d3c4063a",
    "fnit/synthseg_parc/segment.py": "4a57239105012efe7241eb0d1d3c4ba6235a220212595b809ba1a76991cd1259",
    "fnit/synthseg_parc/synthseg.py": "bdfbb91c190c4608d44d9b67890fe82a3a6530948f865dbde2990a1b1c92d33c",
    "fnit/_dmri.py": "b6bf837f8b5a0a8f0d910706631e882a9e3f8dfc1f1ef77e1cfac7efe36fafbf",
}


def _sha256(path: Path) -> str:
    """读取完整文件 SHA-256；只记录摘要，不读取许可证或打印影像内容。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_sidecar(subject: Path, metadata: dict) -> None:
    """仅在旧流程已建立目录后保存当前状态，不创建占位重建输出。"""
    if subject.is_dir():
        (subject / "run-controlled-legacy.json").write_text(
            json.dumps(metadata, indent=2) + "\n")


def _annotate_native_report(subject: Path, control: dict, *, error: str | None = None) -> None:
    """保留旧报告所有字段，增加计算来源；异常时纠正可能遗留的完成状态。"""
    path = subject / "fnit-native-free-run.json"
    if not path.is_file():
        return
    report = json.loads(path.read_text())
    report["benchmark_control"] = control
    if error is not None and report.get("status") == "complete":
        report.update(status="failed", benchmark_wrapper_error=error)
    path.write_text(json.dumps(report, indent=2) + "\n")


def _install_posterior_control(segment_class, torch, metadata: dict, subject: Path):
    """包装实际 e036f57 posterior，在 self.model 真前向前后记录策略。

    只改变 CUDA posterior 作用域内 cuDNN TF32；原 posterior 的模型调用、
    CPU 卸载、flip、blur、候选缓冲区与结果均由原函数执行。不额外同步 GPU。
    返回旧方法，以便调用方在 finally 中恢复类。CPU 调用直接走旧方法。
    """
    original_posterior = segment_class.posterior

    def autocast_enabled(device_type: str) -> bool:
        try:
            return bool(torch.is_autocast_enabled(device_type))
        except TypeError:
            return (bool(torch.is_autocast_enabled()) if device_type == "cuda"
                    else bool(torch.is_autocast_cpu_enabled()))

    @functools.wraps(original_posterior)
    def controlled_posterior(self, image, *, flip=True, smooth=True):
        model = self.model  # 旧 e036f57 源码的真实属性：SegmentUNet。
        model_device = next(model.parameters()).device
        if model_device.type != "cuda":
            return original_posterior(self, image, flip=flip, smooth=smooth)
        previous_cudnn_tf32 = bool(torch.backends.cudnn.allow_tf32)
        scope = {"invocation": len(metadata["synthseg_posterior_scopes"]) + 1,
                 "status": "running", "device": str(model_device),
                 "flip": bool(flip), "smooth": bool(smooth),
                 "expected_model_forwards": 2 if flip else 1,
                 "cudnn_tf32_before_scope": previous_cudnn_tf32,
                 "forwards": []}
        metadata["synthseg_posterior_scopes"].append(scope)
        torch.backends.cudnn.allow_tf32 = False

        def before_forward(module, inputs):
            x = inputs[0]
            parameters = list(module.parameters())
            record = {"index": len(scope["forwards"]) + 1,
                      "completed": False,
                      "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
                      "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                      "input_dtype": str(x.dtype), "input_device": str(x.device),
                      "input_shape": list(x.shape),
                      "parameter_dtypes": sorted({str(p.dtype) for p in parameters}),
                      "parameter_devices": sorted({str(p.device) for p in parameters}),
                      "autocast_cuda": autocast_enabled("cuda"),
                      "autocast_cpu": autocast_enabled("cpu"),
                      "torch_threads": torch.get_num_threads(),
                      "grad_enabled": bool(torch.is_grad_enabled()),
                      "model_training": bool(module.training)}
            scope["forwards"].append(record)
            if record["cudnn_tf32"] or not record["matmul_tf32"] \
                    or record["input_dtype"] != "torch.float32" \
                    or record["parameter_dtypes"] != ["torch.float32"] \
                    or record["autocast_cuda"] or record["autocast_cpu"]:
                raise RuntimeError(f"Controlled SynthSeg precision mismatch: {record}")

        def after_forward(module, inputs, output):
            record = scope["forwards"][-1]
            record.update(completed=True, output_dtype=str(output.dtype),
                          output_device=str(output.device))
            if output.dtype != torch.float32:
                raise RuntimeError("Controlled SynthSeg output is not float32")

        pre_handle = model.register_forward_pre_hook(before_forward)
        post_handle = model.register_forward_hook(after_forward)
        try:
            output = original_posterior(self, image, flip=flip, smooth=smooth)
            if len(scope["forwards"]) != scope["expected_model_forwards"] \
                    or not all(row["completed"] for row in scope["forwards"]):
                raise RuntimeError("SynthSeg model forward count/completion differs from frozen posterior")
            scope["status"] = "complete"
            return output
        except Exception as error:
            scope.update(status="failed", error=repr(error))
            raise
        finally:
            pre_handle.remove()
            post_handle.remove()
            torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
            scope["cudnn_tf32_after_restore"] = bool(torch.backends.cudnn.allow_tf32)
            _write_sidecar(subject, metadata)

    segment_class.posterior = controlled_posterior
    return original_posterior


def main() -> None:
    started = time.perf_counter()  # 早于解析、来源/输入校验、依赖导入和 CUDA 初始化。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("t1", type=Path)
    parser.add_argument("subject", type=Path)
    parser.add_argument("--legacy-source", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--native-bin-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, choices=(4,), default=4)
    args = parser.parse_args()
    subject = args.subject.resolve()
    t1 = args.t1.resolve(strict=True)
    legacy_source = args.legacy_source.resolve(strict=True)
    if not t1.is_file():
        raise FileNotFoundError(t1)
    if subject.exists() and (not subject.is_dir() or any(subject.iterdir())):
        raise ValueError("subject must be nonexistent or an empty directory")
    verified = {name: _sha256(legacy_source / name) for name in EXPECTED_SOURCE_SHA256}
    if verified != EXPECTED_SOURCE_SHA256:
        raise ValueError("Expected exactly the frozen e036f57 controller/SynthSeg/device source hashes")
    metadata = {"status": "running", "benchmark_only": True,
                "calculation_commit": LEGACY_COMMIT,
                "wrapper_sha256": _sha256(Path(__file__)),
                "input": str(t1), "input_sha256": _sha256(t1),
                "subject_dir": str(subject), "legacy_source": str(legacy_source),
                "verified_control_source_sha256": verified,
                "legacy_source_sha256": {str(path.relative_to(legacy_source)): _sha256(path)
                                          for path in sorted((legacy_source / "fnit").rglob("*.py"))},
                "hostname": platform.node(), "python_version": platform.python_version(),
                "argv": sys.argv, "device": args.device,
                "threads": args.threads, "synthseg_posterior_scopes": [],
                "precision_control": "CUDA posterior scope: cuDNN TF32=False; matmul TF32=True; no autocast/half; CPU unchanged",
                "wall_scope": "wrapper main entry through validation/import/context/entire old API and output/report writes; external monitor additionally includes interpreter startup and final stdout flush",
                "prior_NUMBA_NUM_THREADS": os.environ.get("NUMBA_NUM_THREADS"),
                "prior_PYTORCH_NO_CUDA_MEMORY_CACHING": os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING")}
    # 在导入 torch/numba 与初始化 CUDA 之前固定预算及分配策略。
    os.environ["NUMBA_NUM_THREADS"] = str(args.threads)
    cuda_requested = args.device.startswith("cuda")
    if cuda_requested:
        os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    os.environ["PYTHONPATH"] = str(legacy_source)  # 后续 Python 子进程也只取旧 FNIT 源。
    sys.path.insert(0, str(legacy_source))
    import torch
    import numba

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Controlled baseline supports CPU or CUDA")
    retained = None
    if device.type == "cuda":
        if torch.cuda.is_initialized():
            raise RuntimeError("Start this wrapper in a fresh process before CUDA initialization")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        retained = torch.ones(1, dtype=torch.float32, device=device)
        torch.cuda.synchronize(device)
        metadata["device_uuid"] = str(torch.cuda.get_device_properties(device).uuid)
    metadata.update(device=str(device), torch_version=torch.__version__,
                    numba_version=numba.__version__, torch_threads=torch.get_num_threads(),
                    numba_threads=numba.get_num_threads(),
                    cuda_initialized_before_api=torch.cuda.is_initialized(),
                    retained_tensor_bytes=(retained.numel() * retained.element_size()
                                           if retained is not None else 0),
                    CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    NUMBA_NUM_THREADS=os.environ["NUMBA_NUM_THREADS"],
                    PYTORCH_NO_CUDA_MEMORY_CACHING=os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING"),
                    effective_PYTHONPATH=os.environ["PYTHONPATH"])
    if metadata["torch_threads"] != 4 or metadata["numba_threads"] != 4:
        raise RuntimeError("Controlled Torch/Numba thread budget must be exactly four")
    native = importlib.import_module("fnit.recon_all.native_free")
    segment = importlib.import_module("fnit.synthseg_parc.segment")
    for module in (native, segment):
        if not Path(module.__file__).resolve().is_relative_to(legacy_source):
            raise RuntimeError(f"Module resolved outside frozen legacy source: {module.__file__}")
    original_posterior = _install_posterior_control(
        segment.SynthSegSegmenter, torch, metadata, subject)
    control = {"calculation_commit": LEGACY_COMMIT,
               "wrapper_sha256": metadata["wrapper_sha256"],
               "legacy_source": str(legacy_source),
               "verified_control_source_sha256": verified,
               "precision_control": metadata["precision_control"],
               "torch_threads": 4, "numba_threads": 4,
               "sidecar": str(subject / "run-controlled-legacy.json")}
    try:
        print(json.dumps({"validation_device_uuid": metadata.get("device_uuid"),
                          "calculation_commit": LEGACY_COMMIT,
                          "wrapper_sha256": metadata["wrapper_sha256"]}), flush=True)
        result = native.run_recon_all_python(
            t1=t1, subject_dir=subject, weights_dir=args.weights_dir,
            assets_dir=args.assets_dir, device=str(device), threads=args.threads,
            native_bin_dir=args.native_bin_dir)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if result.get("status") != "complete":
            raise RuntimeError("Frozen baseline API did not return complete")
        metadata.update(status="complete", native_api_total_seconds=result.get("total_seconds"),
                        standard_output_validation=result.get("output_validation"),
                        standard_mesh_validation=result.get("mesh_validation"))
        _annotate_native_report(subject, control)
        metadata["seconds_including_validation_loading_context_and_run"] = time.perf_counter() - started
        _write_sidecar(subject, metadata)
        # 再次测量，包含首次侧车写入；最后一次元数据刷新由外层命令墙钟覆盖。
        metadata["seconds_including_first_sidecar_write"] = time.perf_counter() - started
        _write_sidecar(subject, metadata)
        result["benchmark_control"] = control
        print(json.dumps(result, indent=2), flush=True)
    except Exception as error:
        metadata.update(status="failed", error=repr(error),
                        seconds_including_validation_loading_context_and_run=time.perf_counter() - started)
        _annotate_native_report(subject, control, error=repr(error))
        _write_sidecar(subject, metadata)
        raise
    finally:
        segment.SynthSegSegmenter.posterior = original_posterior


if __name__ == "__main__":
    main()
