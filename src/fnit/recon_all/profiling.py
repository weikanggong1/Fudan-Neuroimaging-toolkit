"""recon-all 的可选阶段剖析；计时不改变计算或精度策略。"""

from __future__ import annotations

import os
import resource
import time

import torch


def autocast_state(device_type: str) -> dict:
    """返回调用方 CPU/CUDA autocast 开关与 dtype；兼容 PyTorch 2.1，不修改设置。"""
    try:
        enabled = torch.is_autocast_enabled(device_type)
    except TypeError:
        enabled = (torch.is_autocast_enabled() if device_type == "cuda"
                   else torch.is_autocast_cpu_enabled())
    dtype = (torch.get_autocast_dtype(device_type) if hasattr(torch, "get_autocast_dtype")
             else torch.get_autocast_gpu_dtype() if device_type == "cuda"
             else torch.get_autocast_cpu_dtype())
    return {"enabled": bool(enabled), "dtype": str(dtype)}


def record_network_forward(module, inputs: torch.Tensor, records: list, **metadata) -> None:
    """在调用前记录实际设备、dtype、TF32、cuDNN后端及autocast，不同步CUDA或修改精度。

    module为已构造网络，inputs为真实前向张量，records为调用者列表；
    metadata可附模型名/是否计算逆变换。只追加JSON兼容字典，返回None。
    输出含模型设备集合，以及cudnn_enabled/benchmark/deterministic布尔值；
    列表由调用者保存，不形成全局缓存。读取后端状态，不改变算法、缓存或同步。
    缺少张量属性、模块参数不可遍历或列表不可追加时抛异常；无独立官方命令。
    """
    records.append({
        **metadata, "device": str(inputs.device), "input_dtype": str(inputs.dtype),
        "model_devices": sorted({str(p.device) for p in module.parameters()}),
        "model_dtypes": sorted({str(p.dtype) for p in module.parameters()}),
        "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "autocast": autocast_state(inputs.device.type),
    })


def configure_cuda_allocator(device: str, policy: str = "auto") -> dict:
    """在 CUDA 初始化前选择分配缓存，并记录能够确认的实际状态。

    device 为 CPU 或逻辑 CUDA 设备；policy 为 auto/enabled/disabled。
    auto 保留已有环境，首次 CUDA 调用默认关闭缓存以延续低显存策略；
    已初始化的 API 不修改缓存。显式 enabled/disabled 须在初始化前调用，
    否则抛 ValueError，避免把环境变量误报为已经生效的运行时策略。
    返回请求、初始化状态、环境及统计可信性；不分配张量或改变精度。
    """
    if policy not in {"auto", "enabled", "disabled"}:
        raise ValueError("cuda_allocator_cache must be auto, enabled, or disabled")
    cuda = torch.device(device).type == "cuda"
    initialized = torch.cuda.is_initialized() if cuda else False
    entry_environment = os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING")
    if cuda and initialized and policy != "auto":
        raise ValueError("explicit CUDA allocator policy must be set before CUDA initialization")
    if cuda and not initialized:
        if policy == "auto":
            os.environ.setdefault("PYTORCH_NO_CUDA_MEMORY_CACHING", "1")
        elif policy == "disabled":
            os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
        else:
            os.environ.pop("PYTORCH_NO_CUDA_MEMORY_CACHING", None)
    environment = os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING")
    # PyTorch native allocator checks presence, including "0" and "".
    known_disabled = cuda and not initialized and environment is not None
    known_enabled = cuda and not initialized and not known_disabled
    return {"requested": policy, "cuda_initialized_at_entry": initialized,
            "environment_at_entry": entry_environment, "environment_after_selection": environment,
            "effective": ("not_applicable" if not cuda else "preserved_preinitialized_unknown"
                          if initialized else "disabled" if known_disabled else "enabled"),
            "torch_stats_known_valid": known_enabled,
            "torch_stats_known_unavailable": known_disabled}


class StageProfiler:
    """记录函数全程、父子 CPU 时间及可选 CUDA 同步，结果单位为秒/字节。

    device 必须指向实际逻辑设备；synchronize=False 为生产默认，不插入
    CUDA 同步，不清除缓存。True 在阶段前后同步并单列等待时间，阶段秒数
    包含前同步、函数和后同步。allocator 是 configure_cuda_allocator 的
    返回值。缓存关闭时不把零统计当作零显存；初始化前的 CPU 阶段不建立
    CUDA context。失败时 last_row 仍记录错误，原异常继续抛出。
    """

    def __init__(self, *, device: str, synchronize: bool, allocator: dict):
        self.device = torch.device(device)
        self.synchronize = synchronize
        self.allocator = allocator
        self.last_row: dict = {}

    def _cuda_active(self) -> bool:
        return self.device.type == "cuda" and torch.cuda.is_initialized()

    def _memory(self) -> dict:
        if not self._cuda_active() or self.allocator["torch_stats_known_unavailable"]:
            return {"torch_memory_stats_status": "unavailable"}
        allocated = int(torch.cuda.max_memory_allocated(self.device))
        reserved = int(torch.cuda.max_memory_reserved(self.device))
        if not self.allocator["torch_stats_known_valid"] and not (allocated or reserved):
            return {"torch_memory_stats_status": "unavailable_or_allocator_not_observed"}
        return {"torch_memory_stats_status": "available",
                "gpu_peak_allocated_bytes": allocated, "gpu_peak_reserved_bytes": reserved,
                "gpu_peak_scope": "stage" if self.synchronize else "since_cuda_initialization"}

    def run(self, name: str, function, *args, **kwargs):
        """执行具名阶段并返回原函数结果；last_row 保存同步和 CPU 诊断。"""
        tick = time.perf_counter()
        cpu = time.process_time()
        child = resource.getrusage(resource.RUSAGE_CHILDREN)
        pre = 0.0
        post = 0.0
        synchronized = False
        error = None
        body = 0.0
        body_tick = None
        try:
            if self.synchronize and self._cuda_active():
                sync_tick = time.perf_counter()
                try:
                    torch.cuda.synchronize(self.device)
                    synchronized = True
                finally:
                    pre = time.perf_counter() - sync_tick
                if self._memory()["torch_memory_stats_status"] == "available":
                    torch.cuda.reset_peak_memory_stats(self.device)
            body_tick = time.perf_counter()
            value = function(*args, **kwargs)
            body = time.perf_counter() - body_tick
            body_tick = None
            if self.synchronize and self._cuda_active():
                sync_tick = time.perf_counter()
                try:
                    torch.cuda.synchronize(self.device)
                    synchronized = True
                finally:
                    post = time.perf_counter() - sync_tick
            return value
        except Exception as caught:
            if body_tick is not None:
                body = time.perf_counter() - body_tick
            error = repr(caught)
            raise
        finally:
            child_after = resource.getrusage(resource.RUSAGE_CHILDREN)
            try:
                memory = self._memory()
            except Exception as memory_error:
                memory = {"torch_memory_stats_status": "failed",
                          "torch_memory_stats_error": repr(memory_error)}
            self.last_row = {"name": name, "seconds": time.perf_counter() - tick,
                             "function_seconds": body,
                             "cuda_pre_sync_seconds": pre, "cuda_post_sync_seconds": post,
                             "cuda_synchronized": synchronized,
                             "cuda_synchronization_requested": self.synchronize,
                             "parent_cpu_seconds": time.process_time() - cpu,
                             "child_cpu_seconds": ((child_after.ru_utime + child_after.ru_stime)
                                                   - (child.ru_utime + child.ru_stime)),
                             **memory}
            if error is not None:
                self.last_row["error"] = error
