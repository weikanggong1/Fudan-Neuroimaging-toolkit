"""Python NVML查询：不启动命令、不建立CUDA context、不猜PID归属。

复用项目benchmark已有的pynvml查询接口。单位直接为字节；设备查询与
进程列表是先后两次快照。它只提供原始驱动数据，归属由profiling判断。
"""
from __future__ import annotations

from importlib import import_module


class NvmlProcessQuery:
    """延迟导入已在主页Conda声明的nvidia-ml-py；init失败原样抛异常。

    无构造参数。devices返回物理索引字符串→GPU UUID；snapshot(uuid)
    返回进程(pid/bytes)列表、整卡字节数和整卡查询错误。无法取得进程
    字节数时抛异常，不用0替代；MIG映射由调用者显式拒绝。close释放
    本实例的一次NVML引用；可重复调用，不修改其他实例的引用或CUDA。
    """

    def __init__(self):
        self.closed = True
        self.module = import_module("pynvml")
        self.module.nvmlInit()
        self.closed = False
        self.handles = {}

    def devices(self) -> dict[str, str]:
        if self.closed:
            raise RuntimeError("NVML query is closed")
        devices = {}
        for index in range(self.module.nvmlDeviceGetCount()):
            handle = self.module.nvmlDeviceGetHandleByIndex(index)
            value = self.module.nvmlDeviceGetUUID(handle)
            uuid = value.decode() if isinstance(value, bytes) else str(value)
            devices[str(index)] = uuid
            self.handles[uuid] = handle
        return devices

    def snapshot(self, uuid: str) -> tuple[list[dict], int | None, str | None]:
        if self.closed:
            raise RuntimeError("NVML query is closed")
        handle = self.handles.get(uuid)
        if handle is None:
            handle = self.module.nvmlDeviceGetHandleByUUID(uuid)
            self.handles[uuid] = handle
        rows = self.module.nvmlDeviceGetComputeRunningProcesses(handle)
        unavailable = getattr(self.module, "NVML_VALUE_NOT_AVAILABLE", (1 << 64)-1)
        entries = []
        for row in rows:
            value = row.usedGpuMemory
            if value is None or int(value) < 0 or int(value) == unavailable:
                raise ValueError("NVML process memory is unavailable; not an observed zero")
            entries.append({"pid": int(row.pid), "bytes": int(value)})
        try:
            device_bytes = int(self.module.nvmlDeviceGetMemoryInfo(handle).used)
            if device_bytes < 0 or device_bytes == unavailable:
                raise ValueError("NVML device memory is unavailable; not an observed zero")
            return entries, device_bytes, None
        except Exception as error:
            return entries, None, repr(error)

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.module.nvmlShutdown()

    def __del__(self):
        # 旧调用方仅保存report时也释放本实例引用；显式close可确定释放时机。
        try:
            self.close()
        except Exception:
            pass

