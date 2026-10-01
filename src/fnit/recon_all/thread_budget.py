"""recon-all 的 CPU 线程预算；记录生效范围并保留 Python API 调用方状态。"""

from __future__ import annotations

from contextlib import contextmanager
from numbers import Integral
import os
from typing import Iterator, Mapping

import numba
import torch


NATIVE_THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
    "NUMBA_NUM_THREADS",
)


def _validate_threads(threads: int) -> int:
    """将正整数线程数返回为 int；布尔值、非整数或小于 1 时抛 ValueError。"""
    if isinstance(threads, bool) or not isinstance(threads, Integral) or threads < 1:
        raise ValueError("threads must be a positive integer")
    return int(threads)


@contextmanager
def thread_budget(*, threads: int = 4) -> Iterator[dict]:
    """暂时设置 Torch intraop 与当前调用线程的 Numba 并行掩码。

    输入 threads 是正整数，默认 4，单位为 CPU 工作线程。输出为可写入 JSON
    的报告，记录进入、生效、恢复后的数量，Numba 初始容量、线程后端及未改动
    的 Torch interop 数量；退出时同一个报告会补充恢复状态。没有影像输入或
    坐标变换，不改变 CUDA、精度、环境变量或原生程序的线程设置。

    Numba 掩码不能超过其导入时的 NUMBA_NUM_THREADS 容量；超过时在修改
    Torch 前抛 ValueError，不截断请求。后端设置失败或实际数量与请求不符时
    抛出原异常或 RuntimeError。正常退出和函数抛异常时均恢复调用方设置。
    Torch intraop 是进程设置，本作用域不能与其他线程的线程预算修改并发。
    Numba 掩码不是进程的总线程数，也不保证多个库合计只使用 threads 个线程。
    """
    requested = _validate_threads(threads)
    capacity = int(numba.config.NUMBA_NUM_THREADS)
    if requested > capacity:
        raise ValueError(
            f"threads={requested} exceeds Numba initial capacity {capacity}; "
            "set NUMBA_NUM_THREADS before importing Numba in a fresh process"
        )
    before_torch = int(torch.get_num_threads())
    before_numba = int(numba.get_num_threads())
    report = {
        "requested_threads": requested,
        "scope": "Torch process intraop and current caller Numba thread mask",
        "torch": {"before": before_torch, "effective": None,
                  "interop_unchanged": int(torch.get_num_interop_threads()),
                  "restored": None},
        "numba": {"before": before_numba, "effective": None,
                  "initial_capacity": capacity, "mask_only": True,
                  "threading_layer": numba.threading_layer(), "restored": None},
        "environment_modified": False,
        "restoration_complete": False,
    }
    change_torch = before_torch != requested
    change_numba = before_numba != requested
    try:
        if change_torch:
            torch.set_num_threads(requested)
        if change_numba:
            numba.set_num_threads(requested)
        report["torch"]["effective"] = int(torch.get_num_threads())
        report["numba"]["effective"] = int(numba.get_num_threads())
        if (report["torch"]["effective"] != requested
                or report["numba"]["effective"] != requested):
            raise RuntimeError("Torch or Numba did not apply the requested thread count")
        yield report
    finally:
        try:
            if change_numba:
                numba.set_num_threads(before_numba)
            report["numba"]["restored"] = int(numba.get_num_threads())
        finally:
            if change_torch:
                torch.set_num_threads(before_torch)
            report["torch"]["restored"] = int(torch.get_num_threads())
            report["restoration_complete"] = (
                report["torch"]["restored"] == before_torch
                and report["numba"]["restored"] == before_numba
            )


def native_thread_environment(*, threads: int = 4,
                              environ: Mapping[str, str] | None = None
                              ) -> tuple[dict[str, str], dict]:
    """为新子进程返回线程环境副本和作用范围报告，不修改 os.environ。

    threads 是正整数，默认 4。environ=None 时复制当前环境；显式 Mapping
    用作待复制的完整父环境。返回 (child_env, report)：前者可传给
    subprocess.run(env=...)，后者仅记录线程相关变量的前后字符串值，适合
    JSON；其余环境值不写入报告。OMP、OpenBLAS、MKL、NumExpr、ITK 与
    Numba 的变量设为相同预算。单位为 CPU 工作线程，没有影像空间或输出文件。

    这些变量供新进程初始化对应库时读取，不改变当前已初始化的库，不保证
    程序采用这些变量或已经实测其活跃线程数。显式命令行线程选项仍须匹配；
    并行两个子进程时应各分配总预算的一部分。非法线程数抛 ValueError；
    无法转换输入环境时传播 Python 的类型错误。不会启动程序或改变精度。
    """
    requested = _validate_threads(threads)
    child_env = dict(os.environ if environ is None else environ)
    before = {name: child_env.get(name) for name in NATIVE_THREAD_VARIABLES}
    value = str(requested)
    child_env.update({name: value for name in NATIVE_THREAD_VARIABLES})
    report = {
        "requested_threads": requested,
        "scope": "fresh subprocess initialization only",
        "environment_before": before,
        "environment_after": {name: child_env[name] for name in NATIVE_THREAD_VARIABLES},
        "parent_environment_modified": False,
        "active_worker_counts_verified": False,
    }
    return child_env, report
