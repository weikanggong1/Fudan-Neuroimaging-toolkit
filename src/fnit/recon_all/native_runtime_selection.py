"""选择独立 Conda 构建的已验证热点实现；没有 GPU 或参考结果读取。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess


def select_native_optimizations(native_bin_dir: str | Path, *, threads: int,
                                mode: str = "auto") -> dict:
    """返回完整 GCA 后端与 white 专用程序的实际选择。

    native_bin_dir 为 FNIT Conda 程序目录，threads 为当前总线程预算；
    mode=auto 查询 GCA capability version 2 / upstream_ROMP_partials，
    仅在已配对验证的4线程预算选择 cpu_cached；其他预算使用 original。
    white 专用程序存在时须具有固定源码和未消费 face-MHT 补丁的能力标记。
    mode=original 固定原始评分和放置程序，用于同程序性能控制。
    返回 em_backend、white_binary、pial_binary、能力字典和选择原因；路径
    为绝对路径，程序哈希由主调度独立记录。没有独立官方命令；实际阶段
    仍运行完整 mri_em_register / mris_place_surface，空间与单位不变。
    模式非法、线程非正、已安装 white 快速程序能力不符时明确抛异常。
    旧 GCA 无能力入口时保持 original，不按文件名猜测缓存版本。
    """
    if mode not in ("auto", "original") or threads < 1:
        raise ValueError("native optimization mode must be auto/original with positive threads")
    directory = Path(native_bin_dir).resolve()
    standard = directory / "mris_place_surface"
    result = {"mode": mode, "em_backend": "original", "white_binary": str(standard),
              "pial_binary": str(standard), "gca_capabilities": None,
              "white_capabilities": None, "gca_reason": "original requested"}
    if mode == "original":
        return result
    query_environment = dict(os.environ, FNIT_GCA_QUERY_CAPABILITIES="1")
    query = subprocess.run([str(directory / "mri_em_register")],
                           env=query_environment, capture_output=True, text=True,
                           timeout=30, check=False)
    try:
        capability = json.loads(query.stdout) if query.returncode == 0 else None
    except json.JSONDecodeError:
        capability = None
    if not isinstance(capability, dict):
        capability = None
    result["gca_capabilities"] = capability
    valid = bool(capability and capability.get("version") == 2
                 and capability.get("full_native_em") is True
                 and capability.get("fnit_gca_cached_search") is True
                 and capability.get("reduction") == "upstream_ROMP_partials")
    result["gca_reason"] = ("verified capability and four-thread budget" if valid and threads == 4
                            else "original: unavailable capability or unvalidated thread budget")
    if valid and threads == 4:
        result["em_backend"] = "cpu_cached"
    fast = directory / "mris_place_surface_white_fast"
    if fast.exists():
        query = subprocess.run([str(fast), "--fnit-placement-capabilities"],
                               capture_output=True, text=True, timeout=30, check=True)
        capability = json.loads(query.stdout)
        if (not isinstance(capability, dict) or capability.get("schema_version") != 1
                or capability.get("upstream_commit") != "d932c45b7941662ea380a05efef580568b98d41a"
                or capability.get("features") != ["skip-unconsumed-repulse-face-table"]):
            raise ValueError("installed white optimization capability mismatch")
        result.update(white_binary=str(fast), white_capabilities=capability)
    return result
