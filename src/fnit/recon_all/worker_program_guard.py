"""自有 exec worker 继承生产程序禁令；独立参考测试不自动启用。"""
from __future__ import annotations

from functools import wraps
import os

from .execution_recipe import original_stage_guard


def inherit_original_stage_guard(function):
    """在调用前按环境标志安装程序守卫，正常/异常退出均恢复守卫状态。

    function 为自有 worker 的具名入口；返回保留签名、参数及返回值的
    包装函数。每次调用读取 FNIT_RECON_FORBID_ORIGINAL_STAGES，只有
    字符串1启用；不在导入时冻结，不从后端名称推测生产授权。
    禁止原程序尝试由既有守卫抛 RuntimeError。此函数不设置线程、
    TF32、CUDA、分配器或输入输出，也不拦截外部 shell 子孙/动态库。
    """
    @wraps(function)
    def guarded(*args, **kwargs):
        enabled = os.environ.get('FNIT_RECON_FORBID_ORIGINAL_STAGES') == '1'
        with original_stage_guard(enabled=enabled):
            return function(*args, **kwargs)

    return guarded
