"""保留既有归一化函数的子步骤计时，不触发额外GPU同步。"""
from __future__ import annotations

import math
from numbers import Real


def extract_algorithm_seconds(result) -> dict[str, float]:
    """从阶段返回字典提取已有秒数，返回带来源路径的JSON字典。

    result为既有函数返回值；非dict返回空字典。只读取顶层以_seconds
    结尾的标量、steps及completion.steps中的此类标量；不读取张量、
    控制点、图像或文件。单位为秒，没有坐标空间。保留原数值，不求和，
    不将异步内部计时冒充GPU同步墙钟；整阶段仍由StageProfiler记录。
    布尔值、非有限、负值和非数值项跳过，原字典不修改。不调用外部软件，
    属于报告整理内部步骤，没有对应的独立官方CLI。
    """
    if not isinstance(result, dict):
        return {}
    groups = [("", result), ("steps/", result.get("steps"))]
    completion = result.get("completion")
    if isinstance(completion, dict):
        groups.append(("completion/steps/", completion.get("steps")))
    values = {}
    for prefix, group in groups:
        if not isinstance(group, dict):
            continue
        for name, value in group.items():
            if (isinstance(name, str) and name.endswith("_seconds")
                    and isinstance(value, Real) and not isinstance(value, bool)
                    and math.isfinite(value) and value >= 0):
                values[prefix + name] = float(value)
    return values
