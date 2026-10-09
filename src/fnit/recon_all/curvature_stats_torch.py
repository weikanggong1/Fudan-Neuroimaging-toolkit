"""Torch post-principal curvature maps and diagnostic statistics.

The fixed mris_curvature_stats command defaults to *discrete* principal
curvature.  Existing FNIT anatomical-stats fitting is continuous and is not a
replacement for that upstream step.  This module migrates the complete four-map
post-principal transformation and the map-summary reductions only.  It does not
calculate discrete K/H/k1/k2 or write a placeholder standard curv.stats file.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import nibabel.freesurfer.io as fsio
import numpy as np
import torch


@torch.inference_mode()
def curvature_derivatives_tensor(k1: torch.Tensor, k2: torch.Tensor) -> dict[str, torch.Tensor]:
    """两个同设备 FP32(N,) 主曲率→BE/C/FI/S 四张同设备 FP32(N,) 图。

    输入单位 mm⁻¹，必须同网格/同顶点顺序；k1/k2 的顺序按上游定义保留。
    BE/S/FI 单位 mm⁻²，C 为 mm⁻¹。保留固定 d932 源码的 float/double
    表达式：BE 与 S 为 FP32 算术，C 在 FP32 平方和后用 double sqrt，
    FI 的 fabs/减法/乘法为 double 后转 FP32。不改变输入、TF32或autocast。
    非Tensor抛TypeError；非一维、非FP32、形状或设备不一致抛ValueError。
    不扫描输入数值的有限性，避免内部GPU同步；调用方需保证输入有限。
    """
    if not isinstance(k1, torch.Tensor) or not isinstance(k2, torch.Tensor):
        raise TypeError("k1 and k2 must be torch.Tensor objects")
    if k1.ndim != 1 or k2.shape != k1.shape:
        raise ValueError("k1 and k2 must have identical one-dimensional shapes")
    if k1.dtype != torch.float32 or k2.dtype != torch.float32 or k1.device != k2.device:
        raise ValueError("k1 and k2 must be float32 on the same device")
    bending = k1 * k1 + k2 * k2
    difference = k1 - k2
    absolute1, absolute2 = k1.double().abs(), k2.double().abs()
    return {"BE": bending, "C": torch.sqrt(bending.double() * 0.5).float(),
            "FI": (absolute1 * (absolute1 - absolute2)).float(),
            "S": difference * difference}


@torch.inference_mode()
def curvature_summary_tensor(values: torch.Tensor, vertex_area: torch.Tensor, *,
                              ripped: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """曲率图与顶点面积→驻留同设备的数值摘要，不输出标准curv.stats。

    values 为有限 FP32(N,) 曲率；vertex_area 为同序非负 FP32(N,) mm²；
    ripped=None 包含全部顶点，或传同设备 bool(N,) 排除对应点。没有掩膜
    扩展、重拟合或空间变换。输出count、mean/std、min/max与首次顶点索引，
    integrals为(4,5) FP64：natural/rectified/positive/negative 四行，依次
    为FP32积分值、纳入顶点数、FP32计入面积、FP32积分/点数、FP32积分/面积。
    正值组包含零；负值组严格小于零。少于两点时后两个归一化量为零。
    摘要均驻留同设备，不逐顶点bool()/float()。均值方差和积分使用FP64
    并行归约；积分项先做FP32曲率×面积，与源表达式一致。计入面积为
    并行FP32 sum，不能声称等于官方的顺序FP32累加；完整原格式统计尚未替代。
    空/全排除图count为0，min/max为NaN，索引-1，均值/标准差/积分为0。
    类型、shape、dtype、设备契约不符抛TypeError/ValueError；有限性由调用者检查。
    """
    if not isinstance(values, torch.Tensor) or not isinstance(vertex_area, torch.Tensor):
        raise TypeError("values and vertex_area must be torch.Tensor objects")
    if values.ndim != 1 or vertex_area.shape != values.shape:
        raise ValueError("values and vertex_area must have identical one-dimensional shapes")
    if values.dtype != torch.float32 or vertex_area.dtype != torch.float32 or values.device != vertex_area.device:
        raise ValueError("values and vertex_area must be float32 on the same device")
    if ripped is None:
        active = torch.ones_like(values, dtype=torch.bool)
    else:
        if not isinstance(ripped, torch.Tensor):
            raise TypeError("ripped must be a torch.Tensor or None")
        if ripped.shape != values.shape or ripped.dtype != torch.bool or ripped.device != values.device:
            raise ValueError("ripped must be bool(N,) on the same device")
        active = ~ripped
    count = active.sum()
    denominator = count.clamp_min(1).double()
    selected = torch.where(active, values.double(), 0.0)
    mean = selected.sum() / denominator
    variance = (selected * selected).sum() / denominator - mean * mean
    std = variance.sqrt()
    groups = torch.stack((active, active, active & (values >= 0), active & (values < 0)))
    group_values = torch.stack((values, values.abs(), values.abs(), values.abs()))
    integral = torch.where(groups, group_values * vertex_area[None, :], 0.0).sum(dim=1, dtype=torch.float64).float()
    group_count = groups.sum(dim=1)
    area_counted = torch.where(groups, vertex_area[None, :], 0.0).sum(dim=1)
    normalized_count = torch.where(group_count > 1, integral / group_count.clamp_min(1), 0.0)
    normalized_area = torch.where(group_count > 1, integral / area_counted, 0.0)
    integrals = torch.stack((integral.double(), group_count.double(), area_counted.double(),
                             normalized_count.double(), normalized_area.double()), dim=1)
    if values.numel() == 0:
        minimum = maximum = values.new_tensor(float("nan"))
        minimum_index = maximum_index = count.new_tensor(-1)
    else:
        min_values = torch.where(active, values, torch.inf)
        max_values = torch.where(active, values, -torch.inf)
        minimum = torch.where(count > 0, min_values.min(), values.new_tensor(float("nan")))
        maximum = torch.where(count > 0, max_values.max(), values.new_tensor(float("nan")))
        minimum_index = torch.where(count > 0, min_values.argmin(), count.new_tensor(-1))
        maximum_index = torch.where(count > 0, max_values.argmax(), count.new_tensor(-1))
    return {"count": count, "mean": mean, "std": std, "min": minimum, "max": maximum,
            "min_vertex": minimum_index, "max_vertex": maximum_index, "integrals": integrals}


def write_curvature_derivatives(k1_path: str | Path, k2_path: str | Path,
                               output_prefix: str | Path, *, device: str = "cuda:0",
                               face_count: int = 0) -> dict:
    """读取自产同网格K1/K2 morph并写BE/C/FI/S；返回路径与含读写墙钟。

    输入为FreeSurfer morph标量文件；output_prefix例如/data/new/lh.smoothwm，
    输出是prefix.{BE,C,FI,S}.crv。device默认cuda:0，不回退CPU；face_count
    默认0是morph头中的面数，不用于计算，须非负整数。nibabel读写FP32。
    返回outputs(名称→路径)、vertices、device、read/compute/write/total_seconds。
    compute包含H2D和D2H，下载显式完成GPU计算；total包含读取、校验和输出。
    顶点数不一致、非有限主曲率或非法face_count抛ValueError；I/O或CUDA失败
    直接抛异常，部分输出不能当作完整阶段。此函数不生成离散主曲率或curv.stats。
    """
    started = perf_counter()
    if isinstance(face_count, bool) or not isinstance(face_count, (int, np.integer)) or face_count < 0:
        raise ValueError("face_count must be a nonnegative integer")
    k1 = np.asarray(fsio.read_morph_data(str(k1_path)), dtype=np.float32)
    k2 = np.asarray(fsio.read_morph_data(str(k2_path)), dtype=np.float32)
    if k1.ndim != 1 or k1.shape != k2.shape or not np.isfinite(k1).all() or not np.isfinite(k2).all():
        raise ValueError("input principal maps must be finite with identical vertex counts")
    read_seconds = perf_counter() - started
    tick = perf_counter()
    tensors = curvature_derivatives_tensor(torch.as_tensor(k1, device=device), torch.as_tensor(k2, device=device))
    arrays = {name: tensor.cpu().numpy() for name, tensor in tensors.items()}
    compute_seconds = perf_counter() - tick
    tick = perf_counter()
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, array in arrays.items():
        output = prefix.with_name(prefix.name + f".{name}.crv")
        fsio.write_morph_data(str(output), array, fnum=face_count)
        outputs[name] = str(output)
    return {"outputs": outputs, "vertices": len(k1), "device": device,
            "read_seconds": read_seconds, "compute_seconds_including_transfers": compute_seconds,
            "write_seconds": perf_counter() - tick, "total_seconds_including_io": perf_counter() - started,
            "scope": "post-principal-four-map-transform", "discrete_principal_computation": "not_included",
            "standard_curv_stats": "not_written"}


def main(argv: list[str] | None = None):
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k1", type=Path, required=True)
    parser.add_argument("--k2", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--face-count", type=int, default=0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = write_curvature_derivatives(k1_path=args.k1, k2_path=args.k2,
        output_prefix=args.output_prefix, device=args.device, face_count=args.face_count)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
