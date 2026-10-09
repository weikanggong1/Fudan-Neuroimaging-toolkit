"""Native-free ``mri_normalize -aseg -mask`` for matched T1 voxel grids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch

from ..mgh_compat import save_same_dtype_mgh
from .normalize_aseg_ridge import medial_ridge
from .normalize_aseg_source import (apply_initial_aseg_bias, filter_aseg_ridge,
                                    prepare_aseg_source)
from .normalize_gentle_source import gentle_controls
from .normalize_3d_controls import controls_3d
from .normalize_gaussian_source import (apply_gentle_bias, apply_gentle_bias_float,
                                        smooth_bias, smooth_bias_torch)
from .normalize_voronoi_source import voronoi_fill, voronoi_fill_torch


def _complete_from_initial(initial: np.ndarray, device: str,
                           three_d_iterations: int, *,
                           controls_neighbor_backend: str = "cpu") -> tuple[np.ndarray, dict]:
    """从同网格 float32 初始偏置校正图完成温和/三维迭代。

    initial 是 (x,y,z) NumPy 强度图；device 为调用者设备字符串，
    three_d_iterations 由公开接口限定为0/1/2；controls_neighbor_backend
    默认 cpu，torch 为显式 GPU 邻域统计。输出同 shape uint8 NumPy 图
    及含 steps/gentle_controls 的 dict，步骤单位秒；不负责 affine 或文件。
    保留 CPU 控制点选择、有序离群更新和各轮浮点图，错误原样传播。
    属于 mri_normalize -aseg 内部完成阶段，没有独立官方 CLI。
    """
    source = torch.as_tensor(np.ascontiguousarray(initial, dtype=np.float32), device=device)
    steps = {}
    tick = time.perf_counter()
    control, details = gentle_controls(source)
    steps["gentle_controls_seconds"] = time.perf_counter() - tick
    tick = time.perf_counter()
    if source.is_cuda:
        voronoi, _ = voronoi_fill_torch(source, control)
        bias, _ = smooth_bias_torch(voronoi, source, control)
    else:
        control_cpu = control.numpy()
        voronoi, _ = voronoi_fill(source.numpy(), control_cpu)
        bias, _ = smooth_bias(voronoi, source.numpy(), control_cpu)
        bias = torch.as_tensor(bias, device=device)
    steps["gentle_bias_seconds"] = time.perf_counter() - tick
    if not three_d_iterations:
        return apply_gentle_bias(source, bias).cpu().numpy(), {"steps": steps, "gentle_controls": details}
    current = apply_gentle_bias_float(source, bias)
    for index in range(three_d_iterations):
        tick = time.perf_counter()
        control_3d, detail = controls_3d(current.cpu().numpy(),
            neighbor_backend=controls_neighbor_backend, device=str(device))
        steps[f"three_d_{index + 1}_controls_seconds"] = time.perf_counter() - tick
        tick = time.perf_counter()
        if source.is_cuda:
            control_tensor = torch.as_tensor(control_3d, device=device)
            voronoi, _ = voronoi_fill_torch(current, control_tensor)
            bias, _ = smooth_bias_torch(voronoi, current, control_tensor)
        else:
            current_cpu = current.numpy()
            voronoi, _ = voronoi_fill(current_cpu, control_3d)
            bias_cpu, _ = smooth_bias(voronoi, current_cpu, control_3d)
            bias = torch.as_tensor(bias_cpu, device=device)
        steps[f"three_d_{index + 1}_bias_seconds"] = time.perf_counter() - tick
        current = torch.where(bias == 0, current, current * 110.0 / bias)
        steps[f"three_d_{index + 1}_controls"] = detail
    result = torch.floor(current.clamp(0, 255) + 0.5).to(torch.uint8)
    return result.cpu().numpy(), {"steps": steps, "gentle_controls": details}


def normalize_t1_aseg(norm_file: str | Path, aseg_file: str | Path,
                      brainmask_file: str | Path, output_file: str | Path,
                      device: str | None = None, three_d_iterations: int = 2, *,
                      controls_neighbor_backend: str = "cpu",
                      initial_bias_backend: str = "cpu") -> dict:
    """同网格 norm/aseg/brainmask→uint8 brain.mgz，返回完整步骤记录。

    norm_file 为1mm conform uint8强度，aseg_file 为同网格整数标签，
    brainmask_file 为同网格掩膜，output_file 保留 norm 的毫米 affine/MGH头。
    device=None 有 CUDA 时选 cuda:0 否则 cpu，three_d_iterations 默认2，
    仅接受0/1/2。controls_neighbor_backend 默认 cpu，torch 只替换三维扩展
    邻域统计并要求 CUDA；initial_bias_backend 默认 cpu，torch 复用已有GPU
    初始偏置传播/平滑，算术与zero-control规则不变。ridge、有序过滤保留CPU。
    返回 dict 的 total_seconds、ridge_seconds、initial_bias_seconds 单位秒；
    ridge、removed_controls、wm_peak、completion 记录控制点和后续步骤。
    total_seconds 从图像头与同网格检查后开始，完整文件 API 应外层计时。
    无效参数、网格不匹配和组织峰失败抛异常，不调用 FreeSurfer或静默回退。
    对应 mri_normalize -seed 1234 -mprage -aseg ... -mask ...，详见中文页。
    """
    if three_d_iterations not in (0, 1, 2):
        raise ValueError("three_d_iterations must be 0, 1, or 2")
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if controls_neighbor_backend not in {"cpu", "torch"}:
        raise ValueError("controls_neighbor_backend must be cpu or torch")
    if controls_neighbor_backend == "torch" and not str(device).startswith("cuda:"):
        raise ValueError("torch controls_neighbor_backend requires an explicit CUDA device")
    if initial_bias_backend not in {"cpu", "torch"}:
        raise ValueError("initial_bias_backend must be cpu or torch")
    if initial_bias_backend == "torch" and not str(device).startswith("cuda:"):
        raise ValueError("torch initial_bias_backend requires an explicit CUDA device")
    norm_file, aseg_file, brainmask_file = map(Path, (norm_file, aseg_file, brainmask_file))
    images = [nib.load(str(path)) for path in (norm_file, brainmask_file, aseg_file)]
    if any(image.shape != images[0].shape or not np.array_equal(image.affine, images[0].affine)
           for image in images[1:]):
        raise ValueError("norm, aseg, and brainmask must share a voxel grid")
    started = time.perf_counter()
    norm, brainmask, aseg = (np.asarray(image.dataobj) for image in images)
    masked, _ = prepare_aseg_source(norm, brainmask, aseg)
    tick = time.perf_counter()
    ridge, ridge_details = medial_ridge(aseg)
    controls, removed, wm_peak = filter_aseg_ridge(masked, ridge)
    ridge_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    initial = apply_initial_aseg_bias(masked, controls, backend=initial_bias_backend,
                                    device=str(device))
    initial_bias_seconds = time.perf_counter() - tick
    result, completion = _complete_from_initial(initial, device, three_d_iterations,
        controls_neighbor_backend=controls_neighbor_backend)
    save_same_dtype_mgh(norm_file, output_file, result)
    return {"device": device, "three_d_iterations": three_d_iterations,
            "controls_neighbor_backend": controls_neighbor_backend,
            "initial_bias_backend": initial_bias_backend,
            "ridge_seconds": ridge_seconds, "initial_bias_seconds": initial_bias_seconds,
            "ridge": ridge_details, "removed_controls": int(np.count_nonzero(removed)),
            "wm_peak": wm_peak, "completion": completion,
            "total_seconds": time.perf_counter() - started}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--norm", type=Path, required=True)
    parser.add_argument("--aseg", type=Path, required=True)
    parser.add_argument("--brainmask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--three-d-iterations", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--controls-neighbor-backend", choices=("cpu", "torch"), default="cpu")
    parser.add_argument("--initial-bias-backend", choices=("cpu", "torch"), default="cpu")
    args = parser.parse_args()
    print(json.dumps(normalize_t1_aseg(args.norm, args.aseg, args.brainmask,
                                       args.output, args.device, args.three_d_iterations,
                                       controls_neighbor_backend=args.controls_neighbor_backend,
                                       initial_bias_backend=args.initial_bias_backend)))


if __name__ == "__main__":
    main()
