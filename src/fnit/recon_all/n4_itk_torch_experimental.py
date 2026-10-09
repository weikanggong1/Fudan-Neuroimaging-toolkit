"""固定 recon-all recipe 的完整 N4 Torch 实验后端。

实现 shrink=4、4×50 次反馈、200-bin triangular histogram、double FFT
Wiener sharpening、cubic fitting/reconstruction 和 ITK lattice refinement。
未接入生产默认；控制点与 histogram 改用 FP64 归约，GPU log/exp/FFT
和 convergence 归约也与 ITK 的串行 FP32 有舍入差异，需完整真实回归。
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Callable, Sequence

import nibabel as nib
import numpy as np
import torch

from .n4_bspline_torch import N4DenseBSplineFit, _cubic_kernel, _triple


# 固定 ITK 5.4.7/VNL FP32 SVD 的 cubic 2×4 refinement 表。
# 原理和生成脚本见独立 benchmark 的 dump_refinement.cpp；保留小非零项。
_REFINEMENT_BITS = (1056964608, 1056964608, 868220928, 2969567232,
                    1040187397, 1061158916, 1040187397, 2996830208)


class N4CubicReconstruction:
    """固定控制点/输出网格，按 z→y→x、各轴 4 项 FP32 顺序重建。"""
    def __init__(self, *, control_shape: Sequence[int], output_shape: Sequence[int],
                 spacing: Sequence[float], device: torch.device | str):
        self.control_shape = _triple(control_shape, name="control_shape", integer=True)
        self.output_shape = _triple(output_shape, name="output_shape", integer=True)
        self.spacing = _triple(spacing, name="spacing", integer=False)
        if min(self.control_shape) < 4 or min(self.output_shape) < 2 or min(self.spacing) <= 0:
            raise ValueError("cubic control_shape>=4, output_shape>=2 and spacing>0 required")
        self.device = torch.device(device)
        self.axes = []
        for count, size, spacing in zip(self.control_shape, self.output_shape, self.spacing):
            spans = count - 3
            rate = np.float32(spans / (float(np.float32(size - 1)) * spacing))
            epsilon = np.float32(float(rate) * spacing * float(np.float32(.001)))
            u = (torch.arange(size, dtype=torch.float32, device=self.device) * float(spans)) / float(size - 1)
            u = torch.where((u - float(spans)).abs() <= float(epsilon), float(np.float32(spans) - epsilon), u)
            base = u.long()
            weights = [_cubic_kernel((u - (base + corner).float()) + 1.0) for corner in range(4)]
            self.axes.append((base, weights))

    @torch.no_grad()
    def __call__(self, lattice: torch.Tensor) -> torch.Tensor:
        if lattice.dtype != torch.float32 or tuple(lattice.shape) != self.control_shape:
            raise ValueError("lattice must be FP32 xyz control_shape")
        if lattice.device != self.device:
            raise ValueError("lattice device differs from reconstruction cache")
        current = lattice
        for axis in (2, 1, 0):
            base, weights = self.axes[axis]
            shape = [1, 1, 1]; shape[axis] = self.output_shape[axis]
            result_shape = list(current.shape); result_shape[axis] = self.output_shape[axis]
            out = torch.zeros(result_shape, dtype=torch.float32, device=self.device)
            for corner in range(4):
                out.add_(torch.index_select(current, axis, base + corner) * weights[corner].reshape(shape))
            current = out
        return current


@torch.no_grad()
def refine_lattice(lattice: torch.Tensor) -> torch.Tensor:
    """开放 cubic lattice n→2n−3，保存每个输出的 64 项源顺序。"""
    if lattice.dtype != torch.float32 or lattice.ndim != 3 or min(lattice.shape) < 4:
        raise ValueError("refinement requires FP32 cubic xyz lattice")
    coefficients = torch.from_numpy(np.array(_REFINEMENT_BITS, np.uint32).view(np.float32).reshape(2, 4)).to(lattice.device)
    shape = tuple(2 * n - 3 for n in lattice.shape)
    coords = torch.meshgrid(*(torch.arange(n, device=lattice.device) for n in shape), indexing="ij")
    base = [c // 2 for c in coords]
    parity = [c % 2 for c in coords]
    out = torch.zeros(shape, dtype=torch.float32, device=lattice.device)
    for z in range(4):
        for y in range(4):
            for x in range(4):
                index = [base[0] + x, base[1] + y, base[2] + z]
                valid = (index[0] < lattice.shape[0]) & (index[1] < lattice.shape[1]) & (index[2] < lattice.shape[2])
                value = lattice[index[0].clamp(max=lattice.shape[0]-1),
                                index[1].clamp(max=lattice.shape[1]-1),
                                index[2].clamp(max=lattice.shape[2]-1)]
                weight = (coefficients[parity[0], x] * coefficients[parity[1], y]) * coefficients[parity[2], z]
                out.add_(torch.where(valid, value * weight, 0.0))
    return out


class N4HistogramSharpening:
    """固定 200 bins/.15 FWHM/.01 noise；GPU FP64 histogram/complex128 FFT。"""
    def __init__(self, *, device: torch.device | str):
        self.device = torch.device(device)
        self.n = torch.arange(512, dtype=torch.float32, device=self.device)
        self.distance = torch.minimum(self.n, 512.0 - self.n)

    @torch.no_grad()
    def __call__(self, field: torch.Tensor) -> torch.Tensor:
        flat = field.permute(2, 1, 0).contiguous().reshape(-1)
        # ITK uses `if maximum ... else if minimum`; preserve that prefix rule.
        running_max = torch.cummax(flat, 0).values
        previous_max = torch.cat((torch.full((1,), -float("inf"), device=self.device), running_max[:-1]))
        minimum = torch.where(flat <= previous_max, flat, torch.finfo(torch.float32).max).amin()
        maximum = flat.amax()
        slope = (maximum - minimum) / 199.0
        if not bool(torch.isfinite(slope)) or float(slope) <= 0.0:
            raise ValueError("ITK histogram range must be finite and positive")
        cidx = (flat - minimum) / slope
        index = torch.floor(cidx).long()
        offset = cidx - index.float()
        # FP64 sum changes accumulation rounding but preserves triangular weights.
        lower_valid = (index >= 0) & (index < 200) & ((offset == 0.0) | (index < 199))
        upper_valid = (index >= 0) & (index < 199) & (offset != 0.0)
        histogram = torch.bincount(index.clamp(0, 199),
            weights=torch.where(lower_valid, 1.0 - offset.double(), 0.0), minlength=200)
        histogram += torch.bincount((index + 1).clamp(0, 199),
            weights=torch.where(upper_valid, offset.double(), 0.0), minlength=200)
        histogram = histogram.float().double()
        padded = torch.zeros(512, dtype=torch.float64, device=self.device)
        padded[156:356] = histogram
        # VNL fwd uses + sign and both transforms are unnormalised.
        forward = lambda x: torch.fft.ifft(x, norm="forward")
        backward = lambda x: torch.fft.fft(x, norm="backward")
        vf = forward(padded)
        scaled_fwhm = float(np.float32(.15)) / slope
        exp_factor = (4.0 * math.log(2.0) / (scaled_fwhm * scaled_fwhm).double()).float()
        scale_factor = (2.0 * math.sqrt(math.log(2.0) / math.pi) / scaled_fwhm.double()).float()
        gaussian = (scale_factor * torch.exp(-(self.distance * self.distance) * exp_factor)).double()
        # ITK midpoint expression uses double 0.25 then exp(double).
        gaussian[256] = scale_factor.double() * torch.exp(-0.25 * float(np.float32(512) ** 2) * exp_factor.double())
        ff = forward(gaussian)
        gf = ff.conj() / (ff.conj() * ff + float(np.float32(.01)))
        unblurred = backward(vf * gf.real).real.clamp(min=0.0)
        coordinates = (minimum + (self.n - 156.0) * slope).double()
        numerator = backward(forward(coordinates * unblurred) * ff).real
        denominator = backward(forward(unblurred) * ff).real
        mapping = torch.where(denominator != 0.0, numerator / denominator, 0.0).float()[156:356]
        low = index.clamp(0, 199)
        high = (index + 1).clamp(0, 199)
        corrected = mapping[low] + (mapping[high] - mapping[low]) * (cidx - index.float())
        corrected = torch.where((index >= 0) & (index < 199), corrected, mapping[-1])
        return corrected.reshape(field.shape[2], field.shape[1], field.shape[0]).permute(2, 1, 0).contiguous()


@dataclass
class N4TorchResult:
    """float32 xyz tensors；corrected/field 是原网格，lattice 是 11³ 参数网格。"""
    corrected: torch.Tensor
    log_bias_field: torch.Tensor
    lattice: torch.Tensor
    iteration_count: int
    convergence: list[float]
    timings: dict[str, float]


@torch.no_grad()
def correct_tensor(*, image: torch.Tensor, spacing: Sequence[float] = (1, 1, 1),
                   profile: bool = False,
                   callback: Callable[[int, int, torch.Tensor, torch.Tensor], None] | None = None) -> N4TorchResult:
    """原始/conformed FP32 T1 xyz → 完整固定 N4 输出；不调用 C++。

    image>=0、各轴>=8，identity parametric direction；spacing 为 mm。
    固定 shrink=4、cubic、4 层各最多 50 次、threshold=0、全 1 mask、无
    confidence。尚不支持其他 recipe。profile=True 同步每个 GPU 子段，
    callback(level,iteration,residual,phi) 可导出诊断；默认不开同步剖析。
    输出坐标不变。失败抛异常，没有 blur、参考结果复制或 CPU fallback。
    """
    if image.dtype != torch.float32 or image.ndim != 3 or min(image.shape) < 8:
        raise ValueError("image must be a 3D FP32 xyz tensor with every dimension>=8")
    if not bool(torch.isfinite(image).all()) or bool((image < 0).any()):
        raise ValueError("image must be finite and nonnegative")
    spacing = _triple(spacing, name="spacing", integer=False)
    if min(spacing) <= 0:
        raise ValueError("spacing must be positive")
    device = image.device
    sync = lambda: torch.cuda.synchronize(device) if device.type == "cuda" else None
    timings: dict[str, float] = {}
    def measured(name, function):
        if profile: sync()
        start = time.perf_counter()
        value = function()
        if profile: sync()
        timings[name] = timings.get(name, 0.0) + time.perf_counter() - start
        return value
    started = time.perf_counter()
    shape = tuple(n // 4 for n in image.shape)
    small_spacing = tuple(s * 4 for s in spacing)
    origin = tuple(((n-1) - 4*(m-1)) * s * .5 for n, m, s in zip(image.shape, shape, spacing))
    offset = tuple(int(math.floor(o / s + .5)) for o, s in zip(origin, spacing))
    small = image[offset[0]:offset[0]+shape[0]*4:4,
                  offset[1]:offset[1]+shape[1]*4:4,
                  offset[2]:offset[2]+shape[2]*4:4].contiguous()
    log_input = torch.where(small > 0.0, torch.log(small), small)
    current = log_input.clone()
    previous = torch.zeros_like(current)
    lattice = None
    convergence = []
    sharpening = N4HistogramSharpening(device=device)
    total_iterations = 0
    for level in range(4):
        cp = 4 if lattice is None else lattice.shape[0]
        fit = measured("geometry_cache", lambda: N4DenseBSplineFit(
            field_shape=shape, control_shape=(cp, cp, cp), spacing=small_spacing, origin=origin, device=device))
        reconstruct = N4CubicReconstruction(control_shape=(cp, cp, cp), output_shape=shape,
                                            spacing=small_spacing, device=device)
        for iteration in range(50):
            sharpened = measured("sharpening", lambda: sharpening(current))
            residual = current - sharpened
            phi = measured("fitting", lambda: fit.fit(residual, validate=False))
            lattice = phi if lattice is None else lattice + phi
            new_field = measured("small_reconstruction", lambda: reconstruct(lattice))
            # threshold=0 的固定 recipe：零方差停止；非零协方差继续。
            # FP64 population statistics replaces the source FP32 online mean;
            # nonzero thresholds are deliberately not exposed.
            ratio = torch.exp(previous - new_field).double()
            cv = (ratio.std(correction=1) / ratio.mean()).float()
            convergence.append(float(cv))
            previous = new_field
            current = log_input - new_field
            total_iterations += 1
            if callback is not None:
                callback(level, iteration, residual, phi)
            if not bool(torch.isfinite(cv)):
                raise RuntimeError("N4 convergence statistic became nonfinite")
            if float(cv) <= 0.0:
                break
        del fit, reconstruct
        if level < 3:
            lattice = measured("refinement", lambda: refine_lattice(lattice))
    full_reconstruct = N4CubicReconstruction(control_shape=lattice.shape, output_shape=image.shape,
                                            spacing=spacing, device=device)
    full_field = measured("full_reconstruction", lambda: full_reconstruct(lattice))
    corrected = measured("exp_divide", lambda: image / torch.exp(full_field))
    sync()
    timings["wall_tensor_seconds"] = time.perf_counter() - started
    return N4TorchResult(corrected, full_field, lattice, total_iterations, convergence, timings)


def correct_volume(*, input_path: str | Path, output_path: str | Path,
                   device: str = "cuda:0", profile: bool = False) -> dict:
    """nibabel 输入/输出；整数 MGZ/NIfTI 保留原网格并 floor(clip+0.5) 转 uint8。

    输出路径明确由调用者提供；返回时间、网格、dtype、200 次迭代状态。
    这是实验 API，不替换生产 n4_itk.correct_volume。全部加载、传输、写出
    包含在 total_seconds；失败抛异常，已经写出的文件不会伪装成功报告。
    """
    started = time.perf_counter()
    input_path, output_path = Path(input_path), Path(output_path)
    original = nib.load(str(input_path))
    array = np.asarray(original.dataobj, dtype=np.float32)
    tensor = torch.from_numpy(array.copy()).to(device)
    result = correct_tensor(image=tensor, spacing=original.header.get_zooms()[:3], profile=profile)
    floating = result.corrected.cpu().numpy()
    output = np.floor(np.clip(floating, 0, 255) + .5).astype(np.uint8)
    header = original.header.copy(); header.set_data_dtype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(original, nib.freesurfer.mghformat.MGHImage):
        saved = nib.MGHImage(output, original.affine, header)
    else:
        saved = original.__class__(output, original.affine, header)
    nib.save(saved, str(output_path))
    return {"backend": "experimental_complete_n4_torch_fixed_itk547_recipe", "device": device,
            "input_path": str(input_path), "output_path": str(output_path), "shape": list(array.shape),
            "dtype": "uint8", "iterations": result.iteration_count, "timings": result.timings,
            "total_seconds": time.perf_counter() - started,
            "strict_reproduction": "requires comparison; changed reduction/FFT/log/exp rounding",
            "production_default_changed": False}
