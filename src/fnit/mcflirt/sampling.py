"""MCFLIRT 最终图像插值：float32 坐标、Constant 样条和一层边界延伸。"""

import math

import numpy as np
import torch

from ..flirt.core import _edge_background, _flip_to_radiological, _manual_trilinear


def _cubic_coefficients(values):
    """三次 B 样条的双精度递推，每个空间轴结束后存回 float32。"""
    coefficients = values
    pole = math.sqrt(3.0) - 2.0
    for axis in (-3, -2, -1):
        columns = coefficients.movedim(axis, -1).double().clone()
        length = columns.shape[-1]
        if length < 2:
            continue
        terms = min(length, int(math.log(1e-8) / math.log(abs(pole)) + 1.5))
        initial = columns[..., 0].clone()
        power = pole
        for offset in range(1, terms):
            initial = initial + power * columns[..., offset]
            power *= pole
        columns[..., 0] = initial
        original_last = columns[..., -1].clone()
        for offset in range(1, length):
            columns[..., offset] = columns[..., offset] + pole * columns[..., offset - 1]
        columns[..., -1] = -pole / (1.0 - pole * pole) * (2.0 * columns[..., -1] - original_last)
        for offset in range(length - 2, -1, -1):
            columns[..., offset] = pole * (columns[..., offset + 1] - columns[..., offset])
        coefficients = (columns * 6.0).float().movedim(-1, axis)
    return coefficients


def _coordinates(coefficients, shape, device):
    """NEWIMAGE 按 x/z 起点、沿 y 逐次 float32 加法生成坐标。"""
    x = torch.arange(shape[0], device=device, dtype=torch.float32)[:, None]
    z = torch.arange(shape[2], device=device, dtype=torch.float32)[None, :]
    values = torch.empty((3, *shape), device=device, dtype=torch.float32)
    coefficients = torch.as_tensor(coefficients, dtype=torch.float32, device=device)
    for axis, row in enumerate(coefficients):
        current = x * row[0] + z * row[2]
        current = current + row[3]
        for y in range(shape[1]):
            values[axis, :, y, :] = current
            current = current + row[1]
    return values


def _sample_cubic(coefficients, coordinates):
    shape = coefficients.shape
    output_shape = coordinates.shape[1:]
    coordinates = coordinates.double().reshape(3, -1)
    starts, weights = [], []
    for axis in range(3):
        position = coordinates[axis]
        rounded = torch.trunc(position + 0.5).long()
        start = torch.where(rounded.double() < position, rounded - 1, rounded - 2)
        starts.append(start)
        axis_weights = []
        for offset in range(4):
            distance = torch.abs(position - (start + offset))
            near = 2.0 / 3.0 + 0.5 * distance * distance * (distance - 2.0)
            far = (2.0 - distance) ** 3 / 6.0
            axis_weights.append(torch.where(distance < 1, near, torch.where(distance < 2, far, 0.0)))
        weights.append(axis_weights)
    coefficients = coefficients.reshape(-1)
    result = torch.zeros(coordinates.shape[1], device=coefficients.device, dtype=torch.float64)
    for z in range(4):
        iz = (starts[2] + z).clamp(0, shape[2] - 1)
        for y in range(4):
            iy = (starts[1] + y).clamp(0, shape[1] - 1)
            yz_weight = weights[2][z] * weights[1][y]
            for x in range(4):
                ix = (starts[0] + x).clamp(0, shape[0] - 1)
                index = (ix * shape[1] + iy) * shape[2] + iz
                term = coefficients[index].double() * weights[0][x]
                result = result + term * yz_weight
    return result.float().reshape(output_shape)


def sample_motion_frame(values, input_image, reference_image, fsl_matrix, *, device, interpolation):
    """将一帧按 FSL input→reference 矩阵采样，返回参考网格 float32 Tensor。

    数据类型转换由 TorchMCFLIRT 处理；本函数仅计算插值值。
    """
    data = torch.as_tensor(_flip_to_radiological(np.asarray(values, dtype=np.float32),
                                                input_image.affine), device=device)
    source_sizes = tuple(float(v) for v in input_image.header.get_zooms()[:3])
    target_sizes = tuple(float(v) for v in reference_image.header.get_zooms()[:3])
    pull = np.diag([*(1 / np.asarray(source_sizes)), 1]) @ np.linalg.inv(fsl_matrix)
    pull = pull @ np.diag([*target_sizes, 1])
    if (data.device.type == "cpu" and data.dtype == torch.float32
            and not data.requires_grad
            and (interpolation == "spline" or
                 (interpolation == "linear" and min(data.shape) >= 2))
            and np.isfinite(pull).all()
            and np.max(np.abs(pull[:3])) < (2.0**60) / (4 * sum(reference_image.shape[:3]))
            and bool(torch.isfinite(data).all())
            and float(data.abs().max()) < np.finfo(np.float32).max / 256):
        from ._sampling_cpu import sample
        sampled = torch.from_numpy(sample(data.numpy(), pull[:3], reference_image.shape[:3],
                                          float(_edge_background(data)), interpolation))
        if np.linalg.det(reference_image.affine[:3, :3]) > 0:
            sampled = sampled.flip(0)
        return sampled
    coordinates = _coordinates(pull[:3], reference_image.shape[:3], data.device)
    lower = torch.floor(coordinates)
    valid = torch.ones(reference_image.shape[:3], dtype=torch.bool, device=data.device)
    for axis, size in enumerate(data.shape):
        valid &= (lower[axis] >= -1) & (lower[axis] < size)
    if interpolation == "spline":
        sampled = _sample_cubic(_cubic_coefficients(data), coordinates)
    elif interpolation == "linear":
        bounded = torch.stack([coordinates[axis].clamp(0, size - 1)
                               for axis, size in enumerate(data.shape)]).reshape(3, -1)
        sampled = _manual_trilinear(data, bounded).reshape(reference_image.shape[:3])
    else:
        raise ValueError("interpolation must be linear or spline")
    sampled = torch.where(valid, sampled, _edge_background(data))
    if np.linalg.det(reference_image.affine[:3, :3]) > 0:
        sampled = sampled.flip(0)
    return sampled
