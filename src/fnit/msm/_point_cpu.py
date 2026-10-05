"""Literal double Point arithmetic for CPU sphere geometry.

PyTorch CPU vector division can use a rounded reciprocal. Shared triangle
edges make that last bit observable in newMSM's containing-face choice. These
small geometry operators use NumPy's literal division and ordered products;
CUDA and differentiable tensors keep their existing PyTorch operators.
"""
from __future__ import annotations

import numpy as np
import torch


def enabled(*values):
    return all(value.device.type == "cpu" and value.dtype == torch.float64
               and not value.requires_grad for value in values)


def normalize(values):
    array = values.numpy()
    square = array * array
    length = np.sqrt((square[..., 0] + square[..., 1]) + square[..., 2])
    return torch.from_numpy(array / np.where(length > 1e-8, length, 1)[..., None])


def divide(first, second):
    return torch.from_numpy(np.asarray(first.numpy() / second.numpy()))


def _cross(first, second):
    return np.stack((first[..., 1] * second[..., 2] - first[..., 2] * second[..., 1],
                     first[..., 2] * second[..., 0] - first[..., 0] * second[..., 2],
                     first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]), -1)


def _norm(values):
    square = values * values
    return np.sqrt((square[..., 0] + square[..., 1]) + square[..., 2])


def area_weights(triangles, points):
    array, query = triangles.numpy(), points.numpy()
    first, second, third = array[..., 0, :], array[..., 1, :], array[..., 2, :]
    a = _norm(_cross(second - query, third - query)) * 0.5
    b = _norm(_cross(first - query, third - query)) * 0.5
    c = _norm(_cross(first - query, second - query)) * 0.5
    return torch.from_numpy(np.stack((a, b, c), -1) / ((a + b) + c)[..., None])


def tangent_basis(normals):
    x, y, z = np.moveaxis(normals.numpy(), -1, 0)
    first = (np.abs(x) >= np.abs(y)) & (np.abs(x) >= np.abs(z))
    second = (~first) & (np.abs(y) >= np.abs(x)) & (np.abs(y) >= np.abs(z))
    third = ~(first | second)
    yz, xz, xy = np.sqrt(z*z + y*y), np.sqrt(z*z + x*x), np.sqrt(y*y + x*x)
    yz_safe, xz_safe, xy_safe = (np.where(value == 0, 1, value) for value in (yz, xz, xy))
    e1 = np.stack((np.where(second, -z/xz_safe, np.where(third, -y/xy_safe, 0)),
                   np.where(first, -z/yz_safe, np.where(third, x/xy_safe, 0)),
                   np.where(first, y/yz_safe, np.where(second, x/xz_safe, 0))), -1)
    e1 = np.where(((first & (yz == 0)) | (second & (xz == 0)))[..., None],
                  np.array([0., 0., 1.]), e1)
    e1 = np.where((third & (xy == 0))[..., None], np.array([1., 0., 0.]), e1)
    first_tangent = torch.from_numpy(e1)
    second_tangent = normalize(torch.from_numpy(_cross(normals.numpy(), e1)))
    return first_tangent, second_tangent
