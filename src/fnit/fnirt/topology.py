"""FSL 2203.0 topology projection used by FNIRT.

The implementation follows ``warpfns::constrain_topology``.  Warps are
component-first absolute scaled-mm coordinates on a regular grid.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch


_JACOBIAN_OFFSETS = (
    ((0, 0, 0), (0, 0, 0), (0, 0, 0)),
    ((0, 0, 0), (1, 0, 0), (1, 0, 0)),
    ((0, 1, 0), (0, 0, 0), (0, 1, 0)),
    ((0, 0, 1), (0, 0, 1), (0, 0, 0)),
    ((0, 1, 1), (0, 0, 1), (0, 1, 0)),
    ((0, 0, 1), (1, 0, 1), (1, 0, 0)),
    ((0, 1, 0), (1, 0, 0), (1, 1, 0)),
    ((0, 1, 1), (1, 0, 1), (1, 1, 0)),
)


def _determinant(matrix):
    """Three-by-three determinant with the operation order used by warpfns."""
    return (
        matrix[..., 0, 0]
        * (
            matrix[..., 1, 1] * matrix[..., 2, 2]
            - matrix[..., 1, 2] * matrix[..., 2, 1]
        )
        - matrix[..., 0, 1]
        * (
            matrix[..., 1, 0] * matrix[..., 2, 2]
            - matrix[..., 1, 2] * matrix[..., 2, 0]
        )
        + matrix[..., 0, 2]
        * (
            matrix[..., 1, 0] * matrix[..., 2, 1]
            - matrix[..., 1, 1] * matrix[..., 2, 0]
        )
    )


def gradient_field(warp, voxel_sizes):
    """Return the nine periodic forward differences used by ``grad_calc``."""
    if warp.ndim != 4 or warp.shape[0] != 3:
        raise ValueError("warp must have shape [3, X, Y, Z]")
    voxel_sizes = tuple(float(value) for value in voxel_sizes)
    rows = []
    for component in range(3):
        # warpfns 2203.0 divides every derivative in a row by the voxel
        # dimension corresponding to that output component.
        denominator = voxel_sizes[component]
        rows.extend(
            (torch.roll(warp[component], -1, dims=axis) - warp[component])
            / denominator
            for axis in range(3)
        )
    return torch.stack(rows)


def jacobian_check(warp, voxel_sizes):
    """Return the eight corner Jacobians for every voxel cube."""
    gradient = gradient_field(warp, voxel_sizes)
    cube_shape = tuple(size - 1 for size in warp.shape[1:])
    values = []
    for offsets in _JACOBIAN_OFFSETS:
        columns = []
        for column, offset in enumerate(offsets):
            x, y, z = offset
            columns.append(
                gradient[
                    column :: 3,
                    x : x + cube_shape[0],
                    y : y + cube_shape[1],
                    z : z + cube_shape[2],
                ].movedim(0, -1)
            )
        matrix = torch.stack(columns, dim=-1)
        values.append(_determinant(matrix))
    return torch.stack(values)


def jacobian_stats(jacobians, minimum, maximum):
    """Return FSL's initialized min, max and out-of-range counts."""
    one = jacobians.new_tensor(1.0)
    return {
        "minimum": float(torch.minimum(one, jacobians.min())),
        "maximum": float(torch.maximum(one, jacobians.max())),
        "below": int((jacobians < minimum).sum()),
        "above": int((jacobians > maximum).sum()),
    }


def limit_gradient(
    gradient, jacobians, minimum, maximum, *, return_backend=False
):
    """Apply ``limit_grad`` in FSL's voxel-major, corner-minor order."""
    if gradient.is_cuda:
        try:
            from ._topology_triton import limit_gradient_serial_cuda
        except ImportError:
            cpu = limit_gradient(
                gradient.cpu(), jacobians.cpu(), minimum, maximum
            )
            gradient.copy_(cpu.to(device=gradient.device))
            backend = "numpy-cpu-fallback-no-triton"
        else:
            gradient = limit_gradient_serial_cuda(
                gradient, jacobians, float(minimum), float(maximum)
            )
            backend = "triton-cuda-serial-order"
        return (gradient, backend) if return_backend else gradient

    values = gradient.detach().contiguous().numpy()
    checks = jacobians.detach().contiguous().numpy()
    nx, ny, nz = values.shape[1:]
    identity = np.eye(3, dtype=np.float64)
    # Checks remain fixed throughout the original limiter loop. Filter only
    # its no-op corners, in the same z/y/x/corner order. Matrix extraction,
    # NumPy determinant, alpha rounding and overlapping updates stay serial.
    from ._topology_cpu import out_of_range_corners
    for flat in out_of_range_corners(checks, float(minimum), float(maximum)):
        index = int(flat % 8)
        voxel = int(flat // 8)
        x = voxel % (nx - 1)
        voxel //= nx - 1
        y = voxel % (ny - 1)
        z = voxel // (ny - 1)
        offsets = _JACOBIAN_OFFSETS[index]
        matrix = np.empty((3, 3), dtype=np.float64)
        for row in range(3):
            for column, offset in enumerate(offsets):
                dx, dy, dz = offset
                matrix[row, column] = values[
                    3 * row + column, x + dx, y + dy, z + dz
                ]
        alpha = np.float32(0.0)
        candidate = matrix
        determinant = np.linalg.det(candidate)
        while determinant < minimum or determinant > maximum:
            alpha = np.float32(alpha + np.float32(0.1))
            alpha = min(alpha, np.float32(1.0))
            candidate = (1.0 - float(alpha)) * matrix + float(alpha) * identity
            determinant = np.linalg.det(candidate)
        alpha = min(np.float32(alpha + np.float32(0.1)), np.float32(1.0))
        for row in range(3):
            for column, offset in enumerate(offsets):
                dx, dy, dz = offset
                values[3 * row + column, x + dx, y + dy, z + dz] = np.float32(
                    (1.0 - float(alpha)) * matrix[row, column]
                    + float(alpha) * identity[row, column]
                )
    backend = "numpy-cpu-serial-order-numba-filter"
    return (gradient, backend) if return_backend else gradient


def _rounded_fft(values, inverse=False):
    """Three one-dimensional double FFTs with float storage between axes."""
    result = values.to(torch.complex64)
    transform = torch.fft.ifft if inverse else torch.fft.fft
    for axis in range(3):
        result = transform(result.to(torch.complex128), dim=axis).to(torch.complex64)
    return result


def integrate_gradient_field(gradient, means, voxel_sizes):
    """Project a nine-volume gradient onto integrable periodic fields."""
    shape = gradient.shape[1:]
    frequency_factors = []
    for size, device in zip(shape, (gradient.device,) * 3):
        index = torch.arange(size, dtype=torch.float32, device=device)
        angle = index * (torch.tensor(2.0 * math.pi, device=device) / size)
        frequency_factors.append(
            torch.complex(torch.cos(angle) - 1.0, -torch.sin(angle))
        )
    denominator = sum(
        (factor.real.square() + factor.imag.square()).reshape(
            tuple(size if index == axis else 1 for index, size in enumerate(shape))
        )
        for axis, factor in enumerate(frequency_factors)
    )

    outputs = []
    for component in range(3):
        dot = None
        for axis in range(3):
            transformed = _rounded_fft(gradient[3 * component + axis])
            factor = frequency_factors[axis].reshape(
                tuple(size if index == axis else 1 for index, size in enumerate(shape))
            )
            term = transformed * factor
            dot = term if dot is None else dot + term
        spectrum = torch.where(
            denominator > 1e-12,
            dot / denominator.clamp_min(1e-12),
            torch.zeros_like(dot),
        )
        integrated = _rounded_fft(spectrum, inverse=True).real
        integrated = integrated * float(voxel_sizes[component])
        outputs.append(integrated + means[component])
    return torch.stack(outputs)


def constrain_topology(warp, voxel_sizes, minimum, maximum, max_iterations=9):
    """Port of ``warpfns::constrain_topology`` for an absolute warp."""
    started = time.perf_counter()
    result = warp.clone()
    history = []
    limiter_backend = None
    for _ in range(int(max_iterations)):
        jacobians = jacobian_check(result, voxel_sizes)
        stats = jacobian_stats(jacobians, minimum, maximum)
        history.append(stats)
        if stats["below"] == 0 and stats["above"] == 0:
            break
        means = tuple(
            result[index].to(torch.float64).mean().to(result.dtype)
            for index in range(3)
        )
        gradient = gradient_field(result, voxel_sizes)
        gradient, limiter_backend = limit_gradient(
            gradient,
            jacobians,
            float(minimum),
            float(maximum),
            return_backend=True,
        )
        result = integrate_gradient_field(gradient, means, voxel_sizes)
    final = jacobian_stats(
        jacobian_check(result, voxel_sizes), minimum, maximum
    )
    return result, {
        "iterations": len(history),
        "history": history,
        "final": final,
        "limit_gradient_backend": limiter_backend or "not-required",
        "elapsed_seconds": time.perf_counter() - started,
    }


__all__ = [
    "constrain_topology",
    "gradient_field",
    "integrate_gradient_field",
    "jacobian_check",
    "jacobian_stats",
    "limit_gradient",
]
