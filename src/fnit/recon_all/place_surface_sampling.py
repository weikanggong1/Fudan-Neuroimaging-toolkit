# Modified FNIT implementation of FreeSurfer placement formulas.
# FreeSurfer Software License: licenses/FreeSurfer.txt; upstream d932c45.
"""Owned MRI snapshot for opt-in batched CUDA placement intensity sampling.

Coordinates are supplied afresh on every call. No dynamic mesh index is cached.
The ordered optimizer, collision acceptance and final cleanup remain on CPU.
"""
from __future__ import annotations

import numpy as np
import torch


class PlacementSampling:
    """Cache a uint8 MRI and float32 surface-RAS/mm to voxel affine on CUDA.

    ``device`` must name a CUDA index explicitly. ``chunk_size`` bounds Torch
    transient storage, not the sampling range; the fused Triton kernel consumes
    the complete vertex array and keeps fixed 128-thread blocks. The context owns copies of its inputs;
    construct a new context if the MRI or affine changes. No global precision
    flags, autocast or model caches are modified.
    """

    def __init__(self, volume, affine, *, device: str, chunk_size: int = 16384, implementation: str = "torch"):
        if implementation not in ("torch", "triton"):
            raise ValueError("implementation must be torch or triton")
        self.implementation = implementation
        self.device = torch.device(device)
        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError("PlacementSampling requires an explicit CUDA device")
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA placement sampling requested but unavailable")
        data = np.asarray(volume, dtype=np.uint8)
        matrix = np.asarray(affine, dtype=np.float32)
        if data.ndim != 3 or min(data.shape) < 1 or matrix.shape != (4, 4):
            raise ValueError("expected nonempty 3D MRI and 4x4 affine")
        if not np.isfinite(matrix).all():
            raise ValueError("affine must be finite")
        self.chunk_size = chunk_size
        self.volume = torch.tensor(data, dtype=torch.uint8, device=self.device).contiguous()
        self.affine = torch.tensor(matrix, dtype=torch.float32, device=self.device).contiguous()
        self.shape = data.shape

    def _voxel(self, points):
        # Preserve the four explicit FP32 multiply/add operations in _voxel.
        point = points.to(torch.float32)
        result = torch.zeros_like(point)
        for column in range(4):
            value = point[:, column:column+1] if column < 3 else 1.0
            result = result + value * self.affine[:3, column]
        return result.to(torch.float64)

    def _sample(self, points):
        voxel = self._voxel(points)
        inside = torch.ones(len(voxel), dtype=torch.bool, device=self.device)
        coordinates, lower, upper = [], [], []
        for axis, size in enumerate(self.shape):
            value = voxel[:, axis]
            inside = inside & (value >= -0.5) & (value < size - 0.5)
            value = value.clamp(0, size - 1)
            lo = value.to(torch.int64)
            coordinates.append(value - lo)
            lower.append(lo)
            upper.append((lo + 1).clamp(max=size-1))
        x, y, z = coordinates
        # Sum in native corner order, using double interpolation arithmetic.
        result = torch.zeros(len(voxel), dtype=torch.float64, device=self.device)
        for ix in range(2):
            for iy in range(2):
                for iz in range(2):
                    weight = (x if ix else 1-x) * (y if iy else 1-y) * (z if iz else 1-z)
                    index = [upper[a] if bit else lower[a] for a, bit in enumerate((ix, iy, iz))]
                    result = result + weight * self.volume[index[0], index[1], index[2]].to(torch.float64)
        return torch.where(inside, result, 0.0)

    @torch.no_grad()
    def sample(self, vertices: np.ndarray) -> np.ndarray:
        """Return float64 MRI values for finite (N,3) surface RAS/mm points."""
        xyz = np.asarray(vertices, dtype=np.float32)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
            raise ValueError("vertices must be finite (N,3)")
        if self.implementation == "triton" and len(xyz):
            from .place_surface_sampling_triton import sample_kernel
            points = torch.tensor(xyz, device=self.device).contiguous()
            result = torch.empty(len(xyz), dtype=torch.float64, device=self.device)
            with torch.cuda.device(self.device):
                sample_kernel[((len(xyz)+127)//128,)](self.volume, self.affine, points, result,
                    len(xyz), *self.shape, 128, enable_fp_fusion=False)
            return result.cpu().numpy()
        result = np.empty(len(xyz), dtype=np.float64)
        for start in range(0, len(xyz), self.chunk_size):
            stop = min(start + self.chunk_size, len(xyz))
            points = torch.as_tensor(xyz[start:stop], device=self.device).to(torch.float64)
            result[start:stop] = self._sample(points).cpu().numpy()
        return result

    @torch.no_grad()
    def gradient(self, vertices, normals, ripped, target_values, vertex_sigma,
                 voxel_sizes, *, weight: float = 0.2, sigma_global: float = 2.0):
        """Return native-definition float32 (N,3) intensity displacement.

        MRI sampling runs on the selected GPU; only complete chunk results cross
        to CPU. Variable sigma loops retain inclusive endpoints and repeated
        double additions. No bool/float conversion of CUDA tensors occurs.
        """
        xyz = np.asarray(vertices, dtype=np.float32)
        normal = np.asarray(normals, dtype=np.float32)
        values = np.asarray(target_values, dtype=np.float32)
        skip = np.asarray(ripped, dtype=np.bool_) | (values < 0)
        sigmas = np.asarray(vertex_sigma, dtype=np.float32).astype(np.float64)
        voxel_step = float(np.min(np.asarray(voxel_sizes, dtype=np.float32))) * 0.5
        if xyz.ndim != 2 or xyz.shape[1] != 3 or normal.shape != xyz.shape or any(
            x.shape != (len(xyz),) for x in (values, skip, sigmas)):
            raise ValueError("inconsistent placement vertex arrays")
        if not np.isfinite(xyz).all() or not np.isfinite(normal).all() or not np.isfinite(sigmas).all():
            raise ValueError("coordinates, normals and sigmas must be finite")
        if voxel_step <= 0 or not np.isfinite(voxel_step) or not np.isfinite(sigma_global):
            raise ValueError("voxel sizes must be positive; global sigma must be finite")
        sigmas[np.abs(sigmas) < 1e-10] = sigma_global
        sigmas[np.abs(sigmas) < 1e-10] = 0.25
        # A negative active sigma gives a nonterminating loop in the CPU source.
        if not np.isfinite(values).all() or not np.isfinite(weight):
            raise ValueError("target values and weight must be finite")
        if np.any(sigmas[~skip] <= 0):
            raise ValueError("active vertex sigma must be positive")
        if self.implementation == "triton" and len(xyz):
            from .place_surface_sampling_triton import gradient_kernel
            tensors = [torch.tensor(x, device=self.device).contiguous() for x in
                       (xyz, normal, values, np.asarray(vertex_sigma, dtype=np.float32), skip)]
            result = torch.empty(xyz.shape, dtype=torch.float32, device=self.device)
            with torch.cuda.device(self.device):
                gradient_kernel[((len(xyz)+127)//128,)](self.volume, self.affine,
                    *tensors, result, len(xyz), *self.shape, voxel_step,
                    float(np.float32(weight)), float(sigma_global), 128, enable_fp_fusion=False)
            return result.cpu().numpy()
        result = np.zeros_like(xyz)
        selected = np.flatnonzero(~skip)
        for start in range(0, len(selected), self.chunk_size):
            ids = selected[start:start+self.chunk_size]
            sigma_cpu = sigmas[ids]
            step_cpu = np.minimum(sigma_cpu / 2.0, voxel_step)
            # Find the native repeated-addition loop bound on CPU; never sync a
            # CUDA scalar to drive per-vertex control flow.
            distance_cpu = step_cpu.copy()
            iterations = 0
            while np.any(distance_cpu <= 2.0 * sigma_cpu):
                distance_cpu += step_cpu
                iterations += 1
            points = torch.as_tensor(xyz[ids], device=self.device).to(torch.float64)
            normal_gpu = torch.as_tensor(normal[ids], device=self.device).to(torch.float64)
            sigma = torch.as_tensor(sigma_cpu, device=self.device)
            step = torch.as_tensor(step_cpu, device=self.device)
            distance = step.clone()
            outside = torch.zeros(len(ids), dtype=torch.float64, device=self.device)
            inside = torch.zeros_like(outside)
            total = torch.zeros_like(outside)
            current = self._sample(points)
            for _ in range(iterations):
                kernel = torch.exp(-distance * distance / (2.0 * sigma * sigma))
                kernel = torch.where(distance <= 2.0 * sigma, kernel, 0.0)
                delta = distance[:, None] * normal_gpu
                outside = outside + kernel * self._sample(points + delta)
                inside = inside + kernel * self._sample(points - delta)
                total = total + kernel
                distance = distance + step
            slope = (outside / total - inside / total) / 2.0
            sign = torch.where(slope.abs() >= 1e-10, torch.sign(slope), -1.0)
            value = torch.as_tensor(values[ids], device=self.device).to(torch.float64)
            error = (value-current).clamp(-5, 5)
            displacement = float(np.float32(weight)) * error * sign
            result[ids] = (normal_gpu * displacement[:, None]).to(torch.float32).cpu().numpy()
        return result
