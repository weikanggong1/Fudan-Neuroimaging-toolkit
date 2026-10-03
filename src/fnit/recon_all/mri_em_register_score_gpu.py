"""Resident, chunked GCA search scoring; not a complete EM registration backend.

Keep VNL inverse and final ordered double reduction on CPU. The GPU performs
separate float32 coordinate products/additions, rounding and sample likelihood.
No matmul/autocast is used and the caller's global TF32 policy is unchanged.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from numba import njit

from .mri_em_register import StableSamples, _vnl_affine_inverse


@njit(cache=True)
def _density_log_terms(variances, priors):
    # Preserve the existing Numba scorer's float32 libm overload, including
    # float(np.float32) retaining FP32 under Numba. Widen only its output.
    terms = np.empty((len(variances), 2), np.float64)
    for index in range(len(variances)):
        terms[index, 0] = -math.log(math.sqrt(float(variances[index])))
        terms[index, 1] = math.log(float(priors[index]))
    return terms


class GCASearchScorer:
    """Cache one immutable source/sample set on an explicit CUDA device.

    source: 3D uint8 intensity, x/y/z voxel order; samples: existing fixed-T1
    prior-grid samples (spacing 2 source voxels). Matrices map source voxels to
    atlas voxels. Outputs are CPU float32 scores in input candidate order.
    candidate_chunk=64, sample_chunk=8192 bound temporary GPU tensors; final
    CPU workspace is candidate_chunk * sample_count * 8 bytes. This class does
    not mutate source/samples and must be recreated after either changes.
    Empty/invalid samples, nonfinite/singular matrices or non-CUDA device raise.
    """

    def __init__(self, samples: StableSamples, source: np.ndarray, *,
                 device: str | torch.device, candidate_chunk: int = 64,
                 sample_chunk: int = 8192):
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('GCASearchScorer requires an explicit CUDA device')
        for value in (candidate_chunk, sample_chunk):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError('chunk sizes must be positive integers')
        if source.ndim != 3 or source.dtype != np.uint8 or min(source.shape) < 1:
            raise ValueError('source must be a nonempty 3D uint8 array')
        count = len(samples.means)
        if not count or samples.coordinates.shape != (count, 3):
            raise ValueError('samples must contain nonempty (N,3) coordinates')
        arrays = (samples.means, samples.variances, samples.priors)
        if any(a.dtype != np.float32 for a in arrays):
            raise ValueError('fixed-T1 density arrays must have float32 dtype')
        if any(a.shape != (count,) or not np.isfinite(a).all() for a in arrays):
            raise ValueError('sample density arrays must be finite length N')
        if np.any(samples.variances <= 0) or np.any(samples.priors <= 0):
            raise ValueError('sample variances and priors must be positive')
        if not np.isfinite(samples.coordinates).all() or samples.labels.shape != (count,):
            raise ValueError("sample coordinates must be finite and labels have length N")
        self._samples, self._source = samples, source
        self.count, self.shape = count, source.shape
        self.candidate_chunk, self.sample_chunk = candidate_chunk, sample_chunk
        def upload(array, dtype):
            return torch.tensor(np.asarray(array), dtype=dtype, device=self.device)
        self.source = upload(source, torch.uint8)
        self.coordinates = upload(samples.coordinates, torch.float32)
        self.means = upload(samples.means, torch.float32)
        self.variances = upload(samples.variances, torch.float32)
        # Cache the actual reference arithmetic, not Python's FP64 libm variant.
        terms = _density_log_terms(samples.variances, samples.priors)
        self.log_std = upload(terms[:, 0], torch.float64)
        self.log_prior = upload(terms[:, 1], torch.float64)

    @torch.no_grad()
    def score_many(self, matrices: np.ndarray) -> np.ndarray:
        """Score finite (B,4,4) affine float32 matrices, preserving candidate order."""
        matrices = np.asarray(matrices, dtype=np.float32)
        if matrices.ndim != 3 or matrices.shape[1:] != (4, 4):
            raise ValueError('matrices must have shape (B,4,4)')
        if not np.isfinite(matrices).all() or not np.all(matrices[:, 3] == [0, 0, 0, 1]):
            raise ValueError('matrices must be finite affine transforms')
        scores = np.empty(len(matrices), np.float32)
        for begin in range(0, len(matrices), self.candidate_chunk):
            block = matrices[begin:begin + self.candidate_chunk]
            inverses = np.stack([_vnl_affine_inverse(m) for m in block])
            if not np.isfinite(inverses).all():
                raise ValueError('matrices must be nonsingular')
            # Multiplication by the diagonal prior spacing is exact here.
            inverses[:, :, :3] *= np.float32(2)
            transform = torch.tensor(inverses, device=self.device)
            values = np.empty((len(block), self.count), np.float64)
            for start in range(0, self.count, self.sample_chunk):
                stop = min(self.count, start + self.sample_chunk)
                coordinates = self.coordinates[start:stop]
                axes = []
                for axis in range(3):
                    position = torch.zeros((len(block), stop-start), device=self.device, dtype=torch.float32)
                    for inner in range(3):
                        position = position + transform[:, axis, inner, None] * coordinates[None, :, inner]
                    position = position + transform[:, axis, 3, None]
                    axes.append(torch.where(position < 0, torch.ceil(position.double() - .5),
                                            torch.floor(position.double() + .5)).to(torch.int64))
                inside = torch.ones_like(axes[0], dtype=torch.bool)
                for axis, size in zip(axes, self.shape):
                    inside &= (axis >= 0) & (axis < size)
                intensity = self.source[tuple(a.clamp(0, s-1) for a, s in zip(axes, self.shape))].float()
                residual = intensity - self.means[None, start:stop]
                # CUDA float division can use reciprocal approximation. Divide
                # FP32 operands in FP64 then round once to keep CPU FP32 quotient.
                squared = residual * residual
                mahalanobis = (squared.double() / self.variances[None, start:stop].double()).float()
                likelihood = self.log_std[None, start:stop] - .5 * mahalanobis.double()
                likelihood = likelihood + self.log_prior[None, start:stop]
                likelihood = torch.where(inside, likelihood.clamp_min(-6), -1000000.)
                values[:, start:stop] = likelihood.cpu().numpy()
            scores[begin:begin+len(block)] = (np.sum(values, axis=1, dtype=np.float64).astype(np.float32)
                                            / np.float32(self.count))
        return scores

    def __call__(self, samples, source, matrix):
        """Scalar compatibility hook; use score_many for candidate batches."""
        if samples is not self._samples or source is not self._source:
            raise ValueError("scorer belongs to a different source/sample context")
        return float(self.score_many(np.asarray(matrix)[None])[0])
