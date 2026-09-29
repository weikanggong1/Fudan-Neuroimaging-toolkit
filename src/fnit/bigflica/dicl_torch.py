"""GPU mini-batch dictionary learning with streamed projected modalities.

Sparse codes and sequential dictionary updates follow sklearn's Lasso-LARS
online objective; projected voxels are read from disk in bounded blocks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch

from .pipeline import _device


def _sparse_codes_lars(samples: torch.Tensor, dictionary: torch.Tensor,
                       alpha: float, max_events: int | None = None) -> torch.Tensor:
    """Batched Lasso-LARS path on CUDA, stopping at the sklearn penalty alpha."""
    gram = dictionary @ dictionary.T
    response = samples @ dictionary.T
    batch, atoms = response.shape
    code = torch.zeros_like(response)
    active = torch.zeros((batch, atoms), device=samples.device, dtype=torch.bool)
    signs = torch.zeros_like(response)
    done = response.abs().amax(dim=1) <= alpha
    initial = response.abs().argmax(dim=1)
    rows = torch.arange(batch, device=samples.device)
    active[rows, initial] = ~done
    signs[rows, initial] = torch.sign(response[rows, initial]) * (~done)
    identity = torch.eye(atoms, device=samples.device, dtype=samples.dtype)
    limit = max_events or 3 * atoms
    for _ in range(limit):
        if bool(done.all()):
            break
        correlation = response - code @ gram
        current = (correlation * active).abs().amax(dim=1)
        masked = gram[None] * active[:, :, None] * active[:, None, :]
        masked = masked + torch.diag_embed((~active).to(samples.dtype))
        direction = torch.linalg.solve(masked + identity[None] * 1e-10, signs[..., None])[..., 0]
        scale = torch.rsqrt((direction * signs).sum(dim=1).clamp_min(1e-20))
        direction = direction * scale[:, None]
        correlations_slope = direction @ gram
        target_step = ((current - alpha) / scale).clamp_min(0)
        upper = scale[:, None] - correlations_slope
        lower = scale[:, None] + correlations_slope
        positive = (current[:, None] - correlation) / upper.clamp_min(1e-20)
        negative = (current[:, None] + correlation) / lower.clamp_min(1e-20)
        possible = (~active) & (~done[:, None])
        positive = torch.where(possible & (upper > 1e-12) & (positive > 1e-12),
                               positive, torch.inf)
        negative = torch.where(possible & (lower > 1e-12) & (negative > 1e-12),
                               negative, torch.inf)
        enter_positive, positive_index = positive.min(dim=1)
        enter_negative, negative_index = negative.min(dim=1)
        entering_positive = enter_positive <= enter_negative
        enter_step = torch.minimum(enter_positive, enter_negative)
        enter_index = torch.where(entering_positive, positive_index, negative_index)
        drop = -code / torch.where(direction.abs() > 1e-20, direction,
                                   torch.ones_like(direction))
        drop = torch.where(active & (drop > 1e-12) & (code * direction < 0),
                           drop, torch.inf)
        drop_step, drop_index = drop.min(dim=1)
        step = torch.minimum(target_step, torch.minimum(enter_step, drop_step))
        step = torch.where(done, 0, step)
        code += step[:, None] * direction
        reached_target = (target_step <= enter_step) & (target_step <= drop_step)
        done |= reached_target
        dropping = (~done) & (drop_step < enter_step)
        entering = (~done) & (~dropping)
        active[rows[dropping], drop_index[dropping]] = False
        signs[rows[dropping], drop_index[dropping]] = 0
        code[rows[dropping], drop_index[dropping]] = 0
        active[rows[entering], enter_index[entering]] = True
        signs[rows[entering], enter_index[entering]] = torch.where(
            entering_positive[entering], signs.new_tensor(1.), signs.new_tensor(-1.))
    if not bool(done.all()):
        raise ValueError("LARS path exceeded sparse_iterations; increase the limit")
    return code


def fit_dicl_gpu_streaming(projected_dir: str | Path,
                           modality_names: Sequence[str], dicl_dim: int,
                           *, device: str = "cuda:0", max_iter: int = 1000,
                           batch_size: int = 32, sparse_iterations: int = 120,
                           alpha: float = 1.0, random_state: int = 0,
                           feature_block: int = 4096) -> dict[str, np.ndarray]:
    """Fit K×R dictionaries without reading a full P×R modality into RAM."""
    backend = _device(device)
    if backend.type != "cuda":
        raise ValueError("fit_dicl_gpu_streaming requires CUDA")
    if dicl_dim < 2 or max_iter < 1 or batch_size < 1 or sparse_iterations < 1:
        raise ValueError("Invalid dictionary dimensions or iteration counts")
    output = {}
    for name in modality_names:
        with h5py.File(Path(projected_dir) / f"{name}_projected.h5", "r") as file:
            projected = file["data"]
            n_voxels, n_features = projected.shape
            if n_voxels < dicl_dim:
                raise ValueError(f"Too few masked voxels for DicL: {name}")
            sums = np.zeros(n_features, dtype=np.float64)
            squares = np.zeros(n_features, dtype=np.float64)
            for start in range(0, n_voxels, feature_block):
                chunk = projected[start:start + feature_block].astype(np.float64)
                sums += chunk.sum(axis=0)
                squares += np.square(chunk).sum(axis=0)
            mean = sums / n_voxels
            std = np.sqrt(np.maximum(squares / n_voxels - mean ** 2, 0))
            std[std == 0] = 0.1
            mean_gpu = torch.as_tensor(mean, device=backend, dtype=torch.float64)
            std_gpu = torch.as_tensor(std, device=backend, dtype=torch.float64)

            covariance = torch.zeros((n_features, n_features), device=backend,
                                     dtype=torch.float64)
            for start in range(0, n_voxels, feature_block):
                samples = torch.as_tensor(projected[start:start + feature_block],
                                          device=backend, dtype=torch.float64)
                samples = (samples - mean_gpu) / std_gpu
                covariance.addmm_(samples.T, samples, alpha=1 / n_voxels)
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
            n_initial = min(dicl_dim, n_features)
            dictionary = torch.zeros((dicl_dim, n_features), device=backend,
                                     dtype=torch.float64)
            dictionary[:n_initial] = (
                eigenvectors[:, -n_initial:].flip(1).T *
                torch.sqrt(eigenvalues[-n_initial:].flip(0).clamp_min(0) * n_voxels)[:, None]
            )
            # sklearn's randomized SVD fixes signs from left singular vectors.
            largest = torch.zeros(n_initial, device=backend, dtype=torch.float64)
            orientation = torch.ones(n_initial, device=backend, dtype=torch.float64)
            for start in range(0, n_voxels, feature_block):
                samples = torch.as_tensor(projected[start:start + feature_block],
                                          device=backend, dtype=torch.float64)
                left = ((samples - mean_gpu) / std_gpu) @ dictionary[:n_initial].T
                indices = left.abs().argmax(dim=0)
                values = left[indices, torch.arange(n_initial, device=backend)]
                update = values.abs() > largest
                orientation[update] = torch.sign(values[update])
                largest = torch.maximum(largest, values.abs())
            dictionary[:n_initial] *= orientation[:, None]
            inner_a = torch.zeros((dicl_dim, dicl_dim), device=backend,
                                  dtype=torch.float64)
            inner_b = torch.zeros((n_features, dicl_dim), device=backend,
                                  dtype=torch.float64)
            rng = np.random.RandomState(random_state)
            rng.normal(size=(n_features, dicl_dim + 10))
            projected_bytes = n_voxels * n_features * 8
            keep_on_gpu = projected_bytes < min(4 * 2**30, torch.cuda.mem_get_info(backend)[0] // 4)
            samples_gpu = None
            if keep_on_gpu:
                samples_gpu = torch.as_tensor(projected[:], device=backend,
                                              dtype=torch.float64)
                samples_gpu = (samples_gpu - mean_gpu) / std_gpu
            permutation = torch.as_tensor(rng.permutation(n_voxels).copy())
            n_batches = (n_voxels + batch_size - 1) // batch_size
            step = 0
            ewa_cost = None
            best_cost = None
            no_improvement = 0
            converged = False
            for _ in range(max_iter):
                for batch_index in range(n_batches):
                    start, end = batch_index * batch_size, min((batch_index + 1) * batch_size, n_voxels)
                    indices = permutation[start:end]
                    if samples_gpu is not None:
                        samples = samples_gpu[indices.to(backend)]
                    else:
                        sorted_indices, undo = torch.sort(indices)
                        samples = torch.as_tensor(projected[sorted_indices.numpy()],
                                                  device=backend, dtype=torch.float64)
                        samples = ((samples - mean_gpu) / std_gpu)[torch.argsort(undo).to(backend)]
                    code = _sparse_codes_lars(samples, dictionary, alpha,
                                              sparse_iterations)
                    size = end - start
                    cost = (0.5 * (samples - code @ dictionary).square().sum() +
                            alpha * code.abs().sum()) / size
                    old_dictionary = dictionary.clone()
                    theta = ((step + 1) * size if step < size - 1
                             else size ** 2 + step + 1 - size)
                    beta = (theta + 1 - size) / (theta + 1)
                    inner_a.mul_(beta).addmm_(code.T, code, alpha=1 / size)
                    inner_b.mul_(beta).addmm_(samples.T, code, alpha=1 / size)
                    # sklearn updates each atom against the already updated atoms.
                    for atom in range(dicl_dim):
                        if inner_a[atom, atom] > 1e-6:
                            dictionary[atom] += ((inner_b[:, atom] - inner_a[atom] @ dictionary) /
                                                 inner_a[atom, atom])
                        else:
                            replacement = samples[int(rng.choice(size))]
                            noise_level = 0.01 * float(replacement.std(correction=0))
                            if noise_level == 0:
                                noise_level = 0.01
                            noise = torch.as_tensor(
                                rng.normal(0, noise_level, size=n_features),
                                device=backend, dtype=replacement.dtype)
                            dictionary[atom] = replacement + noise
                        dictionary[atom] /= torch.linalg.vector_norm(dictionary[atom]).clamp_min(1)
                    step += 1
                    if step <= min(100, n_voxels / size):
                        continue
                    current = float(cost)
                    ewa_cost = current if ewa_cost is None else (
                        ewa_cost * (1 - min(size / (n_voxels + 1), 1)) +
                        current * min(size / (n_voxels + 1), 1))
                    if torch.linalg.vector_norm(dictionary - old_dictionary) / dicl_dim <= 1e-3:
                        converged = True
                        break
                    if best_cost is None or ewa_cost < best_cost:
                        best_cost = ewa_cost
                        no_improvement = 0
                    else:
                        no_improvement += 1
                        if no_improvement >= 10:
                            converged = True
                            break
                if converged:
                    break
            dictionary -= dictionary.mean(dim=1, keepdim=True)
            scale = torch.sqrt(torch.mean(dictionary ** 2))
            if not torch.isfinite(scale) or scale <= 0:
                raise ValueError(f"Degenerate GPU dictionary: {name}")
            output[name] = (dictionary / scale).cpu().numpy().astype(np.float64)
    return output
