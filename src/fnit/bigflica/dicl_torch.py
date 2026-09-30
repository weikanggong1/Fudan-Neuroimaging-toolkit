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


def _randomized_svd_dictionary(projected: h5py.Dataset, samples_gpu: torch.Tensor | None,
                               mean_gpu: torch.Tensor, std_gpu: torch.Tensor,
                               n_components: int, rng: np.random.RandomState,
                               feature_block: int) -> torch.Tensor:
    """Match sklearn's seeded randomized SVD with float64 CUDA arithmetic."""
    n_voxels, n_features = projected.shape
    backend = mean_gpu.device
    n_random = n_components + 10
    q_bytes = n_voxels * n_random * 8
    if q_bytes > min(8 * 2**30, torch.cuda.mem_get_info(backend)[0] // 3):
        raise MemoryError("Randomized SVD basis exceeds the GPU memory budget")
    q = torch.as_tensor(rng.normal(size=(n_features, n_random)),
                        device=backend, dtype=torch.float64)

    def multiply_x(matrix: torch.Tensor) -> torch.Tensor:
        if samples_gpu is not None:
            return samples_gpu @ matrix
        output = torch.empty((n_voxels, matrix.shape[1]), device=backend,
                             dtype=torch.float64)
        for start in range(0, n_voxels, feature_block):
            block = torch.as_tensor(projected[start:start + feature_block],
                                    device=backend, dtype=torch.float64)
            output[start:start + block.shape[0]] = ((block - mean_gpu) / std_gpu) @ matrix
        return output

    def multiply_xt(matrix: torch.Tensor) -> torch.Tensor:
        if samples_gpu is not None:
            return samples_gpu.T @ matrix
        output = torch.zeros((n_features, matrix.shape[1]), device=backend,
                             dtype=torch.float64)
        for start in range(0, n_voxels, feature_block):
            block = torch.as_tensor(projected[start:start + feature_block],
                                    device=backend, dtype=torch.float64)
            output += ((block - mean_gpu) / std_gpu).T @ matrix[start:start + block.shape[0]]
        return output

    n_iter = 7 if n_components < 0.1 * min(n_voxels, n_features) else 4
    for _ in range(n_iter):
        q = torch.linalg.qr(multiply_x(q), mode="reduced")[0]
        q = torch.linalg.qr(multiply_xt(q), mode="reduced")[0]
    q = torch.linalg.qr(multiply_x(q), mode="reduced")[0]
    u_small, singular_values, right = torch.linalg.svd(
        multiply_xt(q).T, full_matrices=False)
    left = q @ u_small
    indices = left.abs().argmax(dim=0)
    signs = torch.sign(left[indices, torch.arange(left.shape[1], device=backend)])
    count = min(n_components, right.shape[0])
    dictionary = torch.zeros((n_components, n_features), device=backend,
                             dtype=torch.float64)
    dictionary[:count] = (singular_values[:count, None] * right[:count] *
                          signs[:count, None])
    return dictionary


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
    solve_failed = torch.zeros((), device=samples.device, dtype=torch.bool)
    limit = max_events or 3 * atoms
    for event in range(limit):
        if event % 4 == 0 and bool(done.all()):
            break
        correlation = response - code @ gram
        current = (correlation * active).abs().amax(dim=1)
        masked = gram[None] * active[:, :, None] * active[:, None, :]
        masked = masked + torch.diag_embed((~active).to(samples.dtype))
        solved = torch.linalg.solve_ex(
            masked + identity[None] * 1e-10, signs[..., None],
            check_errors=False)
        solve_failed |= (solved.info != 0).any()
        direction = solved.result[..., 0]
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
        drop_at = drop_index[:, None]
        drop_mask = dropping[:, None]
        active.scatter_(1, drop_at, active.gather(1, drop_at) & ~drop_mask)
        signs.scatter_(1, drop_at, torch.where(drop_mask, 0, signs.gather(1, drop_at)))
        code.scatter_(1, drop_at, torch.where(drop_mask, 0, code.gather(1, drop_at)))
        enter_at = enter_index[:, None]
        enter_mask = entering[:, None]
        active.scatter_(1, enter_at, active.gather(1, enter_at) | enter_mask)
        entering_sign = torch.where(entering_positive[:, None], 1., -1.)
        signs.scatter_(1, enter_at, torch.where(enter_mask, entering_sign,
                                               signs.gather(1, enter_at)))
    if bool(solve_failed):
        raise ValueError("LARS linear solve failed")
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

            projected_bytes = n_voxels * n_features * 8
            keep_on_gpu = projected_bytes < min(4 * 2**30, torch.cuda.mem_get_info(backend)[0] // 4)
            samples_gpu = None
            if keep_on_gpu:
                samples_gpu = torch.as_tensor(projected[:], device=backend,
                                              dtype=torch.float64)
                samples_gpu = (samples_gpu - mean_gpu) / std_gpu
            rng = np.random.RandomState(random_state)
            dictionary = _randomized_svd_dictionary(
                projected, samples_gpu, mean_gpu, std_gpu, dicl_dim, rng, feature_block)
            inner_a = torch.zeros((dicl_dim, dicl_dim), device=backend,
                                  dtype=torch.float64)
            inner_b = torch.zeros((n_features, dicl_dim), device=backend,
                                  dtype=torch.float64)
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
                    all_atoms_alive = bool((torch.diagonal(inner_a) > 1e-6).all())
                    for atom in range(dicl_dim):
                        if all_atoms_alive or inner_a[atom, atom] > 1e-6:
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
