"""CUDA tensor updates for the single-group, scalar-noise FLICA model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import h5py
import numpy as np
import torch

from .pipeline import _device


@dataclass
class RawVoxelMatrix:
    """A normalized subject×voxel HDF5 dataset viewed as voxel×subject."""

    dataset: h5py.Dataset
    feature_block: int
    squared_sum: float = 0.0

    @property
    def shape(self) -> tuple[int, int]:
        return self.dataset.shape[1], self.dataset.shape[0]

    def blocks(self, device: torch.device):
        for start in range(0, self.shape[0], self.feature_block):
            end = min(start + self.feature_block, self.shape[0])
            yield start, torch.as_tensor(self.dataset[:, start:end].T.copy(),
                                         device=device, dtype=torch.float64)


def _tensor(value: object, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float64).clone()
    return torch.as_tensor(np.asarray(value, dtype=np.float64).copy(),
                           dtype=torch.float64, device=device)


def _scaled_inverse(matrix: torch.Tensor) -> torch.Tensor:
    scale = torch.rsqrt(torch.diagonal(matrix))
    return scale[:, None] * torch.linalg.inv(scale[:, None] * matrix * scale[None, :]) * scale[None, :]


def _estimate_dd_gpu(data: torch.Tensor) -> torch.Tensor:
    """Match the two-point Marchenko–Pastur eigenspectrum DOF estimate on CUDA."""
    n_voxels, n_reduced = data.shape
    eigenvalues = torch.linalg.eigvalsh(data.T @ data).flip(0).clamp_min(0)
    return _estimate_dd_eigenvalues(eigenvalues, n_voxels, n_reduced)


def _estimate_dd_eigenvalues(eigenvalues: torch.Tensor, n_voxels: int,
                             n_reduced: int) -> torch.Tensor:
    if n_reduced >= n_voxels:
        return eigenvalues.new_tensor(1.0)
    indices = (int(np.ceil(n_reduced * 0.25) - 1),
               int(np.floor(n_reduced * 0.75) - 1))
    observed = eigenvalues[list(indices)]
    quantiles = eigenvalues.new_tensor([1 - index / (n_reduced - 1) for index in indices])

    def score(gamma: torch.Tensor) -> torch.Tensor:
        root = torch.sqrt(gamma)[:, None]
        lower = (1 - root) ** 2
        upper = (1 + root) ** 2
        positions = torch.linspace(0, 1, 4096, device=eigenvalues.device,
                                   dtype=torch.float64)[None, :]
        grid = lower + (upper - lower) * positions
        density = (torch.sqrt(((grid - lower) * (upper - grid)).clamp_min(0)) /
                   (2 * torch.pi * gamma[:, None] * grid))
        step = (upper - lower) / 4095
        cdf = torch.cumsum(density, dim=1) * step
        cdf[:, -1] = 1
        positions = torch.searchsorted(cdf.contiguous(), quantiles.expand(len(gamma), -1).contiguous())
        positions = positions.clamp(1, 4095)
        before = positions - 1
        low_cdf = torch.gather(cdf, 1, before)
        high_cdf = torch.gather(cdf, 1, positions)
        low_grid = torch.gather(grid, 1, before)
        high_grid = torch.gather(grid, 1, positions)
        estimate = low_grid + (high_grid - low_grid) * (
            (quantiles[None, :] - low_cdf) / (high_cdf - low_cdf).clamp_min(1e-12)
        )
        scale = (estimate * observed[None, :]).sum(dim=1) / estimate.square().sum(dim=1)
        error = ((estimate * scale[:, None] - observed[None, :]) ** 2).sum(dim=1)
        return error

    coarse = torch.linspace(0.001, 0.999, 1000, device=eigenvalues.device, dtype=torch.float64)
    best = coarse[torch.argmin(score(coarse))]
    fine = torch.linspace(-0.01, 0.01, 1000, device=eigenvalues.device, dtype=torch.float64)
    candidates = (best + fine).clamp(0.001, 0.999)
    gamma = candidates[torch.argmin(score(candidates))]
    return n_reduced / (gamma * n_voxels)


def initialize_flica_torch(y_arrays: Sequence[np.ndarray], n_components: int,
                           *, device: str = "cuda:0") -> tuple[dict, dict, dict]:
    """Initialize the original scalar-noise FLICA model using CUDA tensors."""
    backend = _device(device)
    if backend.type != "cuda":
        raise ValueError("CUDA device required for FLICA GPU initialization")
    y = [_tensor(value, backend) for value in y_arrays]
    n_modalities, n_reduced = len(y), y[0].shape[1]
    if not 1 <= n_components < n_reduced:
        raise ValueError("Invalid FLICA component count")
    dd = torch.stack([_estimate_dd_gpu(value) for value in y])
    covariance = sum(dd[k] * (value.T @ value) for k, value in enumerate(y))
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    dominant = eigenvalues[-n_components:].flip(0)
    h = dominant[:, None] * eigenvectors[:, -n_components:].flip(1).T
    x = [((torch.linalg.pinv(h @ h.T) @ h) @ (value.T * torch.sqrt(dd[k]))).T
         for k, value in enumerate(y)]
    return _initial_state(h, x, dd, [value.shape[0] for value in y],
                          [value.square().mean() for value in y], backend)


def _initial_state(h: torch.Tensor, x: list[torch.Tensor], dd: torch.Tensor,
                   n_voxels: Sequence[int], mean_squares: Sequence[torch.Tensor],
                   backend: torch.device) -> tuple[dict, dict, dict]:
    n_modalities, n_components, n_reduced = len(x), h.shape[0], h.shape[1]
    unit = torch.eye(n_components, device=backend, dtype=torch.float64)
    mean_dd = dd.mean()
    w = [torch.ones(n_components, device=backend, dtype=torch.float64) *
         torch.sqrt(mean_dd / dd[k]) for k in range(n_modalities)]
    w_cov = [unit * 1e-12 for _ in range(n_modalities)]
    wtw = [torch.outer(weight, weight) + w_cov[k] for k, weight in enumerate(w)]
    xtdx = [dd[k] * (x[k].T @ x[k]) for k in range(n_modalities)]
    prior_pi = [torch.full((3, n_components), n_voxels[k] * 0.1,
                           device=backend, dtype=torch.float64) for k in range(n_modalities)]
    prior_beta_b = [torch.tensor([1000., 1., 1000.], device=backend,
                                 dtype=torch.float64)[:, None].expand(-1, n_components)
                    for _ in range(n_modalities)]
    prior_beta_c = [torch.tensor([1e-6, 1e6, 1e-6], device=backend,
                                 dtype=torch.float64)[:, None].expand(-1, n_components)
                    for _ in range(n_modalities)]
    beta = [torch.tensor([100., 1e-6, 1.], device=backend,
                         dtype=torch.float64)[:, None].expand(-1, n_components).clone()
            for _ in range(n_modalities)]
    prior_lambda_b = [h.new_tensor(1e12) for _ in range(n_modalities)]
    prior_lambda_c = [h.new_tensor(1e-12) for _ in range(n_modalities)]
    lam = [torch.reciprocal(mean_squares[k]) for k in range(n_modalities)]
    priors = {"prior_pi_weights": prior_pi, "prior_beta_b": prior_beta_b,
              "prior_beta_c": prior_beta_c, "prior_W_var": 1 / dd[-1],
              "prior_mu_var": h.new_tensor(1e4),
              "prior_eta_b": h.new_tensor(1e6),
              "prior_eta_c": h.new_tensor(1e-3),
              "prior_lambda_b": prior_lambda_b,
              "prior_lambda_c": prior_lambda_c}
    posteriors = {"H": h, "H_colcov": unit * 1e-12,
                  "H2Gmat": h.square().sum(dim=1),
                  "eta": torch.ones(n_components, device=backend,
                                    dtype=torch.float64) * 1e3,
                  "X": x, "X2": [value.square() for value in x],
                  "W": w, "W_rowcov": w_cov, "WtW": wtw, "XtDX": xtdx,
                  "Lambda": lam,
                  "mu": [torch.zeros((3, n_components), device=backend,
                                     dtype=torch.float64) for _ in range(n_modalities)],
                  "mu2": [torch.zeros((3, n_components), device=backend,
                                      dtype=torch.float64) for _ in range(n_modalities)],
                  "beta": beta,
                  "beta_log": [torch.log(value) for value in beta],
                  "beta_c": [torch.full((3, n_components), 1e6, device=backend,
                                          dtype=torch.float64) for _ in range(n_modalities)],
                  "pi_weights": [value.clone() for value in prior_pi],
                  "pi_log": [torch.full((3, n_components), -np.log(3),
                                          device=backend, dtype=torch.float64) for _ in range(n_modalities)]}
    constants = {"K": n_modalities, "L": n_components, "R": n_reduced,
                 "DD": dd}
    return priors, posteriors, constants


def initialize_flica_raw(y: Sequence[RawVoxelMatrix], n_components: int,
                         *, device: str = "cuda:0", max_gpu_gb: float = 28.0
                         ) -> tuple[dict, dict, dict]:
    """Initialize direct-voxel FLICA without holding any modality matrix in RAM."""
    backend = _device(device)
    if backend.type != "cuda" or not y:
        raise ValueError("Direct-voxel FLICA requires CUDA and modalities")
    n_subjects = y[0].shape[1]
    if not 1 <= n_components < n_subjects or any(value.shape[1] != n_subjects for value in y):
        raise ValueError("Invalid raw FLICA dimensions")
    dtype = torch.float64
    matrix_bytes = n_subjects ** 2 * 8
    spatial_bytes = (sum(value.shape[0] for value in y) * n_components * 32 +
                     max(value.shape[0] for value in y) * n_components * 8)
    if matrix_bytes * 2 + spatial_bytes + 2 * 2**30 > max_gpu_gb * 2**30:
        raise MemoryError("Direct FLICA exceeds max_gpu_gb; enable mMIGP+DicL")
    total_covariance = torch.zeros((n_subjects, n_subjects),
                                   device=backend, dtype=dtype)
    dd_values = []
    mean_squares = []
    for value in y:
        covariance = torch.zeros_like(total_covariance)
        total_squares = 0.0
        for _, block in value.blocks(backend):
            covariance.addmm_(block.T, block)
            total_squares += float(block.square().sum())
        value.squared_sum = total_squares
        mean_squares.append(torch.as_tensor(
            total_squares / (value.shape[0] * n_subjects),
            device=backend, dtype=torch.float64))
        if n_subjects <= 2048:
            spectrum = torch.linalg.eigvalsh(covariance).flip(0).clamp_min(0)
            dd = _estimate_dd_eigenvalues(spectrum, *value.shape)
        else:
            # The original full-spectrum estimator is cubic in subject count.
            dd = covariance.new_tensor(1.0)
        dd_values.append(dd.to(torch.float64))
        total_covariance += covariance * dd.to(dtype)
    dd = torch.stack(dd_values)
    if n_subjects <= 2048:
        eigenvalues, eigenvectors = torch.linalg.eigh(total_covariance)
        eigenvalues = eigenvalues[-n_components:].flip(0)
        eigenvectors = eigenvectors[:, -n_components:].flip(1)
    else:
        rank = min(n_subjects, n_components + 32)
        generator = torch.Generator(device=backend).manual_seed(0)
        basis = torch.randn((n_subjects, rank), device=backend,
                            dtype=dtype, generator=generator)
        for iteration in range(30):
            basis, _ = torch.linalg.qr(total_covariance @ basis, mode="reduced")
            if iteration >= 5 and iteration % 3 == 2:
                reduced = basis.T @ total_covariance @ basis
                values, vectors = torch.linalg.eigh(reduced)
                trial_values = values[-n_components:].flip(0)
                trial_vectors = basis @ vectors[:, -n_components:].flip(1)
                residual = torch.linalg.vector_norm(
                    total_covariance @ trial_vectors - trial_vectors * trial_values)
                if residual / torch.linalg.vector_norm(trial_vectors * trial_values) < 1e-8:
                    break
        reduced = basis.T @ total_covariance @ basis
        values, vectors = torch.linalg.eigh(reduced)
        eigenvalues = values[-n_components:].flip(0)
        eigenvectors = basis @ vectors[:, -n_components:].flip(1)
    h = (eigenvalues[:, None] * eigenvectors.T).to(torch.float64)
    del covariance, total_covariance
    projector = torch.linalg.pinv(h @ h.T) @ h
    spatial = []
    for k, value in enumerate(y):
        x = torch.empty((value.shape[0], n_components), device=backend,
                        dtype=torch.float64)
        for start, block in value.blocks(backend):
            x[start:start + block.shape[0]] = (
                block @ projector.T) * torch.sqrt(dd[k])
        spatial.append(x)
    return _initial_state(h, spatial, dd, [value.shape[0] for value in y],
                          mean_squares, backend)


def iterate_flica_torch(y_arrays: Sequence[np.ndarray | RawVoxelMatrix], priors: dict,
                        posteriors: dict, constants: dict, max_iter: int,
                        *, device: str = "cuda:0") -> dict[str, object]:
    """Run original FLICA coordinate updates with the compressed arrays on GPU.

    Initial state uses the source-compatible FLICA initializer; this function
    supports the BigFLICA option ``lambda_dims='o'`` only.
    """
    backend = _device(device)
    if backend.type != "cuda":
        raise ValueError("CUDA device required for FLICA GPU iteration")
    if max_iter < 0:
        raise ValueError("max_iter must be nonnegative")
    n_modalities, n_components, n_reduced = (int(constants[key]) for key in ("K", "L", "R"))
    y = [value if isinstance(value, RawVoxelMatrix) else _tensor(value, backend)
         for value in y_arrays]

    def cross(index: int, spatial: torch.Tensor) -> torch.Tensor:
        if isinstance(y[index], RawVoxelMatrix):
            result = torch.zeros((n_components, n_reduced), device=backend,
                                 dtype=torch.float64)
            for start, block in y[index].blocks(backend):
                result += spatial[start:start + block.shape[0]].T @ block
            return result
        return spatial.T @ y[index]

    def projected(index: int, temporal: torch.Tensor) -> torch.Tensor:
        if isinstance(y[index], RawVoxelMatrix):
            result = torch.empty((y[index].shape[0], n_components),
                                 device=backend, dtype=torch.float64)
            for start, block in y[index].blocks(backend):
                result[start:start + block.shape[0]] = block @ temporal
            return result
        return y[index] @ temporal

    def squared_sum(index: int) -> torch.Tensor:
        if isinstance(y[index], RawVoxelMatrix):
            return torch.as_tensor(y[index].squared_sum, device=backend,
                                   dtype=torch.float64)
        return y[index].square().sum()
    dd = _tensor(constants["DD"], backend)
    h = _tensor(posteriors["H"], backend)
    h_cov = _tensor(posteriors["H_colcov"], backend)
    h2 = _tensor(posteriors["H2Gmat"], backend).reshape(-1)
    eta = _tensor(posteriors["eta"], backend).reshape(-1)
    x = [_tensor(value, backend) for value in posteriors["X"]]
    x2 = [_tensor(value, backend) for value in posteriors["X2"]]
    w = [_tensor(value, backend).reshape(-1) for value in posteriors["W"]]
    w_cov = [_tensor(value, backend) for value in posteriors["W_rowcov"]]
    wtw = [_tensor(value, backend) for value in posteriors["WtW"]]
    xtdx = [_tensor(value, backend) for value in posteriors["XtDX"]]
    lam = [_tensor(value, backend).reshape(-1)[0] for value in posteriors["Lambda"]]
    mu = [_tensor(value, backend) for value in posteriors["mu"]]
    mu2 = [_tensor(value, backend) for value in posteriors["mu2"]]
    beta = [_tensor(value, backend) for value in posteriors["beta"]]
    beta_log = [_tensor(value, backend) for value in posteriors["beta_log"]]
    pi_log = [_tensor(value, backend) for value in posteriors["pi_log"]]
    beta_c = [_tensor(value, backend) for value in posteriors["beta_c"]]
    pi_weights = [_tensor(value, backend) for value in posteriors["pi_weights"]]
    prior_pi = [_tensor(value, backend) for value in priors["prior_pi_weights"]]
    prior_beta_b = [_tensor(value, backend) for value in priors["prior_beta_b"]]
    prior_beta_c = [_tensor(value, backend) for value in priors["prior_beta_c"]]
    prior_w_var = _tensor(priors["prior_W_var"], backend).reshape(-1)[0]
    prior_mu_var = _tensor(priors["prior_mu_var"], backend).reshape(-1)[0]
    prior_eta_b = _tensor(priors["prior_eta_b"], backend).reshape(-1)[0]
    prior_eta_c = _tensor(priors["prior_eta_c"], backend).reshape(-1)[0]
    prior_lambda_b = [_tensor(value, backend).reshape(-1)[0]
                      for value in priors["prior_lambda_b"]]
    prior_lambda_c = [_tensor(value, backend).reshape(-1)[0]
                      for value in priors["prior_lambda_c"]]
    unit = torch.eye(n_components, device=backend, dtype=torch.float64)
    h_pcs = torch.zeros((n_modalities + 1, n_components), device=backend,
                        dtype=torch.float64)
    sum_y2 = [squared_sum(index) for index in range(n_modalities)]

    # Upstream indexes from zero and exits after maxits + 1 updates.
    for _ in range(max(2, max_iter + 1)):
        eta_c = prior_eta_c + n_reduced / 2
        eta_binv = 1 / prior_eta_b + h2 / 2
        eta = eta_c / eta_binv

        precision = torch.diag(eta)
        mean_term = torch.zeros((n_components, n_reduced), device=backend,
                                dtype=torch.float64)
        old_cross = [cross(k, x[k]) for k in range(n_modalities)]
        for k in range(n_modalities):
            precision += lam[k] * (wtw[k] * xtdx[k].T)
            mean_term += dd[k] * lam[k] * (w[k][:, None] * old_cross[k])
            h_pcs[k] = torch.diagonal(wtw[k]) * torch.diagonal(xtdx[k]) * lam[k]
        h_cov = _scaled_inverse(precision)
        h = h_cov @ mean_term
        h2 = h.square().sum(dim=1) + n_reduced * torch.diagonal(h_cov)
        h_pcs[-1] = eta

        hlambda = []
        for k in range(n_modalities):
            weighted_h = lam[k] * (h @ h.T + n_reduced * h_cov)
            hlambda.append(weighted_h)
            w_cov[k] = _scaled_inverse(xtdx[k] * weighted_h + unit / prior_w_var)
            target = dd[k] * lam[k] * torch.diagonal(old_cross[k] @ h.T)
            w[k] = target @ w_cov[k]
            wtw[k] = torch.outer(w[k], w[k]) + w_cov[k]

        for k in range(n_modalities):
            projected_values = projected(k, h.T * (lam[k] * w[k])[None, :])
            sum_q = torch.zeros((3, n_components), device=backend, dtype=torch.float64)
            sum_qx = torch.zeros_like(sum_q)
            sum_qx2 = torch.zeros_like(sum_q)
            for component in range(n_components):
                x[k][:, component] = 0
                residual = (projected_values[:, component] - x[k] @
                            (wtw[k][:, component] * hlambda[k][:, component]))
                precision_three = (wtw[k][component, component] *
                                   hlambda[k][component, component] + beta[k][:, component])
                variance_three = 1 / precision_three
                component_mean = ((residual[:, None] +
                                   (beta[k][:, component] * mu[k][:, component])[None, :]) /
                                  precision_three[None, :])
                component_second = component_mean.square() + variance_three[None, :]
                logits = (0.5 * (torch.log(variance_three) + beta_log[k][:, component] -
                                 beta[k][:, component] * mu2[k][:, component]) +
                          pi_log[k][:, component])[None, :]
                logits = logits + component_mean.square() / (2 * variance_three[None, :])
                responsibilities = torch.softmax(logits, dim=1)
                sum_q[:, component] = dd[k] * responsibilities.sum(dim=0)
                sum_qx[:, component] = dd[k] * (responsibilities * component_mean).sum(dim=0)
                sum_qx2[:, component] = dd[k] * (responsibilities * component_second).sum(dim=0)
                x[k][:, component] = (responsibilities * component_mean).sum(dim=1)
                x2[k][:, component] = (responsibilities * component_second).sum(dim=1)

            xtdx[k] = dd[k] * (x[k].T @ x[k])
            xtdx[k].diagonal().copy_(dd[k] * x2[k].sum(dim=0))
            pi_weights[k] = prior_pi[k] + sum_q
            pi_log[k] = (torch.special.digamma(pi_weights[k]) -
                         torch.special.digamma(pi_weights[k].sum(dim=0))[None, :])
            beta_c[k] = prior_beta_c[k] + sum_q / 2
            beta_binv = (1 / prior_beta_b[k] +
                         (sum_q * mu2[k] + sum_qx2 - 2 * mu[k] * sum_qx) / 2)
            beta[k] = beta_c[k] / beta_binv
            beta_log[k] = torch.special.digamma(beta_c[k]) - torch.log(beta_binv)
            mu_precision = 1 / prior_mu_var + beta[k] * sum_q
            mu[k] = beta[k] * sum_qx / mu_precision
            mu2[k] = mu[k].square() + 1 / mu_precision

        for k in range(n_modalities):
            covariance_weight = wtw[k] * xtdx[k]
            diagonal_term = (h * (covariance_weight @ h)).sum(dim=0)
            diagonal_term = diagonal_term + (covariance_weight * h_cov).sum()
            new_cross = cross(k, x[k])
            fitted = ((new_cross * h) * w[k][:, None]).sum(dim=0) * dd[k]
            residual_sum = dd[k] * sum_y2[k] / 2 - fitted.sum() + diagonal_term.sum() / 2
            shape = dd[k] * y[k].shape[0] * n_reduced / 2 + prior_lambda_c[k]
            rate = residual_sum + 1 / prior_lambda_b[k]
            lam[k] = shape / rate
            if not torch.isfinite(lam[k]) or lam[k] <= 0:
                raise ValueError("GPU FLICA noise precision diverged")

    result = {"H": h.cpu().numpy(), "H_PCs": h_pcs.cpu().numpy(),
              "W": [value.cpu().numpy() for value in w],
              "X": [value.cpu().numpy() for value in x],
              "lambda": np.asarray([value.item() for value in lam]),
              "DD": dd.cpu().numpy()}
    return result
