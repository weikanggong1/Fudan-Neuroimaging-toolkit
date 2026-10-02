"""CUDA tensor updates for the single-group FLICA model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import h5py
import numpy as np
import torch

from .flica_vb import fit_eigenspectrum
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


def _scaled_inverse(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = torch.rsqrt(torch.diagonal(matrix))
    inverse, info = torch.linalg.inv_ex(
        scale[:, None] * matrix * scale[None, :], check_errors=False)
    return scale[:, None] * inverse * scale[None, :], info


def _scaled_inverse_batch(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = torch.rsqrt(torch.diagonal(matrix, dim1=-2, dim2=-1))
    inverse, info = torch.linalg.inv_ex(
        scale[:, :, None] * matrix * scale[:, None, :], check_errors=False)
    return scale[:, :, None] * inverse * scale[:, None, :], info


def _estimate_dd_gpu(data: torch.Tensor) -> torch.Tensor:
    """Compute the spectrum on CUDA and source-compatible scalar DOF on CPU."""
    n_voxels, n_reduced = data.shape
    eigenvalues = torch.linalg.eigvalsh(data.T @ data).flip(0).clamp_min(0)
    return _estimate_dd_eigenvalues(eigenvalues, n_voxels, n_reduced)


def _estimate_dd_eigenvalues(eigenvalues: torch.Tensor, n_voxels: int,
                             n_reduced: int) -> torch.Tensor:
    """The small CPU fmin step preserves the original FLICA stopping point."""
    if n_reduced >= n_voxels:
        return eigenvalues.new_tensor(1.0)
    indices = (int(np.ceil(n_reduced * 0.25) - 1),
               int(np.floor(n_reduced * 0.75) - 1))
    spectrum = np.full(n_reduced, np.nan, dtype=np.float64)
    values = eigenvalues.detach().cpu().numpy()
    spectrum[list(indices)] = values[list(indices)]
    gamma = fit_eigenspectrum(spectrum)[0]
    return eigenvalues.new_tensor(n_reduced / (gamma * n_voxels))


def initialize_flica_torch(y_arrays: Sequence[np.ndarray], n_components: int,
                           *, device: str = "cuda:0", lambda_dims: str = "o",
                           init_h: np.ndarray | torch.Tensor | None = None
                           ) -> tuple[dict, dict, dict]:
    """Initialize FLICA; numeric init_h uses the source's pre-scaled H convention."""
    backend = _device(device)
    if backend.type != "cuda":
        raise ValueError("CUDA device required for FLICA GPU initialization")
    y = [_tensor(value, backend) for value in y_arrays]
    n_modalities, n_reduced = len(y), y[0].shape[1]
    if not 1 <= n_components < n_reduced:
        raise ValueError("Invalid FLICA component count")
    if lambda_dims not in ("o", "R"):
        raise ValueError("lambda_dims must be 'o' or 'R'")
    dd = torch.stack([_estimate_dd_gpu(value) for value in y])
    if init_h is None:
        covariance = sum(dd[k] * (value.T @ value) for k, value in enumerate(y))
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        dominant = eigenvalues[-n_components:].flip(0)
        h_pre = (dominant.clamp_min(0).sqrt()[:, None] *
                 eigenvectors[:, -n_components:].flip(1).T /
                 np.sqrt(sum(value.shape[0] for value in y)))
    else:
        h_pre = _tensor(init_h, backend)
        if h_pre.shape != (n_components, n_reduced):
            raise ValueError("init_h must have shape (n_components, n_reduced)")
    h = h_pre / torch.sqrt(dd.mean())
    projector = torch.linalg.pinv(h_pre)
    x = [(value * torch.sqrt(dd[k])) @ projector for k, value in enumerate(y)]
    return _initial_state(h, x, dd, [value.shape[0] for value in y],
                          [value.square().mean(dim=0 if lambda_dims == "R" else None)
                           for value in y], backend, lambda_dims)


def _initial_state(h: torch.Tensor, x: list[torch.Tensor], dd: torch.Tensor,
                   n_voxels: Sequence[int], mean_squares: Sequence[torch.Tensor],
                   backend: torch.device, lambda_dims: str = "o"
                   ) -> tuple[dict, dict, dict]:
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
              "prior_beta_c": prior_beta_c, "prior_W_var": 1 / dd,
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
                 "DD": dd, "lambda_dims": lambda_dims}
    return priors, posteriors, constants


def initialize_flica_raw(y: Sequence[RawVoxelMatrix], n_components: int,
                         *, device: str = "cuda:0", max_gpu_gb: float = 28.0,
                         lambda_dims: str = "o",
                         init_h: np.ndarray | torch.Tensor | None = None
                         ) -> tuple[dict, dict, dict]:
    """Initialize direct-voxel FLICA without holding any modality matrix in RAM."""
    backend = _device(device)
    if backend.type != "cuda" or not y:
        raise ValueError("Direct-voxel FLICA requires CUDA and modalities")
    n_subjects = y[0].shape[1]
    if not 1 <= n_components < n_subjects or any(value.shape[1] != n_subjects for value in y):
        raise ValueError("Invalid raw FLICA dimensions")
    if lambda_dims not in ("o", "R"):
        raise ValueError("lambda_dims must be 'o' or 'R'")
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
        if lambda_dims == "R":
            subject_squares = torch.zeros(n_subjects, device=backend, dtype=dtype)
        for _, block in value.blocks(backend):
            covariance.addmm_(block.T, block)
            if lambda_dims == "R":
                squares = block.square().sum(dim=0)
                subject_squares += squares
                total_squares += float(squares.sum())
            else:
                total_squares += float(block.square().sum())
        value.squared_sum = total_squares
        if lambda_dims == "R":
            mean_squares.append(subject_squares / value.shape[0])
        else:
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
    if init_h is None:
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
        h_pre = (eigenvalues.clamp_min(0).sqrt()[:, None] * eigenvectors.T /
                 np.sqrt(sum(value.shape[0] for value in y)))
    else:
        h_pre = _tensor(init_h, backend)
        if h_pre.shape != (n_components, n_subjects):
            raise ValueError("init_h must have shape (n_components, n_reduced)")
    h = h_pre / torch.sqrt(dd.mean())
    projector = torch.linalg.pinv(h_pre)
    del covariance, total_covariance
    spatial = []
    for k, value in enumerate(y):
        x = torch.empty((value.shape[0], n_components), device=backend,
                        dtype=torch.float64)
        for start, block in value.blocks(backend):
            x[start:start + block.shape[0]] = (
                block * torch.sqrt(dd[k])) @ projector
        spatial.append(x)
    return _initial_state(h, spatial, dd, [value.shape[0] for value in y],
                          mean_squares, backend, lambda_dims)


def iterate_flica_torch(y_arrays: Sequence[np.ndarray | RawVoxelMatrix], priors: dict,
                        posteriors: dict, constants: dict, max_iter: int,
                        *, device: str = "cuda:0") -> dict[str, object]:
    """Run single-group FLICA coordinate updates with scalar or subjectwise noise."""
    backend = _device(device)
    if backend.type != "cuda":
        raise ValueError("CUDA device required for FLICA GPU iteration")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    n_modalities, n_components, n_reduced = (int(constants[key]) for key in ("K", "L", "R"))
    lambda_dims = constants.get("lambda_dims", "o")
    if lambda_dims not in ("o", "R"):
        raise ValueError("lambda_dims must be 'o' or 'R'")
    subjectwise = lambda_dims == "R"
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
            if subjectwise:
                result = torch.zeros(n_reduced, device=backend, dtype=torch.float64)
                for _, block in y[index].blocks(backend):
                    result += block.square().sum(dim=0)
                return result
            return torch.as_tensor(y[index].squared_sum, device=backend,
                                   dtype=torch.float64)
        if subjectwise:
            return y[index].square().sum(dim=0)
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
    lam = [(_tensor(value, backend).reshape(-1) if subjectwise else
            _tensor(value, backend).reshape(-1)[0])
           for value in posteriors["Lambda"]]
    if subjectwise and any(value.numel() != n_reduced for value in lam):
        raise ValueError("Subjectwise Lambda must have one value per reduced subject")
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
    prior_w_var = _tensor(priors["prior_W_var"], backend).reshape(-1)
    if prior_w_var.numel() == 1:
        prior_w_var = prior_w_var.expand(n_modalities)
    elif prior_w_var.numel() != n_modalities:
        raise ValueError("prior_W_var must be scalar or have one value per modality")
    if not bool(torch.all(torch.isfinite(prior_w_var) & (prior_w_var > 0))):
        raise ValueError("prior_W_var must contain finite positive variances")
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

    # Run the requested number of complete coordinate updates.
    for _ in range(max_iter):
        eta_c = prior_eta_c + n_reduced / 2
        eta_binv = 1 / prior_eta_b + h2 / 2
        eta = eta_c / eta_binv

        precision = (torch.diag(eta)[None].expand(n_reduced, -1, -1).clone()
                     if subjectwise else torch.diag(eta))
        mean_term = torch.zeros((n_components, n_reduced), device=backend,
                                dtype=torch.float64)
        old_cross = [cross(k, x[k]) for k in range(n_modalities)]
        for k in range(n_modalities):
            if subjectwise:
                precision += lam[k][:, None, None] * (wtw[k] * xtdx[k].T)[None]
                mean_term += dd[k] * (w[k][:, None] * old_cross[k]) * lam[k][None, :]
                h_pcs[k] = torch.diagonal(wtw[k]) * torch.diagonal(xtdx[k]) * lam[k].mean()
            else:
                precision += lam[k] * (wtw[k] * xtdx[k].T)
                mean_term += dd[k] * lam[k] * (w[k][:, None] * old_cross[k])
                h_pcs[k] = torch.diagonal(wtw[k]) * torch.diagonal(xtdx[k]) * lam[k]
        h_cov, h_inverse_info = (_scaled_inverse_batch(precision) if subjectwise
                                 else _scaled_inverse(precision))
        inverse_info = [h_inverse_info]
        if subjectwise:
            h = (h_cov @ mean_term.T[:, :, None]).squeeze(-1).T
            h2 = h.square().sum(dim=1) + torch.diagonal(h_cov, dim1=-2, dim2=-1).sum(dim=0)
        else:
            h = h_cov @ mean_term
            h2 = h.square().sum(dim=1) + n_reduced * torch.diagonal(h_cov)
        h_pcs[-1] = eta

        hlambda = []
        for k in range(n_modalities):
            weighted_h = ((h * lam[k][None, :]) @ h.T +
                          torch.einsum("r,rij->ij", lam[k], h_cov)) if subjectwise else (
                          lam[k] * (h @ h.T + n_reduced * h_cov))
            hlambda.append(weighted_h)
            w_cov[k], w_inverse_info = _scaled_inverse(
                xtdx[k] * weighted_h + unit / prior_w_var[k])
            inverse_info.append(w_inverse_info)
            target = (dd[k] * torch.diagonal((old_cross[k] * lam[k][None, :]) @ h.T)
                      if subjectwise else
                      dd[k] * lam[k] * torch.diagonal(old_cross[k] @ h.T))
            w[k] = target @ w_cov[k]
            wtw[k] = torch.outer(w[k], w[k]) + w_cov[k]

        for k in range(n_modalities):
            projected_values = (projected(k, h.T * lam[k][:, None] * w[k][None, :])
                                if subjectwise else
                                projected(k, h.T * (lam[k] * w[k])[None, :]))
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
            diagonal_term = (diagonal_term +
                             torch.einsum("ij,rij->r", covariance_weight, h_cov)
                             if subjectwise else
                             diagonal_term + (covariance_weight * h_cov).sum())
            new_cross = cross(k, x[k])
            fitted = ((new_cross * h) * w[k][:, None]).sum(dim=0) * dd[k]
            if subjectwise:
                shape = dd[k] * y[k].shape[0] / 2 + prior_lambda_c[k]
                rate = (dd[k] * sum_y2[k] / 2 - fitted + diagonal_term / 2 +
                        1 / prior_lambda_b[k])
            else:
                residual_sum = dd[k] * sum_y2[k] / 2 - fitted.sum() + diagonal_term.sum() / 2
                shape = dd[k] * y[k].shape[0] * n_reduced / 2 + prior_lambda_c[k]
                rate = residual_sum + 1 / prior_lambda_b[k]
            lam[k] = shape / rate
        noise_precision = torch.stack(lam)
        inverse_ok = (torch.all(torch.cat([value.reshape(-1) for value in inverse_info]) == 0)
                      if subjectwise else torch.all(torch.stack(inverse_info) == 0))
        if not bool(inverse_ok &
                    torch.all(torch.isfinite(noise_precision) & (noise_precision > 0))):
            raise ValueError("GPU FLICA covariance or noise precision diverged")

    result = {"H": h.cpu().numpy(), "H_PCs": h_pcs.cpu().numpy(),
              "W": [value.cpu().numpy() for value in w],
              "X": [value.cpu().numpy() for value in x],
              "lambda": (noise_precision.cpu().numpy() if subjectwise else
                         np.asarray([value.item() for value in lam])),
              "DD": dd.cpu().numpy()}
    return result
