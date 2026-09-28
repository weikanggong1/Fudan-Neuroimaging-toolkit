"""Gaussian likelihood and EM updates for GEMS anatomical labels."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass
class GaussianParameters:
    means: torch.Tensor       # Cclass,M
    covariances: torch.Tensor # Cclass,M,M


def _as_modalities(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 3:
        return image[None]
    if image.ndim == 4:
        return image
    raise ValueError("image must be [X,Y,Z] or [M,X,Y,Z]")


def initialise_gaussians(image: torch.Tensor, priors: torch.Tensor,
                         label_classes: torch.Tensor,
                         variance_floor: float = 1e-4) -> GaussianParameters:
    image = _as_modalities(image)
    label_classes = label_classes.long()
    n_classes = int(label_classes.max().item()) + 1
    class_resp = torch.zeros((n_classes, *priors.shape[1:]), device=priors.device, dtype=priors.dtype)
    class_resp.index_add_(0, label_classes, priors)
    return update_gaussians(image, class_resp, variance_floor=variance_floor)


def update_gaussians(image: torch.Tensor, responsibilities: torch.Tensor, *,
                     mean_hyper: torch.Tensor | None = None,
                     n_hyper: torch.Tensor | None = None,
                     variance_floor: float = 1e-4) -> GaussianParameters:
    image = _as_modalities(image).to(dtype=responsibilities.dtype)
    m = image.shape[0]
    flat_x = image.reshape(m, -1).T  # N,M
    r = responsibilities.reshape(responsibilities.shape[0], -1)
    finite = torch.isfinite(flat_x).all(1)
    nonzero = flat_x.abs().sum(1) > 0
    keep = finite & nonzero
    flat_x = flat_x[keep]
    r = r[:, keep]
    mass = r.sum(1).clamp_min(1e-8)
    weighted = r @ flat_x
    if mean_hyper is not None and n_hyper is not None:
        mh = mean_hyper.to(device=flat_x.device, dtype=flat_x.dtype)
        nh = n_hyper.to(device=flat_x.device, dtype=flat_x.dtype).reshape(-1, 1)
        means = (weighted + nh * mh) / (mass[:, None] + nh)
    else:
        means = weighted / mass[:, None]
    diff = flat_x[None] - means[:, None, :]
    cov = torch.einsum("cn,cnm,cnp->cmp", r, diff, diff) / mass[:, None, None]
    eye = torch.eye(m, device=cov.device, dtype=cov.dtype)
    scale = torch.diagonal(cov, dim1=-2, dim2=-1).mean(-1).clamp_min(1.0)
    cov = cov + eye[None] * (float(variance_floor) * scale)[:, None, None]
    return GaussianParameters(means, cov)


def gaussian_log_likelihood(image: torch.Tensor, params: GaussianParameters) -> torch.Tensor:
    image = _as_modalities(image).to(dtype=params.means.dtype)
    m = image.shape[0]
    x = image.reshape(m, -1).T
    diff = x[None] - params.means[:, None, :]
    chol = torch.linalg.cholesky(params.covariances)
    solved = torch.cholesky_solve(diff.unsqueeze(-1), chol[:, None]).squeeze(-1)
    maha = (diff * solved).sum(-1)
    logdet = 2 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
    ll = -0.5 * (maha + logdet[:, None] + m * math.log(2 * math.pi))
    return ll.reshape(params.means.shape[0], *image.shape[1:])


def label_posterior(priors: torch.Tensor, class_log_likelihood: torch.Tensor,
                    label_classes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    log_prior = torch.log(priors.clamp_min(torch.finfo(priors.dtype).tiny))
    log_joint = log_prior + class_log_likelihood[label_classes.long()]
    log_norm = torch.logsumexp(log_joint, dim=0)
    posterior = torch.exp(log_joint - log_norm[None])
    return posterior, -log_norm.sum()
