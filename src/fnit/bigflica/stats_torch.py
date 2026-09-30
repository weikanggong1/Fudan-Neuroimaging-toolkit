"""CUDA regression and Student-t to normal-z conversion for spatial maps."""

from __future__ import annotations

import numpy as np
import torch

from .pipeline import _device


class SpatialRegression:
    def __init__(self, h: np.ndarray, *, device: str = "cuda:0") -> None:
        self.backend = _device(device)
        self.design = torch.as_tensor(np.column_stack((h, np.ones(h.shape[0]))),
                                      device=self.backend, dtype=torch.float64)
        self.df = self.design.shape[0] - self.design.shape[1]
        if self.df < 1:
            raise ValueError("At least one residual degree of freedom is required")
        self.inverse = torch.linalg.pinv(self.design)
        self.se_multiplier = torch.sqrt(torch.diagonal(
            torch.linalg.pinv(self.design.T @ self.design))[:-1, None])

    def t(self, projected: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(projected, torch.Tensor):
            y = projected.T.to(device=self.backend, dtype=torch.float64).contiguous()
        else:
            y = torch.as_tensor(projected.T.copy(), device=self.backend, dtype=torch.float64)
        coefficients = self.inverse @ y
        residual = y - self.design @ coefficients
        sigma = torch.sqrt(residual.square().sum(dim=0) / self.df)
        se = self.se_multiplier * sigma
        t = torch.where(se > 0, coefficients[:-1] / se.clamp_min(1e-300), 0)
        return t.T.cpu().numpy()


def spatial_t_gpu(h: np.ndarray, projected: np.ndarray,
                  *, device: str = "cuda:0") -> np.ndarray:
    return SpatialRegression(h, device=device).t(projected)


def _beta_fraction(a: float, b: float, x: torch.Tensor) -> torch.Tensor:
    """Modified Lentz continued fraction for incomplete beta on CUDA."""
    tiny = 1e-300
    qap = a + 1
    qab = a + b
    c = torch.ones_like(x)
    d = 1 - qab * x / qap
    d = torch.where(d.abs() < tiny, torch.full_like(d, tiny), d)
    d = 1 / d
    result = d.clone()
    for m in range(1, 81):
        numerator = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        d = 1 + numerator * d
        d = torch.where(d.abs() < tiny, torch.full_like(d, tiny), d)
        c = 1 + numerator / c
        c = torch.where(c.abs() < tiny, torch.full_like(c, tiny), c)
        d = 1 / d
        result = result * d * c
        numerator = -(a + m) * (qab + m) * x / ((a + 2 * m) * (qap + 2 * m))
        d = 1 + numerator * d
        d = torch.where(d.abs() < tiny, torch.full_like(d, tiny), d)
        c = 1 + numerator / c
        c = torch.where(c.abs() < tiny, torch.full_like(c, tiny), c)
        d = 1 / d
        result = result * d * c
    return result


def t_to_z_gpu(t_values: np.ndarray, df: int,
               *, device: str = "cuda:0") -> np.ndarray:
    if df < 1:
        raise ValueError("df must be positive")
    backend = _device(device)
    t = torch.as_tensor(t_values, device=backend, dtype=torch.float64)
    a, b = df / 2, 0.5
    x = (df / (df + t.square())).clamp(1e-300, 1 - 1e-15)
    log_beta = (torch.lgamma(t.new_tensor(a)) + torch.lgamma(t.new_tensor(b)) -
                torch.lgamma(t.new_tensor(a + b)))
    factor = torch.exp(a * torch.log(x) + b * torch.log1p(-x) - log_beta)
    direct = factor * _beta_fraction(a, b, x) / a
    complement = 1 - factor * _beta_fraction(b, a, 1 - x) / b
    p_two_sided = torch.where(x < (a + 1) / (a + b + 2), direct, complement)
    p_two_sided = p_two_sided.clamp(np.finfo(np.float64).tiny, 1)
    z = -torch.sign(t) * torch.special.ndtri(p_two_sided / 2)
    return z.float().cpu().numpy()
