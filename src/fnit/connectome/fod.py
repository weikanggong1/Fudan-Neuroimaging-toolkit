"""Fixed-response, three-tissue spherical deconvolution in PyTorch.

All diffusion signals are normalized by their mean b=0 signal. Gradient
directions are unit vectors in the image's world (RAS) frame and b-values are
in s/mm². The supplied response values are dimensionless S/S0 attenuations;
they are not estimated here. The returned even-order, orthonormal real SH
coefficients have shape ``[X, Y, Z, C]`` and order ``l=0,2,...`` then
``m=-l,...,l``. The integral of the WM FOD over the unit sphere is its tissue
fraction. GM and CSF outputs have shape ``[X, Y, Z]``.

The WM FOD is a nonnegative mixture of normalized even cosine kernels. This
gives an exactly nonnegative, band-limited FOD at the requested SH order, but
is not the MRtrix MSMT-CSD optimization or its response estimation.
"""

from __future__ import annotations

import math

import torch


def _associated_legendre(l: int, m: int, z: torch.Tensor) -> torch.Tensor:
    """Condon-Shortley associated Legendre polynomial P_l^m(z)."""
    pmm = torch.ones_like(z)
    if m:
        root = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
        for i in range(1, m + 1):
            pmm = -(2 * i - 1) * root * pmm
    if l == m:
        return pmm
    pm1 = (2 * m + 1) * z * pmm
    if l == m + 1:
        return pm1
    pm2 = pmm
    for degree in range(m + 2, l + 1):
        current = ((2 * degree - 1) * z * pm1 - (degree + m - 1) * pm2) / (degree - m)
        pm2, pm1 = pm1, current
    return pm1


def real_sh(directions: torch.Tensor, lmax: int = 4) -> torch.Tensor:
    """Evaluate even-degree orthonormal real SH at RAS world directions.

    ``directions`` is float32 with shape ``[..., 3]``; nonzero vectors are
    normalized internally. Coefficients are ordered by even ``l``, then by
    ``m=-l,...,l``. The basis includes the Condon-Shortley phase.
    """
    if lmax < 0 or lmax % 2:
        raise ValueError("lmax must be a nonnegative even integer")
    if directions.ndim < 2 or directions.shape[-1] != 3 or directions.dtype != torch.float32:
        raise ValueError("directions must be a float32 tensor with shape [..., 3]")
    lengths = torch.linalg.vector_norm(directions, dim=-1)
    if bool(torch.any(lengths <= 0)):
        raise ValueError("directions must be nonzero")
    unit = directions / lengths[..., None]
    z = torch.clamp(unit[..., 2], -1.0, 1.0)
    phi = torch.atan2(unit[..., 1], unit[..., 0])
    values = []
    for l in range(0, lmax + 1, 2):
        for m in range(-l, l + 1):
            order = abs(m)
            norm = math.sqrt((2 * l + 1) / (4 * math.pi) * math.factorial(l - order) / math.factorial(l + order))
            value = norm * _associated_legendre(l, order, z)
            if m < 0:
                value = math.sqrt(2.0) * value * torch.sin(order * phi)
            elif m > 0:
                value = math.sqrt(2.0) * value * torch.cos(order * phi)
            values.append(value)
    return torch.stack(values, dim=-1)


def _orientation_dictionary(lmax: int, device: torch.device) -> torch.Tensor:
    """SH coefficients of unit-integral, nonnegative cos(theta)^lmax kernels."""
    n_coeff = sum(2 * l + 1 for l in range(0, lmax + 1, 2))
    n_dirs = max(64, 2 * n_coeff)
    i = torch.arange(n_dirs, device=device, dtype=torch.float32)
    z = (i + 0.5) / n_dirs  # one representative per antipodal orientation
    phi = i * (math.pi * (3.0 - math.sqrt(5.0)))
    radius = torch.sqrt(1.0 - z * z)
    directions = torch.stack((radius * torch.cos(phi), radius * torch.sin(phi), z), dim=-1)
    basis = real_sh(directions, lmax)
    factors = []
    for l in range(0, lmax + 1, 2):
        # Funk-Hecke coefficient for (lmax+1)/(4*pi) * cos(theta)^lmax.
        left = math.prod(range(lmax - l, 0, -2))
        right = math.prod(range(lmax + l + 1, 0, -2))
        factor = (lmax + 1) * math.factorial(lmax) / (left * right)
        factors.extend([factor] * (2 * l + 1))
    return basis * torch.tensor(factors, device=device, dtype=torch.float32)


def fit_three_tissue_csd(
    signal: torch.Tensor,
    bvals: torch.Tensor,
    bvecs: torch.Tensor,
    shell_bvals: torch.Tensor,
    wm_response: torch.Tensor,
    gm_response: torch.Tensor,
    csf_response: torch.Tensor,
    mask: torch.Tensor | None = None,
    lmax: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit WM FOD and isotropic GM/CSF fractions from multi-shell DWI.

    ``signal`` is float32 ``[X,Y,Z,N]`` on the desired CPU/CUDA device.
    ``bvals`` and ``shell_bvals`` use s/mm². ``wm_response`` is ``[S,L]``
    for even orders 0..lmax; ``gm_response`` and ``csf_response`` are ``[S]``.
    Responses are fixed, unit-fraction S/S0 values, with b=0 row ``[1,0,...]``
    for WM and ``1`` for GM/CSF. Each volume maps to its nearest shell within
    100 s/mm². No response estimation or intensity normalization is performed.
    """
    if signal.ndim != 4 or signal.dtype != torch.float32:
        raise ValueError("signal must be float32 [X,Y,Z,N]")
    if lmax < 0 or lmax % 2:
        raise ValueError("lmax must be a nonnegative even integer")
    device = signal.device
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    bvals = torch.as_tensor(bvals, device=device, dtype=torch.float32)
    bvecs = torch.as_tensor(bvecs, device=device, dtype=torch.float32)
    shell_bvals = torch.as_tensor(shell_bvals, device=device, dtype=torch.float32)
    wm_response = torch.as_tensor(wm_response, device=device, dtype=torch.float32)
    gm_response = torch.as_tensor(gm_response, device=device, dtype=torch.float32)
    csf_response = torch.as_tensor(csf_response, device=device, dtype=torch.float32)
    n_volumes = signal.shape[-1]
    n_orders = lmax // 2 + 1
    n_shells = shell_bvals.numel()
    if (
        bvals.shape != (n_volumes,)
        or bvecs.shape != (n_volumes, 3)
        or shell_bvals.ndim != 1
        or wm_response.shape != (n_shells, n_orders)
        or gm_response.shape != (n_shells,)
        or csf_response.shape != (n_shells,)
    ):
        raise ValueError("gradient and response shapes do not match signal and lmax")
    if not bool(torch.isfinite(bvals).all() and torch.isfinite(bvecs).all() and torch.isfinite(wm_response).all()):
        raise ValueError("b-values, b-vectors, and responses must be finite")
    b0 = bvals < 50.0
    if not bool(b0.any()):
        raise ValueError("at least one b=0 volume is required")
    distances = torch.abs(bvals[:, None] - shell_bvals[None, :])
    shell_index = torch.argmin(distances, dim=1)
    if bool(torch.any(distances.gather(1, shell_index[:, None]) > 100.0)):
        raise ValueError("each b-value must be within 100 s/mm² of a response shell")
    b0_shell = int(torch.argmin(torch.abs(shell_bvals)).item())
    expected_wm_b0 = torch.zeros(n_orders, device=device)
    expected_wm_b0[0] = 1.0
    if (
        abs(float(shell_bvals[b0_shell])) > 50.0
        or not bool(torch.allclose(wm_response[b0_shell], expected_wm_b0, atol=1e-4))
        or not bool(torch.allclose(gm_response[b0_shell], torch.ones((), device=device), atol=1e-4))
        or not bool(torch.allclose(csf_response[b0_shell], torch.ones((), device=device), atol=1e-4))
    ):
        raise ValueError("responses must be normalized to unit signal at b=0")
    if mask is not None:
        mask = torch.as_tensor(mask, device=device, dtype=torch.bool)
        if mask.shape != signal.shape[:3]:
            raise ValueError("mask must have shape [X,Y,Z]")

    # A unit-integral FOD has c00 = WM_fraction / sqrt(4*pi).
    unit_bvecs = bvecs.clone()
    unit_bvecs[b0] = torch.tensor([0.0, 0.0, 1.0], device=device)
    basis = real_sh(unit_bvecs, lmax)
    response_per_coefficient = torch.cat(
        [wm_response[shell_index, order:order + 1].expand(-1, 2 * l + 1)
         for order, l in enumerate(range(0, lmax + 1, 2))],
        dim=1,
    )
    orientation_sh = _orientation_dictionary(lmax, device)
    wm_design = (4.0 * math.pi) * (basis * response_per_coefficient) @ orientation_sh.T
    design = torch.cat((wm_design, gm_response[shell_index, None], csf_response[shell_index, None]), dim=1)
    design = design / math.sqrt(n_volumes)
    gram = design.T @ design
    rho = 0.001
    factor = torch.linalg.cholesky(gram + rho * torch.eye(gram.shape[0], device=device))

    flat_signal = signal.reshape(-1, n_volumes)
    s0 = flat_signal[:, b0].mean(dim=1)
    valid = torch.isfinite(flat_signal).all(dim=1) & (s0 > 0)
    if mask is not None:
        valid &= mask.reshape(-1)
    n_coeff = orientation_sh.shape[1]
    wm_flat = torch.zeros((flat_signal.shape[0], n_coeff), device=device)
    gm_flat = torch.zeros(flat_signal.shape[0], device=device)
    csf_flat = torch.zeros(flat_signal.shape[0], device=device)
    valid_indices = torch.nonzero(valid).flatten()
    with torch.no_grad():
        for block in valid_indices.split(4096):
            data = (flat_signal[block] / s0[block, None]).T / math.sqrt(n_volumes)
            rhs = design.T @ data
            z = torch.zeros_like(rhs)
            dual = torch.zeros_like(rhs)
            for _ in range(500):
                unconstrained = torch.cholesky_solve(rhs + rho * (z - dual), factor)
                z = torch.clamp(unconstrained + dual, min=0.0)
                dual = dual + unconstrained - z
            wm_flat[block] = z[:-2].T @ orientation_sh
            gm_flat[block] = z[-2]
            csf_flat[block] = z[-1]
    return (
        wm_flat.reshape(*signal.shape[:3], n_coeff),
        gm_flat.reshape(signal.shape[:3]),
        csf_flat.reshape(signal.shape[:3]),
    )
