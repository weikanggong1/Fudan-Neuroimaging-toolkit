"""Mask-guided three-tissue response estimation for the local PyTorch FOD fit.

This uses registered tissue labels and, optionally, FA-ranked tissue voxels.
WM directions come from a low-shell tensor fit, followed by an axial Legendre
fit to the normalized signal. It is not the unsupervised MRtrix Dhollander
algorithm and its responses are not numerically equivalent to that method.
Crossing fibres, partial volume, misregistration, and poor angular coverage
can bias the WM response.
"""

from __future__ import annotations

import torch


def _even_legendre(cosine: torch.Tensor, lmax: int) -> torch.Tensor:
    """Columns (2l+1) P_l(cosine), for even l from 0 through lmax."""
    terms = [torch.ones_like(cosine)]
    previous, current = terms[0], cosine
    for degree in range(2, lmax + 1):
        following = ((2 * degree - 1) * cosine * current
                     - (degree - 1) * previous) / degree
        if degree % 2 == 0:
            terms.append((2 * degree + 1) * following)
        previous, current = current, following
    return torch.stack(terms, dim=-1)


def estimate_three_tissue_response(
    signal: torch.Tensor,
    bvals: torch.Tensor,
    bvecs: torch.Tensor,
    tissue_labels: torch.Tensor,
    fa: torch.Tensor | None = None,
    lmax: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate responses compatible with ``fit_three_tissue_csd``.

    ``signal`` is float32 ``[X,Y,Z,N]``; ``bvals`` use s/mm² and ``bvecs``
    are RAS-world directions ``[N,3]``. Registered ``tissue_labels`` are 0
    background, 1 GM, 2 WM, 3 CSF. If FA is supplied, the highest-FA 20% of
    WM and lowest-FA 20% of GM/CSF mask voxels are used. Selection requires
    finite signals and positive mean b0 signal. Returns float32
    ``(shell_bvals[S], wm_response[S,L], gm_response[S], csf_response[S])``
    on ``signal.device``, with ``L=lmax//2+1``. Shells within 100 s/mm² are
    grouped; the b0 responses are exactly WM ``[1,0,...]`` and GM/CSF ``1``.
    """
    if signal.ndim != 4 or signal.dtype != torch.float32:
        raise ValueError("signal must be float32 [X,Y,Z,N]")
    if lmax < 0 or lmax % 2:
        raise ValueError("lmax must be a nonnegative even integer")
    device = signal.device
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    n_volumes = signal.shape[-1]
    bvals = torch.as_tensor(bvals, device=device, dtype=torch.float32)
    bvecs = torch.as_tensor(bvecs, device=device, dtype=torch.float32)
    tissue_labels = torch.as_tensor(tissue_labels, device=device)
    if bvals.shape != (n_volumes,) or bvecs.shape != (n_volumes, 3):
        raise ValueError("bvals and bvecs must match the DWI volume count")
    if tissue_labels.shape != signal.shape[:3] or tissue_labels.is_floating_point():
        raise ValueError("tissue_labels must be integer [X,Y,Z]")
    if not bool(torch.isfinite(bvals).all() and torch.isfinite(bvecs).all()):
        raise ValueError("bvals and bvecs must be finite")
    baseline = bvals < 50
    diffusion = ~baseline
    if not bool(baseline.any()) or not bool(diffusion.any()):
        raise ValueError("response estimation requires b0 and diffusion volumes")
    if bool((torch.linalg.vector_norm(bvecs[diffusion], dim=-1) < 1e-6).any()):
        raise ValueError("diffusion bvecs must be nonzero")
    directions = torch.nn.functional.normalize(bvecs, dim=-1)

    # Shell centres are the mean acquired b-value within each sorted 100-unit
    # group. b0 is represented by zero to satisfy the FOD fitter's contract.
    groups: list[list[float]] = []
    for value in torch.sort(bvals[diffusion]).values.tolist():
        if not groups or value - groups[-1][0] > 100:
            groups.append([value])
        else:
            groups[-1].append(value)
    if len(groups) < 2:
        raise ValueError("three-tissue response estimation requires two diffusion shells")
    shell_bvals = torch.tensor(
        [0.0] + [sum(group) / len(group) for group in groups],
        device=device, dtype=torch.float32,
    )
    shell_distance = torch.abs(bvals[:, None] - shell_bvals[None, :])
    shell_index = shell_distance.argmin(dim=1)
    if (bool((shell_index[baseline] != 0).any())
            or bool((shell_index[diffusion] == 0).any())
            or bool((shell_distance.gather(1, shell_index[:, None]) > 100).any())):
        raise ValueError("b-values cannot map to the inferred FOD response shells")
    flat = signal.reshape(-1, n_volumes)
    s0 = flat[:, baseline].mean(dim=1)
    valid = torch.isfinite(flat).all(dim=1) & (s0 > 0)
    labels = tissue_labels.reshape(-1)
    if fa is not None:
        fa = torch.as_tensor(fa, device=device, dtype=torch.float32)
        if fa.shape != signal.shape[:3]:
            raise ValueError("fa must match the DWI spatial shape")
        flat_fa = fa.reshape(-1)
        valid &= torch.isfinite(flat_fa)

    selected = []
    for label in (2, 1, 3):
        chosen = valid & (labels == label)
        if not bool(chosen.any()):
            raise ValueError(f"no usable tissue-label {label} voxels")
        if fa is not None:
            threshold = torch.quantile(flat_fa[chosen], 0.8 if label == 2 else 0.2)
            chosen &= flat_fa >= threshold if label == 2 else flat_fa <= threshold
        selected.append(flat[chosen] / s0[chosen, None])
    wm_signal, gm_signal, csf_signal = selected

    n_orders = lmax // 2 + 1
    wm_response = torch.zeros((shell_bvals.numel(), n_orders), device=device)
    gm_response = torch.ones(shell_bvals.numel(), device=device)
    csf_response = torch.ones(shell_bvals.numel(), device=device)
    wm_response[0, 0] = 1.0

    if lmax:
        low_shell = shell_index == 1
        if int(low_shell.sum()) < 6:
            raise ValueError("WM orientation fit requires at least six low-shell directions")
        gx, gy, gz = directions[low_shell].unbind(-1)
        tensor_design = -bvals[low_shell, None] * torch.stack(
            (gx.square(), gy.square(), gz.square(), 2 * gx * gy, 2 * gx * gz, 2 * gy * gz), dim=1
        )
        if int(torch.linalg.matrix_rank(tensor_design)) < 6:
            raise ValueError("low-shell directions cannot identify a diffusion tensor")
        log_signal = (wm_signal[:, low_shell].clamp_min(1e-6)).log()
        coefficients = log_signal @ torch.linalg.pinv(tensor_design).T
        dxx, dyy, dzz, dxy, dxz, dyz = coefficients.unbind(-1)
        tensors = torch.stack(
            (dxx, dxy, dxz, dxy, dyy, dyz, dxz, dyz, dzz), dim=-1
        ).reshape(-1, 3, 3)
        fibre = torch.linalg.eigh(tensors).eigenvectors[:, :, -1]

    for shell in range(1, shell_bvals.numel()):
        in_shell = shell_index == shell
        if int(in_shell.sum()) < n_orders:
            raise ValueError(f"shell {shell} has too few directions for lmax={lmax}")
        gm_response[shell] = gm_signal[:, in_shell].mean()
        csf_response[shell] = csf_signal[:, in_shell].mean()
        if lmax:
            cosine = directions[in_shell] @ fibre.T
            design = _even_legendre(cosine, lmax).reshape(-1, n_orders)
        else:
            design = torch.ones((int(in_shell.sum()) * wm_signal.shape[0], 1), device=device)
        target = wm_signal[:, in_shell].T.reshape(-1)
        if int(torch.linalg.matrix_rank(design)) < n_orders:
            raise ValueError(f"shell {shell} cannot identify the WM response")
        wm_response[shell] = torch.linalg.lstsq(design, target).solution
    return shell_bvals, wm_response, gm_response, csf_response
