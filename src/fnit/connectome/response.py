"""PyTorch MRtrix-compatible Dhollander three-tissue response estimation.

The complete response path includes shell grouping, mask/SDM preparation,
IWLS tensor FA, tissue and single-fibre voxel selection, and amp2response.
"""

from __future__ import annotations

import math

import torch


def _even_legendre(cosine: torch.Tensor, lmax: int) -> torch.Tensor:
    """Evaluate the even Legendre orders used by MRtrix amp2response.

    Input cosine is a float64 CPU/CUDA tensor of any shape and lmax is
    an even nonnegative SH degree. Output is same-shape-prefix float64
    [...,lmax//2+1] on the input device, with columns (2l+1)P_l for
    l=0,2,...,lmax. This is the zonal basis in the amp2response fit.
    """
    terms = [torch.ones_like(cosine)]
    previous, current = terms[0], cosine
    for degree in range(2, lmax + 1):
        following = ((2 * degree - 1) * cosine * current
                     - (degree - 1) * previous) / degree
        if degree % 2 == 0:
            terms.append((2 * degree + 1) * following)
        previous, current = current, following
    return torch.stack(terms, dim=-1)


def _mrtrix_zonal_basis(cosine: torch.Tensor, lmax: int) -> torch.Tensor:
    """MRtrix even m=0 SH basis.

    Input cosine is a float64 tensor of any shape on CPU/CUDA; even lmax
    selects the last SH degree. Output is same-device float64
    ``[...,lmax//2+1]`` with coefficients ordered l=0,2,...,lmax.
    """
    degrees = range(0, lmax + 1, 2)
    scale = torch.tensor(
        [math.sqrt((2 * l + 1) / (4 * math.pi)) / (2 * l + 1) for l in degrees],
        device=cosine.device, dtype=cosine.dtype,
    )
    return _even_legendre(cosine, lmax) * scale


def _mrtrix_response_constraints(lmax: int, device: torch.device) -> torch.Tensor:
    """MRtrix response constraints at 0..90 degree elevations.

    Input even lmax and torch device select the zonal SH basis. Returns
    float64 ``[K,lmax//2+1]`` on that device; K excludes a zero derivative
    row at elevation zero.
    """
    theta = torch.arange(91, device=device, dtype=torch.float64) * (math.pi / 180)
    cosine = theta.cos()
    sine = theta.sin()
    amplitude = _mrtrix_zonal_basis(cosine, lmax)
    p_previous = torch.ones_like(cosine)
    p_current = cosine
    d_previous = torch.zeros_like(cosine)
    d_current = torch.ones_like(cosine)
    derivative = [torch.zeros_like(cosine)]
    for degree in range(2, lmax + 1):
        p_next = ((2 * degree - 1) * cosine * p_current
                  - (degree - 1) * p_previous) / degree
        d_next = ((2 * degree - 1) * (p_current + cosine * d_current)
                  - (degree - 1) * d_previous) / degree
        if degree % 2 == 0:
            derivative.append(
                -sine * d_next * math.sqrt((2 * degree + 1) / (4 * math.pi))
            )
        p_previous, p_current = p_current, p_next
        d_previous, d_current = d_current, d_next
    constraints = torch.cat((amplitude, torch.stack(derivative, dim=1)))
    return constraints[torch.linalg.vector_norm(constraints, dim=1) > 1e-12]


def _mrtrix_response_icls(
    normal: torch.Tensor,
    right: torch.Tensor,
    constraints: torch.Tensor,
) -> torch.Tensor:
    """Solve the MRtrix response active-set problem.

    Inputs: float64 normal matrix ``[L,L]``, right-hand vector ``[L]``
    and amplitude/derivative constraints ``[K,L]`` on one CPU/CUDA device.
    Returns same-device float64 zonal coefficients ``[L]``.
    """
    normal = normal.clone()
    normal.diagonal().add_(1e-10 * normal.diagonal().max())
    chol = torch.linalg.cholesky(normal)
    projected = torch.linalg.solve_triangular(
        chol, constraints.T, upper=False,
    ).T
    projected /= torch.linalg.vector_norm(projected, dim=1, keepdim=True)
    unconstrained = torch.linalg.solve_triangular(
        chol, right[:, None], upper=False,
    ).flatten()
    original_violation = projected @ unconstrained
    violation = original_violation.clone()
    active: list[int] = []
    prior = torch.zeros_like(original_violation)
    solution = unconstrained.clone()
    for _ in range(10 * normal.shape[0] + 1):
        index = int(violation.argmin())
        if bool(violation[index] >= 0):
            break
        changed = index not in active
        if changed:
            active.append(index)
            active.sort()
        while True:
            selected = torch.tensor(active, device=normal.device)
            rows = projected[selected]
            matrix = rows @ rows.T
            matrix.diagonal().add_(1e-10)
            multipliers = torch.linalg.solve(matrix, -original_violation[selected])
            negative = multipliers < 0
            if not bool(negative.any()):
                solution = unconstrained + rows.T @ multipliers
                prior.zero_()
                prior[selected] = multipliers
                break
            negative_indices = selected[negative]
            ratio = prior[negative_indices] / (
                prior[negative_indices] - multipliers[negative]
            )
            active.remove(int(negative_indices[int(ratio.argmin())]))
            changed = True
        if not changed:
            break
        violation = projected @ solution
    return torch.linalg.solve_triangular(
        chol.T, solution[:, None], upper=True,
    ).flatten()


def estimate_mrtrix_selected_response(
    signal: torch.Tensor,
    grad_mrtrix: torch.Tensor,
    shell_bvals: torch.Tensor,
    voxel_mask: torch.Tensor,
    fibre_directions: torch.Tensor | None = None,
    isotropic: bool = False,
) -> torch.Tensor:
    """Reproduce MRtrix amp2response for an already selected tissue mask.

    Inputs: raw float32 DWI ``[X,Y,Z,N]`` on CPU/CUDA; MRtrix-exported
    gradient scheme ``[N,4]`` in its gradient frame; shell centres ``[S]``
    in s/mm²; selected binary voxel mask ``[X,Y,Z]`` in the same image grid;
    and principal fibre directions ``[X,Y,Z,3]`` in the gradient frame for
    anisotropic WM. With ``isotropic=True`` the direction image is unused.
    Outputs are same-device float64 MRtrix zonal SH coefficients ``[S,6]``
    for WM (b0 lmax=0, diffusion lmax=10) or ``[S,1]`` for GM/CSF.
    The DWI can remain float32 on CUDA; regression uses float64.

    Equivalent reference commands:
    amp2response dwi.mif voxels_sfwm.mif safe_vecs.mif response_wm.txt
    amp2response dwi.mif voxels_gm.mif safe_vecs.mif response_gm.txt -isotropic

    This is the final response regression only. Dhollander's unsupervised
    voxel selection and tensor direction estimation are separate operators.
    """
    if signal.ndim != 4 or signal.dtype != torch.float32:
        raise ValueError("signal must be float32 [X,Y,Z,N]")
    device = signal.device
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    grad = torch.as_tensor(grad_mrtrix, device=device, dtype=torch.float64)
    shells = torch.as_tensor(shell_bvals, device=device, dtype=torch.float64)
    mask = torch.as_tensor(voxel_mask, device=device, dtype=torch.bool)
    if grad.shape != (signal.shape[-1], 4) or mask.shape != signal.shape[:3]:
        raise ValueError("gradient or selected mask shape differs from DWI")
    if shells.ndim != 1 or shells.numel() < 2 or abs(float(shells[0])) > 50:
        raise ValueError("response shells must start with a b0 shell")
    if not bool(mask.any()):
        raise ValueError("selected voxel mask is empty")
    shell_distance = torch.abs(grad[:, 3, None] - shells[None, :])
    assignment = shell_distance.argmin(dim=1)
    if bool((shell_distance.gather(1, assignment[:, None]) > 100).any()):
        raise ValueError("gradient b-values do not match response shells")
    selected_signal = signal[mask].to(torch.float64)
    if not bool(torch.isfinite(selected_signal).all()):
        raise ValueError("selected DWI contains nonfinite signals")
    directions = grad[:, :3].clone()
    directions[torch.linalg.vector_norm(directions, dim=1) < 1e-8] = torch.tensor(
        [0., 0., 1.], device=device, dtype=torch.float64,
    )
    directions = torch.nn.functional.normalize(directions, dim=1)
    if not isotropic:
        if fibre_directions is None:
            raise ValueError("anisotropic response requires fibre directions")
        fibres = torch.as_tensor(fibre_directions, device=device, dtype=torch.float64)
        if fibres.shape != (*signal.shape[:3], 3):
            raise ValueError("fibre directions must have shape [X,Y,Z,3]")
        fibres = torch.nn.functional.normalize(fibres[mask], dim=1)
    output = torch.zeros(
        (shells.numel(), 1 if isotropic else 6),
        device=device, dtype=torch.float64,
    )
    for shell in range(shells.numel()):
        indices = assignment == shell
        lmax = 0 if isotropic or shell == 0 else 10
        if lmax:
            cosine = fibres @ directions[indices].T
        else:
            cosine = torch.zeros(
                (selected_signal.shape[0], int(indices.sum())),
                device=device, dtype=torch.float64,
            )
        design = _mrtrix_zonal_basis(cosine, lmax).reshape(-1, lmax // 2 + 1)
        values = selected_signal[:, indices].reshape(-1)
        normal = design.T @ design
        right = design.T @ values
        if lmax:
            solution = _mrtrix_response_icls(
                normal, right, _mrtrix_response_constraints(lmax, device),
            )
        else:
            solution = right / normal.diagonal()
        output[shell, :solution.numel()] = solution
    return output


def _erode_six_neighbour(mask: torch.Tensor, passes: int) -> torch.Tensor:
    """Erode a bool [X,Y,Z] CPU/CUDA mask with MRtrix's six face neighbours.

    Input passes is a nonnegative integer. Output is a same-device bool
    [X,Y,Z] mask; image boundaries are false after each positive pass.
    """
    result = mask
    for _ in range(passes):
        interior = (
            result[1:-1, 1:-1, 1:-1]
            & result[:-2, 1:-1, 1:-1]
            & result[2:, 1:-1, 1:-1]
            & result[1:-1, :-2, 1:-1]
            & result[1:-1, 2:, 1:-1]
            & result[1:-1, 1:-1, :-2]
            & result[1:-1, 1:-1, 2:]
        )
        next_result = torch.zeros_like(result)
        next_result[1:-1, 1:-1, 1:-1] = interior
        result = next_result
    return result


def prepare_mrtrix_dhollander_sdm(
    signal: torch.Tensor,
    grad_mrtrix: torch.Tensor,
    shell_bvals: torch.Tensor,
    brain_mask: torch.Tensor,
    erosion_passes: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match the mask erosion and signal decay metric preparation in Dhollander.

    Inputs: corrected float32 DWI [X,Y,Z,N] on CPU/CUDA; MRtrix exported
    gradient scheme [N,4] with b-values in s/mm²; shell centres [S] in
    ascending order starting at b0; bool brain mask [X,Y,Z] aligned to the
    DWI voxel grid; and number of six-neighbour mask erosion passes.
    NIfTI affine alignment belongs to the nibabel caller. Gradient directions
    are unused in this preparation step. TF32 is enabled on CUDA.

    Returns (eroded_mask, safe_mask, safe_sdm), each [X,Y,Z] on signal.device.
    The first two are bool; safe_sdm is float32, zero outside safe_mask and
    capped at 10 inside it. The SDM is the diffusion-volume-count-weighted
    mean of log(mean_b0/mean_shell) after each DWI volume is clamped at zero.

    MRtrix dwi2response dhollander calls:
    maskfilter mask.mif erode eroded_mask.mif -npass 3
    dwiextract dwi.mif -shells B - | mrcalc - 0 -max - |
        mrmath - mean mean_bB.mif -axis 3
    mrcalc mean_b0.mif mean_bB.mif -divide -log sdm_bB.mif
    mrcalc safe_mask.mif full_sdm.mif 0 -if 10 -min safe_sdm.mif

    This operator stops before tensor fitting and tissue voxel selection.
    """
    if signal.ndim != 4 or signal.dtype != torch.float32:
        raise ValueError("signal must be float32 [X,Y,Z,N]")
    if erosion_passes < 0:
        raise ValueError("erosion_passes must be nonnegative")
    device = signal.device
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    grad = torch.as_tensor(grad_mrtrix, device=device, dtype=torch.float64)
    shells = torch.as_tensor(shell_bvals, device=device, dtype=torch.float64)
    mask = torch.as_tensor(brain_mask, device=device, dtype=torch.bool)
    if grad.shape != (signal.shape[-1], 4) or mask.shape != signal.shape[:3]:
        raise ValueError("gradient or brain mask shape differs from DWI")
    if shells.ndim != 1 or shells.numel() < 2 or abs(float(shells[0])) > 50:
        raise ValueError("shells must start at b0 and include diffusion data")
    distance = torch.abs(grad[:, 3, None] - shells[None, :])
    assignment = distance.argmin(dim=1)
    if bool((distance.gather(1, assignment[:, None]) > 100).any()):
        raise ValueError("gradient b-values do not match requested shells")
    eroded = _erode_six_neighbour(mask, erosion_passes)
    means = []
    for shell in range(shells.numel()):
        indices = assignment == shell
        if not bool(indices.any()):
            raise ValueError(f"shell {shell} has no DWI volumes")
        means.append(signal[..., indices].clamp_min(0).mean(dim=-1))
    safe = eroded.clone()
    for mean in means:
        safe &= torch.isfinite(mean) & (mean > 0)
    n_diffusion = signal.shape[-1] - int((assignment == 0).sum())
    sdm = torch.zeros(signal.shape[:3], dtype=torch.float32, device=device)
    for shell in range(1, shells.numel()):
        n = int((assignment == shell).sum())
        sdm += (means[0] / means[shell]).log() * n
    sdm /= n_diffusion
    safe &= torch.isfinite(sdm) & (sdm > 0)
    safe_sdm = torch.where(safe, sdm.clamp_max(10), 0)
    return eroded, safe, safe_sdm


def fit_mrtrix_dhollander_tensor(
    signal: torch.Tensor,
    grad_mrtrix: torch.Tensor,
    safe_mask: torch.Tensor,
    batch_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match Dhollander's default MRtrix IWLS tensor, FA, and principal vector.

    Inputs: corrected float32 DWI [X,Y,Z,N] on CPU/CUDA; MRtrix-exported
    gradient scheme [N,4] in the gradient frame with b-values in s/mm²;
    affine-aligned bool safe mask [X,Y,Z]; maximum voxels per solve.
    Returns (fa, principal_vector): float32 [X,Y,Z] and [X,Y,Z,3] on the
    input device, zero outside the mask. Vectors use the MRtrix gradient
    frame, have unit length within the mask and may have either antipodal
    sign. Tensor parameters are rounded to float32 before tensor2metric,
    matching MRtrix's intermediate image. Voxels with no positive DWI
    measurement return NaN FA/direction, as in MRtrix. CUDA TF32 is enabled.

    Equivalent MRtrix commands:
    dwi2tensor dwi.mif - -mask safe_mask.mif |
        tensor2metric - -fa safe_fa.mif -vector safe_vecs.mif
        -modulate none -mask safe_mask.mif

    The initial signal-weighted least squares fit is followed by two
    predicted-signal-weighted fits, MRtrix's default -iter 2. Singular
    normal equations use a QR least-squares fallback.
    """
    if signal.ndim != 4 or signal.dtype != torch.float32:
        raise ValueError("signal must be float32 [X,Y,Z,N]")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    device = signal.device
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    grad = torch.as_tensor(grad_mrtrix, device=device, dtype=torch.float64)
    mask = torch.as_tensor(safe_mask, device=device, dtype=torch.bool)
    if grad.shape != (signal.shape[-1], 4) or mask.shape != signal.shape[:3]:
        raise ValueError("gradient or safe mask shape differs from DWI")
    gx, gy, gz, b = grad.unbind(dim=1)
    design = torch.stack((
        -b * gx.square(), -b * gy.square(), -b * gz.square(),
        -2 * b * gx * gy, -2 * b * gx * gz, -2 * b * gy * gz,
        torch.ones_like(b),
    ), dim=1)
    flat_signal = signal.reshape(-1, signal.shape[-1])
    flat_fa = torch.zeros(flat_signal.shape[0], device=device, dtype=torch.float32)
    flat_vec = torch.zeros((flat_signal.shape[0], 3), device=device, dtype=torch.float32)
    voxels = torch.nonzero(mask.reshape(-1)).flatten()
    for block in voxels.split(batch_size):
        samples = flat_signal[block].to(torch.float64)
        valid = samples.amax(dim=1) > 0
        flat_fa[block[~valid]] = torch.nan
        flat_vec[block[~valid]] = torch.nan
        block = block[valid]
        if block.numel() == 0:
            continue
        samples = samples[valid]
        smallest = 1e-6 * samples.amax(dim=1, keepdim=True)
        samples = torch.maximum(samples, smallest)
        log_signal = samples.log()
        weights = samples
        for iteration in range(3):
            weighted_design = design[None, :, :] * weights[:, :, None]
            gram = weighted_design.transpose(1, 2) @ weighted_design
            right = weighted_design.transpose(1, 2) @ (weights * log_signal)[:, :, None]
            factor, info = torch.linalg.cholesky_ex(gram)
            parameters = torch.empty((block.numel(), 7), device=device, dtype=torch.float64)
            good = info == 0
            if bool(good.any()):
                parameters[good] = torch.cholesky_solve(right[good], factor[good]).squeeze(-1)
            if bool((~good).any()):
                parameters[~good] = torch.linalg.lstsq(
                    weighted_design[~good], (weights * log_signal)[~good],
                ).solution
            if iteration < 2:
                weights = (parameters @ design.T).exp()
        d = parameters[:, :6].to(torch.float32).to(torch.float64)
        d11, d22, d33, d12, d13, d23 = d.unbind(dim=1)
        mean = (d11 + d22 + d33) / 3
        off = 2 * (d12.square() + d13.square() + d23.square())
        numerator = 1.5 * (
            (d11 - mean).square() + (d22 - mean).square()
            + (d33 - mean).square() + off
        )
        denominator = d11.square() + d22.square() + d33.square() + off
        fa = torch.where(denominator > 0, (numerator / denominator).sqrt(), 0)
        tensor = torch.stack((
            d11, d12, d13, d12, d22, d23, d13, d23, d33,
        ), dim=1).reshape(-1, 3, 3)
        values, vectors = torch.linalg.eigh(tensor)
        principal = values.abs().argmax(dim=1)
        vec = vectors.gather(2, principal[:, None, None].expand(-1, 3, 1)).squeeze(2)
        flat_fa[block] = fa.to(torch.float32)
        flat_vec[block] = vec.to(torch.float32)
    return flat_fa.reshape(signal.shape[:3]), flat_vec.reshape(*signal.shape[:3], 3)


def _mrtrix_optimal_threshold(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return MRtrix's float32 golden-section correlation threshold.

    Inputs are float32 values [X,Y,Z] and bool selection mask [X,Y,Z] on
    one CPU/CUDA device. Output is a same-device float32 scalar threshold.
    The sorted cumulative sums only accelerate the cost evaluations; the
    golden-section points and 0.01 stopping rule follow MRtrix.
    """
    selected = values[mask & torch.isfinite(values)].reshape(-1)
    if selected.numel() < 2:
        raise ValueError("automatic threshold requires at least two finite values")
    ordered = selected.sort().values
    cumulative = ordered.to(torch.float64).cumsum(dim=0)
    count = ordered.numel()
    mean = cumulative[-1] / count
    one = torch.ones((), device=values.device, dtype=torch.float32)
    g1 = one * 0.61803399
    g2 = one - g1
    low, high = ordered[0], ordered[-1]
    span = high - low
    x0 = low + 0.001 * span
    x3 = high - 0.001 * span
    initial = (low + high) * 0.5
    if bool((high - initial).abs() > (initial - low).abs()):
        x1 = initial
        x2 = initial + g2 * (x3 - initial)
    else:
        x2 = initial
        x1 = initial - g2 * (initial - x0)

    def cost(threshold: torch.Tensor) -> torch.Tensor:
        below = int(torch.searchsorted(ordered, threshold, right=True))
        above = count - below
        if above == 0 or below == 0:
            return torch.tensor(float("inf"), device=values.device)
        proportion = above / count
        sum_above = cumulative[-1] - (cumulative[below - 1] if below else 0)
        covariance = sum_above / count - proportion * mean
        return -covariance / math.sqrt(proportion * (1 - proportion))

    f1, f2 = cost(x1), cost(x2)
    while bool(0.01 * (x1.abs() + x2.abs()) < (x3 - x0).abs()):
        if bool(f2 < f1):
            x0, x1 = x1, x2
            x2 = g1 * x1 + g2 * x3
            f1, f2 = f2, cost(x2)
        else:
            x3, x2 = x2, x1
            x1 = g1 * x2 + g2 * x0
            f2, f1 = f1, cost(x1)
    return x1 if bool(f1 < f2) else x2


def segment_mrtrix_dhollander(
    fa: torch.Tensor,
    safe_sdm: torch.Tensor,
    safe_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Match Dhollander's FA/SDM crude, refined, GM and CSF voxel selection.

    Inputs: MRtrix-equivalent float32 FA and SDM maps [X,Y,Z] plus bool
    safe mask [X,Y,Z] on one CPU/CUDA device in the same affine-aligned
    DWI voxel grid. Returns a dictionary of same-device bool [X,Y,Z]
    masks named crude_wm, crude_gm, crude_csf, refined_wm, refined_gm,
    refined_csf, voxels_gm, and voxels_csf. The final single-fibre WM
    selection is a separate FOD and peak-analysis stage.

    Equivalent MRtrix commands are the crude/refined mrcalc and
    mrthreshold pipelines inside dwi2response dhollander; defaults are
    FA > 0.2, final GM 2% and final CSF 10% of refined masks.
    """
    if fa.shape != safe_sdm.shape or fa.shape != safe_mask.shape or fa.ndim != 3:
        raise ValueError("FA, SDM and safe mask must share a 3D DWI grid")
    if fa.dtype != torch.float32 or safe_sdm.dtype != torch.float32:
        raise ValueError("FA and SDM must be float32")
    if fa.device != safe_sdm.device or fa.device != safe_mask.device:
        raise ValueError("FA, SDM and safe mask must share a device")
    safe = safe_mask.to(torch.bool)
    crude_wm = safe & (fa > 0.2)
    non_wm = safe & ~crude_wm
    non_wm_median = torch.quantile(safe_sdm[non_wm], 0.5)
    crude_score = torch.where(non_wm, safe_sdm - non_wm_median, 0)
    crude_csf = non_wm & (crude_score >= _mrtrix_optimal_threshold(crude_score, non_wm))
    crude_gm = non_wm & ~crude_csf

    wm_median = torch.quantile(safe_sdm[crude_wm], 0.5)
    wm_mad = torch.quantile((safe_sdm[crude_wm] - wm_median).abs(), 0.5)
    wm_outliers = crude_wm & (safe_sdm > wm_median + 1.4826 * wm_mad * 2)
    refined_wm = crude_wm & ~wm_outliers

    gm_median = torch.quantile(safe_sdm[crude_gm], 0.5)
    gm_high = crude_gm & (safe_sdm > gm_median)
    gm_low = crude_gm & ~gm_high
    high_score = torch.where(gm_high, safe_sdm - gm_median, 0)
    low_score = torch.where(gm_low, gm_median - safe_sdm, 0)
    high_selected = gm_high & (high_score < _mrtrix_optimal_threshold(high_score, gm_high))
    low_selected = gm_low & (low_score < _mrtrix_optimal_threshold(low_score, gm_low))
    refined_gm = high_selected | low_selected

    csf_min = safe_sdm[crude_csf].min()
    csf_extra = crude_csf | (wm_outliers & (safe_sdm > csf_min))
    csf_score = torch.where(csf_extra, safe_sdm - csf_min, 0)
    refined_csf = csf_extra & (
        csf_score >= _mrtrix_optimal_threshold(csf_score, csf_extra)
    )

    gm_count = round(int(refined_gm.sum()) * 0.02)
    refined_gm_median = torch.quantile(safe_sdm[refined_gm], 0.5)
    gm_distance = torch.where(
        refined_gm, (safe_sdm - refined_gm_median).abs() + 1, 0,
    )
    gm_values = gm_distance[refined_gm & (gm_distance != 0)].sort().values
    gm_cutoff = gm_values[gm_count - 1]
    voxels_gm = refined_gm & (gm_distance <= gm_cutoff)

    csf_count = round(int(refined_csf.sum()) * 0.1)
    csf_values = safe_sdm[refined_csf & (safe_sdm != 0)].sort().values
    csf_cutoff = csf_values[-csf_count]
    voxels_csf = refined_csf & (safe_sdm >= csf_cutoff)
    return {
        "crude_wm": crude_wm,
        "crude_gm": crude_gm,
        "crude_csf": crude_csf,
        "refined_wm": refined_wm,
        "refined_gm": refined_gm,
        "refined_csf": refined_csf,
        "voxels_gm": voxels_gm,
        "voxels_csf": voxels_csf,
    }


def select_mrtrix_single_fibre_wm(
    signal: torch.Tensor,
    grad_mrtrix: torch.Tensor,
    shell_bvals: torch.Tensor,
    refined_wm: torch.Tensor,
    csf_response: torch.Tensor,
    batch_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select Dhollander 2019 single-fibre WM voxels by two CSD peak fits.

    Inputs: corrected float32 DWI [X,Y,Z,N] on CPU/CUDA, MRtrix gradient
    scheme [N,4], shell centres [S] s/mm², aligned bool refined-WM mask
    [X,Y,Z], isotropic CSF response [S,1], and CSD batch size. The image
    affine must already match the DWI voxel grid; directions remain in the
    MRtrix gradient frame. Returns (empirical_wm_response float64 [S,4],
    metric_lmax2 float32 [X,Y,Z], metric_lmax6 float32 [X,Y,Z],
    refined_single_fibre_wm bool [X,Y,Z], selected_single_fibre_wm bool
    [X,Y,Z]), all on signal.device. Metrics are zero outside their search
    masks. The empirical response stores four zonal orders, and b0 uses
    its first coefficient only. CUDA TF32 is enabled; no float16 is used.

    Equivalent Dhollander MRtrix sequence: mrmath dwi.mif mean mean_sig.mif
    -axis 3; dwi2fod msmt_csd dwi.mif ewmrf.txt abs_ewm2.mif
    response_csf.txt abs_csf2.mif -mask refined_wm.mif -lmax 2,0;
    sh2peaks abs_ewm2.mif - -num 1 -mask refined_wm.mif |
    peaks2amp - -; mrthreshold metric_sfwm2.mif -top 2N -ignorezero;
    repeat dwi2fod/sh2peaks/peaks2amp at -lmax 6,0 on those voxels,
    then mrthreshold metric_sfwm6.mif -top N -ignorezero.
    """
    from .fod import fit_mrtrix_two_tissue_csd, mrtrix_fod_peak_amplitude

    if signal.ndim != 4 or signal.dtype != torch.float32:
        raise ValueError("signal must be float32 [X,Y,Z,N]")
    mask = torch.as_tensor(refined_wm, device=signal.device, dtype=torch.bool)
    if mask.shape != signal.shape[:3] or not bool(mask.any()):
        raise ValueError("refined WM mask must match DWI and be nonempty")
    if signal.device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    shells = torch.as_tensor(shell_bvals, device=signal.device, dtype=torch.float64)
    centre = torch.quantile(signal.mean(dim=3)[mask], 0.5).to(torch.float64)
    coefficient = centre * math.sqrt(4 * math.pi)
    empirical = coefficient.repeat(shells.numel(), 4)
    for index in range(shells.numel()):
        if abs(float(shells[index])) < 50:
            empirical[index, 1:] = 0
        else:
            empirical[index, 1] = -coefficient
            empirical[index, 3] = -coefficient
    csf = torch.as_tensor(csf_response, device=signal.device, dtype=torch.float64).reshape(-1)
    if csf.shape != shells.shape:
        raise ValueError("CSF response must have one coefficient per shell")
    selected_count = round(int(mask.sum()) * 0.005)
    if selected_count < 1:
        raise ValueError("refined WM mask is too small for 0.5% selection")

    metric_maps = []
    current_mask = mask
    masks = []
    for lmax, number in ((2, 2 * selected_count), (6, selected_count)):
        wm_fod, csf_fod = fit_mrtrix_two_tissue_csd(
            signal, grad_mrtrix, shells, empirical, csf, current_mask,
            lmax=lmax, batch_size=batch_size,
        )
        amplitudes = mrtrix_fod_peak_amplitude(wm_fod[current_mask], lmax)
        denominator = wm_fod[..., 0][current_mask].to(torch.float64) + csf_fod[current_mask].to(torch.float64)
        scores = (amplitudes / denominator).to(torch.float32)
        metric = torch.zeros(signal.shape[:3], device=signal.device, dtype=torch.float32)
        metric[current_mask] = scores
        eligible = torch.nonzero(current_mask.reshape(-1)).flatten()
        count = min(number, eligible.numel())
        selected = torch.zeros(signal.shape[:3], device=signal.device, dtype=torch.bool)
        selected.reshape(-1)[eligible[torch.topk(scores, count).indices]] = True
        metric_maps.append(metric)
        masks.append(selected)
        current_mask = selected
    return empirical, metric_maps[0], metric_maps[1], masks[0], masks[1]


def estimate_mrtrix_dhollander(
    signal: torch.Tensor,
    grad_mrtrix: torch.Tensor,
    shell_bvals: torch.Tensor,
    brain_mask: torch.Tensor,
    batch_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Run Dhollander three-tissue response estimation on corrected DWI.

    Inputs: float32 corrected DWI [X,Y,Z,N] on CPU/CUDA; MRtrix exported
    gradient table [N,4] in gradient frame; shell centres [S] in s/mm²;
    affine-aligned bool brain mask [X,Y,Z]; maximum tensor/CSD batch
    size. The caller must align NIfTI affines before tensor conversion.
    Returns (shells float64 [S], WM float64 [S,6], GM float64 [S,1],
    CSF float64 [S,1], masks dict). Every tensor is on signal.device.
    The dictionary contains bool [X,Y,Z] voxel masks named voxels_sfwm,
    voxels_gm, voxels_csf, refined_sfwm, safe_mask, and tissue selection
    masks from segment_mrtrix_dhollander. It also contains float32 maps
    safe_sdm, fa [X,Y,Z], principal_directions [X,Y,Z,3],
    metric_sfwm2 and metric_sfwm6 [X,Y,Z]. TF32 is enabled on CUDA.

    Equivalent MRtrix: dwi2response dhollander corrected.mif
    response_wm.txt response_gm.txt response_csf.txt -mask brain_mask.mif
    -voxels voxels.mif -scratch scratch -nocleanup.
    The complete fixed-parameter default path includes mask erosion,
    SDM, IWLS tensor/FA, tissue selection, two-tissue CSD, FOD peaks,
    and amp2response. Use identical corrected DWI, gradient frame,
    brain mask, and shell centres for numerical comparison.
    """
    shells = torch.as_tensor(shell_bvals, device=signal.device, dtype=torch.float64)
    _, safe_mask, safe_sdm = prepare_mrtrix_dhollander_sdm(
        signal, grad_mrtrix, shells, brain_mask,
    )
    fa, directions = fit_mrtrix_dhollander_tensor(
        signal, grad_mrtrix, safe_mask, batch_size=batch_size,
    )
    masks = segment_mrtrix_dhollander(fa, safe_sdm, safe_mask)
    csf = estimate_mrtrix_selected_response(
        signal, grad_mrtrix, shells, masks["voxels_csf"], isotropic=True,
    )
    gm = estimate_mrtrix_selected_response(
        signal, grad_mrtrix, shells, masks["voxels_gm"], isotropic=True,
    )
    _, metric2, metric6, refined_sfwm, voxels_sfwm = select_mrtrix_single_fibre_wm(
        signal, grad_mrtrix, shells, masks["refined_wm"], csf,
        batch_size=batch_size,
    )
    wm = estimate_mrtrix_selected_response(
        signal, grad_mrtrix, shells, voxels_sfwm, directions,
    )
    masks.update({
        "safe_mask": safe_mask, "safe_sdm": safe_sdm, "fa": fa,
        "principal_directions": directions, "metric_sfwm2": metric2,
        "metric_sfwm6": metric6, "refined_sfwm": refined_sfwm,
        "voxels_sfwm": voxels_sfwm,
    })
    return shells, wm, gm, csf, masks


def mrtrix_shell_centres(
    grad_mrtrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cluster gradient b-values as MRtrix 3.0.3 and label Dhollander shells.

    Input: MRtrix-exported float32/float64 gradient table [N,4] on
    CPU/CUDA; column 3 holds b-values in s/mm². Directions and affine
    are irrelevant to this calculation. Returns same-device tensors
    (raw_means float64 [S], dhollander_selection float64 [S],
    amp2response_header float64 [S], shell_sizes int64 [S]), sorted by
    raw mean. Dhollander rounds MRtrix mrinfo output to nearest integer
    for -shells selection. amp2response truncates each raw mean to an
    integer when writing the response text header. This routine uses
    MRtrix's b<=10 b0 rule and density clustering with epsilon 80 and
    minimum linkage 3, including its original volume order. Volumes that
    remain unassigned are omitted from the reported shells, as in MRtrix.

    Equivalent MRtrix commands: mrinfo corrected.mif -shell_bvalues
    -shell_sizes; dwi2response dhollander corrected.mif wm.txt gm.txt
    csf.txt; head -1 wm.txt. The raw means can have decimals even when
    the response header contains integer labels.
    """
    grad = torch.as_tensor(grad_mrtrix)
    if grad.ndim != 2 or grad.shape[1] != 4 or grad.dtype not in (torch.float32, torch.float64):
        raise ValueError("gradient must be float32/float64 [N,4]")
    bvals = grad[:, 3].detach().cpu().to(torch.float64).tolist()
    n = len(bvals)
    if not n or any(not math.isfinite(value) for value in bvals):
        raise ValueError("gradient b-values must be finite and nonempty")
    clusters = [0] * n
    visited = [False] * n
    label = 0
    for index, value in enumerate(bvals):
        if value <= 10:
            visited[index] = True
            clusters[index] = 1
            label = 1
    for index, value in enumerate(bvals):
        if visited[index]:
            continue
        visited[index] = True
        neighbours = [i for i, other in enumerate(bvals)
                      if other > 10 and abs(value - other) < 80]
        if len(neighbours) < 3:
            continue
        label += 1
        clusters[index] = label
        position = 0
        while position < len(neighbours):
            neighbour = neighbours[position]
            if not visited[neighbour]:
                visited[neighbour] = True
                nearby = [i for i, other in enumerate(bvals)
                          if other > 10 and abs(bvals[neighbour] - other) < 80]
                if len(nearby) >= 3:
                    neighbours.extend(nearby)
            if clusters[neighbour] == 0:
                clusters[neighbour] = label
            position += 1
    if label < 1 or label > math.sqrt(n):
        raise ValueError("gradient cannot be classified into MRtrix shells")
    shells = [
        (sum(bvals[i] for i in range(n) if clusters[i] == group)
         / sum(clusters[i] == group for i in range(n)),
         sum(clusters[i] == group for i in range(n)))
        for group in range(1, label + 1)
    ]
    shells.sort(key=lambda item: item[0])
    means = torch.tensor([item[0] for item in shells], device=grad.device, dtype=torch.float64)
    sizes = torch.tensor([item[1] for item in shells], device=grad.device, dtype=torch.int64)
    selection = torch.tensor(
        [round(float(format(item[0], ".6g"))) for item in shells],
        device=grad.device, dtype=torch.float64,
    )
    return means, selection, means.trunc(), sizes
