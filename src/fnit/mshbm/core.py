"""Single-subject 17-network MS-HBM adapted from CBIG (MIT license).

Original: ThomasYeoLab/CBIG, Kong2019_MSHBM, commit
b69b822a15e2a94f1e439606552fc44b6858cf3c. See licenses/CBIG-MIT.txt.
The implementation uses NumPy/SciPy on CPU and requires no MATLAB or CBIG
installation. Only the fs_LR_32k/HCP_40 17-network case is supported.
"""

from importlib.resources import files

import numpy as np
from scipy import sparse, special, optimize


def load_assets(path=None):
    """Load CBIG HCP_40 priors, cortex mask, 900-mesh seeds and mesh graph."""
    path = path or files(__package__).joinpath("assets/hcp40_fslr32k_17.npz")
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    graph = sparse.csr_matrix(
        (data["adj_data"], data["adj_indices"], data["adj_indptr"]),
        shape=(64984, 64984),
    )
    mask = data["cortex_mask"].astype(bool)
    seeds = data["seed_vertices"].astype(np.int64)
    if (mask.shape != (64984,) or mask.sum() != 59412
            or seeds.shape != (1483,) or not mask[seeds].all()
            or data["mu"].shape != (1483, 17)
            or data["theta"].shape != (64984, 17)):
        raise ValueError("Incompatible CBIG fsLR32k HCP_40 assets")
    data["graph"] = graph
    data["cortex_mask"] = mask
    data["seed_vertices"] = seeds
    return data


def _unit_columns(x):
    x = np.asarray(x, dtype=np.float32).copy()
    x -= x.mean(axis=0, keepdims=True)
    norm = np.sqrt(np.einsum("ij,ij->j", x, x, optimize=True))
    x /= np.maximum(norm, 1e-20)
    return x


def _profile(part, seed_positions, full_vertex_count=64984, threshold=0.1):
    """CBIG_corr plus a single global top-10% binarization per session."""
    if len(part) < 4:
        raise ValueError("At least four uncensored frames are needed per session")
    target = _unit_columns(part)
    seed = target[:, seed_positions]
    corr = np.asarray(target.T @ seed, dtype=np.float32)
    # MATLAB sort(descend) then round(numel * 0.1), with 1-based indexing.
    rank = int(np.floor(full_vertex_count * len(seed_positions) * threshold + 0.5))
    if not 1 <= rank <= corr.size:
        raise ValueError("Invalid global profile rank")
    cutoff = np.partition(corr.ravel(), corr.size - rank)[corr.size - rank]
    binary = (corr >= cutoff).astype(np.float32)
    binary -= binary.mean(axis=1, keepdims=True)
    norm = np.sqrt(np.einsum("ij,ij->i", binary, binary, optimize=True))
    binary /= np.maximum(norm[:, None], 1e-20)
    return binary


def profiles_from_timeseries(series, assets, censor=None):
    """Make two CBIG-style pseudo-sessions from one T×59412 cortical run."""
    series = np.asarray(series, dtype=np.float32)
    if series.ndim != 2 or series.shape[1] != 59412:
        raise ValueError("Expected [time, 59412] fsLR32k cortical series")
    if not np.isfinite(series).all():
        raise ValueError("Time series contains nonfinite values")
    if censor is not None:
        censor = np.asarray(censor).ravel()
        if censor.shape != (len(series),) or not np.isin(censor, [0, 1]).all():
            raise ValueError("Censor must contain one 0/1 value per frame")
        series = series[censor.astype(bool)]
    if len(series) < 8:
        raise ValueError("At least eight uncensored frames are needed")
    mask = assets["cortex_mask"]
    positions = np.searchsorted(np.flatnonzero(mask), assets["seed_vertices"])
    half = len(series) // 2
    return [_profile(part, positions) for part in (series[:half], series[half:])]


def _log_iv(order, kappa):
    """Log modified Bessel I, avoiding scaled-I underflow at small kappa."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        raw = special.iv(order, kappa)
        scaled = special.ive(order, kappa)
        return np.where(np.isfinite(raw) & (raw > 0), np.log(raw),
                        np.log(scaled) + kappa)


def _cdln(kappa, dim):
    """CBIG Cdln, including its 1000-point high-kappa approximation."""
    kappa = np.asarray(kappa, dtype=np.float64)
    k0 = 500.0 if dim < 1200 else 650.0
    order = dim / 2 - 1
    base = order * np.log(k0) - _log_iv(order, k0)
    out = order * np.log(kappa) - _log_iv(order, kappa)
    large = kappa > k0
    if np.any(large):
        steps = (kappa[large] - k0) / 1000
        grid = k0 + steps[:, None] * (np.arange(1000) + 0.5)
        ratio = 0.5 * (dim - 1) / grid
        out[large] = base - steps * np.sum(1 / (ratio + np.sqrt(1 + ratio**2)), axis=1)
    return out


def _inv_ad(dim, rbar):
    """CBIG invAd with its overflow fallback for high-dimensional profiles."""
    rbar = float(np.clip(rbar, 1e-8, 1 - 1e-8))
    guess = (dim - 1) * rbar / (1 - rbar * rbar) + dim / (dim - 1) * rbar
    iv = special.iv(dim / 2 - 1, guess)
    if not np.isfinite(iv) or iv == 0:
        return guess - dim / (dim - 1) * rbar / 2
    def ratio(k):
        numerator, denominator = special.iv(dim / 2, k), special.iv(dim / 2 - 1, k)
        if not np.isfinite(denominator) or denominator == 0:
            numerator, denominator = special.ive(dim / 2, k), special.ive(dim / 2 - 1, k)
        return numerator / denominator - rbar

    try:
        answer = optimize.newton(ratio, guess, x1=guess * 1.01, maxiter=100)
        if np.isfinite(answer) and answer > 0:
            return answer
    except (RuntimeError, ValueError, ZeroDivisionError):
        pass
    return guess - dim / (dim - 1) * rbar / 2


def _softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def parcellate(profiles, assets, w=200.0, c=50.0, *, max_outer=50,
               max_em=101, max_m=300, max_lambda=101):
    """Run CBIG single-subject MS-HBM and return full 64984-vertex labels.

    Profiles are row-normalized cortex×1483 matrices, one per session. Network
    labels use CBIG's fixed 1..17 HCP_40 ordering; medial/invalid rows are 0.
    The official wrapper always supplies at least two sessions, splitting a
    single run in half. Convergence tolerances match the MATLAB implementation.
    """
    if len(profiles) < 2:
        raise ValueError("MS-HBM requires at least two session profiles")
    n = 59412
    mu = np.asarray(assets["mu"], dtype=np.float64)
    sigma = np.asarray(assets["sigma"], dtype=np.float64).ravel()
    epsil = np.asarray(assets["epsil"], dtype=np.float64).ravel()
    mask = assets["cortex_mask"]
    theta = np.asarray(assets["theta"][mask], dtype=np.float64)
    graph = assets["graph"][mask][:, mask].tocsr().astype(np.float64)
    x = [np.asarray(p, dtype=np.float32) for p in profiles]
    if any(p.shape != (n, 1483) or not np.isfinite(p).all() for p in x):
        raise ValueError("Each profile must be finite [59412, 1483]")
    valid = np.any(np.stack([np.any(p != 0, axis=1) for p in x]), axis=0)
    x = [p[valid] for p in x]
    theta = theta[valid]
    graph = graph[valid][:, valid].tocsr()
    degree = np.asarray(graph.sum(axis=1))
    log_theta = np.log(theta, where=theta > 0, out=np.full_like(theta, -np.inf))
    dim = 1482
    sessions = len(x)
    kappa = 650.0
    psi = mu.copy()
    nu = [mu.copy() for _ in x]
    lamb = _softmax(sum(p @ mu for p in x) * kappa)
    prev_outer = 0.0
    history = []
    for outer in range(max_outer):
        kappa = 650.0
        nu = [mu.copy() for _ in x]
        prev_em = 0.0
        for em in range(max_em):
            for m in range(max_m):
                dots = sum(np.sum((p @ direction) * lamb) for p, direction in zip(x, nu))
                rbar = dots / (sessions * lamb.sum())
                updated_kappa = _inv_ad(dim, rbar)
                new_nu = []
                for p in x:
                    direction = updated_kappa * (p.T @ lamb) + psi * sigma
                    direction /= np.maximum(np.linalg.norm(direction, axis=0), 1e-20)
                    new_nu.append(direction)
                converged_nu = all(np.all(1 - np.sum(a * b, axis=0) < 1e-4)
                                   for a, b in zip(new_nu, nu))
                converged_kappa = abs(updated_kappa - kappa) / kappa < 1e-4
                nu, kappa = new_nu, updated_kappa
                if converged_nu and converged_kappa:
                    break
            else:
                raise RuntimeError("MS-HBM M-step did not converge")
            data_logits = kappa * sum(p @ direction for p, direction in zip(x, nu))
            prior_logits = w * log_theta
            prior_logits[~np.isfinite(prior_logits)] = -1e20
            previous_change = 0.0
            for lam_iter in range(max_lambda):
                neighbor_disagreement = degree - graph @ lamb
                updated_lamb = _softmax(data_logits + prior_logits - 2 * c * neighbor_disagreement)
                change = np.mean(np.abs(updated_lamb - lamb))
                lamb = updated_lamb
                if abs(change - previous_change) <= 1e-4:
                    break
                previous_change = change
            entropy = -np.sum(lamb * np.log(np.maximum(lamb, np.finfo(float).eps**20)))
            cd_kappa = float(_cdln(np.array([kappa]), dim)[0])
            em_cost = (np.sum(lamb * data_logits) + lamb.sum() * sessions * cd_kappa
                       + np.sum(lamb * prior_logits) + entropy
                       - c * np.sum(lamb * neighbor_disagreement))
            if em and abs(em_cost - prev_em) / max(abs(prev_em), 1) <= 1e-4:
                em_record_cost = prev_em  # CBIG records the previous EM cost.
                break
            prev_em = em_cost
        else:
            em_record_cost = em_cost
        psi = epsil * mu + sum(sigma * direction for direction in nu)
        psi /= np.maximum(np.linalg.norm(psi, axis=0), 1e-20)
        outer_cost = (em_record_cost + sum(np.sum(sigma * psi * direction) for direction in nu)
                      + sessions * np.sum(_cdln(sigma, dim))
                      + np.sum(epsil * mu * psi) + np.sum(_cdln(epsil, dim)))
        history.append({"outer": outer + 1, "em": em + 1, "kappa": float(kappa),
                        "cost": float(outer_cost)})
        if outer and abs(outer_cost - prev_outer) / max(abs(prev_outer), 1) <= 1e-4:
            break
        prev_outer = outer_cost
    labels = np.zeros(64984, dtype=np.uint8)
    cortex = np.flatnonzero(mask)
    labels[cortex[valid]] = np.argmax(lamb, axis=1).astype(np.uint8) + 1
    return labels, history
