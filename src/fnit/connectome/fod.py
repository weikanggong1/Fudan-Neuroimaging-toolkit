"""PyTorch MRtrix-compatible three-tissue spherical deconvolution.

The default lmax=8 MSMT-CSD uses raw DWI amplitudes, official response
coefficients, and MRtrix's 300-direction active-set constraints.
"""

from __future__ import annotations

import math
from functools import lru_cache

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
    """Evaluate even-degree orthonormal real SH in the caller's frame.

    ``directions`` is float32 or float64 with shape ``[..., 3]``; nonzero vectors are
    normalized internally. Coefficients are ordered by even ``l``, then by
    ``m=-l,...,l``. Returns a same-device tensor ``[...,C]`` where
    ``C=(lmax+1)*(lmax+2)//2``. The basis includes the Condon-Shortley phase.
    """
    if lmax < 0 or lmax % 2:
        raise ValueError("lmax must be a nonnegative even integer")
    if directions.ndim < 2 or directions.shape[-1] != 3 or directions.dtype not in (torch.float32, torch.float64):
        raise ValueError("directions must be float32 or float64 with shape [..., 3]")
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


@lru_cache(maxsize=8)
def _tracking_sh_table(lmax: int, device: torch.device) -> torch.Tensor:
    """Build MRtrix iFOD2's 512-elevation float32 Legendre lookup table."""
    count = 512
    inc = torch.tensor(math.pi / (count - 1), device=device, dtype=torch.float32)
    z = (torch.arange(count, device=device, dtype=torch.float32) * inc).cos()
    width = (lmax + 1) * (lmax + 2) // 2
    table = torch.zeros((count, width), device=device, dtype=torch.float32)
    for l in range(0, lmax + 1, 2):
        for m in range(l + 1):
            norm = math.sqrt((2 * l + 1) / (4 * math.pi) *
                             math.factorial(l - m) / math.factorial(l + m))
            index = l * (l - 1) // 2 + l + m
            table[:, index] = norm * _associated_legendre(l, m, z) * (math.sqrt(2) if m else 1)
    return table



@lru_cache(maxsize=8)
def _tracking_sh_indices(lmax: int, device: torch.device):
    """Cache coefficient columns without changing SH recurrence or ordering."""
    centres = torch.tensor([l * (l - 1) // 2 + l
                            for l in range(0, lmax + 1, 2)],
                           dtype=torch.long, device=device)
    orders = []
    for m in range(1, lmax + 1):
        positive = torch.tensor([l * (l - 1) // 2 + l + m
                                 for l in range(m + (m & 1), lmax + 1, 2)],
                                dtype=torch.long, device=device)
        orders.append((positive, positive - 2 * m))
    return centres, tuple(orders)


def tracking_sh_precomputed(directions: torch.Tensor, lmax: int = 8) -> torch.Tensor:
    """Evaluate tracking SH via MRtrix iFOD2's 512-elevation lookup rule.

    Input is nonzero float32 directions ``[...,3]`` on CPU/CUDA; output is
    float32 real SH ``[...,C]`` in the same coefficient order as ``real_sh``.
    The table is cached per device and lmax. Equivalent original operation:
    ``Math::SH::PrecomputedAL<float>::value`` in MRtrix3 iFOD2, enabled by
    ``tckgen -algorithm iFOD2 ...``. Real-FOD single-arc comparison is in
    ``validation/connectome/ds004666/ifod2_single_arc_20260929.md``.
    """
    if lmax < 0 or lmax % 2 or directions.shape[-1] != 3 or directions.dtype != torch.float32:
        raise ValueError("expected float32 directions [...,3] and nonnegative even lmax")
    shape = directions.shape[:-1]
    flat = directions.reshape(-1, 3)
    unit = flat / torch.linalg.vector_norm(flat, dim=-1).clamp_min(1e-20)[:, None]
    table = _tracking_sh_table(lmax, unit.device)
    inc = unit.new_tensor(math.pi / 511)
    position = unit[:, 2].clamp(-1, 1).acos() / inc
    lower = position.long().clamp(0, 511)
    fraction = (position - lower).clamp(0, 1)
    upper = (lower + 1).clamp_max(511)
    basis = (1 - fraction)[:, None] * table[lower] + fraction[:, None] * table[upper]
    radius = torch.linalg.vector_norm(unit[:, :2], dim=-1)
    cosine = torch.where(radius > 0, unit[:, 0] / radius.clamp_min(1e-20), 1)
    sine = torch.where(radius > 0, unit[:, 1] / radius.clamp_min(1e-20), 0)
    output = torch.zeros_like(basis)
    centres, orders = _tracking_sh_indices(lmax, unit.device)
    output.index_copy_(1, centres, basis.index_select(1, centres))
    cos_m = torch.ones_like(cosine)
    sin_m = torch.zeros_like(sine)
    for positive, negative in orders:
        next_cos = cos_m * cosine - sin_m * sine
        next_sin = sin_m * cosine + cos_m * sine
        values = basis.index_select(1, positive)
        output.index_copy_(1, positive, values * next_cos[:, None])
        output.index_copy_(1, negative, values * next_sin[:, None])
        cos_m, sin_m = next_cos, next_sin
    return output.reshape(*shape, -1)


# MRtrix3 electrostatic_repulsion_300_data (azimuth, elevation; radians).
# Copyright 2008-2026 MRtrix3 contributors; MPL-2.0.
# https://github.com/MRtrix3/mrtrix3/blob/master/src/dwi/directions/predefined.cpp
_MRTRIX300_ANGLES = """
2.832910411 1.21211455 -2.74511538 1.363952022 -2.204084013 1.425059005 -2.148346504 1.004302434
2.251025375 0.6174449251 -1.786980409 1.024014087 2.481189158 0.6889487595 -3.026542367 0.6217626872
-0.1563077565 1.002601793 -0.4647931006 1.301328499 1.069159292 0.2467513039 1.490547947 1.313067057
-1.609622922 1.084310198 -0.0004986942567 0.9284414331 0.6780965978 0.6321788036 0.2064533907 1.440152241
-0.1005703034 0.3273867762 1.890696276 1.510135764 -0.9020555768 0.810401083 -1.037187785 1.276317725
2.510108617 1.130408667 -3.019306162 1.230113421 -1.478787467 0.8400208601 2.961926464 1.107880087
1.004868663 1.298490352 0.7383696941 1.170000081 1.238613461 1.713432069 1.592726689 0.7241099974
3.045189314 1.574862216 -3.120725891 1.483623564 -2.724138463 1.216574225 -0.6488243793 0.5468382434
-2.304077619 0.604577541 3.116144069 1.171023784 0.5200311385 0.5074509463 -0.1776445613 1.326926147
1.722052153 0.8318688419 -0.7708479928 1.354721326 3.013127399 1.421374167 -1.994898461 0.3406173931
-2.879022586 1.293394538 1.554943994 0.3202461057 -0.3720471332 1.535812283 2.079096701 0.3193853107
-1.137598729 1.013421277 2.305937572 0.9223543757 0.568358547 1.578181919 -1.642685525 0.9302839417
2.501254221 0.8473314642 -1.971637726 1.288389796 1.328209573 1.306525472 1.408565095 0.6144164165
-2.049098408 0.8688347278 -0.05969996249 1.413143868 -1.844636682 0.8829552164 -1.304360476 1.072348115
1.662334937 1.33491501 -2.156525364 0.7335925293 0.4607724913 1.477721271 -0.01442374758 0.7753128409
0.05482592601 1.501835338 -2.842599666 0.9937532989 0.6782082395 1.305091242 -1.530444363 0.5420175914
2.861469444 1.364292405 0.7067737669 0.3690864835 2.337695953 1.085975307 2.706124032 1.312128011
-1.839264222 0.4869958516 1.681673207 0.5769940073 2.67207084 0.7760403202 -0.01059485785 1.090313816
0.4215118976 0.6561689907 -0.8421832358 1.077942855 1.085448462 1.165126191 1.191306788 0.709828936
0.7193991674 1.572807371 -1.210876423 1.479084373 0.4466357116 1.076916227 0.8528047789 0.5016712981
-2.025469163 0.6036329164 2.752250827 0.9040064257 -2.6085475 0.7815756692 0.6239088671 1.047542499
-2.79749403 0.6913676339 0.9445314904 0.6396550135 1.458240988 0.4660430908 -2.811322004 0.8466080623
-1.064787582 1.432682632 0.2379422876 1.279081318 1.389657788 0.7658180314 -1.474407133 1.282913028
-1.465773555 1.425868537 -0.6511385683 1.471768729 -0.2993375131 1.077105211 -0.03630640307 0.6255265495
2.423365383 1.378968033 1.558507571 1.447470351 -1.462844011 1.141840835 -1.239721608 0.4772513637
-0.6773458919 0.7804795558 -2.287459848 1.28638671 0.569910669 0.777236151 -1.900837718 1.152842986
-0.7748140099 0.9240593031 -0.9187001443 1.39079111 -1.621609685 0.2381382959 2.58089914 1.416290038
-1.106661168 0.8599239828 1.243305801 1.437687916 1.180727888 1.033460785 0.3181295082 0.9668715249
-0.06310447498 0.4765729285 1.837084154 0.7003090114 -2.130229143 1.284494385 -1.563297858 0.3903004688
1.092263032 0.90440959 0.7876927647 0.7663029724 2.547380636 1.273576138 2.577018549 0.2384208894
2.850058144 0.09955760825 0.2776141307 1.1229415 0.5116196487 1.332459251 1.015919787 1.574789947
-0.4042572656 1.681896261 -2.44544071 1.295991185 0.3626128999 0.8103410101 1.007613126 0.779002203
-2.515530393 0.4804682132 -0.5187651881 0.6705736361 1.834622232 1.369508879 -0.8437098778 0.4337077459
1.978669178 0.5751224243 -1.424701953 0.06917390096 -1.463090842 0.9895237278 -2.860260824 1.456311155
-0.7388212471 1.20212028 -1.285830779 0.7709172196 -1.688873575 0.7817859985 -0.5935019503 0.314154943
2.106573129 1.29755113 -2.303284614 0.1873976217 -0.8893622886 1.235669114 2.495674418 0.5374548213
-2.698672249 1.49816587 2.725985273 0.6359006202 2.887062397 0.7808329911 -1.546827231 1.942372946
0.3514024933 1.37731788 1.641689698 0.958130962 1.317058595 1.575223377 -0.3373381026 1.390232552
2.612637864 1.55837929 -0.672506152 1.046671362 -2.865989146 1.141513944 1.818620348 0.181910502
1.16272067 0.5491598367 1.888452529 1.115921467 -0.1585830655 1.164810295 -2.560759144 1.181570618
2.892220217 1.514660665 2.383454269 1.233520288 -1.610114573 1.406304316 2.21872195 1.190267314
-3.134630214 0.4817619796 -2.993655476 0.7786292516 2.193004223 0.4660824923 -0.4947538648 1.028850845
-2.803442977 0.5348464929 -2.183980289 0.4676591963 -1.807531345 1.296157179 0.09011459489 1.34529017
0.2168012355 0.552225771 2.673610191 1.162048539 0.189036651 0.7060285858 -0.03041700312 1.25365758
-0.6200889758 1.325368762 2.295732283 0.7732012654 -0.9615514424 0.9629977396 1.724236704 1.472115801
2.056894646 1.146662488 0.9119715726 1.165347221 2.00452577 0.9946875306 -1.163569752 1.168112177
1.401122748 1.443695427 3.104779566 1.018113979 -2.456180444 0.3325861891 -1.9763898 1.56717627
1.390681112 1.714067423 1.775619215 1.23136107 -0.215043731 1.477494271 0.5005650593 0.9270638765
0.1676854336 1.593049538 2.305766378 1.487497239 0.6227146792 1.440968844 -0.3136084408 0.5765043079
-2.253308928 0.8708885791 1.4186025 1.175389703 -1.50191445 0.6914910009 0.118425973 1.183144662
-1.750899093 0.6346324164 0.9935954328 1.034089034 1.930783106 0.8422226779 -2.819530622 1.602796121
2.968898396 0.6055606566 -2.510781235 1.436621153 -0.2440180225 0.7146578925 2.962216197 0.3485001896
1.481485574 0.8938483896 -2.355476857 1.427779336 -1.326157252 1.373285803 -1.099116768 1.586150528
0.276940008 0.4000588264 -3.016952487 1.077196632 0.8068446545 1.036033042 -1.967446472 1.008602606
0.9340550343 1.435050644 -2.604152821 1.330192986 -2.857751492 0.3854054916 1.088335949 1.434972021
-0.353620567 0.935265291 -1.005484813 1.119053813 -0.5754532629 0.9022833725 2.125626026 0.8743942664
0.8551725856 0.09946507294 2.466658736 0.9861454111 -1.065040368 0.7092429288 2.932065366 0.9461806989
1.712085819 1.094846898 -2.454818321 0.8914308479 1.129606523 0.3965596366 -0.8014930137 1.505035935
-2.511528461 1.036822988 2.794908881 0.4831137974 3.135588276 1.327359699 0.6923309363 0.9057827073
2.985972513 1.26620029 -1.35847526 1.530259933 -0.9930820929 0.5636921299 -2.392478663 1.154152808
1.522201845 1.060647972 2.153099327 1.447106597 1.355855539 1.020889178 -0.4294951078 1.16043297
-0.8112917019 0.6660442728 -2.69114299 1.073167031 -0.9604950696 0.2027712467 3.083999858 0.7279084433
-3.002227432 1.389472551 -2.22653226 1.144698717 2.075010695 0.7231111118 2.175485406 1.032920384
0.8422612475 1.299027344 1.826448715 0.9713631715 -1.272827977 0.6229774124 -2.062682237 1.144821661
1.944715153 1.260164652 -2.743962874 1.915140215 -2.388935599 0.7466681265 -2.361888158 1.706759535
0.8889768777 0.9038757515 -3.008254805 0.9275289509 -1.317745192 1.224116549 1.823504987 0.4400858514
-2.565795661 0.6308614059 -1.626896793 1.259954901 -0.6831984006 1.61734329 -0.9502034716 1.54358224
1.257063922 1.165787745 0.867601728 1.574827054 -0.1815315567 0.1766637885 -1.5224926 1.558299787
-1.920673206 0.7427644488 -2.05383815 1.425724978 -1.293269388 0.9205186701 1.998216504 1.405230931
2.51071955 0.384770373 0.1476508105 1.022508571 1.271921602 0.8816492104 1.167386076 1.299387241
-0.4974579947 1.441891468 -0.4167658657 0.4419641864 0.4240045867 0.2507420401 0.1718848515 0.8622683095
-1.148908852 0.3379573252 -2.976449663 0.2375465318 -1.182483732 1.322966996 -0.3004018302 1.242825303
-0.5856199546 1.17685907 -0.1922746014 0.8558516864 0.5653267258 1.189673224 2.632998793 1.006878133
-1.740965097 1.166705435 3.092569002 0.8700702716 -2.329885664 1.011896463 -0.4253251876 0.8005358047
2.265511385 1.339200324 1.47106441 1.582903696 2.802135393 1.057963848 -2.649071 0.930030914
"""


def _mrtrix_msmt_design(
    grad_mrtrix: torch.Tensor,
    shell_bvals: torch.Tensor,
    wm_response: torch.Tensor,
    gm_response: torch.Tensor,
    csf_response: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build MRtrix MSMT-CSD signal and nonnegative-amplitude matrices.

    Gradients are the four columns exported by mrconvert -export_grad_mrtrix.
    Response rows follow their Shells header. WM default lmax is 8; the first
    five even-order zonal coefficients are used. GM/CSF are isotropic.
    Inputs: grad ``[N,4]``; shell centres ``[S]``; WM ``[S,>=5]``;
    GM/CSF ``[S]``. All directions share the MRtrix gradient frame.
    Returns same-device float64 signal design ``[N,47]`` and amplitude
    constraint design ``[302,47]``; no image affine is used here.
    """
    device = grad_mrtrix.device
    grad = grad_mrtrix.to(torch.float64)
    shell_bvals = shell_bvals.to(device=device, dtype=torch.float64)
    wm = wm_response.to(device=device, dtype=torch.float64)
    gm = gm_response.to(device=device, dtype=torch.float64).reshape(-1)
    csf = csf_response.to(device=device, dtype=torch.float64).reshape(-1)
    if grad.ndim != 2 or grad.shape[1] != 4:
        raise ValueError("grad_mrtrix must have shape [N,4]")
    if wm.ndim != 2 or wm.shape[0] != shell_bvals.numel() or wm.shape[1] < 5:
        raise ValueError("WM response needs at least five zonal columns per shell")
    if gm.shape != shell_bvals.shape or csf.shape != shell_bvals.shape:
        raise ValueError("GM/CSF responses must have one value per shell")
    if not bool(torch.isfinite(grad).all() and torch.isfinite(wm).all()
                and torch.isfinite(gm).all() and torch.isfinite(csf).all()):
        raise ValueError("gradients and responses must be finite")
    shell_distance = torch.abs(grad[:, 3, None] - shell_bvals[None, :])
    shell = shell_distance.argmin(dim=1)
    if bool((shell_distance.gather(1, shell[:, None]) > 100).any()):
        raise ValueError("gradient b-values do not match response shells")
    directions = grad[:, :3].clone()
    lengths = torch.linalg.vector_norm(directions, dim=1)
    directions[lengths < 1e-8] = torch.tensor(
        [0., 0., 1.], device=device, dtype=torch.float64,
    )
    y = real_sh(directions, 8)
    degrees = torch.tensor(
        [l // 2 for l in range(0, 9, 2) for _ in range(2 * l + 1)],
        device=device,
    )
    poles = torch.tensor(
        [math.sqrt((2 * l + 1) / (4 * math.pi)) for l in range(0, 9, 2)],
        device=device, dtype=torch.float64,
    )
    wm_design = y * (wm[shell][:, degrees] / poles[degrees])
    design = torch.cat((wm_design, gm[shell, None], csf[shell, None]), dim=1)
    angles = torch.tensor(
        [float(v) for v in _MRTRIX300_ANGLES.split()],
        device=device, dtype=torch.float64,
    ).reshape(300, 2)
    azimuth, elevation = angles.unbind(dim=1)
    directions = torch.stack((
        torch.sin(elevation) * torch.cos(azimuth),
        torch.sin(elevation) * torch.sin(azimuth),
        torch.cos(elevation),
    ), dim=1)
    constraints = torch.zeros((302, 47), device=device, dtype=torch.float64)
    constraints[:300, :45] = real_sh(directions, 8)
    constraints[300, 45] = poles[0]
    constraints[301, 46] = poles[0]
    return design, constraints


def _mrtrix_icls_design_matrices(
    design: torch.Tensor, constraints: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """每次设计矩阵只计算一次 ICLS 的 Cholesky 与投影约束。"""
    gram = design.T @ design
    gram.diagonal().add_(1e-10 * gram.diagonal().max())
    chol = torch.linalg.cholesky(gram)
    projected = torch.linalg.solve_triangular(chol, constraints.T, upper=False).T
    projected /= torch.linalg.vector_norm(projected, dim=1, keepdim=True)
    return chol, projected


def _mrtrix_icls_batch(
    signal: torch.Tensor,
    design: torch.Tensor,
    constraints: torch.Tensor,
    *,
    prepared: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Solve MRtrix ICLS for one batch.

    ``signal`` is float64 ``[B,N]``, ``design`` is float64 ``[N,47]``, and
    ``constraints`` is float64 ``[302,47]`` on one device. Returns float64
    coefficients ``[B,47]`` in WM SH (45), GM (1), CSF (1) order.
    ``prepared`` reuses the same design-dependent matrices across batches.
    """
    n = signal.shape[0]
    n_params = design.shape[1]
    chol, projected = (
        _mrtrix_icls_design_matrices(design, constraints)
        if prepared is None else prepared
    )
    y = torch.linalg.solve_triangular(
        chol, (signal @ design).T, upper=False,
    ).T
    original_violation = y @ projected.T
    violation = original_violation.clone()
    active = torch.zeros((n, constraints.shape[0]), device=signal.device, dtype=torch.bool)
    prior_multipliers = torch.zeros_like(original_violation)
    solution = y.clone()
    running = torch.ones(n, device=signal.device, dtype=torch.bool)
    rows = torch.arange(n, device=signal.device)
    constraint_indices = torch.arange(constraints.shape[0], device=signal.device)
    for iteration in range(10 * n_params + 1):
        minimum, index = violation.min(dim=1)
        step = running & (minimum < 0)
        if not bool(step.any()):
            break
        was_active = active[rows, index]
        # Reuse ordered integer indices: boolean indexing otherwise repeats
        # nonzero (and CUDA host synchronisation) for the same selection.
        step_rows = torch.nonzero(step, as_tuple=True)[0]
        active[step_rows, index[step_rows]] = True
        changed = step & ~was_active
        pending = step.clone()
        for _ in range(constraints.shape[0] + 1):
            if not bool(pending.any()):
                break
            count = int(active[pending].sum(dim=1).max())
            selected = constraint_indices[None, :].expand(n, -1).masked_fill(
                ~active, constraints.shape[0],
            ).topk(count, dim=1, largest=False, sorted=True).values
            valid = selected < constraints.shape[0]
            selected = selected.clamp_max(constraints.shape[0] - 1)
            selected_rows = projected[selected] * valid[..., None]
            normal = selected_rows @ selected_rows.transpose(1, 2)
            diagonal = torch.where(
                valid, torch.full_like(valid, 1e-10, dtype=torch.float64),
                torch.ones_like(valid, dtype=torch.float64),
            )
            normal.diagonal(dim1=1, dim2=2).add_(diagonal)
            multipliers = torch.linalg.solve(
                normal, -original_violation.gather(1, selected) * valid,
            )
            negative = (multipliers < 0) & valid & pending[:, None]
            needs_removal = negative.any(dim=1)
            completed = pending & ~needs_removal
            completed_rows = torch.nonzero(completed, as_tuple=True)[0]
            if completed_rows.numel():
                solution[completed_rows] = (
                    y[completed_rows]
                    + (selected_rows[completed_rows].transpose(1, 2)
                       @ multipliers[completed_rows, :, None]).squeeze(-1)
                )
                updated = torch.zeros_like(prior_multipliers[completed_rows])
                updated.scatter_add_(
                    1, selected[completed_rows],
                    multipliers[completed_rows] * valid[completed_rows],
                )
                prior_multipliers[completed_rows] = updated
            removal_rows = torch.nonzero(needs_removal, as_tuple=True)[0]
            if removal_rows.numel():
                prior = prior_multipliers.gather(1, selected)
                ratio = torch.where(
                    negative,
                    prior / (prior - multipliers),
                    torch.full_like(multipliers, torch.inf),
                )
                remove = ratio.argmin(dim=1)
                constraint_to_remove = selected.gather(1, remove[:, None]).squeeze(1)
                active[removal_rows, constraint_to_remove[removal_rows]] = False
                changed |= needs_removal
            pending = needs_removal
        else:
            raise RuntimeError("MRtrix ICLS active-set update did not converge")
        running = step & changed & (iteration < 10 * n_params)
        violation = solution @ projected.T
    return torch.linalg.solve_triangular(
        chol.T, solution.T, upper=True,
    ).T


def fit_mrtrix_msmt_csd(
    signal: torch.Tensor,
    grad_mrtrix: torch.Tensor,
    shell_bvals: torch.Tensor,
    wm_response: torch.Tensor,
    gm_response: torch.Tensor,
    csf_response: torch.Tensor,
    mask: torch.Tensor,
    batch_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit raw-signal MSMT-CSD with MRtrix default lmax=8 and ICLS.

    Inputs: raw float32 DWI ``[X,Y,Z,N]`` on CPU/CUDA; MRtrix-exported
    gradient scheme ``[N,4]`` (unit vectors and b-values in s/mm²);
    shell centres ``[S]``; raw WM response ``[S,>=5]`` and GM/CSF
    responses ``[S]``; binary mask ``[X,Y,Z]`` already aligned to the DWI
    voxel grid. Gradient vectors use the same coordinate frame as the MRtrix
    exported scheme. ``batch_size`` is the maximum fitted voxel count per
    solve. Outputs are float32 WM SH ``[X,Y,Z,45]``, GM ``[X,Y,Z]`` and CSF
    ``[X,Y,Z]`` on the input signal device, in MRtrix SH order.
    Float64 is used for the 47-parameter solve; CUDA TF32 remains enabled.
    batch_size bounds peak VRAM.

    Equivalent reference command:
    dwi2fod msmt_csd dwi.mif wm.txt wm.mif gm.txt gm.mif csf.txt csf.mif
        -mask mask.nii.gz

    Dhollander response estimation and mtnormalise are separate stages.
    """
    if signal.ndim != 4 or signal.dtype != torch.float32:
        raise ValueError("signal must be float32 [X,Y,Z,N]")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    device = signal.device
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    grad = torch.as_tensor(grad_mrtrix, device=device, dtype=torch.float64)
    shell = torch.as_tensor(shell_bvals, device=device, dtype=torch.float64)
    wm = torch.as_tensor(wm_response, device=device, dtype=torch.float64)
    gm = torch.as_tensor(gm_response, device=device, dtype=torch.float64)
    csf = torch.as_tensor(csf_response, device=device, dtype=torch.float64)
    mask = torch.as_tensor(mask, device=device, dtype=torch.bool)
    if grad.shape != (signal.shape[-1], 4) or mask.shape != signal.shape[:3]:
        raise ValueError("gradient or mask shape does not match signal")
    design, constraints = _mrtrix_msmt_design(grad, shell, wm, gm, csf)
    flat_signal = signal.reshape(-1, signal.shape[-1])
    output = torch.zeros((flat_signal.shape[0], 47), device=device, dtype=torch.float32)
    voxels = torch.nonzero(mask.reshape(-1)).flatten()
    prepared = _mrtrix_icls_design_matrices(design, constraints) if voxels.numel() else None
    for block in voxels.split(batch_size):
        data = flat_signal[block].to(torch.float64)
        if not bool(torch.isfinite(data).all()):
            raise ValueError("masked DWI contains nonfinite values")
        output[block] = _mrtrix_icls_batch(
            data, design, constraints, prepared=prepared,
        ).to(torch.float32)
    return (
        output[:, :45].reshape(*signal.shape[:3], 45),
        output[:, 45].reshape(signal.shape[:3]),
        output[:, 46].reshape(signal.shape[:3]),
    )


def fit_mrtrix_two_tissue_csd(
    signal: torch.Tensor,
    grad_mrtrix: torch.Tensor,
    shell_bvals: torch.Tensor,
    wm_response: torch.Tensor,
    csf_response: torch.Tensor,
    mask: torch.Tensor,
    lmax: int,
    batch_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit the two-tissue CSD stages used by Dhollander WM voxel selection.

    Inputs: corrected float32 DWI [X,Y,Z,N] on CPU/CUDA; MRtrix-exported
    gradient table [N,4] in its gradient frame; shell centres [S] in
    s/mm²; empirical WM zonal response [S,>=lmax//2+1]; isotropic CSF
    response [S] or [S,1]; affine-aligned bool mask [X,Y,Z]; even lmax
    of 2 or 6; maximum voxels per constrained solve. Outputs are
    (WM SH float32 [X,Y,Z,C], CSF float32 [X,Y,Z]) on signal.device,
    with C=(lmax+1)*(lmax+2)//2 and zeros outside the mask. SH order,
    default 300-direction constraints and active-set solve match MRtrix;
    the solve uses float64 and CUDA TF32 is enabled.

    Equivalent Dhollander internal MRtrix commands:
    dwi2fod msmt_csd dwi.mif ewmrf.txt abs_ewm2.mif response_csf.txt
        abs_csf2.mif -mask refined_wm.mif -lmax 2,0
    dwi2fod msmt_csd dwi.mif ewmrf.txt abs_ewm6.mif response_csf.txt
        abs_csf6.mif -mask refined_sfwm.mif -lmax 6,0
    """
    if signal.ndim != 4 or signal.dtype != torch.float32:
        raise ValueError("signal must be float32 [X,Y,Z,N]")
    if lmax not in (2, 6) or batch_size < 1:
        raise ValueError("lmax must be 2 or 6 and batch_size positive")
    device = signal.device
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    grad = torch.as_tensor(grad_mrtrix, device=device, dtype=torch.float64)
    shells = torch.as_tensor(shell_bvals, device=device, dtype=torch.float64)
    wm = torch.as_tensor(wm_response, device=device, dtype=torch.float64)
    csf = torch.as_tensor(csf_response, device=device, dtype=torch.float64).reshape(-1)
    mask = torch.as_tensor(mask, device=device, dtype=torch.bool)
    n_orders = lmax // 2 + 1
    n_coeff = (lmax + 1) * (lmax + 2) // 2
    if grad.shape != (signal.shape[-1], 4) or mask.shape != signal.shape[:3]:
        raise ValueError("gradient or mask shape does not match DWI")
    if wm.ndim != 2 or wm.shape[0] != shells.numel() or wm.shape[1] < n_orders:
        raise ValueError("WM response lacks required shell and zonal columns")
    if csf.shape != shells.shape:
        raise ValueError("CSF response must have one value per shell")
    distance = torch.abs(grad[:, 3, None] - shells[None, :])
    assignment = distance.argmin(dim=1)
    if bool((distance.gather(1, assignment[:, None]) > 100).any()):
        raise ValueError("gradient b-values do not match response shells")
    directions = grad[:, :3].clone()
    directions[torch.linalg.vector_norm(directions, dim=1) < 1e-8] = torch.tensor(
        [0., 0., 1.], device=device, dtype=torch.float64,
    )
    degrees = torch.tensor(
        [degree // 2 for degree in range(0, lmax + 1, 2)
         for _ in range(2 * degree + 1)],
        device=device,
    )
    poles = torch.tensor(
        [math.sqrt((2 * degree + 1) / (4 * math.pi))
         for degree in range(0, lmax + 1, 2)],
        device=device, dtype=torch.float64,
    )
    design = torch.cat((
        real_sh(directions, lmax) * (wm[assignment][:, degrees] / poles[degrees]),
        csf[assignment, None],
    ), dim=1)
    angles = torch.tensor(
        [float(value) for value in _MRTRIX300_ANGLES.split()],
        device=device, dtype=torch.float64,
    ).reshape(300, 2)
    azimuth, elevation = angles.unbind(dim=1)
    constraint_directions = torch.stack((
        torch.sin(elevation) * torch.cos(azimuth),
        torch.sin(elevation) * torch.sin(azimuth),
        torch.cos(elevation),
    ), dim=1)
    constraints = torch.zeros((301, n_coeff + 1), device=device, dtype=torch.float64)
    constraints[:300, :n_coeff] = real_sh(constraint_directions, lmax)
    constraints[300, n_coeff] = poles[0]
    flat = signal.reshape(-1, signal.shape[-1])
    output = torch.zeros((flat.shape[0], n_coeff + 1), device=device, dtype=torch.float32)
    voxels = torch.nonzero(mask.reshape(-1)).flatten()
    prepared = _mrtrix_icls_design_matrices(design, constraints) if voxels.numel() else None
    for block in voxels.split(batch_size):
        output[block] = _mrtrix_icls_batch(
            flat[block].to(torch.float64), design, constraints, prepared=prepared,
        ).to(torch.float32)
    return (
        output[:, :n_coeff].reshape(*signal.shape[:3], n_coeff),
        output[:, n_coeff].reshape(signal.shape[:3]),
    )


def mrtrix_fod_peak_amplitude(sh: torch.Tensor, lmax: int) -> torch.Tensor:
    """Return the largest FOD peak used by Dhollander's single-fibre metric.

    Input: MRtrix-ordered float32/float64 SH [V,C] on CPU/CUDA, C=6 for
    lmax=2 or C=28 for lmax=6. Output: float64 amplitudes [V] on the same
    device. Coefficients and directions share the DWI gradient frame; no
    affine is applied. The lmax=2 quadratic maximum is analytic. For
    lmax=6, 3000 sphere samples initialize tangent-plane Newton steps.
    CUDA TF32 is enabled by default.

    Equivalent MRtrix: sh2peaks fod.mif - -num 1 -mask mask.mif |
    peaks2amp - peak_amp.mif. Extract masked [V,C] coefficients first.
    """
    if sh.ndim != 2 or lmax not in (2, 6) or sh.shape[1] != (lmax + 1) * (lmax + 2) // 2:
        raise ValueError("SH must be [V,6] for lmax=2 or [V,28] for lmax=6")
    if sh.dtype not in (torch.float32, torch.float64):
        raise ValueError("SH must be float32 or float64")
    if sh.device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    coefficients = sh.to(torch.float64)
    if lmax == 2:
        axes = torch.eye(3, device=sh.device, dtype=torch.float64)
        pairs = torch.stack((axes[0] + axes[1], axes[0] + axes[2], axes[1] + axes[2]))
        basis = real_sh(torch.cat((axes, pairs / math.sqrt(2)), dim=0), 2)
        result = []
        for block in coefficients.split(32768):
            values = block @ basis.T
            matrix = torch.diag_embed(values[:, :3])
            off = values[:, 3:] - torch.stack((
                (values[:, 0] + values[:, 1]) / 2,
                (values[:, 0] + values[:, 2]) / 2,
                (values[:, 1] + values[:, 2]) / 2,
            ), dim=1)
            matrix[:, 0, 1] = matrix[:, 1, 0] = off[:, 0]
            matrix[:, 0, 2] = matrix[:, 2, 0] = off[:, 1]
            matrix[:, 1, 2] = matrix[:, 2, 1] = off[:, 2]
            result.append(torch.linalg.eigvalsh(matrix)[:, -1])
        return torch.cat(result) if result else torch.empty(0, device=sh.device, dtype=torch.float64)

    indices = torch.arange(3000, device=sh.device, dtype=torch.float64)
    z = (indices + 0.5) / 3000
    phi = indices * (math.pi * (3 - math.sqrt(5)))
    radius = torch.sqrt(1 - z * z)
    directions = torch.stack((radius * phi.cos(), radius * phi.sin(), z), dim=1)
    basis = real_sh(directions, 6)

    def amplitude(values: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        return (values * real_sh(points, 6)).sum(dim=1)

    peaks = []
    h = 0.001
    for block in coefficients.split(4096):
        direction = directions[(block @ basis.T).argmax(dim=1)].clone()
        for _ in range(8):
            axis = torch.eye(3, device=sh.device, dtype=torch.float64)[direction.abs().argmin(dim=1)]
            u = torch.nn.functional.normalize(torch.linalg.cross(direction, axis), dim=1)
            v = torch.linalg.cross(direction, u)

            def shifted(du: float, dv: float) -> torch.Tensor:
                points = torch.nn.functional.normalize(direction + h * (du * u + dv * v), dim=1)
                return amplitude(block, points)

            centre = amplitude(block, direction)
            xp, xm = shifted(1, 0), shifted(-1, 0)
            yp, ym = shifted(0, 1), shifted(0, -1)
            gx, gy = (xp - xm) / (2 * h), (yp - ym) / (2 * h)
            hxx, hyy = (xp - 2 * centre + xm) / h**2, (yp - 2 * centre + ym) / h**2
            hxy = (shifted(1, 1) - shifted(1, -1) - shifted(-1, 1) + shifted(-1, -1)) / (4 * h**2)
            determinant = hxx * hyy - hxy.square()
            stable = (determinant > 1e-12) & (hxx < 0) & (hyy < 0)
            denominator = torch.where(stable, determinant, 1)
            du = torch.where(stable, (-hyy * gx + hxy * gy) / denominator, torch.zeros_like(gx))
            dv = torch.where(stable, (hxy * gx - hxx * gy) / denominator, torch.zeros_like(gy))
            scale = torch.clamp(0.2 / torch.sqrt(du.square() + dv.square()).clamp_min(1e-12), max=1)
            direction = torch.nn.functional.normalize(
                direction + scale[:, None] * (du[:, None] * u + dv[:, None] * v), dim=1,
            )
        peaks.append(amplitude(block, direction))
    return torch.cat(peaks) if peaks else torch.empty(0, device=sh.device, dtype=torch.float64)
