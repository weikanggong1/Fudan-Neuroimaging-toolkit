"""First spherical-registration SSE and quadratic line search (FreeSurfer 8.2)."""

from __future__ import annotations

import math

import torch

from .mris_register_nonlinear import (
    _float32, apply_spherical_gradient, face_area_normals, sample_correlation_atlas,
    sphere_arc_distances,
)


@torch.no_grad()
def first_registration_sse(positions: torch.Tensor, faces: torch.Tensor,
                           neighbors: torch.Tensor, degrees: torch.Tensor,
                           original_distances: torch.Tensor,
                           original_face_areas: torch.Tensor,
                           curvature: torch.Tensor,
                           target_mean: torch.Tensor,
                           target_variance: torch.Tensor,
                           original_area: float,
                           total_area: float,
                           *, return_terms: bool = False,
                           current_distances: torch.Tensor | None = None
                           ) -> float | dict[str, float]:
    """计算面积、非线性面积、弧距和曲率相关目标，保持原累计顺序。

    positions 为 (N,3) surface RAS/mm；faces 为 (F,3) 有序整数面，
    neighbors/degrees 为 (N,K)/(N,) 有序邻接。original_distances 为
    同形弧距目标/mm，original_face_areas 为 (F,) 原面积/mm²，
    curvature 为 (N,) 标量，target_mean/target_variance 为同尺寸二维
    图谱。original_area/total_area 是原始及标准球面总面积/mm²。
    return_terms 默认 False 返回四项总和；True 返回四个具名 Python
    float，定义不变。current_distances 默认 None 现场计算弧距；调用者
    可传同一 positions、同一邻接刚生成的 float32(N,K) 张量，避免
    smoothwm 的距离项与弹簧项重复计算。不允许跨试步或坐标更新复用。
    形状、类型或设备不匹配时抛 ValueError；未改变精度、权重或停止规则。
    属于 mris_register 内部目标函数，没有独立官方 CLI。
    """
    current_area, _ = face_area_normals(positions, faces, signed_sphere=True)
    area_scale = _float32(_float32(original_area) / _float32(total_area))
    difference = area_scale * current_area.double() - original_face_areas.double()
    area_sse = 0.2 * float((difference * difference).sum())
    ratio = (area_scale * current_area.double()).clamp(-40.0, 40.0)
    nonlinear_area_sse = float((torch.nn.functional.softplus(10.0 * ratio) / 10.0
                                - ratio).sum())
    if current_distances is None:
        current_distances = sphere_arc_distances(positions, neighbors, degrees)
    elif (current_distances.shape != neighbors.shape
          or current_distances.dtype != torch.float32
          or current_distances.device != positions.device):
        raise ValueError("current_distances must be float32 (N,K) on positions.device")
    delta = (math.sqrt(area_scale) * current_distances.double()
             - original_distances.double())
    active = torch.arange(neighbors.shape[1], device=positions.device)[None, :] < degrees[:, None]
    distance_sse = 5.0 * float(((delta * delta) * active).sum())
    target = sample_correlation_atlas(target_mean, positions)
    variance = sample_correlation_atlas(target_variance, positions)
    standard_deviation = variance.double().sqrt()
    standard_deviation = torch.where(standard_deviation.abs() < torch.finfo(torch.float32).eps,
                                     4.0, standard_deviation)
    residual = (curvature.double() - target.double()) / standard_deviation
    correlation_sse = float((residual * residual).sum())
    if return_terms:
        return {"sse_area": area_sse, "sse_nl_area": nonlinear_area_sse,
                "sse_dist": distance_sse, "sse_corr": correlation_sse}
    return area_sse + nonlinear_area_sse + distance_sse + correlation_sse


def _native_quadratic_candidate(samples: list[tuple[float, float]]) -> float | None:
    """FreeSurfer MatrixMultiply plus VNL fixed 3x3 inverse in float32 order."""
    def f(value: float) -> float:
        return _float32(value)

    def multiply(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
        result = [[0.0] * len(right[0]) for _ in left]
        for row in range(len(left)):
            for col in range(len(right[0])):
                value = 0.0
                for index in range(len(right)):
                    value = f(value + f(left[row][index] * right[index][col]))
                result[row][col] = value
        return result

    design = [[f(dt * dt), f(2.0 * dt), 1.0] for dt, _ in samples]
    observed = [[f(sse)] for _, sse in samples]
    transposed = [list(row) for row in zip(*design)]
    normal = multiply(transposed, design)
    right = multiply(transposed, observed)
    (a, b, c), (d, e, g), (h, i, j) = normal

    # Preserve VXL's term order; changing the order changes this ill-conditioned fit.
    determinant = 0.0
    for x, y, z, sign in ((a, e, j, 1), (a, i, g, -1), (d, b, j, -1),
                          (d, i, c, 1), (h, b, g, 1), (h, e, c, -1)):
        determinant = f(determinant + sign * f(f(x * y) * z))
    if determinant == 0:
        return None
    reciprocal = f(1.0 / determinant)

    def subtract(x: float, y: float, z: float, w: float) -> float:
        return f(f(x * y) - f(z * w))

    cofactors = [
        [subtract(e, j, g, i), subtract(i, c, j, b), subtract(b, g, c, e)],
        [subtract(g, h, d, j), subtract(a, j, c, h), subtract(d, c, g, a)],
        [subtract(d, i, e, h), subtract(b, h, a, i), subtract(a, e, b, d)],
    ]
    inverse = [[f(value * reciprocal) for value in row] for row in cofactors]
    coefficients = multiply(inverse, right)
    quadratic, half_linear = coefficients[0][0], coefficients[1][0]
    if not math.isfinite(quadratic) or quadratic == 0:
        return None
    return f(-half_linear / quadratic)


def first_registration_line_search(positions: torch.Tensor, gradient: torch.Tensor,
                                   objective) -> tuple[float, list[tuple[float, float]]]:
    """Select the first step using native decade probes and a quadratic candidate.

    ``objective`` receives a projected trial surface and returns its scalar SSE.
    The returned samples include the initial SSE and all evaluated candidates.
    """
    squared = (gradient[:, 0] * gradient[:, 0] + gradient[:, 1] * gradient[:, 1]
               + gradient[:, 2] * gradient[:, 2]).double()
    # Native accumulation is an ordered double sum of double square roots.
    mean_delta = sum(math.sqrt(value) for value in squared.cpu().tolist()) / len(gradient)
    if mean_delta == 0:
        return 0.0, [(0.0, objective(positions))]
    min_dt = 0.001 / mean_delta
    max_dt = float(torch.tensor(12.2, dtype=torch.float32)) / mean_delta
    samples = [(0.0, objective(positions))]

    def evaluate(dt: float) -> float:
        value = objective(apply_spherical_gradient(positions, gradient, dt))
        samples.append((dt, value))
        return value

    best_sse, best_dt = samples[0][1], 0.0
    dt = min_dt
    while dt < max_dt:
        sse = evaluate(dt)
        if sse <= best_sse:
            best_sse, best_dt = sse, dt
        dt *= 10.0
    if best_dt == 0.0:
        best_dt = min_dt / 10.0
        best_sse = evaluate(best_dt)
    dt0, dt2 = best_dt - best_dt / 2.0, best_dt + best_dt / 2.0
    sse0, sse2 = evaluate(dt0), evaluate(dt2)
    choices = [(dt0, sse0), (best_dt, best_sse), (dt2, sse2), samples[0]]
    candidate = _native_quadratic_candidate(choices[:3])
    if candidate is not None and best_dt / 10.0 < candidate < 10.0 * best_dt:
        choices.append((candidate, evaluate(candidate)))
    return min(choices, key=lambda item: item[1])[0], samples
