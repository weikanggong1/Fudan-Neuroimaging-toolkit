"""Native-free geometry translation of FreeSurfer 8.2 ``mris_sphere -q``.

The connected inflation and four-epoch optimizer match the frozen bilateral
``fs_sub01`` ordered geometry. This NumPy CPU stage is independent of a
FreeSurfer runtime and is used for ``qsphere.nofix`` by the recon-all runner.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from .inflate_python import _accumulate_corner_normals, average_gradient, matrix
from .sphere_python import inflate_before_quick_sphere
from .smooth_surface_python import ordered_neighbors
from .sphere_python import initial_scale, project_radially


_SPHERE_AREA = np.float32(4.0 * np.pi * 100.0 * 100.0)


def _face_geometry_from_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                                           np.ndarray, np.ndarray]:
    """复用一次面坐标 gather，返回两边向量、叉积和绝对面积。

    points 是 float32(F,3,3) surface RAS/mm；返回向量为 (F,3)，面积为
    (F,) mm²，均保留既有 float32 运算顺序。调用者负责合法形状/有限输入；
    无写出，非法数组会抛 NumPy 异常。这是 mris_sphere -q 的内部步骤。
    """
    edge_a = points[:, 1] - points[:, 0]
    edge_b = points[:, 2] - points[:, 0]
    cross = np.cross(edge_a, edge_b)
    squared = cross[:, 0] * cross[:, 0]
    squared += cross[:, 1] * cross[:, 1]
    squared += cross[:, 2] * cross[:, 2]
    length = np.sqrt(squared)
    return edge_a, edge_b, cross, length * np.float32(0.5)


def _face_geometry(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从 (N,3) mm 坐标和 (F,3) 有序面返回两边向量及 mm² 面积。

    兼容既有返回结构；float32 运算及异常行为由面坐标内核提供。
    """
    edge_a, edge_b, _, area = _face_geometry_from_points(
        np.asarray(vertices, np.float32)[faces])
    return edge_a, edge_b, area


def reference_face_areas(input_vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Face areas stored by the native inflation setup after its initial scale."""
    return _face_geometry(initial_scale(input_vertices), faces)[2]


def nonlinear_area_gradient(
    vertices: np.ndarray,
    faces: np.ndarray,
    original_face_area: np.ndarray,
    original_total_area: float,
    k: float,
) -> tuple[np.ndarray, float, int]:
    """计算 mris_sphere -q 的非线性面积梯度、能量及翻折面数。

    vertices 为 float32(N,3) surface RAS/mm，faces 为有序 (F,3) 顶点索引。
    original_face_area 为球面初始化时固定的 (F,) mm² 面积，
    original_total_area 为相应总面积 mm²；k 为原四轮 10/40/160/640。
    返回 float32(N,3) 梯度、float 能量和 int 负面积面数，不改输入。
    复用 inflate 的顺序角点累加内核，保持面序/角点序 float32 加法。
    非法形状/索引会抛异常；这里没有独立官方 CLI。
    """
    xyz = np.asarray(vertices, np.float32)
    points = xyz[faces]
    a, b, cross, absolute_area = _face_geometry_from_points(points)
    length = absolute_area * np.float32(2)
    center = points[:, 0] + points[:, 1] + points[:, 2]
    outward = center[:, 0] * cross[:, 0]
    outward += center[:, 1] * cross[:, 1]
    outward += center[:, 2] * cross[:, 2]
    sign = np.where(outward < 0, -1, 1).astype(np.float32)
    signed_area = absolute_area * sign
    reciprocal = np.divide(np.float32(1), length, out=np.ones_like(length),
                           where=length >= np.finfo(np.float32).eps)
    normal = cross * sign[:, None] * reciprocal[:, None]
    # Both surface areas are float32 in FreeSurfer; their ratio is then used
    # as a double in the nonlinear term.
    area_scale = np.float32(np.float32(original_total_area) / _SPHERE_AREA)
    area = float(area_scale) * signed_area.astype(np.float64)
    delta = ((area - original_face_area.astype(np.float64))
             / (1.0 + np.exp(np.clip(k * area, -400, 400))))
    a_cross_n = np.cross(a, normal)
    b_cross_n = np.cross(b, normal)
    terms = np.stack(((b_cross_n - a_cross_n) * delta[:, None],
                      b_cross_n * (-delta[:, None]),
                      a_cross_n * delta[:, None]), axis=1).astype(np.float32)
    gradient = _accumulate_corner_normals(
        faces.reshape(-1).astype(np.int32, copy=False),
        terms.reshape(-1, 3), len(xyz))
    sse = float(np.sum(np.logaddexp(0, -k * area) / k))
    return gradient, sse, int(np.count_nonzero(signed_area < 0))


def average_gradients(gradient: np.ndarray, faces: np.ndarray, iterations: int,
                      neighbor_table: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """Use the native face-order one-ring and include each vertex itself."""
    current = np.asarray(gradient, np.float32).copy()
    if iterations == 0:
        return current
    indices, degree = (neighbor_table if neighbor_table is not None
                       else matrix(ordered_neighbors(faces, len(current))))
    return average_gradient(current, indices, degree, iterations)


def projected_step(vertices: np.ndarray, gradient: np.ndarray, dt: float) -> np.ndarray:
    """Apply the native float32 coordinate update and sphere reprojection."""
    moved = (np.asarray(vertices, np.float32).astype(np.float64)
             + float(dt) * np.asarray(gradient, np.float32).astype(np.float64)).astype(np.float32)
    return project_radially(moved, already_sphere=True)


def nonlinear_area_sse(vertices: np.ndarray, faces: np.ndarray,
                       original_total_area: float, k: float) -> float:
    """返回 quick-sphere 试探坐标的负面积能量 float；不改输入。

    vertices 为 (N,3) surface RAS/mm，faces 为 (F,3) 有序面，
    original_total_area 为初始化 mm² 总面积，k 为当前固定轮次权重。
    面 gather/叉积只算一次，float32 面面积和 float64 能量公式不变；
    非法形状/索引抛异常。属于 mris_sphere -q 的内部线搜索，无独立 CLI。
    """
    points = np.asarray(vertices, np.float32)[faces]
    _, _, cross, absolute_area = _face_geometry_from_points(points)
    center = points[:, 0] + points[:, 1] + points[:, 2]
    outward = center[:, 0] * cross[:, 0]
    outward += center[:, 1] * cross[:, 1]
    outward += center[:, 2] * cross[:, 2]
    area = absolute_area * np.where(outward < 0, -1, 1).astype(np.float32)
    area_scale = np.float32(original_total_area / _SPHERE_AREA)
    ratio = area_scale * area
    return float(np.sum(np.logaddexp(0, -k * ratio.astype(np.float64)) / k))


def _matrix_multiply32(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = np.empty((left.shape[0], right.shape[1]), np.float32)
    for row in range(left.shape[0]):
        for col in range(right.shape[1]):
            value = np.float32(0)
            for inner in range(left.shape[1]):
                value = np.float32(value + np.float32(left[row, inner] * right[inner, col]))
            result[row, col] = value
    return result


def _quadratic_minimum32(steps: tuple[float, float, float],
                         energies: tuple[float, float, float]) -> float | None:
    """Replay FreeSurfer's MATRIX_REAL normal equations and VNL 3x3 inverse."""
    x = np.asarray([[dt * dt, 2 * dt, 1] for dt in steps], np.float32)
    y = np.asarray(energies, np.float32)[:, None]
    xtx = _matrix_multiply32(x.T, x)
    a, b, c = xtx
    det = (a[0] * b[1] * c[2] - a[0] * c[1] * b[2] -
           b[0] * a[1] * c[2] + b[0] * c[1] * a[2] +
           c[0] * a[1] * b[2] - c[0] * b[1] * a[2])
    if det == 0 or not np.isfinite(det):
        return None
    reciprocal = np.float32(1) / det
    inverse = np.empty((3, 3), np.float32)
    for row in range(3):
        for col in range(3):
            minor = xtx[[i for i in range(3) if i != col]][
                :, [i for i in range(3) if i != row]]
            cofactor = np.float32(minor[0, 0] * minor[1, 1] -
                                  minor[0, 1] * minor[1, 0])
            inverse[row, col] = np.float32(-cofactor if (row + col) & 1
                                          else cofactor) * reciprocal
    coefficients = _matrix_multiply32(
        inverse, _matrix_multiply32(x.T, y)).ravel()
    if not np.isfinite(coefficients[0]) or abs(float(coefficients[0])) < np.finfo(np.float32).eps:
        return None
    return float(np.float32(-float(coefficients[1]) / float(coefficients[0])))


def _line_minimize(vertices: np.ndarray, gradient: np.ndarray, faces: np.ndarray,
                   original_total_area: float, k: float) -> float:
    """保持原候选顺序选择线搜索步长，返回 dt float，无文件写出。

    vertices/gradient 为 (N,3) float32、faces 为 (F,3) 有序面；坐标 mm，
    original_total_area 单位 mm²，k 为当前轮权重。只在一次调用内缓存两份
    float64 试探基础数组，每个候选仍按相同表达式更新、量化和径向投影；
    试探范围/二次拟合/接受规则不变。零平均梯度返回 0，其他非法输入抛异常。
    对应 mris_sphere -q 内部线搜索，没有独立 CLI。
    """
    squared = gradient[:, 0] * gradient[:, 0]
    squared += gradient[:, 1] * gradient[:, 1]
    squared += gradient[:, 2] * gradient[:, 2]
    mean = sum(math.sqrt(float(value)) for value in squared) / len(squared)
    if mean == 0:
        return 0.0
    min_dt, max_dt = .001 / mean, 12.2 / mean
    starting = nonlinear_area_sse(vertices, faces, original_total_area, k)
    xyz64 = np.asarray(vertices, np.float32).astype(np.float64)
    gradient64 = np.asarray(gradient, np.float32).astype(np.float64)

    def energy(dt: float) -> float:
        moved = (xyz64 + float(dt) * gradient64).astype(np.float32)
        projected = project_radially(moved, already_sphere=True)
        return nonlinear_area_sse(projected, faces,
                                  original_total_area, k)

    best_dt, best_sse = 0.0, starting
    dt = min_dt
    while dt < max_dt:
        sse = energy(dt)
        if sse <= best_sse:
            best_dt, best_sse = dt, sse
        dt *= 10.0
    if best_dt == 0:
        best_dt = min_dt / 10.0
        best_sse = energy(best_dt)
    dt0, dt2 = best_dt * .5, best_dt * 1.5
    candidates = [(dt0, energy(dt0)), (best_dt, best_sse),
                  (dt2, energy(dt2)), (0.0, starting)]
    predicted = _quadratic_minimum32((dt0, best_dt, dt2),
                                     (candidates[0][1], best_sse, candidates[2][1]))
    if predicted is not None:
        if best_dt / 10 < predicted < best_dt * 10:
            candidates.append((predicted, energy(predicted)))
    return min(candidates, key=lambda item: item[1])[0]


def quick_sphere_from_projected(vertices: np.ndarray, faces: np.ndarray,
                                original_face_area: np.ndarray,
                                original_total_area: float, niterations: int = 25,
                                initial_momentum: np.ndarray | None = None
                                ) -> tuple[np.ndarray, list[tuple[int, int, str, float]]]:
    """Run the fixed four-epoch optimizer from an already projected sphere.

    ``niterations=1`` reproduces the isolated native ``-N 1 -in 0`` probe.
    The default is the actual recon-all call. The caller must provide the
    native-equivalent projected coordinates and inherited inflation momentum.
    """
    xyz = np.asarray(vertices, np.float32).copy()
    table = matrix(ordered_neighbors(faces, len(xyz)))
    momentum = (np.zeros_like(xyz) if initial_momentum is None
                else np.asarray(initial_momentum, np.float32).copy())
    if momentum.shape != xyz.shape:
        raise ValueError("initial momentum must have one xyz vector per vertex")
    trace: list[tuple[int, int, str, float]] = []
    first_phase = True
    base_tol = .1 * 1024 / np.sqrt(129.0)
    for k in (10, 40, 160, 640):
        for averages in (128, 32, 8, 2, 0):
            tol = base_tol * np.sqrt((averages + 1.0) / 1024.0)
            for mode in (("line", "momentum", "line") if averages else ("line",)):
                if first_phase:
                    first_phase = False
                else:
                    xyz = project_radially(xyz, already_sphere=True)
                previous_sse = nonlinear_area_sse(xyz, faces, original_total_area, k)
                for _ in range(10 if mode == "momentum" else niterations):
                    gradient, _, _ = nonlinear_area_gradient(
                        xyz, faces, original_face_area, original_total_area, k)
                    gradient = average_gradients(gradient, faces, averages, table)
                    if mode == "momentum":
                        dt = float(np.float32(.05)) * np.sqrt(averages + 1.0)
                        momentum = (dt * gradient.astype(np.float64)
                                    + (np.float32(.9) * momentum).astype(np.float64)).astype(np.float32)
                        squared = momentum[:, 0] * momentum[:, 0]
                        squared += momentum[:, 1] * momentum[:, 1]
                        squared += momentum[:, 2] * momentum[:, 2]
                        magnitude = np.sqrt(squared.astype(np.float64))
                        large = magnitude > 1
                        momentum[large] = (momentum[large].astype(np.float64)
                                           / magnitude[large, None]).astype(np.float32)
                        xyz = project_radially((xyz + momentum).astype(np.float32),
                                               already_sphere=True)
                        xyz = project_radially(xyz, already_sphere=True)
                    else:
                        dt = _line_minimize(xyz, gradient, faces, original_total_area, k)
                        xyz = projected_step(xyz, gradient, dt)
                    current_sse = nonlinear_area_sse(xyz, faces, original_total_area, k)
                    trace.append((k, averages, mode, dt))
                    if current_sse == 0 or dt == 0 or (100 * (previous_sse - current_sse)
                                                        / current_sse) < tol:
                        break
                    previous_sse = current_sse
    return project_radially(xyz, already_sphere=True), trace


def quick_sphere_from_inflated(vertices: np.ndarray, faces: np.ndarray,
                               niterations: int = 25
                               ) -> tuple[np.ndarray, list[tuple[int, int, str, float]]]:
    """Run the fixed native-free inflation and quick optimizer in sequence."""
    faces = np.asarray(faces, np.int32)
    original_area = reference_face_areas(vertices, faces)
    projected, momentum = inflate_before_quick_sphere(
        vertices, faces, return_momentum=True)
    # MRISintegrate projects once more before its first gradient evaluation.
    projected = project_radially(projected, already_sphere=True)
    return quick_sphere_from_projected(
        projected, faces, original_area,
        float(np.sum(original_area, dtype=np.float64)),
        niterations=niterations, initial_momentum=momentum)


def write_quick_sphere(input_path: str | Path, output_path: str | Path) -> None:
    """Write the fixed native-free q-sphere as a FreeSurfer triangle surface."""
    raw = Path(input_path).read_bytes()
    if raw[:3] != b"\xff\xff\xfe":
        raise ValueError("expected FreeSurfer triangular surface")
    start = raw.index(b"\n\n", 3) + 2
    nvertices = int.from_bytes(raw[start:start + 4], "big")
    nfaces = int.from_bytes(raw[start + 4:start + 8], "big")
    xyz_start = start + 8
    xyz_end = xyz_start + 12 * nvertices
    faces_end = xyz_end + 12 * nfaces
    if len(raw) < faces_end:
        raise ValueError("truncated triangular surface")
    vertices = np.frombuffer(raw[xyz_start:xyz_end], dtype=">f4").reshape(-1, 3)
    faces = np.frombuffer(raw[xyz_end:faces_end], dtype=">i4").reshape(-1, 3)
    output, _ = quick_sphere_from_inflated(vertices, faces)
    with Path(output_path).open("wb") as stream:
        stream.write(b"\xff\xff\xfecreated by fnit\n\n")
        stream.write(raw[start:xyz_start])
        stream.write(np.asarray(output, dtype=">f4").tobytes())
        stream.write(raw[xyz_end:])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_inflated", type=Path)
    parser.add_argument("output_qsphere", type=Path)
    args = parser.parse_args(argv)
    write_quick_sphere(args.input_inflated, args.output_qsphere)


if __name__ == "__main__":
    main()
