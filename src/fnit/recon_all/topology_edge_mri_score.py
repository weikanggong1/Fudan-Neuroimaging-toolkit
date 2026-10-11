"""固定拓扑源码的 MRI 边评分；复用已有边评分诊断和几何函数。

只覆盖候选 EDGE.len 的全部 22 次采样，非完整 patch fitness 或 GA。
原坐标的两轮平均、未标顶点的两轮中值与原法向显式分开保存。
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .place_surface_normals import initial_vertex_normals
from .place_surface_repulsion import original_vertex_normals
from .smooth_surface_python import average_positions, ordered_neighbors
from .mri_em_register import _vnl_affine_inverse
from .place_surface_border import _sample as _sample_voxel


def source_topology_ras_to_voxel(*, volume_header) -> np.ndarray:
    """MRI MGH头→拓扑实际 tkregister RAS 到体素 float32(4,4)矩阵。

    volume_header为nibabel MGHHeader，必须包含三维尺寸及正有限体素
    大小；函数复用其get_vox2ras_tkr及FNIT有序VNL仿射逆。输入空间
    是该MRI的tkregister RAS/mm，输出为连续体素索引，无量纲。
    固定源码MATRIX_ALLOCATION=1调用GetSurfaceRASToVoxelMatrix，
    不读取表面footer或桥接scanner RAS。没有设备/容差/精度参数，
    非法头/尺寸/间距抛ValueError，不改原头、不调用原程序。
    """
    try:
        dimensions = np.asarray(volume_header["dims"][:3])
        spacing = np.asarray(volume_header["delta"], np.float32)
        tkreg = np.asarray(volume_header.get_vox2ras_tkr(), np.float32)
    except (AttributeError, KeyError, TypeError) as error:
        raise ValueError("nibabel MGHHeader with tkregister geometry required") from error
    if dimensions.shape != (3,) or np.any(dimensions <= 0) or spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("positive dimensions and finite positive voxel spacing required")
    if tkreg.shape != (4, 4) or not np.isfinite(tkreg).all():
        raise ValueError("finite tkregister vox-to-RAS (4,4) matrix required")
    return np.asarray(_vnl_affine_inverse(tkreg), np.float32)


@njit(cache=True, fastmath=False)
def _sample(volume, matrix, r, a, s):
    # mriSurfaceRASToVoxel默认new-way：输入先F32，每项系数先F64，
    # 按行F64乘加，结果才F32。不可借placement逐项F32的另一分支。
    rr, aa, ss = np.float64(np.float32(r)), np.float64(np.float32(a)), np.float64(np.float32(s))
    x = np.float64(np.float32(np.float64(matrix[0, 0]) * rr + np.float64(matrix[0, 1]) * aa + np.float64(matrix[0, 2]) * ss + np.float64(matrix[0, 3])))
    y = np.float64(np.float32(np.float64(matrix[1, 0]) * rr + np.float64(matrix[1, 1]) * aa + np.float64(matrix[1, 2]) * ss + np.float64(matrix[1, 3])))
    z = np.float64(np.float32(np.float64(matrix[2, 0]) * rr + np.float64(matrix[2, 1]) * aa + np.float64(matrix[2, 2]) * ss + np.float64(matrix[2, 3])))
    return _sample_voxel(volume, x, y, z)


@njit(cache=True, fastmath=False)
def _border_values(xyz, normals, ripped, volume, matrix):
    # 从已存在的 topology_edge_score_probe 迁移；被标顶点原 val 保持零。
    white = np.zeros(len(xyz), np.float32)
    gray = np.zeros(len(xyz), np.float32)
    for vertex in range(len(xyz)):
        if ripped[vertex]:
            continue
        for side in range(2):
            sign = -1.0 if side == 0 else 1.0
            value = np.float32(_sample(volume, matrix,
                np.float64(xyz[vertex, 0]) + sign * .5 * np.float64(normals[vertex, 0]),
                np.float64(xyz[vertex, 1]) + sign * .5 * np.float64(normals[vertex, 1]),
                np.float64(xyz[vertex, 2]) + sign * .5 * np.float64(normals[vertex, 2])))
            if side == 0:
                white[vertex] = value
            else:
                gray[vertex] = value
    return white, gray


@njit(cache=True, fastmath=False)
def _median_twice(values, ripped, neighbors, degree):
    # 保留诊断实现的自身+未rip邻接、中间 float32 舍入及两轮状态。
    current = values.copy()
    for _ in range(2):
        updated = current.copy()
        for vertex in range(len(current)):
            if ripped[vertex]:
                continue
            buf = np.empty(degree[vertex] + 1, np.float32)
            buf[0] = current[vertex]
            count = 1
            for slot in range(degree[vertex]):
                other = neighbors[vertex, slot]
                if not ripped[other]:
                    buf[count] = current[other]
                    count += 1
            ordered = np.sort(buf[:count])
            if count % 2:
                updated[vertex] = ordered[count // 2]
            else:
                updated[vertex] = np.float32(
                    np.float32(ordered[count // 2] + ordered[count // 2 - 1]) / np.float32(2))
        current = updated
    return current


@njit(cache=True, fastmath=False)
def _score_edges(original, normals, edges, white, gray, original_edge, volume, matrix):
    scores = np.empty(len(edges), np.float32)
    for index in range(len(edges)):
        a, b = edges[index]
        direction = np.empty(3, np.float32)
        for axis in range(3):
            direction[axis] = np.float32(
                np.float32(normals[a, axis] + normals[b, axis]) / np.float32(2))
        squared = np.float32(np.float32(direction[0] * direction[0] +
            direction[1] * direction[1]) + direction[2] * direction[2])
        # sqrt(float) 先返回 float，再赋给源码 double total；不能 sqrt(double)。
        length = np.float64(np.sqrt(squared))
        # 固定源码 FZERO 分支，而非添加 epsilon 或改变正常法向。
        if abs(length) < 1.1920928955078125e-7:  # FLT_EPSILON, macros.h:FZERO
            length = 1.0
        for axis in range(3):
            direction[axis] = np.float32(np.float64(direction[axis]) / length)
        wval = float(np.float32(np.float32(white[a] + white[b]) / np.float32(2)))
        gval = float(np.float32(np.float32(gray[a] + gray[b]) / np.float32(2)))
        delta = np.empty(3, np.float64)
        for axis in range(3):
            delta[axis] = float(np.float32(original[b, axis] - original[a, axis]))
        total = 0.0
        fraction = .1
        for slot in range(11):
            point = np.empty(3, np.float64)
            for axis in range(3):
                if slot == 0:
                    point[axis] = float(original[a, axis])
                elif slot == 1:
                    point[axis] = float(original[b, axis])
                else:
                    point[axis] = float(original[a, axis]) + fraction * delta[axis]
            if slot >= 2:
                fraction += .1
            gray_value = _sample(volume, matrix,
                point[0] + .5 * float(direction[0]),
                point[1] + .5 * float(direction[1]),
                point[2] + .5 * float(direction[2]))
            white_value = _sample(volume, matrix,
                point[0] - .5 * float(direction[0]),
                point[1] - .5 * float(direction[1]),
                point[2] - .5 * float(direction[2]))
            total += abs(gray_value - gval)
            total += abs(white_value - wval)
        scores[index] = np.float32(total / 22.0)
        if not original_edge[index]:
            scores[index] = np.float32(scores[index] + np.float32(100))
    return scores


def _validated_volume(volume, ras_to_voxel):
    image = np.asarray(volume)
    matrix = np.asarray(ras_to_voxel)
    if image.ndim != 3 or image.dtype != np.uint8 or not all(image.shape):
        raise ValueError("volume must be nonempty 3D uint8 MRI")
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("finite surface-RAS to voxel (4,4) matrix required")
    return np.ascontiguousarray(image), np.ascontiguousarray(matrix, np.float32)


def prepare_topology_edge_inputs(*, vertices: np.ndarray, faces: np.ndarray,
                                defect_vertices: np.ndarray, volume: np.ndarray,
                                ras_to_voxel: np.ndarray) -> dict:
    """原 nofix 网格/MRI→灰白强度、两轮平均坐标及原法向，不进行边选择。

    vertices(N,3)有限坐标为surface RAS/mm，faces(F,3)为有序整数面；
    defect_vertices(N,)必须是布尔标记，所有已发现缺陷均标True，不能
    仅标准备修复的一个patch。volume为3D uint8归一化MRI，ras_to_voxel
    为surface RAS到该MRI体素的4×4矩阵。原法向、±0.5mm强度、两次
    中值及两次含自身的坐标平均顺序固定，没有精度/设备/迭代开关。
    返回字典包含float32 original_vertices/original_normals(N,3)、
    white_values/gray_values(N,)、ripped(N,)及有序neighbors列表；所有
    顶点索引对应原网格。原坐标未被改写；非法输入抛ValueError。
    这是mris_fix_topology内部准备，不读取参考结果、不调用原程序。
    """
    xyz = np.asarray(vertices, np.float32)
    triangles = np.asarray(faces)
    ripped = np.asarray(defect_vertices)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("finite (N,3) vertices required")
    if ripped.shape != (len(xyz),) or ripped.dtype != np.bool_:
        raise ValueError("defect_vertices must be bool (N,)")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or not np.issubdtype(triangles.dtype, np.integer):
        raise ValueError("integer faces (F,3) required")
    if triangles.size and (triangles.min() < 0 or triangles.max() >= len(xyz)):
        raise ValueError("face index outside original vertex range")
    image, matrix = _validated_volume(volume, ras_to_voxel)
    triangles = np.ascontiguousarray(triangles, np.int32)
    neighbors = ordered_neighbors(triangles, len(xyz))
    normals = initial_vertex_normals(vertices=xyz, triangles=triangles)
    white, gray = _border_values(np.ascontiguousarray(xyz), normals, ripped, image, matrix)
    degree = np.asarray([len(row) for row in neighbors], np.int32)
    rings = np.zeros((len(xyz), int(degree.max(initial=0))), np.int32)
    for vertex, row in enumerate(neighbors):
        rings[vertex, :len(row)] = row
    white = _median_twice(white, ripped, rings, degree)
    gray = _median_twice(gray, ripped, rings, degree)
    original = average_positions(vertices=xyz, faces=triangles, iterations=2, device="cpu")
    return {"original_vertices": original,
            "original_normals": original_vertex_normals(vertices=original, triangles=triangles),
            "white_values": white, "gray_values": gray,
            "ripped": ripped.copy(), "neighbors": neighbors}


def source_mri_edge_scores(*, original_vertices: np.ndarray, original_normals: np.ndarray,
                           edge_vertices: np.ndarray, white_values: np.ndarray,
                           gray_values: np.ndarray, original_edge_present: np.ndarray,
                           volume: np.ndarray, ras_to_voxel: np.ndarray) -> np.ndarray:
    """同序完整候选边→float32(E,)原 MRI 评分；不排序、删边或求解GA。

    original_vertices/original_normals(N,3)为已两轮平均坐标与其原法向；
    坐标surface RAS/mm，法向无量纲。edge_vertices(E,2)为同网格整数
    端点，white_values/gray_values(N,)为原网格±0.5mm采样经两轮未rip
    中值的灰白强度。original_edge_present(E,)布尔True表示原nofix确有
    此边，与修复base中的used标记不同；False固定加100。MRI与变换同
    prepare_topology_edge_inputs；默认无调参。保留端点+0.1…0.9九点、
    每点±0.5mm、22次绝对偏差，FP32字段与FP64累计。非法/非有限输入
    抛ValueError；不支持半精度。返回输入边顺序评分，不能作为完整
    候选fitness或几何质量结论；不执行原程序、不产生表面文件。
    """
    xyz = np.asarray(original_vertices, np.float32)
    normals = np.asarray(original_normals, np.float32)
    edges = np.asarray(edge_vertices)
    white = np.asarray(white_values, np.float32)
    gray = np.asarray(gray_values, np.float32)
    original_edge = np.asarray(original_edge_present)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or normals.shape != xyz.shape:
        raise ValueError("matching (N,3) original coordinates and normals required")
    if edges.ndim != 2 or edges.shape[1] != 2 or not np.issubdtype(edges.dtype, np.integer):
        raise ValueError("integer edge_vertices (E,2) required")
    if edges.size and (edges.min() < 0 or edges.max() >= len(xyz) or np.any(edges[:, 0] == edges[:, 1])):
        raise ValueError("valid distinct edge endpoints required")
    if white.shape != (len(xyz),) or gray.shape != white.shape:
        raise ValueError("white_values/gray_values must be (N,)")
    if original_edge.shape != (len(edges),) or original_edge.dtype != np.bool_:
        raise ValueError("original_edge_present must be bool (E,)")
    if any(not np.isfinite(a).all() for a in (xyz, normals, white, gray)):
        raise ValueError("finite coordinate/normal/intensity fields required")
    image, matrix = _validated_volume(volume, ras_to_voxel)
    return _score_edges(np.ascontiguousarray(xyz), np.ascontiguousarray(normals),
        np.ascontiguousarray(edges, np.int32), np.ascontiguousarray(white),
        np.ascontiguousarray(gray), original_edge, image, matrix)
