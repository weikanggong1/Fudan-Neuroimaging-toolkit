"""MRI coordinate matching for a retessellated FreeSurfer defect patch.

This reproduces the 40-step ``defectMaximizeLikelihood_new`` transition for
frozen first-candidate LH/RH patches. The caller supplies independently made
patch topology and white/gray histogram means.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from .topology_defect_smooth import _incident_arrays


@njit
def _sample(volume, inv, r, a, s):
    rr, aa, ss = np.float32(r), np.float32(a), np.float32(s)
    x = float(np.float32(inv[0, 0] * rr + inv[0, 1] * aa + inv[0, 2] * ss + inv[0, 3]))
    y = float(np.float32(inv[1, 0] * rr + inv[1, 1] * aa + inv[1, 2] * ss + inv[1, 3]))
    z = float(np.float32(inv[2, 0] * rr + inv[2, 1] * aa + inv[2, 2] * ss + inv[2, 3]))
    width, height, depth = volume.shape
    if (x < -0.5 or x > width - 0.5 or y < -0.5 or y > height - 0.5
            or z < -0.5 or z > depth - 0.5):
        return 0.0
    x = min(max(x, 0.0), width - 1.0)
    y = min(max(y, 0.0), height - 1.0)
    z = min(max(z, 0.0), depth - 1.0)
    xm, ym, zm = int(x), int(y), int(z)
    xp, yp, zp = min(xm + 1, width - 1), min(ym + 1, height - 1), min(zm + 1, depth - 1)
    xd, yd, zd = x - float(np.float32(xm)), y - float(np.float32(ym)), z - float(np.float32(zm))
    xa, ya, za = 1.0 - xd, 1.0 - yd, 1.0 - zd
    return (xa * ya * za * volume[xm, ym, zm] + xa * ya * zd * volume[xm, ym, zp]
            + xa * yd * za * volume[xm, yp, zm] + xa * yd * zd * volume[xm, yp, zp]
            + xd * ya * za * volume[xp, ym, zm] + xd * ya * zd * volume[xp, ym, zp]
            + xd * yd * za * volume[xp, yp, zm] + xd * yd * zd * volume[xp, yp, zp])


@njit
def _normals(xyz, faces, inside, offsets, face_ids):
    face_normals = np.zeros((len(faces), 3), np.float32)
    for fi in range(len(faces)):
        a, b, c = faces[fi]
        ax = np.float32(xyz[b, 0] - xyz[a, 0])
        ay = np.float32(xyz[b, 1] - xyz[a, 1])
        az = np.float32(xyz[b, 2] - xyz[a, 2])
        bx = np.float32(xyz[c, 0] - xyz[a, 0])
        by = np.float32(xyz[c, 1] - xyz[a, 1])
        bz = np.float32(xyz[c, 2] - xyz[a, 2])
        nx = np.float32(ay * bz - az * by)
        ny = np.float32(az * bx - ax * bz)
        nz = np.float32(ax * by - ay * bx)
        length = np.sqrt(np.float32(nx * nx + ny * ny + nz * nz))
        if length < 1e-5:
            length = np.float32(1.0)
        face_normals[fi, 0] = np.float32(nx / length)
        face_normals[fi, 1] = np.float32(ny / length)
        face_normals[fi, 2] = np.float32(nz / length)
    normals = np.zeros((len(inside), 3), np.float32)
    for i in range(len(inside)):
        v = inside[i]
        nx = ny = nz = np.float32(0)
        for slot in range(offsets[v], offsets[v + 1]):
            fi = face_ids[slot]
            nx = np.float32(nx + face_normals[fi, 0])
            ny = np.float32(ny + face_normals[fi, 1])
            nz = np.float32(nz + face_normals[fi, 2])
        length = np.sqrt(np.float32(nx * nx + ny * ny + nz * nz))
        if length < 1e-5:
            length = np.float32(1.0)
        normals[i, 0] = np.float32(nx / length)
        normals[i, 1] = np.float32(ny / length)
        normals[i, 2] = np.float32(nz / length)
    return normals


@njit
def _match(xyz, faces, inside, neighbors, degree, offsets, face_ids, volume, inv, wm, gm,
           iterations=40):
    current = xyz.copy()
    mean = np.float32(np.float32(wm + gm) * np.float32(0.5))
    for _ in range(iterations):
        normals = _normals(current, faces, inside, offsets, face_ids)
        next_xyz = current.copy()
        for i in range(len(inside)):
            v = inside[i]
            x = float(current[v, 0])
            y = float(current[v, 1])
            z = float(current[v, 2])
            nx = float(normals[i, 0])
            ny = float(normals[i, 1])
            nz = float(normals[i, 2])
            xm = ym = zm = 0.0
            for slot in range(degree[i]):
                other = neighbors[i, slot]
                xm += float(current[other, 0])
                ym += float(current[other, 1])
                zm += float(current[other, 2])
            if degree[i]:
                scale = 1.0 / float(degree[i])
                xm *= scale
                ym *= scale
                zm *= scale
            white = _sample(volume, inv, x - 0.5 * nx, y - 0.5 * ny, z - 0.5 * nz)
            mid = _sample(volume, inv, x, y, z)
            gray = _sample(volume, inv, x + 0.5 * nx, y + 0.5 * ny, z + 0.5 * nz)
            g = (white - gray) * (mid - float(mean))
            if abs(g) > 0.2:
                g = 0.2 if g > 0 else -0.2
            dx = 0.5 * (xm - x) + g * nx
            dy = 0.5 * (ym - y) + g * ny
            dz = 0.5 * (zm - z) + g * nz
            next_xyz[v, 0] = np.float32(x + 0.5 * dx)
            next_xyz[v, 1] = np.float32(y + 0.5 * dy)
            next_xyz[v, 2] = np.float32(z + 0.5 * dz)
        current = next_xyz
    return current



def defect_mri_match(xyz: np.ndarray, faces: np.ndarray, inside_vertices: np.ndarray,
                     runtime_neighbors: list[list[int]], volume: np.ndarray,
                     ras_to_voxel: np.ndarray, white_mean: float,
                     gray_mean: float, *, iterations: int = 40,
                     step_acceptor=None, topology=None) -> np.ndarray:
    """复用完整 MRI 拟合；未选顶点固定，坐标为 surface RAS/mm。

    xyz(N,3)/faces(F,3)及inside_vertices(K,)对应同一有序网格；
    runtime_neighbors给出每个选择顶点的原顺序邻接。volume为三维
    uint8归一化MRI，ras_to_voxel为surface RAS→MRI体素的4×4矩阵；
    white_mean/gray_mean为已有直方图求得的强度均值。iterations默认40，
    保留原完整算法；0返回独立坐标副本。非法迭代数在计算前报错。
    step_acceptor默认None，保持原40轮浮点更新；实验调用者可提供
    (current,proposal,iteration)→accepted坐标回调，在每轮同一MRI
    提案后执行明确的碰撞接受规则。回调不得改变shape/dtype、产生
    非有限坐标或移动未选择顶点；失败抛ValueError。返回float32(N,3)，
    不读写文件、不执行原版命令。回调版本是独立算法候选，不能冒充
    已冻结的原40轮复现结果。
    topology默认None，保留完整有序入射构造；显式FaceNormalTopology须
    与当前有序整数面及N完全一致，不兼容抛ValueError。它只缓存整数CSR，
    不缓存逐轮法向或坐标，不跨候选面版本复用。可与同候选25轮平滑共享。
    """
    if isinstance(iterations, (bool, np.bool_)) or not isinstance(iterations, (int, np.integer)) or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer")
    if step_acceptor is not None and not callable(step_acceptor):
        raise ValueError("step_acceptor must be callable or None")
    points = np.ascontiguousarray(xyz, np.float32)
    triangles = np.ascontiguousarray(faces, np.int32)
    inside = np.ascontiguousarray(inside_vertices, np.int32)
    degree = np.asarray([len(row) for row in runtime_neighbors], np.int32)
    neighbors = np.zeros((len(inside), int(degree.max(initial=0))), np.int32)
    for i, row in enumerate(runtime_neighbors):
        neighbors[i, :len(row)] = row
    offsets, face_ids = _incident_arrays(faces=triangles if topology is None else faces,
        vertex_count=len(points), topology=topology)
    arguments = (triangles, inside, neighbors, degree, offsets, face_ids,
                 np.ascontiguousarray(volume, np.uint8),
                 np.ascontiguousarray(ras_to_voxel, np.float64),
                 np.float32(white_mean), np.float32(gray_mean))
    if step_acceptor is None:
        return _match(points, *arguments, iterations)
    current = points.copy()
    fixed = np.ones(len(points), dtype=bool)
    fixed[inside] = False
    for iteration in range(iterations):
        proposal = _match(current, *arguments, 1)
        accepted = np.asarray(step_acceptor(current.copy(), proposal, iteration))
        if (accepted.shape != points.shape or accepted.dtype != np.float32
                or not np.isfinite(accepted).all()
                or not np.array_equal(accepted[fixed], points[fixed])):
            raise ValueError("step_acceptor changed shape, precision, fixed vertices or finite coordinates")
        current = np.ascontiguousarray(accepted)
    return current
