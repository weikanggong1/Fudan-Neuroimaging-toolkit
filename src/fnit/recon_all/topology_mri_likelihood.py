"""缺陷 patch 的完整 MRI face/vertex likelihood 子项；不是完整 GA fitness。

提供原顺序灰白联合 PDF 和两个 MRI 子项；几何、匹配、主曲率、unMRI
体积代价和遗传控制由调用者负责，不能以此替代完整拓扑质量验收。
"""
from __future__ import annotations
import numpy as np
from numba import njit
from .topology_edge_mri_score import _sample as _sample_ras
from .place_surface_border import _sample as _sample_voxel


@njit(cache=True, fastmath=False)
def _joint_pdf(white, gray, excluded):
    joint = np.zeros((256, 256, 1), np.float32)
    for vertex in range(len(white)):
        if excluded[vertex]:
            continue
        w = int(np.floor(float(white[vertex]) + .5))
        g = int(np.floor(float(gray[vertex]) + .5))
        joint[w, g, 0] = np.float32(joint[w, g, 0] + np.float32(1))
    norm = np.float32(0)
    for w in range(256):
        for g in range(256):
            if joint[w, g, 0] == 0:
                joint[w, g, 0] = np.float32(.1)
            norm = np.float32(norm + joint[w, g, 0])
    for w in range(256):
        for g in range(256):
            joint[w, g, 0] = np.float32(joint[w, g, 0] / norm)
    return joint


def joint_gray_white_pdf(*, white_values, gray_values, excluded_vertices) -> np.ndarray:
    """float32(N,)灰白中值与bool(N,)排除标记→(256,256,1)float32 PDF。

    两强度数组使用原MRI的0…255单位，顶点顺序必须一致；排除全部
    marked或rip顶点。按原顶点顺序计数，空bin赋0.1，再按白质行/灰质
    列的顺序float32归一化。无平滑、设备和参数开关，不改变输入；
    非有限、形状/布尔标记错误、未排除强度越界或全部排除时抛ValueError。
    属于mris_fix_topology内部统计，没有独立原软件命令。
    """
    white, gray = np.asarray(white_values, np.float32), np.asarray(gray_values, np.float32)
    excluded = np.asarray(excluded_vertices)
    if white.ndim != 1 or gray.shape != white.shape or excluded.shape != white.shape or excluded.dtype != bool:
        raise ValueError('matching white/gray vectors and bool excluded_vertices required')
    if not np.isfinite(white).all() or not np.isfinite(gray).all():
        raise ValueError('finite white/gray values required')
    selected = ~excluded
    if not selected.any() or any(np.any((x[selected] <= -.5) | (x[selected] >= 255.5)) for x in (white, gray)):
        raise ValueError('at least one unexcluded value within histogram range required')
    return _joint_pdf(np.ascontiguousarray(white), np.ascontiguousarray(gray), excluded)


@njit(cache=True, fastmath=False, error_model="numpy")
def _likelihood(xyz, faces, vertex_normals, face_normals, vertex_areas, face_areas,
                vertices, patch_faces, volume, matrix, pdf):
    vertex_total = weighted_vertices = vertex_area = 0.0
    for vertex in vertices:
        x, y, z = np.float64(xyz[vertex, 0]), np.float64(xyz[vertex, 1]), np.float64(xyz[vertex, 2])
        nx, ny, nz = np.float64(vertex_normals[vertex, 0]), np.float64(vertex_normals[vertex, 1]), np.float64(vertex_normals[vertex, 2])
        white = _sample_ras(volume, matrix, x-.5*nx, y-.5*ny, z-.5*nz)
        gray = _sample_ras(volume, matrix, x+.5*nx, y+.5*ny, z+.5*nz)
        probability = _sample_voxel(pdf, white, gray, 0.0)
        likelihood = np.log(probability)
        vertex_total += likelihood
        weighted_vertices += likelihood * np.float64(vertex_areas[vertex])
        vertex_area += np.float64(vertex_areas[vertex])
    face_total = weighted_faces = face_area = 0.0
    for face in patch_faces:
        a, b, c = faces[face]
        x = np.float64(np.float32(np.float32(np.float32(xyz[a, 0]+xyz[b, 0])+xyz[c, 0])/np.float32(3)))
        y = np.float64(np.float32(np.float32(np.float32(xyz[a, 1]+xyz[b, 1])+xyz[c, 1])/np.float32(3)))
        z = np.float64(np.float32(np.float32(np.float32(xyz[a, 2]+xyz[b, 2])+xyz[c, 2])/np.float32(3)))
        nx, ny, nz = np.float64(face_normals[face, 0]), np.float64(face_normals[face, 1]), np.float64(face_normals[face, 2])
        white = _sample_ras(volume, matrix, x-.5*nx, y-.5*ny, z-.5*nz)
        gray = _sample_ras(volume, matrix, x+.5*nx, y+.5*ny, z+.5*nz)
        likelihood = np.log(_sample_voxel(pdf, white, gray, 0.0))
        face_total += likelihood
        weighted_faces += likelihood * np.float64(face_areas[face])
        face_area += np.float64(face_areas[face])
    vertex_ll = np.float32(vertex_total / len(vertices)) if len(vertices) else np.float32(0)
    # face_ll 源码先(float)sum，再(float)count除法；vertex_ll是double除后存float。
    face_ll = np.float32(np.float32(face_total) / np.float32(len(patch_faces))) if len(patch_faces) else np.float32(0)
    # 源候选可能尚未刷新面积：主似然有限，面积加权诊断仍按 IEEE
    # 除法保存 NaN/Inf。它们不参与 fitness，不能拒绝这一中间候选。
    # 分母仍用上面的源顺序 double 累计，不另建归约树或替换为 0。
    vertex_weighted = np.float32(weighted_vertices / vertex_area) if len(vertices) else np.float32(0)
    face_weighted = np.float32(weighted_faces / face_area) if len(patch_faces) else np.float32(0)
    return face_ll, vertex_ll, face_weighted, vertex_weighted


def patch_mri_likelihood(*, original_vertices, faces, vertex_normals, face_normals,
                         vertex_areas_mm2, face_areas_mm2, patch_vertices, patch_faces,
                         volume, ras_to_voxel, gray_white_pdf) -> dict:
    """已拟合patch几何/MRI/自产PDF→完整face_ll、vertex_ll及面积加权诊断。

    original_vertices(N,3)为本候选的surface RAS/mm坐标；faces(F,3)为同序
    整数面，vertex_normals(N,3)/face_normals(F,3)为当前已计算的无量纲
    法向，vertex_areas_mm2(N,)/face_areas_mm2(F,)为源metric字段的面积。
    topology仍处于球面status时面面积可有方向符号，须完整保留，
    不能丢弃负面或把本诊断加权值解释为物理脑区面积。
    patch_vertices(K,)/patch_faces(M,)为完整、有序、唯一的修改索引；
    不能只选择质量好的子集。volume为3D uint8 MRI，ras_to_voxel为
    本MRI的tkregister RAS到连续体素的4×4矩阵（调用
    source_topology_ras_to_voxel生成，不用placement的scanner桥）。gray_white_pdf为自产正值float32
    (256,256,1)联合PDF。±0.5mm采样、face centroid FP32及累计规则固定。
    返回Python字典face_ll/vertex_ll（源float32子项转Python float）、
    mri_ll（两子项相加的double）、area_weighted_face_ll和vertex_ll；
    加权项仅为诊断，不替换原fitness子项。其分母为零时保留源 IEEE
    NaN/Inf，并返回 area_weighted_*_defined=False 及源 float32 *_bits；
    不把未定义诊断改成0，也不据此拒绝主似然有限的中间候选。
    空patch的源诊断仍为0且defined=True。JSON调用者应将未定义值
    写成null并保留bits。非法形状/索引、非有限输入、非正PDF抛
    ValueError；采样后非有限主似然抛FloatingPointError。
    不移动几何、不修交、不组合unMRI/曲率/normal项，不写文件、不运行
    原软件。只有本MRI子项完成，不能把它作为完整GA或质量通过。
    """
    xyz = np.asarray(original_vertices, np.float32); triangles = np.asarray(faces)
    normal = np.asarray(vertex_normals, np.float32); fnormal = np.asarray(face_normals, np.float32)
    area = np.asarray(vertex_areas_mm2, np.float32); farea = np.asarray(face_areas_mm2, np.float32)
    vertices = np.asarray(patch_vertices); selected_faces = np.asarray(patch_faces)
    image = np.asarray(volume); matrix = np.asarray(ras_to_voxel, np.float32); pdf = np.asarray(gray_white_pdf)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or normal.shape != xyz.shape or area.shape != (len(xyz),):
        raise ValueError('matching vertex coordinates/normals/areas required')
    if triangles.ndim != 2 or triangles.shape[1] != 3 or not np.issubdtype(triangles.dtype, np.integer) or fnormal.shape != (len(triangles), 3) or farea.shape != (len(triangles),):
        raise ValueError('matching integer faces, normals and areas required')
    if triangles.size and (triangles.min() < 0 or triangles.max() >= len(xyz)):
        raise ValueError('face indices outside vertex range')
    for indices, limit in [(vertices, len(xyz)), (selected_faces, len(triangles))]:
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer) or len(np.unique(indices)) != len(indices) or indices.size and (indices.min() < 0 or indices.max() >= limit):
            raise ValueError('unique ordered integer patch indices required')
    if image.ndim != 3 or image.dtype != np.uint8 or not all(image.shape) or matrix.shape != (4, 4) or pdf.shape != (256, 256, 1) or pdf.dtype != np.float32:
        raise ValueError('uint8 MRI, (4,4) matrix, float32 (256,256,1) PDF required')
    if any(not np.isfinite(a).all() for a in (xyz, normal, fnormal, area, farea, matrix, pdf)) or np.any(pdf <= 0):
        raise ValueError('finite fields and positive PDF required')
    values = _likelihood(np.ascontiguousarray(xyz), np.ascontiguousarray(triangles, np.int32),
        np.ascontiguousarray(normal), np.ascontiguousarray(fnormal), area, farea,
        np.asarray(vertices, np.int32), np.asarray(selected_faces, np.int32),
        np.ascontiguousarray(image), matrix, np.ascontiguousarray(pdf))
    if not np.isfinite(values[:2]).all():
        raise FloatingPointError('MRI likelihood sampling outside supported PDF produced nonfinite values')
    face_ll, vertex_ll, weighted_face, weighted_vertex = map(float, values)
    return {'face_ll': face_ll, 'vertex_ll': vertex_ll,
        'mri_ll': vertex_ll + face_ll,
        'area_weighted_face_ll': weighted_face, 'area_weighted_vertex_ll': weighted_vertex,
        'area_weighted_face_defined': bool(np.isfinite(weighted_face)),
        'area_weighted_vertex_defined': bool(np.isfinite(weighted_vertex)),
        'area_weighted_face_bits': int(np.float32(weighted_face).view(np.uint32)),
        'area_weighted_vertex_bits': int(np.float32(weighted_vertex).view(np.uint32)),
        'scope': 'MRI component only; full fitness/GA not implemented'}
