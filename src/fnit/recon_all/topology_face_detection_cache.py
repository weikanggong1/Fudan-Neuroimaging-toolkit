"""同一不可变基底的面发现索引复用；动态邻接与有序判定每次重新消费。"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
import hashlib
import time

import numpy as np

from .topology_retessellation_python import _contains_neighbor_on_sphere, _canonical_winding

_TOKEN = object()


@dataclass(frozen=True)
class FaceDetectionContext:
    """由prepare_face_detection_context创建的独立、不可变保留面索引。"""
    faces: np.ndarray
    vertex_count: int
    face_identities: frozenset
    edge_counts: MappingProxyType
    faces_sha256: str
    preparation_seconds: float
    _token: object


def prepare_face_detection_context(*, base_faces, vertex_count):
    """有序整数(F,3)保留面及N→不可变面身份/无向边计数。

    vertex_count为非负整数；面索引在[0,N)，包括重复面和重复边的计数。
    返回FaceDetectionContext，独立bytes-backed int32 faces、frozenset
    面身份、只读边计数及准备时间/SHA。面序不改变，原输入后续修改不
    影响它。只缓存保留面，不缓存canonical、候选邻接、接受/拒绝或
    增加面。无默认阈值、精度/设备设置和文件I/O；非法输入抛ValueError。
    内部步骤没有独立原软件CLI，对应detectDefectFaces的固定保留面索引。
    """
    tick = time.perf_counter()
    faces = np.asarray(base_faces)
    if (isinstance(vertex_count, (bool, np.bool_)) or
            not isinstance(vertex_count, (int, np.integer)) or vertex_count < 0 or
            faces.ndim != 2 or faces.shape[1] != 3 or
            not np.issubdtype(faces.dtype, np.integer) or
            faces.size and (faces.min() < 0 or faces.max() >= vertex_count)):
        raise ValueError("integer base_faces (F,3) and valid vertex_count required")
    frozen = np.asarray(faces, np.int32)
    frozen = np.frombuffer(frozen.tobytes(), np.int32).reshape(frozen.shape)
    identities = frozenset(tuple(sorted(face)) for face in frozen)
    edges = np.sort(np.concatenate((frozen[:, [0, 1]], frozen[:, [1, 2]],
                                    frozen[:, [2, 0]])), axis=1)
    keys, counts = np.unique(edges, axis=0, return_counts=True)
    table = MappingProxyType({tuple(row): int(count) for row, count in zip(keys, counts)})
    return FaceDetectionContext(frozen, int(vertex_count), identities, table,
        hashlib.sha256(frozen.tobytes()).hexdigest(), time.perf_counter()-tick, _TOKEN)


def detect_patch_faces_cached(*, canonical, vertices, neighbors, external_faces, context):
    """复用保留面索引，完整执行有序局部面发现→(int32(K,3),判定dict)。

    canonical有限(N,3)，按成熟原语转换为float32，单位canonical RAS；
    vertices整数有序修改顶点；neighbors为全部N行当前邻接，包含未被
    面支撑的边。external_faces必须与context完整保留面逐元素相同。
    context必填且来自上述工厂；动态邻接只在真正消费时转换集合，不
    复用另一候选。新增面身份/边计数采用本次独立overlay；保持重复面、
    每条边的更新依赖、全部判定顺序与源float32定向/球面包含谓词。
    返回同成熟detect_patch_faces完整报告；输入、缓存与其他候选不变。
    不产生最终网格、不运行GA或原程序。非法/变化的基底或元数据报错。
    """
    xyz, verts, outside = np.asarray(canonical, np.float32), np.asarray(vertices), np.asarray(external_faces, np.int32)
    if (not isinstance(context, FaceDetectionContext) or context._token is not _TOKEN or
            context.faces.dtype != np.int32 or context.faces.ndim != 2 or
            context.faces.shape[1] != 3 or context.faces.flags.writeable or
            not context.faces.flags.c_contiguous or not isinstance(context.edge_counts, MappingProxyType) or
            not isinstance(context.face_identities, frozenset)):
        raise ValueError("unchanged immutable FaceDetectionContext required")
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all() or
            len(neighbors) != len(xyz) or len(xyz) != context.vertex_count):
        raise ValueError("finite canonical (N,3) and current N neighbor rows required")
    if (verts.ndim != 1 or not np.issubdtype(verts.dtype, np.integer) or
            verts.size and (verts.min() < 0 or verts.max() >= len(xyz))):
        raise ValueError("valid source-order modified vertices required")
    if outside.shape != context.faces.shape or not np.array_equal(outside, context.faces):
        raise ValueError("complete external_faces must match the frozen base")
    links, added_identities, added_edge_counts = {}, set(), {}
    output, decisions, contained, occupied = [], [], 0, 0
    for a in verts:
        a = int(a)
        for b in neighbors[a]:
            if b not in links:
                links[b] = set(neighbors[b])
            for c in neighbors[a]:
                if b == c or c not in links[b]:
                    continue
                identity = tuple(sorted((a, b, c)))
                if identity in context.face_identities or identity in added_identities:
                    continue
                seen, checks = {a, b, c}, []
                for vertex in (a, b, c):
                    for other in neighbors[vertex]:
                        if other not in seen:
                            seen.add(other); checks.append(other)
                if _contains_neighbor_on_sphere(xyz, np.asarray([a, b, c], np.int32),
                                               np.asarray(checks, np.int32)):
                    contained += 1
                    decisions.append(dict(vertices=[a, b, c], decision="contains_neighbor_on_sphere"))
                    continue
                edges = (tuple(sorted((a, b))), tuple(sorted((b, c))), tuple(sorted((c, a))))
                if any(context.edge_counts.get(edge, 0) + added_edge_counts.get(edge, 0) >= 2 for edge in edges):
                    occupied += 1
                    decisions.append(dict(vertices=[a, b, c], decision="edge_already_has_two_faces"))
                    continue
                face = [a, c, b] if _canonical_winding(xyz, a, b, c) < 0 else [a, b, c]
                output.append(face); added_identities.add(identity)
                for edge in edges:
                    added_edge_counts[edge] = added_edge_counts.get(edge, 0) + 1
                decisions.append(dict(vertices=[a, b, c], decision="accepted", oriented_face=face))
    return np.asarray(output, np.int32).reshape(-1, 3), dict(
        source_semantics="detectDefectFaces + containsAnotherVertexOnSphere + mrisCanAttachFaceToVertices + orientDefectFaces",
        fixed_source_commit="d932c45b7941662ea380a05efef580568b98d41a",
        fixed_defect_cpp_sha256="55ace70ab09b1868a8cb152a9547879abdce790635b484bef1bf056986958187",
        modified_vertices=len(verts), accepted_faces=len(output),
        rejected_contains_neighbor=contained, rejected_fully_occupied_edge=occupied,
        complete_ordered_face_decisions=decisions)
