"""动态局部重三角化候选：复用拓扑缺陷与有序边接受，不调用原版程序。

这是完整首个缺陷候选的生成/三角面组合/QC原语，不是完整GA搜索。
MRI边代价是明确的独立配方；尚无原GA的所有fitness、mutation/crossover。
输入原始nofix几何与同序qsphere及自产uint8 MRI，修改边/面而非固定面形变。
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np
from numba import njit
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .place_surface_geometry import surface_ras_to_voxel
from .place_surface_normals import initial_vertex_normals
from .smooth_surface_python import ordered_neighbors
from .topology_defect_mri_match import _sample
from .topology_first_candidate import accept_first_candidate_edges
from .topology_intensity_histogram import first_defect_intensity_histograms
from .topology_preflight_python import (
    center_sphere, defect_component_labels, defect_regions,
    defect_retention_status, genetic_base_translation,
    genetic_candidate_edge_table, project_and_smooth_sphere, topology_counts,
)
from .topology_tissue_patch_python import _sample_sides, _median_twice
from .topology_volume_python import _nonmanifold_vertex_links


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def graph_triangles(*, canonical: np.ndarray, edges: np.ndarray,
                    external_faces: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """将局部球面嵌入图的有向边循环恢复为正向三角面。

    canonical为(N,3)有限非零球面点，edges为(K,2)无向全局索引。
    在每顶点的径向切平面按atan2和索引确定邻居顺序，逐half-edge
    选择前一邻居；每条有向边恰好访问一次。只接受长度3且正径向
    面循环，外部负循环不作为patch面；正向非三角循环明确标为未完成。
    external_faces可提供保留的完整外部三角面。它们的有向half-edge
    后继是硬约束；在每顶点把固定后继合成链，仅在链之间按切平面
    角度排列新边。这样不以canonical角度重排既有外部面。此模式仍
    会返回原外部面供调用者去重，并报告违反固定后继的结构错误。
    不改变坐标、不对空洞添加fan/convex-hull占位面，不隐藏退化循环。
    """
    xyz = np.asarray(canonical, np.float64)
    rows = np.asarray(edges)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("finite canonical (N,3) required")
    if rows.ndim != 2 or rows.shape[1] != 2 or not np.issubdtype(rows.dtype, np.integer):
        raise ValueError("integer edges (K,2) required")
    if rows.size and (rows.min() < 0 or rows.max() >= len(xyz) or np.any(rows[:, 0] == rows[:, 1])):
        raise ValueError("invalid edge indices")
    exterior = np.empty((0, 3), np.int32) if external_faces is None else np.asarray(external_faces)
    if exterior.ndim != 2 or exterior.shape[1] != 3 or not np.issubdtype(exterior.dtype, np.integer):
        raise ValueError("integer external_faces (M,3) required")
    if exterior.size and (exterior.min() < 0 or exterior.max() >= len(xyz) or np.any(np.diff(np.sort(exterior, axis=1), axis=1) == 0)):
        raise ValueError("invalid external face indices")
    if len(exterior):
        outside_edges = np.concatenate((exterior[:, [0, 1]], exterior[:, [1, 2]], exterior[:, [2, 0]]))
        rows = np.concatenate((rows, outside_edges))
    rows = np.unique(np.sort(rows.astype(np.int32), axis=1), axis=0)
    neighbors = {}
    for a, b in rows:
        neighbors.setdefault(int(a), []).append(int(b))
        neighbors.setdefault(int(b), []).append(int(a))
    forced = {}
    for face in exterior:
        for a, b, c in (face, np.roll(face, -1), np.roll(face, -2)):
            key, following = (int(a), int(b)), (int(b), int(c))
            if key in forced and forced[key] != following:
                raise ValueError("external faces assign conflicting directed-edge successors")
            forced[key] = following
    successor, rotation_conflicts = {}, []
    for vertex, adjacent in neighbors.items():
        point = xyz[vertex]
        radius = np.linalg.norm(point)
        if radius == 0:
            raise ValueError("zero canonical radius")
        normal = point / radius
        axis = np.zeros(3); axis[int(np.argmin(np.abs(normal)))] = 1
        tangent = np.cross(normal, axis); tangent /= np.linalg.norm(tangent)
        second = np.cross(normal, tangent)
        delta = xyz[adjacent] - point
        angle = np.arctan2(delta @ second, delta @ tangent)
        permutation = np.lexsort((np.asarray(adjacent), angle))
        ordered = [adjacent[int(index)] for index in permutation]
        angle_by_neighbor = {adjacent[index]: float(angle[index]) for index in range(len(adjacent))}
        fixed = {a: following[1] for (a, b), following in forced.items() if b == vertex}
        if len(set(fixed.values())) != len(fixed):
            raise ValueError("external faces assign conflicting vertex-star predecessors")
        if not fixed:
            for slot, a in enumerate(ordered):
                successor[(a, vertex)] = (vertex, ordered[(slot - 1) % len(ordered)])
            continue
        # 外部面的有向扇区不可拆开；新边只进入尚无固定后继的扇区。
        heads = [a for a in adjacent if a not in set(fixed.values())]
        seen, chains = set(), []
        for head in heads:
            chain, node = [], head
            while node not in seen:
                chain.append(node); seen.add(node)
                if node not in fixed:
                    break
                node = fixed[node]
            chains.append(chain)
        remaining = set(adjacent) - seen
        if remaining:
            # 已封闭的外部顶点扇区不能再插入一条候选边。
            if heads or len(remaining) != len(adjacent):
                rotation_conflicts.append({"vertex": int(vertex), "reason": "closed external star and additional neighbors"})
            for a in remaining:
                if a not in fixed:
                    raise ValueError("unvisited external star neighbor has no successor")
                successor[(a, vertex)] = (vertex, fixed[a])
        chains.sort(key=lambda chain: (-angle_by_neighbor[chain[0]], chain[0]))
        for slot, chain in enumerate(chains):
            for a, b in zip(chain, chain[1:]):
                successor[(a, vertex)] = (vertex, b)
            successor[(chain[-1], vertex)] = (vertex, chains[(slot + 1) % len(chains)][0])
    visited, output, trace = set(), [], []
    for first, second in rows:
        for start in ((int(first), int(second)), (int(second), int(first))):
            if start in visited:
                continue
            loop, current, valid = [], start, True
            while current not in visited:
                visited.add(current); loop.append(current[0])
                current = successor[current]
                if len(loop) > 2 * len(rows):
                    raise RuntimeError("half-edge walk exceeded complete directed edge count")
            valid = current == start and len(set(loop)) == len(loop) and len(loop) >= 3
            orientation = 0.
            if valid:
                points = xyz[loop]
                orientation = float(np.dot(np.cross(points, np.roll(points, -1, axis=0)).sum(axis=0), points.mean(axis=0)))
            triangular = valid and len(loop) == 3 and (orientation > 0 if external_faces is None else orientation != 0)
            if triangular:
                output.append(loop)
            trace.append({"vertices": loop, "valid_simple_cycle": bool(valid),
                          "radial_orientation": orientation, "triangle_accepted": bool(triangular)})
    triangles = np.asarray(output, np.int32).reshape(-1, 3)
    report = {"input_unique_edges": len(rows), "directed_edges_visited": len(visited),
              "complete_directed_edge_visit": len(visited) == 2 * len(rows),
              "candidate_triangles": len(triangles), "invalid_cycles": sum(not row["valid_simple_cycle"] for row in trace),
              "positive_nontriangle_cycles": sum(row["valid_simple_cycle"] and row["radial_orientation"] > 0 and not row["triangle_accepted"] for row in trace),
              "zero_orientation_cycles": sum(row["radial_orientation"] == 0 for row in trace),
              "fixed_external_faces": len(exterior), "fixed_external_halfedges": len(forced),
              "fixed_external_successors_preserved": all(successor.get(key) == value for key, value in forced.items()),
              "vertex_star_rotation_conflicts": rotation_conflicts,
              "negative_radial_triangles": sum(row["triangle_accepted"] and row["radial_orientation"] < 0 for row in trace), "cycles": trace}
    return triangles, report


def constrain_patch_boundary(*, patch: np.ndarray, external_faces: np.ndarray,
                             nvertices: int) -> tuple[np.ndarray, dict]:
    """按实际保留面边界限定候选，不能将球面凸包外侧填成patch。

    外部边已有两面时禁止新增面；只有一面时候选必须反向共边。
    无外部面时保留，是否恰好两面仍由最终全网格闭合/流形QC决定。
    不复制原面，不生成填充fan，不按MRI/被试坐标裁面。
    """
    candidate, outside = np.asarray(patch, np.int32), np.asarray(external_faces, np.int32)
    if candidate.ndim != 2 or candidate.shape[1] != 3 or outside.ndim != 2 or outside.shape[1] != 3:
        raise ValueError("triangular patch/external faces required")
    if nvertices < 1 or any(row.size and (row.min() < 0 or row.max() >= nvertices) for row in (candidate, outside)):
        raise ValueError("invalid face indices")
    exterior_edges = np.concatenate((outside[:,[0,1]],outside[:,[1,2]],outside[:,[2,0]]))
    exterior_keys = exterior_edges.min(axis=1).astype(np.int64)*nvertices+exterior_edges.max(axis=1)
    keys, inverse, incidence = np.unique(exterior_keys, return_inverse=True, return_counts=True)
    sign = np.bincount(inverse, weights=np.where(exterior_edges[:,0]<exterior_edges[:,1],1,-1))
    rows = np.concatenate((candidate[:,[0,1]],candidate[:,[1,2]],candidate[:,[2,0]]))
    query = rows.min(axis=1).astype(np.int64)*nvertices+rows.max(axis=1)
    insertion = np.searchsorted(keys, query)
    exists = insertion < len(keys)
    exists[exists] &= keys[insertion[exists]] == query[exists]
    count = np.zeros(len(rows),np.int64); direction = np.zeros(len(rows),np.int64)
    count[exists]=incidence[insertion[exists]]; direction[exists]=sign[insertion[exists]]
    occupied = np.any((count>=2).reshape(3,-1),axis=0)
    wrong_direction = np.any(((count==1)&(direction==np.where(rows[:,0]<rows[:,1],1,-1))).reshape(3,-1),axis=0)
    keep = ~(occupied|wrong_direction)
    return candidate[keep], {"input_triangles":len(candidate),"kept_triangles":int(keep.sum()),
                            "triangles_on_fully_occupied_exterior_edges":int(occupied.sum()),
                            "triangles_with_wrong_boundary_orientation":int(wrong_direction.sum()),
                            "rejected_triangle_indices":np.flatnonzero(~keep).tolist()}


@njit(cache=True, fastmath=False)
def _contains_neighbor_on_sphere(canonical, triangle, neighbors):
    """固定源码containsAnotherVertexOnSphere的三半空间检查，边界计inside。

    canonical是float32的(N,3)，triangle是3个索引，neighbors是三角顶点
    邻居并集且已去除三角顶点。cx+cx及cx-cx先float32，再进入double
    orgn/tangent数组；不添加epsilon、不默认把折叠边界视为凸多边形。
    """
    normals = np.empty((3, 3), np.float64)
    origins = np.empty((3, 3), np.float64)
    signs = np.empty(3, np.float64)
    for slot in range(3):
        a, b, c = triangle[slot], triangle[(slot + 1) % 3], triangle[(slot + 2) % 3]
        edge = np.empty(3, np.float64)
        for axis in range(3):
            origins[slot, axis] = np.float64(np.float32(canonical[b, axis] + canonical[c, axis])) / 2.
            edge[axis] = np.float64(np.float32(canonical[c, axis] - canonical[b, axis]))
        org = origins[slot]
        normals[slot, 0] = org[1] * edge[2] - org[2] * edge[1]
        normals[slot, 1] = org[2] * edge[0] - org[0] * edge[2]
        normals[slot, 2] = org[0] * edge[1] - org[1] * edge[0]
        delta = canonical[a].astype(np.float64) - org
        signs[slot] = (delta[0] * normals[slot, 0] + delta[1] * normals[slot, 1]) + delta[2] * normals[slot, 2]
    for vertex in neighbors:
        inside = True
        for slot in range(3):
            delta = canonical[vertex].astype(np.float64) - origins[slot]
            test = (delta[0] * normals[slot, 0] + delta[1] * normals[slot, 1]) + delta[2] * normals[slot, 2]
            if signs[slot] * test < 0:
                inside = False
                break
        if inside:
            return True
    return False


@njit(cache=True, fastmath=False)
def _canonical_winding(canonical, a, b, c):
    """固定orientDefectFaces的float32 chord cross / centroid dot。"""
    first, second = canonical[b] - canonical[a], canonical[c] - canonical[a]
    normal = np.empty(3, np.float32)
    normal[0] = first[1] * second[2] - first[2] * second[1]
    normal[1] = first[2] * second[0] - first[0] * second[2]
    normal[2] = first[0] * second[1] - first[1] * second[0]
    center = (canonical[a] + canonical[b]) + canonical[c]
    return np.float32(np.float32(normal[0] * center[0] + normal[1] * center[1]) + normal[2] * center[2])


def detect_patch_faces(*, canonical: np.ndarray, vertices: np.ndarray,
                       neighbors: list[list[int]], external_faces: np.ndarray) -> tuple[np.ndarray, dict]:
    """固定源detectDefectFaces的完整有序面发现原语，不是完整GA。

    vertices是retain-inside随后border的源顺序；neighbors是保留基底
    加已接受边后的有序一环。逐顶点/邻居对检查3-clique、既有面、
    三顶点邻居并集的球面包含关系、三条边的当前面数(<2)，随后按
    orientDefectFaces浮点规则定向。外部面从不重排或改变；新增面
    更新incidence，后候选必须读取更新状态。不筛掉困难retain顶点。
    这段不含USE_SOME动态顶点重分类、完整fitness/mutation/controller。
    """
    xyz = np.asarray(canonical, np.float32)
    verts = np.asarray(vertices)
    outside = np.asarray(external_faces, np.int32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all() or len(neighbors) != len(xyz):
        raise ValueError("finite canonical (N,3) and one ordered neighbor row per vertex required")
    if verts.ndim != 1 or not np.issubdtype(verts.dtype, np.integer) or (verts.size and (verts.min() < 0 or verts.max() >= len(xyz))):
        raise ValueError("valid source-order modified vertices required")
    if outside.ndim != 2 or outside.shape[1] != 3 or (outside.size and (outside.min() < 0 or outside.max() >= len(xyz))):
        raise ValueError("valid external triangular faces required")
    linked = [set(row) for row in neighbors]
    existing = {tuple(sorted(face)) for face in outside}
    pairs = np.sort(np.concatenate((outside[:, [0, 1]], outside[:, [1, 2]], outside[:, [2, 0]])), axis=1)
    keys, counts = np.unique(pairs, axis=0, return_counts=True)
    incidence = {tuple(row): int(count) for row, count in zip(keys, counts)}
    output, decisions = [], []
    contained, occupied = 0, 0
    for a in verts:
        a = int(a)
        for b in neighbors[a]:
            for c in neighbors[a]:
                if b == c or c not in linked[b]:
                    continue
                identity = tuple(sorted((a, b, c)))
                if identity in existing:
                    continue
                seen = {a, b, c}; checks = []
                for vertex in (a, b, c):
                    for other in neighbors[vertex]:
                        if other not in seen:
                            seen.add(other); checks.append(other)
                if _contains_neighbor_on_sphere(xyz, np.asarray([a, b, c], np.int32), np.asarray(checks, np.int32)):
                    contained += 1
                    decisions.append({"vertices": [a, b, c], "decision": "contains_neighbor_on_sphere"})
                    continue
                edges = (tuple(sorted((a, b))), tuple(sorted((b, c))), tuple(sorted((c, a))))
                if any(incidence.get(edge, 0) >= 2 for edge in edges):
                    occupied += 1
                    decisions.append({"vertices": [a, b, c], "decision": "edge_already_has_two_faces"})
                    continue
                face = [a, c, b] if _canonical_winding(xyz, a, b, c) < 0 else [a, b, c]
                output.append(face); existing.add(identity)
                for edge in edges:
                    incidence[edge] = incidence.get(edge, 0) + 1
                decisions.append({"vertices": [a, b, c], "decision": "accepted", "oriented_face": face})
    return np.asarray(output, np.int32).reshape(-1, 3), {
        "source_semantics": "detectDefectFaces + containsAnotherVertexOnSphere + mrisCanAttachFaceToVertices + orientDefectFaces",
        "fixed_source_commit": "d932c45b7941662ea380a05efef580568b98d41a",
        "fixed_defect_cpp_sha256": "55ace70ab09b1868a8cb152a9547879abdce790635b484bef1bf056986958187",
        "modified_vertices": len(verts), "accepted_faces": len(output),
        "rejected_contains_neighbor": contained, "rejected_fully_occupied_edge": occupied,
        "complete_ordered_face_decisions": decisions}


def removed_patch_domain(*, faces: np.ndarray) -> dict:
    """原移除子域的Euler和实际boundary图；分支边界不冒称简单loop。"""
    triangles = np.asarray(faces, np.int32)
    vertices = np.unique(triangles)
    compact = np.searchsorted(vertices, triangles).astype(np.int32)
    counts = topology_counts(compact, len(vertices))
    directed = np.concatenate((compact[:, [0, 1]], compact[:, [1, 2]], compact[:, [2, 0]]))
    edges, inverse, incidence = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True)
    boundary = edges[incidence == 1]
    ids = np.unique(boundary)
    degree = np.bincount(boundary.ravel(), minlength=len(vertices))
    graph = coo_matrix((np.ones(len(boundary) * 2),
                       (np.r_[boundary[:, 0], boundary[:, 1]], np.r_[boundary[:, 1], boundary[:, 0]])),
                      shape=(len(vertices), len(vertices))).tocsr()
    components = int(connected_components(graph[ids][:, ids], directed=False)[0]) if len(ids) else 0
    winding = np.bincount(inverse, weights=np.where(directed[:, 0] < directed[:, 1], 1, -1))
    return {"mesh_counts": asdict(counts), "boundary_components": components,
            "boundary_vertices_degree_not_two": int(np.count_nonzero(degree[ids] != 2)),
            "simple_boundary_loop_count": components if np.all(degree[ids] == 2) else None,
            "nonboundary_orientation_conflicts": int(np.count_nonzero(winding[incidence != 1])),
            "boundary_directed_edges_source_indices": vertices[directed[incidence[inverse] == 1]].tolist()}


def source_base_neighbors(*, source_neighbors: list[list[int]], labels: np.ndarray,
                          translation: np.ndarray) -> list[list[int]]:
    """保留未标记顶点间的源有序邻接，不能只从剩余三角面重建。

    固定源码分别复制face和neighbor表。一条border-border边的两面
    都被删除时，该边仍属于base图；它必须参与阻挡及面发现。
    labels为源顶点缺陷号(0表示未标记)，translation为源→base完整
    索引置换。输出每个base顶点的有序一环，缺陷顶点行为空。
    """
    lab, trans = np.asarray(labels), np.asarray(translation)
    size = len(source_neighbors)
    if lab.shape != (size,) or trans.shape != (size,) or not np.issubdtype(lab.dtype, np.integer) or not np.issubdtype(trans.dtype, np.integer):
        raise ValueError("integer source labels and source-to-base permutation required")
    if np.any(lab < 0) or not np.array_equal(np.sort(trans), np.arange(size)):
        raise ValueError("nonnegative defect labels and complete translation permutation required")
    output = [[] for _ in range(size)]
    for vertex, row in enumerate(source_neighbors):
        if any(not isinstance(other, (int, np.integer)) or other < 0 or other >= size or other == vertex for other in row) or len(row) != len(set(row)):
            raise ValueError("valid distinct source one-ring neighbor indices required")
        if lab[vertex] == 0:
            output[int(trans[vertex])] = [int(trans[other]) for other in row if lab[other] == 0]
    return output


@njit(cache=True, fastmath=False)
def _mri_edge_cost(original, edges, volume, matrix, white_probability, spacing, tissue_scale):
    """完整边线段间隔≤spacing采样；length(mm)−scale(mm)×平均log p(WM)。"""
    costs = np.empty(len(edges), np.float64)
    for index in range(len(edges)):
        a, b = original[edges[index, 0]], original[edges[index, 1]]
        delta = b.astype(np.float64) - a.astype(np.float64)
        length = np.sqrt(np.dot(delta, delta))
        count = max(2, int(np.ceil(length / spacing)) + 1)
        likelihood = 0.
        for slot in range(count):
            point = a.astype(np.float64) + delta * (slot / (count - 1))
            intensity = _sample(volume, matrix, point[0], point[1], point[2])
            bin_index = min(255, max(0, int(np.floor(intensity + .5))))
            probability = max(float(white_probability[bin_index]), 1e-30)
            likelihood += np.log(probability)
        costs[index] = length - tissue_scale * likelihood / count
    return costs


def run_first_retessellation(*, input_surface: str | Path, input_qsphere: str | Path,
                            intensity_volume: str | Path, output_surface: str | Path,
                            report_path: str | Path, tissue_scale_mm: float = 1.,
                            edge_sample_spacing_mm: float = .5,
                            require_quality: bool = True) -> dict:
    """自产首缺陷边/面候选，非完整拓扑GA；返回映射、全部输入SHA及QC。

    两表面须同序同面；surface RAS/mm。MRI须同空间MGZ/MGH uint8，
    几何由原表面元数据与MRI头变换核对。首先发现全部球面缺陷，再
    固定最低源序号组件，使用该组件全部retain/border候选，不挑参考边。
    使用现有归一化球面、状态/基底/边接受和白质直方图。MRI边代价为
    独立显式配方，非原GA len或完整fitness。排序按代价、原表序稳定tie。
    输出保留全部未受影响外部面的原相对顺序；新增局部面在后，删除
    不再使用的顶点，报告真实source index映射。其余缺陷仍存在，Euler
    不必为2，不能把一个patch通过视为全脑拓扑修复完成。
    require_quality=True仅在完整闭合、流形、连通、有限、非退化且
    无几何相交时写阶段候选；失败只写报告并抛异常，无原生回退。
    """
    tick = time.perf_counter()
    inp, qsphere, mri, out, report_path = map(Path, (input_surface, input_qsphere, intensity_volume, output_surface, report_path))
    if len({path.resolve() for path in (inp, qsphere, mri, out, report_path)}) != 5:
        raise ValueError("five distinct paths required")
    if out.exists() or report_path.exists():
        raise FileExistsError("new output/report paths required")
    if not np.isfinite(tissue_scale_mm) or tissue_scale_mm < 0 or not np.isfinite(edge_sample_spacing_mm) or edge_sample_spacing_mm <= 0:
        raise ValueError("nonnegative finite tissue scale and positive sampling spacing required")
    if not isinstance(require_quality, bool):
        raise ValueError("require_quality must be bool")
    original, faces, metadata = fsio.read_geometry(str(inp), read_metadata=True)
    canonical_input, qfaces = fsio.read_geometry(str(qsphere))
    if original.shape != canonical_input.shape or not np.array_equal(faces, qfaces) or not metadata:
        raise ValueError("same ordered input mesh with MRI geometry required")
    original = np.asarray(original, np.float32); faces = np.asarray(faces, np.int32)
    if not np.isfinite(original).all() or not np.isfinite(canonical_input).all():
        raise ValueError("finite input coordinates required")
    image = nib.load(str(mri)); volume = np.asarray(image.dataobj)
    if volume.ndim != 3 or volume.dtype != np.uint8 or not hasattr(image.header, "get_vox2ras_tkr"):
        raise ValueError("3D uint8 MGH/MGZ MRI required")
    matrix = surface_ras_to_voxel(image.header, metadata)
    canonical, center_iterations = center_sphere(project_and_smooth_sphere(canonical_input, faces))
    labels = defect_component_labels(canonical, faces)
    regions = defect_regions(labels, faces)
    if not regions:
        raise ValueError("no discovered sphere defect; this primitive does not copy a surface")
    status = defect_retention_status(canonical, original, faces, labels)
    translation, face_translation = genetic_base_translation(labels, faces)
    inverse = np.argsort(translation)
    sphere_ordered, original_ordered = canonical[inverse], original[inverse]
    table, unpruned = genetic_candidate_edge_table(canonical, faces, labels, status, defect_number=0)
    inside, border = regions[0]
    neighbors = ordered_neighbors(faces, len(original))
    normals = initial_vertex_normals(vertices=original, triangles=faces)
    white, gray = _sample_sides(original, normals, np.ascontiguousarray(volume), matrix, np.float32(.5))
    degree = np.asarray([len(row) for row in neighbors], np.int32)
    adjacent = np.zeros((len(neighbors), int(degree.max(initial=0))), np.int32)
    for vertex, row in enumerate(neighbors):
        adjacent[vertex, :len(row)] = row
    white, gray = _median_twice(white, adjacent, degree), _median_twice(gray, adjacent, degree)
    histogram = first_defect_intensity_histograms(white, gray, border, neighbors, labels > 0)
    if not len(histogram["selected"]):
        raise ValueError("no nondefect tissue histogram support")
    costs = _mri_edge_cost(original_ordered, table[:, :2], np.ascontiguousarray(volume), matrix,
                           histogram["white_smooth"], edge_sample_spacing_mm, tissue_scale_mm)
    order = np.lexsort((np.arange(len(costs)), costs))
    table = table[order]; costs = costs[order]
    base_local = table[table[:, 2] == 1, :2]
    # 仅替换首组件接触面；未处理的其余缺陷仍保留原始面，不将其删除。
    removed = np.any(labels[faces] == 1, axis=1)
    external_ids = np.flatnonzero(~removed)
    external = translation[faces[external_ids]]
    # intersectDefectEdges遍历inside及完整chull的邻边，不能仅取border内边。
    occupied = set(inside) | set(border)
    hull = list(border)
    for vertex in border:
        for neighbor in neighbors[vertex]:
            if neighbor not in occupied:
                occupied.add(neighbor); hull.append(neighbor)
    outside_neighbors = source_base_neighbors(source_neighbors=neighbors, labels=labels,
                                               translation=translation)
    outside_edges = np.asarray([(vertex, other) for vertex, row in enumerate(outside_neighbors)
                                for other in row if vertex < other], np.int32).reshape(-1, 2)
    blocker_vertices = translation[np.asarray(inside + hull, np.int32)]
    blocking_edges = outside_edges[np.any(np.isin(outside_edges, blocker_vertices), axis=1)]
    selected = accept_first_candidate_edges(sphere_ordered, table[:, :2], table[:, 2], blocking_edges)
    active_edges = np.concatenate((base_local, table[selected, :2]))
    patch_vertices = np.unique(active_edges)
    local_external = external[np.any(np.isin(external, patch_vertices), axis=1)]
    _, walks = graph_triangles(canonical=sphere_ordered, edges=active_edges,
                               external_faces=local_external)
    walks["scope"] = "half-edge diagnostic only; actual faces are selected by ordered source face discovery"
    for a, b in table[selected, :2]:
        a, b = int(a), int(b)
        if b not in outside_neighbors[a]:
            outside_neighbors[a].append(b)
        if a not in outside_neighbors[b]:
            outside_neighbors[b].append(a)
    retained_ids = np.asarray(inside, np.int32)[status[inside] == 1]
    modified_vertices = translation[np.r_[retained_ids, np.asarray(border, np.int32)]]
    patch, detection = detect_patch_faces(canonical=sphere_ordered, vertices=modified_vertices,
                                          neighbors=outside_neighbors, external_faces=external)
    _, boundary = constrain_patch_boundary(patch=patch, external_faces=external, nvertices=len(original))
    boundary["scope"] = "diagnostic only; source-generated faces remain unchanged and full QC rejects conflicts"
    combined = np.concatenate((external, patch))
    used = np.unique(combined)
    compact = np.full(len(original), -1, np.int32); compact[used] = np.arange(len(used), dtype=np.int32)
    corrected, corrected_faces = original_ordered[used], compact[combined]
    source_indices = inverse[used]
    missing_retained = np.setdiff1d(retained_ids, source_indices)
    generation_done = time.perf_counter()
    counts = topology_counts(corrected_faces, len(corrected))
    edges = np.concatenate((corrected_faces[:, [0, 1]], corrected_faces[:, [1, 2]], corrected_faces[:, [2, 0]]))
    graph = coo_matrix((np.ones(len(edges) * 2), (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])), shape=(len(corrected), len(corrected))).tocsr()
    components = int(connected_components(graph, directed=False)[0])
    nonmanifold = int(_nonmanifold_vertex_links(corrected_faces, len(corrected)))
    crosses = np.cross(corrected[corrected_faces[:, 1]].astype(np.float64) - corrected[corrected_faces[:, 0]], corrected[corrected_faces[:, 2]].astype(np.float64) - corrected[corrected_faces[:, 0]])
    zero_faces = int(np.count_nonzero(np.linalg.norm(crosses, axis=1) == 0))
    signs = np.where(edges[:, 0] < edges[:, 1], 1, -1)
    _, edge_index = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True)
    conflicts = int(np.count_nonzero(np.bincount(edge_index, weights=signs)))
    duplicate_faces = len(corrected_faces) - len(np.unique(np.sort(corrected_faces, axis=1), axis=0))
    structural = (len(patch) > 0 and counts.boundary_edges == 0 and counts.nonmanifold_edges == 0 and nonmanifold == 0 and components == 1 and zero_faces == 0 and conflicts == 0 and duplicate_faces == 0 and len(missing_retained) == 0)
    intersections = None
    if structural:
        from .mris_remove_intersection_python import mark_intersections
        _, intersections = mark_intersections(corrected, corrected_faces)
    quality = structural and intersections == 0
    external_face_edges = np.unique(np.sort(np.concatenate((external[:, [0, 1]],
                         external[:, [1, 2]], external[:, [2, 0]])), axis=1), axis=0)
    floating_base_edges = len(set(map(tuple, outside_edges)) - set(map(tuple, external_face_edges)))
    report = {"scope": "first discovered defect dynamic edge/face candidate primitive, not complete GA or full topology repair", "execution_status": "complete_primitive" if quality else "failed_quality",
              "native_program_called": False, "complete_ga_solver": False, "overall_metric_equivalence": "not_assessed", "input_sha256": {"orig_nofix": _sha(inp), "qsphere_nofix": _sha(qsphere), "MRI": _sha(mri)},
              "source_sha256": _sha(__file__), "defect_components": len(regions), "defect_number_source_order": 1, "inside_vertices": len(inside), "border_vertices": len(border), "retained_inside_vertices": int(np.count_nonzero(status[inside] == 1)),
              "candidate_edges_before_prune": unpruned, "candidate_edges_after_prune": len(table), "candidate_edges_selected": int(selected.sum()), "base_edges": len(base_local), "edge_cost_definition": "length_mm - tissue_scale_mm * mean log p(WM) along complete segment", "tissue_scale_mm": tissue_scale_mm,
              "actual_chull_vertices": len(hull), "actual_chull_blocking_edges": len(blocking_edges), "actual_chull_blocking_edge_sha256": hashlib.sha256(blocking_edges.astype('<i4').tobytes()).hexdigest(), "retessellation_vertex_mode": "USE_ALL-style fixed initially retained vertices; no USE_SOME dynamic reclassification",
              "base_neighbor_source": "original ordered one-ring with both endpoints unmarked; independent of preserved face incidence",
              "base_edges_without_preserved_faces": floating_base_edges,
              "edge_sample_spacing_mm": edge_sample_spacing_mm, "canonical_center_iterations": center_iterations, "histogram_support_vertices": len(histogram["selected"]), "white_histogram_mean": float(histogram["white_mean"]), "gray_histogram_mean": float(histogram["gray_mean"]),
              "candidate_edge_order_sha256": hashlib.sha256(table.astype('<i4').tobytes()).hexdigest(), "candidate_edge_cost_sha256": hashlib.sha256(costs.astype('<f8').tobytes()).hexdigest(), "accepted_edge_flags_sha256": hashlib.sha256(selected.tobytes()).hexdigest(),
              "removed_faces": int(removed.sum()), "added_patch_faces": len(patch), "outside_faces_order_preserved": bool(np.array_equal(source_indices[corrected_faces[:len(external_ids)]], faces[external_ids])), "outside_face_source_indices": external_ids.tolist(),
              "output_vertex_source_indices": source_indices.tolist(), "removed_vertex_source_indices": np.setdiff1d(np.arange(len(original)), source_indices).tolist(), "retained_inside_source_indices": retained_ids.tolist(), "missing_retained_source_indices": missing_retained.tolist(), "mesh_input": asdict(topology_counts(faces, len(original))), "mesh_output": asdict(counts), "connected_components": components, "nonmanifold_vertices": nonmanifold, "zero_area_faces": zero_faces, "duplicate_faces": duplicate_faces, "edge_orientation_conflicts": conflicts,
              "geometric_intersecting_faces": intersections, "patch_face_walk": walks, "actual_boundary_constraint": boundary, "source_face_discovery": detection, "original_removed_patch_domain": removed_patch_domain(faces=faces[removed]), "mesh_quality_status": "passed_first_patch" if quality else "failed", "full_topology_status": "not_completed", "output_written": False,
              "candidate_generation_IO_JIT_seconds": generation_done - tick, "quality_seconds": time.perf_counter() - generation_done}
    if quality:
        out.parent.mkdir(parents=True, exist_ok=True)
        fsio.write_geometry(str(out), corrected, corrected_faces, volume_info=metadata)
        report.update(output_written=True, output_sha256=_sha(out))
    report["total_seconds_including_IO_JIT_quality"] = time.perf_counter() - tick
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if require_quality and not quality:
        raise RuntimeError("dynamic first patch failed structural/intersection QC; no candidate written")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input-surface", "input-qsphere", "intensity-volume", "output-surface", "report-path"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--tissue-scale-mm", type=float, default=1.)
    parser.add_argument("--edge-sample-spacing-mm", type=float, default=.5)
    args = parser.parse_args()
    run_first_retessellation(input_surface=args.input_surface, input_qsphere=args.input_qsphere,
        intensity_volume=args.intensity_volume, output_surface=args.output_surface, report_path=args.report_path,
        tissue_scale_mm=args.tissue_scale_mm, edge_sample_spacing_mm=args.edge_sample_spacing_mm)


if __name__ == "__main__":
    main()
