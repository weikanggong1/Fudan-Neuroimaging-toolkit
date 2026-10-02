"""Python translation of FreeSurfer 8.2 ``mris_remesh --remesh`` geometry steps.

Follows the pinned FreeSurfer remesher source at commit d932c45. This CPU
implementation keeps the source's edge ordering and in-place smoothing.
"""

from __future__ import annotations

import heapq
import math
import struct
from pathlib import Path

import nibabel.freesurfer as fs
import numpy as np
from numba import njit

def edge_key(a, b):
    return (a, b) if a < b else (b, a)


def edge_length(points, a, b):
    u, v = points[a], points[b]
    x = u[0] - v[0]
    y = u[1] - v[1]
    z = u[2] - v[2]
    return math.sqrt(x * x + y * y + z * z)


def initial_topology(faces):
    edge_index = {}
    edge_vertices = []
    edge_faces = []
    face_edges = []
    for ti, (a, b, c) in enumerate(faces):
        row = []
        for u, v in ((a, b), (b, c), (c, a)):
            key = edge_key(u, v)
            ei = edge_index.get(key)
            if ei is None:
                ei = len(edge_vertices)
                edge_index[key] = ei
                edge_vertices.append([key[0], key[1]])
                edge_faces.append([ti])
            else:
                edge_faces[ei].append(ti)
            row.append(ei)
        face_edges.append(row)
    return edge_index, edge_vertices, edge_faces, face_edges


def split_edge(points, faces, edge_index, edge_vertices, edge_faces, face_edges, ei):
    a, b = edge_vertices[ei]
    v1, v2 = edge_key(a, b)
    assert edge_index[(v1, v2)] == ei
    ni = len(points)
    points.append(tuple(v1c * 0.5 + v2c * (1.0 - 0.5)
                        for v1c, v2c in zip(points[v1], points[v2])))
    en = len(edge_vertices)
    edge_vertices.append([ni, v2])
    edge_index[edge_key(ni, v2)] = en
    if edge_vertices[ei][0] == v2:
        edge_vertices[ei][0] = ni
    else:
        assert edge_vertices[ei][1] == v2
        edge_vertices[ei][1] = ni
    edge_index[edge_key(v1, ni)] = ei
    del edge_index[(v1, v2)]
    edge_faces.append([])

    replacement = []
    for ti in edge_faces[ei]:
        face = faces[ti]
        other = next(k for k in range(3) if face[k] != v1 and face[k] != v2)
        reverse = face[(other + 1) % 3] == v2
        assert face[(other + 2) % 3] == (v1 if reverse else v2)
        if not reverse:
            assert face[(other + 1) % 3] == v1
        eint = len(edge_vertices)
        edge_vertices.append([face[other], ni])
        edge_index[edge_key(face[other], ni)] = eint
        t2i = len(faces)
        faces.append([ni, face[other], face[(other + 1) % 3]])
        face_edges.append([eint, face_edges[ti][other], en if reverse else ei])
        tte = face_edges[ti][other]
        neighbours = edge_faces[tte]
        neighbours[neighbours.index(ti)] = t2i
        edge_faces.append([ti, t2i])
        face[(other + 1) % 3] = ni
        face_edges[ti][other] = eint
        face_edges[ti][(other + 1) % 3] = ei if reverse else en
        if reverse:
            edge_faces[en].append(t2i)
            replacement.append(ti)
        else:
            edge_faces[en].append(ti)
            replacement.append(t2i)
    edge_faces[ei] = replacement


def split_pass(points, faces, edge_index, edge_vertices, edge_faces, face_edges, threshold):
    """按长度降序、边编号降序拆边；threshold 为 surface RAS 的 mm。

    输入是原位更新的 points、faces 和四份可变拓扑索引，返回新增顶点数。
    初始堆只改用线性 heapify，后续重新入堆和拆边规则保持不变。
    需要有限坐标及一致的拓扑；非法索引或拓扑断言失败会抛异常。
    """
    queue = []
    for ei, (a, b) in enumerate(edge_vertices):
        length = edge_length(points, a, b)
        queue.append((-length, -ei))
    heapq.heapify(queue)
    inserted = 0
    while queue and -queue[0][0] > threshold:
        old_neg_length, neg_ei = heapq.heappop(queue)
        ei = -neg_ei
        if not edge_faces[ei]:
            continue
        a, b = edge_vertices[ei]
        current = edge_length(points, a, b)
        old = -old_neg_length
        if current < old:
            heapq.heappush(queue, (-current, -ei))
        else:
            split_edge(points, faces, edge_index, edge_vertices, edge_faces, face_edges, ei)
            inserted += 1
    return inserted


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def norm(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def subtract(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


class Mesh:
    def __init__(self, points, faces):
        self.points = points
        self.faces = faces
        self.rebuild()

    def rebuild(self):
        self.edge_index, self.edge_vertices, self.edge_faces, self.face_edges = initial_topology(self.faces)
        self.vertex_faces = [[] for _ in self.points]
        self.vertex_local = [[] for _ in self.points]
        for ti, face in enumerate(self.faces):
            for local, vertex in enumerate(face):
                if ti not in self.vertex_faces[vertex]:
                    self.vertex_faces[vertex].append(ti)
                    self.vertex_local[vertex].append(local)
        self.vertex_edges = [dict() for _ in self.points]
        for ei, (a, b) in enumerate(self.edge_vertices):
            self.vertex_edges[a][b] = ei
            self.vertex_edges[b][a] = ei
        self.onboundary = [False] * len(self.points)
        for ei, (a, b) in enumerate(self.edge_vertices):
            if len(self.edge_faces[ei]) != 2:
                self.onboundary[a] = self.onboundary[b] = True

    def face_normals(self):
        result = []
        for a, b, c in self.faces:
            p1, p2, p3 = self.points[a], self.points[b], self.points[c]
            n = cross(subtract(p2, p1), subtract(p3, p1))
            length = norm(n)
            result.append((0.0, 0.0, 0.0) if length < 1e-11
                          else tuple((1.0 / length) * value for value in n))
        return result

    def remove_edge(self, ei):
        a, b = self.edge_vertices[ei]
        assert self.vertex_edges[a].pop(b) == ei
        assert self.vertex_edges[b].pop(a) == ei
        assert self.edge_index.pop(edge_key(a, b)) == ei
        return a, b

    def contract_in_face(self, ei, ti):
        edges = self.face_edges[ti]
        lv = edges.index(ei)
        e0 = edges[(lv + 2) % 3]
        e1 = edges[(lv + 1) % 3]
        assert e0 != e1
        adjacent0 = self.edge_faces[e0]
        adjacent1 = self.edge_faces[e1]
        assert len(adjacent0) == 2 and len(adjacent1) == 2
        if adjacent0[0] == ti:
            neighbour0, local = adjacent0[1], 0
        else:
            assert adjacent0[1] == ti
            neighbour0, local = adjacent0[0], 1
        neighbour1 = adjacent1[1] if adjacent1[0] == ti else adjacent1[0]
        assert neighbour0 != neighbour1
        adjacent0[local] = neighbour1
        self.face_edges[neighbour1][self.face_edges[neighbour1].index(e1)] = e0
        self.remove_edge(e1)
        self.edge_faces[e1].clear()
        self.edge_vertices[e1].clear()

    def remove_face_from_vertex(self, vertex, ti):
        row = self.vertex_faces[vertex]
        try:
            index = row.index(ti)
        except ValueError:
            return
        row[index] = row[-1]
        row.pop()
        local = self.vertex_local[vertex]
        local[index] = local[-1]
        local.pop()

    def can_contract(self, ei, normals):
        if not self.edge_faces[ei]:
            return None
        assert len(self.edge_faces[ei]) == 2
        v0, v1 = self.edge_vertices[ei]
        t0, t1 = self.edge_faces[ei]
        tip = []
        for ti in (t0, t1):
            face = self.faces[ti]
            opposite = face[0]
            if opposite == v0 or opposite == v1:
                opposite = face[1]
            if opposite == v0 or opposite == v1:
                opposite = face[2]
            tip.append(opposite)
        for other in self.vertex_edges[v1]:
            if other not in (v0, tip[0], tip[1]) and other in self.vertex_edges[v0]:
                return None
        p0, p1 = self.points[v0], self.points[v1]
        midpoint = tuple(0.5 * (a + b) for a, b in zip(p0, p1))
        epsilon = math.cos(math.pi / 3.0)
        for vertex in (v0, v1):
            for ti, local in zip(self.vertex_faces[vertex], self.vertex_local[vertex]):
                if ti == t0 or ti == t1:
                    continue
                face = self.faces[ti]
                assert face[local] == vertex
                pb = self.points[face[(local + 1) % 3]]
                pc = self.points[face[(local + 2) % 3]]
                n = cross(subtract(pb, midpoint), subtract(pc, midpoint))
                length = norm(n)
                dot = n[0] * normals[ti][0] + n[1] * normals[ti][1] + n[2] * normals[ti][2]
                projection = dot / length if length else float('nan')
                if projection < epsilon:
                    return None
        return v0, v1, t0, t1, tip, midpoint

    def contract(self, ei, normals):
        decision = self.can_contract(ei, normals)
        if decision is None:
            return False
        v0, v1, t0, t1, tip, midpoint = decision
        self.points[v0] = midpoint
        self.contract_in_face(ei, t0)
        self.contract_in_face(ei, t1)
        for ti, local in zip(self.vertex_faces[v1], self.vertex_local[v1]):
            self.faces[ti][local] = v0
        for vertex, ti in ((v0, t0), (v0, t1), (v1, t0), (v1, t1),
                           (tip[0], t0), (tip[1], t1)):
            self.remove_face_from_vertex(vertex, ti)
        self.vertex_faces[v0].extend(self.vertex_faces[v1])
        self.vertex_local[v0].extend(self.vertex_local[v1])
        self.vertex_faces[v1].clear()
        self.vertex_local[v1].clear()
        self.remove_edge(ei)
        for other, edge in list(self.vertex_edges[v1].items()):
            assert other != v0 and other not in self.vertex_edges[v0]
            pair = self.edge_vertices[edge]
            pair[pair.index(v1)] = v0
            assert self.vertex_edges[v1].pop(other) == edge
            assert self.vertex_edges[other].pop(v1) == edge
            self.vertex_edges[v0][other] = edge
            self.vertex_edges[other][v0] = edge
            assert self.edge_index.pop(edge_key(v1, other)) == edge
            self.edge_index[edge_key(v0, other)] = edge
        self.edge_faces[ei].clear()
        self.edge_vertices[ei].clear()
        self.faces[t0].clear()
        self.faces[t1].clear()
        self.face_edges[t0].clear()
        self.face_edges[t1].clear()
        return True

    def compact(self):
        ti = 0
        while ti < len(self.faces):
            if not self.faces[ti]:
                self.faces[ti] = self.faces[-1]
                self.faces.pop()
            else:
                ti += 1
        used = [False] * len(self.points)
        for face in self.faces:
            for vertex in face:
                used[vertex] = True
        mapping = [-1] * len(self.points)
        new_points = []
        for old, active in enumerate(used):
            if active:
                mapping[old] = len(new_points)
                new_points.append(self.points[old])
        self.points = new_points
        for face in self.faces:
            for index, vertex in enumerate(face):
                face[index] = mapping[vertex]
        self.rebuild()

    def collapse_pass(self, threshold):
        """按长度升序、边编号升序缩边，返回接受次数并紧凑重建拓扑。

        threshold 单位为 mm；使用本轮初始面法线及原有几何/连接性检查。
        只将初始队列逐项 heappush 改为 heapify，动态重新入堆不变。
        非流形拓扑或无效索引会触发原有断言/异常。
        """
        normals = self.face_normals()
        queue = []
        for ei, pair in enumerate(self.edge_vertices):
            if not self.edge_faces[ei]:
                continue
            a, b = pair
            if len(self.edge_faces[ei]) < 2 or self.onboundary[a] or self.onboundary[b]:
                continue
            queue.append((edge_length(self.points, a, b), ei))
        heapq.heapify(queue)
        accepted = 0
        while queue and queue[0][0] < threshold:
            old, ei = heapq.heappop(queue)
            if not self.edge_faces[ei]:
                continue
            a, b = self.edge_vertices[ei]
            current = edge_length(self.points, a, b)
            if current > old:
                heapq.heappush(queue, (current, ei))
            elif self.contract(ei, normals):
                accepted += 1
        self.compact()
        return accepted


@njit(cache=True, fastmath=False)
def _smooth_ordered(points: np.ndarray, faces: np.ndarray,
                    offsets: np.ndarray, neighbours: np.ndarray,
                    boundary: np.ndarray, repeats: int) -> None:
    """编译后的顺序平滑内核；points float64(N,3) 原位更新，单位 mm。

    faces int32(F,3)、offsets int64(N+1)、neighbours int32(E) 描述升序 CSR
    邻接，boundary bool(N) 标记边界。每轮重算面积和法线；保持面序累加、
    顶点升序原位更新及双精度表达式，不并行、不使用 fastmath。repeats 是轮数。
    无返回值；零顶点法线或零邻接总面积与旧 Python 路径一样触发除零异常。
    这是 mris_remesh 内部平滑步骤，没有独立官方 CLI。
    """
    for _ in range(repeats):
        areas = np.zeros(len(points), np.float64)
        vertex_normals = np.zeros((len(points), 3), np.float64)
        # 面面积和单位法线原先各自重算同一个 cross/length。合并计算但仍按
        # 原面序/角点序累加；顶点法线归一化前不会使用中间累加结果。
        for face in range(len(faces)):
            a, b, c = faces[face]
            ax = points[b, 0] - points[a, 0]
            ay = points[b, 1] - points[a, 1]
            az = points[b, 2] - points[a, 2]
            bx = points[c, 0] - points[a, 0]
            by = points[c, 1] - points[a, 1]
            bz = points[c, 2] - points[a, 2]
            nx = ay * bz - az * by
            ny = az * bx - ax * bz
            nz = ax * by - ay * bx
            length = math.sqrt(nx * nx + ny * ny + nz * nz)
            area = 0.5 * length / 3.0
            areas[a] += area
            areas[b] += area
            areas[c] += area
            if length < 1e-11:
                nx, ny, nz = 0.0, 0.0, 0.0
            else:
                inverse = 1.0 / length
                nx, ny, nz = inverse * nx, inverse * ny, inverse * nz
            for local in range(3):
                vertex = faces[face, local]
                vertex_normals[vertex, 0] += nx
                vertex_normals[vertex, 1] += ny
                vertex_normals[vertex, 2] += nz
        for normal in vertex_normals:
            length = math.sqrt(normal[0] * normal[0] + normal[1] * normal[1]
                               + normal[2] * normal[2])
            inverse = 1.0 / length
            normal[0] *= inverse
            normal[1] *= inverse
            normal[2] *= inverse
        for vertex in range(len(points)):
            if boundary[vertex]:
                continue
            start, end = offsets[vertex], offsets[vertex + 1]
            if start == end:
                continue
            gx, gy, gz = 0.0, 0.0, 0.0
            total = 0.0
            for p in range(start, end):
                other = neighbours[p]
                weight = areas[other]
                total += weight
                gx += weight * points[other, 0]
                gy += weight * points[other, 1]
                gz += weight * points[other, 2]
            inverse = 1.0 / total
            dx = inverse * gx - points[vertex, 0]
            dy = inverse * gy - points[vertex, 1]
            dz = inverse * gz - points[vertex, 2]
            normal = vertex_normals[vertex]
            for i in range(3):
                t0 = (1.0 if i == 0 else 0.0) - normal[i] * normal[0]
                t1 = (1.0 if i == 1 else 0.0) - normal[i] * normal[1]
                t2 = (1.0 if i == 2 else 0.0) - normal[i] * normal[2]
                movement = (t0 * dx + t1 * dy) + t2 * dz
                points[vertex, i] += 0.99 * movement


def smooth(mesh, repeats=2):
    """对 Mesh 执行顺序切向平滑，原位更新 points，默认两轮。

    points 为 surface RAS 的 mm 坐标；faces 和 vertex_edges 采用原有顶点编号。
    本次调用内拓扑固定，因此只构建一次升序 CSR 邻接；每轮的几何量仍重算。
    输出仍为 Mesh.points 的 float64 坐标三元组列表；无返回值。repeats<=0
    不修改输入。退化法线或零邻接总面积抛 ZeroDivisionError；Numba 首调用
    编译包含在阶段耗时内。对应 mris_remesh 的内部平滑，无独立官方命令。
    """
    if repeats <= 0:
        return
    points = np.asarray(mesh.points, np.float64)
    faces = np.asarray(mesh.faces, np.int32)
    rows = [sorted(row) for row in mesh.vertex_edges]
    offsets = np.zeros(len(rows) + 1, np.int64)
    offsets[1:] = np.cumsum([len(row) for row in rows], dtype=np.int64)
    neighbours = np.fromiter((other for row in rows for other in row),
                             dtype=np.int32, count=int(offsets[-1]))
    _smooth_ordered(points, faces, offsets, neighbours,
                    np.asarray(mesh.onboundary, np.bool_), repeats)
    mesh.points = [tuple(row) for row in points]



def remesh_geometry(vertices: np.ndarray, faces: np.ndarray, iterations: int = 3
                    ) -> tuple[np.ndarray, np.ndarray]:
    """按固定上游拆边、缩边及顺序平滑规则重新划分三角网格。

    vertices 是 (N,3) 有限 surface RAS/mm 坐标，faces 是 (F,3) 有序三角面
    顶点索引；内部坐标为 float64，iterations 默认 3 且不得为负。目标边长
    固定为初始平均边长的 0.8，不改变拆边/缩边阈值或顺序。
    返回 float32(Nnew,3) 坐标与 int32(Fnew,3) 面，保留算法生成的编号顺序；
    不读写文件、不单独修复相交。非法迭代数抛 ValueError，索引/非流形拓扑
    和退化法线抛原有异常。对应 mris_remesh --remesh --iters 3。
    """
    if iterations < 0:
        raise ValueError("iterations must be nonnegative")
    mesh = Mesh([tuple(row) for row in np.asarray(vertices, dtype=np.float64)],
                np.asarray(faces, dtype=np.int32).tolist())
    total_length = 0.0
    for a, b in mesh.edge_vertices:
        total_length += edge_length(mesh.points, a, b)
    target = 0.8 * total_length / len(mesh.edge_vertices)
    for _ in range(iterations):
        edge_index, edge_vertices, edge_faces, face_edges = initial_topology(mesh.faces)
        while split_pass(mesh.points, mesh.faces, edge_index, edge_vertices,
                         edge_faces, face_edges, target * 4.0 / 3.0):
            pass
        mesh = Mesh(mesh.points, mesh.faces)
        while mesh.collapse_pass(target * 4.0 / 5.0):
            pass
        smooth(mesh)
    return (np.asarray(mesh.points, dtype=np.float32),
            np.asarray(mesh.faces, dtype=np.int32))


def _footer_offset(path: str | Path) -> int:
    with open(path, "rb") as stream:
        if stream.read(3) != b"\xff\xff\xfe":
            raise ValueError("expected FreeSurfer triangle surface")
        stream.readline()
        stream.readline()
        nv, nf = struct.unpack(">ii", stream.read(8))
        return stream.tell() + 12 * (nv + nf)


def _copy_footer(input_path: str | Path, output_path: str | Path) -> None:
    with open(input_path, "rb") as stream:
        stream.seek(_footer_offset(input_path))
        footer = stream.read()
    with open(output_path, "r+b") as stream:
        stream.seek(_footer_offset(output_path))
        stream.truncate()
        stream.write(footer)


def remesh_surface(input_path: str | Path, output_path: str | Path,
                   iterations: int = 3) -> None:
    """读取三角表面，重划分后写出坐标、面和输入的原始几何尾部。

    input_path/output_path 为文件路径；surface RAS 坐标单位 mm，iterations
    默认 3。使用 nibabel 读写 float32 坐标/int32 面，随后保留 volume-info
    等尾部；返回 None。输入格式、算法或文件读写失败会抛异常，失败文件不
    应视为有效输出。对应 mris_remesh --remesh --iters 3 INPUT OUTPUT。
    """
    vertices, faces = fs.read_geometry(input_path)
    result_vertices, result_faces = remesh_geometry(vertices, faces, iterations)
    fs.write_geometry(output_path, result_vertices, result_faces)
    _copy_footer(input_path, output_path)
