"""Verified geometry update used by spherical registration line minimization."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import numpy as np
import torch

from .mris_register_kernels import project_sphere
from .mris_register_objective import _atan_table
from .mris_register_average_numba import three_hop_neighbor_total
from .place_surface_normals import ordered_face_csr


@torch.no_grad()
def ordered_neighbors_from_faces(faces: torch.Tensor, nvertices: int
                                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """Build FreeSurfer's ordered one-ring from the input face order."""
    rows: list[list[int]] = [[] for _ in range(nvertices)]
    for a, b, c in faces.cpu().tolist():
        for vertex, first, second in ((a, c, b), (b, a, c), (c, b, a)):
            row = rows[vertex]
            if first not in row:
                row.append(first)
            if second not in row:
                row.append(second)
    degrees = torch.tensor([len(row) for row in rows], dtype=torch.int64)
    neighbors = torch.zeros((nvertices, int(degrees.max())), dtype=torch.int64)
    for vertex, row in enumerate(rows):
        neighbors[vertex, :len(row)] = torch.tensor(row, dtype=torch.int64)
    return neighbors.to(faces.device), degrees.to(faces.device)


@dataclass(frozen=True)
class RegistrationForceCache:
    neighbors: torch.Tensor
    degrees: torch.Tensor
    original_distances: torch.Tensor
    original_areas: torch.Tensor
    three_hop_neighbors: float
    orig_area: float
    incidence: tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@torch.no_grad()
def ordered_face_incidence(faces: torch.Tensor, nvertices: int
                           ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """保留每顶点 face/corner 次序，返回用于配准法向的整数表。

    ``faces`` 是任意 CPU/CUDA 设备上的 (F,3) 整数三角面张量；
    ``nvertices`` 为非负总顶点数，包含孤立顶点。共享 NumPy CSR 仅在
    CPU 建立一次，再向量化填入表并返回原 faces.device。返回 int64
    ``face_indices``、``corner_indices`` (N,K) 和 ``degrees`` (N,)，
    K 为最大关联面数量；仅前 degree 列有效，其余列填零。重复角点及
    原面序完整保留；无面时 K=0。不改变浮点累计、坐标空间或单位，
    没有算法默认参数。形状、dtype、顶点数或索引非法时抛 ValueError。
    属于 mris_register 内部准备阶段，无独立官方 CLI。
    """
    offsets, face_ids, corners = ordered_face_csr(
        faces.detach().cpu().numpy(), nvertices=nvertices)
    degrees = np.diff(offsets)
    width = int(degrees.max(initial=0))
    face_indices = np.zeros((nvertices, width), dtype=np.int64)
    corner_indices = np.zeros_like(face_indices)
    rows = np.repeat(np.arange(nvertices, dtype=np.int64), degrees)
    columns = np.arange(len(face_ids), dtype=np.int64) - offsets[rows]
    face_indices[rows, columns] = face_ids
    corner_indices[rows, columns] = corners
    return (torch.from_numpy(face_indices).to(faces.device),
            torch.from_numpy(corner_indices).to(faces.device),
            torch.from_numpy(degrees).to(faces.device))


@torch.no_grad()
def original_chord_distances(positions: torch.Tensor, neighbors: torch.Tensor,
                             degrees: torch.Tensor) -> torch.Tensor:
    """Ordered original-surface edge lengths used by registration."""
    displacement = positions[:, None, :] - positions[neighbors]
    squared = (displacement[..., 0] * displacement[..., 0]
               + displacement[..., 1] * displacement[..., 1]
               + displacement[..., 2] * displacement[..., 2])
    lengths = squared.double().sqrt().float()
    active = torch.arange(neighbors.shape[1], device=positions.device)[None, :] < degrees[:, None]
    return torch.where(active, lengths, 0.0)


@torch.no_grad()
def sphere_arc_distances(positions: torch.Tensor, neighbors: torch.Tensor,
                         degrees: torch.Tensor) -> torch.Tensor:
    """FreeSurfer's small-angle sphere distance in native float32 order."""
    xyz = positions.double()
    radius = (xyz[:, 0] * xyz[:, 0] + xyz[:, 1] * xyz[:, 1]
              + xyz[:, 2] * xyz[:, 2]).sqrt().float()
    normalized = positions * (1.0 / radius)[:, None]
    first = normalized[:, None, :].double()
    second = positions[neighbors].double()
    dot = (first[..., 0] * second[..., 0] + first[..., 1] * second[..., 1]
           + first[..., 2] * second[..., 2]).float()
    norm = torch.maximum(radius[neighbors], dot.abs())
    ratio = (dot / norm).double()
    angle = torch.where(ratio < 0.99, torch.acos(ratio),
                        torch.sqrt(2.0 * (1.0 - ratio))).float()
    distances = angle * radius[:, None]
    active = torch.arange(neighbors.shape[1], device=positions.device)[None, :] < degrees[:, None]
    return torch.where(active, distances, 0.0)


@torch.no_grad()
def sphere_vertex_normals(positions: torch.Tensor,
                          faces: torch.Tensor, *,
                          incidence: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
                          ) -> torch.Tensor:
    """Default FreeSurfer metric-properties corner normals in face order."""
    corner0 = faces[:, [2, 0, 1]]
    corner1 = faces[:, [1, 2, 0]]

    def unit(vectors: torch.Tensor) -> torch.Tensor:
        squared = (vectors[..., 0] * vectors[..., 0]
                   + vectors[..., 1] * vectors[..., 1]
                   + vectors[..., 2] * vectors[..., 2])
        return vectors / squared.double().sqrt().float()[..., None]

    first = unit(positions[faces] - positions[corner0])
    second = unit(positions[corner1] - positions[faces])
    corner_normals = torch.empty_like(first)
    corner_normals[..., 0] = -second[..., 1] * first[..., 2] + first[..., 1] * second[..., 2]
    corner_normals[..., 1] = second[..., 0] * first[..., 2] - first[..., 0] * second[..., 2]
    corner_normals[..., 2] = -second[..., 0] * first[..., 1] + first[..., 0] * second[..., 1]
    corner_normals = unit(corner_normals)

    if incidence is None:
        face_indices, corner_indices, degrees = ordered_face_incidence(
            faces.to(positions.device), len(positions))
    else:
        face_indices, corner_indices, degrees = incidence
    width = face_indices.shape[1]
    normals = torch.zeros_like(positions)
    for index in range(width):
        normals = torch.where((degrees > index)[:, None],
                              normals + corner_normals[face_indices[:, index], corner_indices[:, index]],
                              normals)
    return unit(normals)


def three_hop_avg_nbrs(neighbors: torch.Tensor, degrees: torch.Tensor) -> float:
    """返回三跳可达邻点数的顶点均值，沿用 float32 存储和舍入顺序。

    neighbors 为 (N, K) 的有序 int64 邻接；degrees 为 (N,) 有效列数。
    自身不计入邻点，重复路径只计一次；输出是 Python float。整数 BFS
    在 CPU Numba 执行，CUDA 输入只将静态邻接搬回一次；不缓存不同网格。
    调用者提供合法非空邻接；属于 mris_register 内部距离权重准备步骤。
    """
    total = three_hop_neighbor_total(neighbors.cpu().numpy(), degrees.cpu().numpy())
    return _float32(_float32(total) / _float32(len(degrees)))


@torch.no_grad()
def registration_orig_area(input_sphere: torch.Tensor, faces: torch.Tensor) -> float:
    """FreeSurfer's preserved original sphere face-area sum."""
    first = input_sphere[faces[:, 1]] - input_sphere[faces[:, 0]]
    second = input_sphere[faces[:, 2]] - input_sphere[faces[:, 0]]
    x = first[:, 1] * second[:, 2] - first[:, 2] * second[:, 1]
    y = first[:, 2] * second[:, 0] - first[:, 0] * second[:, 2]
    z = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    face_areas = (x * x + y * y + z * z).double().sqrt().float() * 0.5
    return _float32(float(face_areas.double().sum()))


def registration_total_area() -> float:
    """Theoretical area of FreeSurfer's 100 mm canonical registration sphere."""
    return _float32(4.0 * math.pi * 100.0 * 100.0)


@torch.no_grad()
def first_distance_gradient(input_sphere: torch.Tensor,
                            original_surface: torch.Tensor,
                            registered_sphere: torch.Tensor,
                            faces: torch.Tensor, *, project: bool = True,
                            weight: float = 5.0,
                            cache: RegistrationForceCache | None = None) -> torch.Tensor:
    """Distance-force gradient from the frozen registration surface inputs."""
    input_sphere = input_sphere.float()
    original_surface = original_surface.float()
    registered_sphere = (project_sphere(registered_sphere.float()) if project
                         else registered_sphere.float())
    if cache is None:
        neighbors, degrees = ordered_neighbors_from_faces(faces, len(input_sphere))
        normals = sphere_vertex_normals(registered_sphere, faces)
        current = sphere_arc_distances(registered_sphere, neighbors, degrees)
        original = original_chord_distances(original_surface, neighbors, degrees)
        return distance_gradient(registered_sphere, normals, neighbors, degrees,
                                 current, original, three_hop_avg_nbrs(neighbors, degrees),
                                 registration_orig_area(input_sphere, faces),
                                 registration_total_area(), weight)
    normals = sphere_vertex_normals(
        registered_sphere, faces, incidence=cache.incidence)
    current = sphere_arc_distances(registered_sphere, cache.neighbors, cache.degrees)
    return distance_gradient(
        registered_sphere, normals, cache.neighbors, cache.degrees,
        current, cache.original_distances, cache.three_hop_neighbors,
        cache.orig_area, registration_total_area(), weight)


@torch.no_grad()
def face_area_normals(positions: torch.Tensor, faces: torch.Tensor,
                      *, signed_sphere: bool = False
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """Triangle areas and unit normals; orient spherical faces outward."""
    first = positions[faces[:, 1]] - positions[faces[:, 0]]
    second = positions[faces[:, 2]] - positions[faces[:, 0]]
    cross = torch.stack((first[:, 1] * second[:, 2] - first[:, 2] * second[:, 1],
                         first[:, 2] * second[:, 0] - first[:, 0] * second[:, 2],
                         first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]), dim=1)
    squared = (cross[:, 0] * cross[:, 0] + cross[:, 1] * cross[:, 1]
               + cross[:, 2] * cross[:, 2])
    length = squared.double().sqrt().float()
    if signed_sphere:
        centroid = positions[faces].sum(dim=1)
        orientation = torch.where((centroid * cross).sum(dim=1) < 0.0, -1.0, 1.0)
        return length * 0.5 * orientation, cross * (orientation / length)[:, None]
    return length * 0.5, cross * (1.0 / length)[:, None]


@torch.no_grad()
def area_gradient_add(gradient: torch.Tensor, positions: torch.Tensor,
                      faces: torch.Tensor, current_areas: torch.Tensor,
                      original_areas: torch.Tensor, face_normals: torch.Tensor,
                      orig_area: float, total_area: float, *,
                      l_nlarea: float = 1.0, l_parea: float = 0.2,
                      incidence: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
                      ) -> torch.Tensor:
    """Add nonlinear and percentage area forces in native face order."""
    first = positions[faces[:, 1]] - positions[faces[:, 0]]
    second = positions[faces[:, 2]] - positions[faces[:, 0]]

    def cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.stack((a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
                            a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
                            a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]), dim=1)

    first_cross = cross(first, face_normals)
    second_cross = cross(second, face_normals)
    corner0 = -first_cross + second_cross
    scale = _float32(_float32(orig_area) / _float32(total_area))
    scaled_nonlinear = float(scale) * current_areas.double()
    ratio = scaled_nonlinear.clamp(-40.0, 40.0)
    delta_nonlinear = (_float32(l_nlarea) / (1.0 + torch.exp(10.0 * ratio))) * (
        scaled_nonlinear - original_areas.double())
    nonlinear = torch.stack(((corner0.double() * delta_nonlinear[:, None]).float(),
                             (-second_cross.double() * delta_nonlinear[:, None]).float(),
                             (first_cross.double() * delta_nonlinear[:, None]).float()), dim=1)
    delta_percent = _float32(l_parea) * (current_areas * scale - original_areas)
    percent = torch.stack((corner0 * delta_percent[:, None],
                           second_cross * (-delta_percent)[:, None],
                           first_cross * delta_percent[:, None]), dim=1)
    if incidence is None:
        face_indices, corner_indices, degrees = ordered_face_incidence(
            faces.to(positions.device), len(positions))
    else:
        face_indices, corner_indices, degrees = incidence
    width = face_indices.shape[1]
    for contributions in (nonlinear, percent):
        for index in range(width):
            gradient = torch.where((degrees > index)[:, None],
                                   gradient + contributions[face_indices[:, index], corner_indices[:, index]],
                                   gradient)
    return gradient


@torch.no_grad()
def first_area_gradient(input_sphere: torch.Tensor,
                        original_surface: torch.Tensor,
                        registered_sphere: torch.Tensor, faces: torch.Tensor,
                        gradient_after_distance: torch.Tensor, *,
                        project: bool = True, l_nlarea: float = 1.0,
                        l_parea: float = 0.2,
                        cache: RegistrationForceCache | None = None) -> torch.Tensor:
    """First area-force update following the native-free distance gradient."""
    input_sphere = input_sphere.float()
    original_surface = original_surface.float()
    positions = (project_sphere(registered_sphere.float()) if project
                 else registered_sphere.float())
    current_areas, face_normals = face_area_normals(positions, faces, signed_sphere=True)
    if cache is None:
        original_areas, _ = face_area_normals(original_surface, faces)
        orig_area = registration_orig_area(input_sphere, faces)
        incidence = None
    else:
        original_areas, orig_area = cache.original_areas, cache.orig_area
        incidence = cache.incidence
    return area_gradient_add(gradient_after_distance, positions, faces,
                             current_areas, original_areas, face_normals,
                             orig_area, registration_total_area(),
                             l_nlarea=l_nlarea, l_parea=l_parea,
                             incidence=incidence)


@torch.no_grad()
def prepare_registration_force_cache(input_sphere: torch.Tensor,
                                     original_surface: torch.Tensor,
                                     faces: torch.Tensor) -> RegistrationForceCache:
    """Prepare static topology and original-surface terms once per hemisphere."""
    neighbors, degrees = ordered_neighbors_from_faces(faces, len(input_sphere))
    original_distances = original_chord_distances(original_surface, neighbors, degrees)
    original_areas, _ = face_area_normals(original_surface, faces)
    return RegistrationForceCache(
        neighbors, degrees, original_distances, original_areas,
        three_hop_avg_nbrs(neighbors, degrees),
        registration_orig_area(input_sphere, faces),
        ordered_face_incidence(faces, len(input_sphere)))


@torch.no_grad()
def average_gradients_once(gradient: torch.Tensor, neighbors: torch.Tensor,
                           degrees: torch.Tensor) -> torch.Tensor:
    """One source-order float32 MRISaverageGradients iteration without rips."""
    averaged = gradient.clone()
    for index in range(neighbors.shape[1]):
        averaged = torch.where((degrees > index)[:, None],
                               averaged + gradient[neighbors[:, index]], averaged)
    return averaged * (1.0 / (degrees + 1).to(gradient.dtype))[:, None]


@torch.no_grad()
def average_gradients(gradient: torch.Tensor, neighbors: torch.Tensor,
                      degrees: torch.Tensor, iterations: int) -> torch.Tensor:
    """Repeat FreeSurfer's source-order one-ring gradient average."""
    for _ in range(iterations):
        gradient = average_gradients_once(gradient, neighbors, degrees)
    return gradient



@torch.no_grad()
def spring_gradient_add(gradient: torch.Tensor, positions: torch.Tensor,
                        neighbors: torch.Tensor, degrees: torch.Tensor,
                        dist_scale: torch.Tensor, l_spring: float) -> torch.Tensor:
    """Add the smoothwm registration spring after gradient averaging."""
    displacement = torch.zeros_like(positions)
    for index in range(neighbors.shape[1]):
        active = degrees > index
        delta = positions[neighbors[:, index]] - positions
        displacement = torch.where(active[:, None], displacement + delta, displacement)
    displacement = ((dist_scale * displacement) / degrees[:, None].float()).float()
    return gradient + (displacement.double() * _float32(l_spring)).float()


def _float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


@torch.no_grad()
def distance_gradient(positions: torch.Tensor, normals: torch.Tensor,
                      neighbors: torch.Tensor, degrees: torch.Tensor,
                      current_distances: torch.Tensor, original_distances: torch.Tensor,
                      avg_nbrs: float, orig_area: float, total_area: float,
                      weight: float) -> torch.Tensor:
    """First source-order spherical registration distance force, no ripped vertices.

    Distances have shape ``(vertices, max_neighbors)`` in the same neighbor
    order. They use FreeSurfer's current sphere arcs and original-surface edges.
    """
    scale = _float32(math.sqrt(_float32(_float32(orig_area) / _float32(total_area))))
    norm = _float32(1.0 / _float32(avg_nbrs))
    delta_sum = torch.zeros_like(positions)
    for index in range(neighbors.shape[1]):
        displacement = positions[neighbors[:, index]] - positions
        squared = (displacement[:, 0] * displacement[:, 0]
                   + displacement[:, 1] * displacement[:, 1]
                   + displacement[:, 2] * displacement[:, 2])
        length = squared.double().sqrt().float()
        direction = displacement * (1.0 / length)[:, None]
        mismatch = current_distances[:, index] - original_distances[:, index] / scale
        update = direction * mismatch[:, None]
        delta_sum = torch.where((degrees > index)[:, None], delta_sum + update, delta_sum)
    delta_sum = delta_sum * norm
    normal_component = (normals[:, 0] * delta_sum[:, 0]
                        + normals[:, 1] * delta_sum[:, 1]
                        + normals[:, 2] * delta_sum[:, 2])
    return _float32(weight) * (delta_sum + normals * (-normal_component)[:, None])


def apply_spherical_gradient(positions: torch.Tensor,
                             gradient: torch.Tensor, dt: float) -> torch.Tensor:
    """Apply one native float32-stored gradient step and radial projection."""
    translated = (positions.double() + float(dt) * gradient.double()).float()
    return project_sphere(translated)


@torch.no_grad()
def tangent_basis(normals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Source-order tangent axes used by the scalar curvature force."""
    n = normals.float()

    def cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.stack((a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
                            a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
                            a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]), dim=1)

    def normalize(a: torch.Tensor) -> torch.Tensor:
        squared = a[:, 0] * a[:, 0] + a[:, 1] * a[:, 1] + a[:, 2] * a[:, 2]
        length = torch.sqrt(squared.double()).float()
        return a * (1.0 / length)[:, None]

    e1 = cross(n, n[:, [1, 2, 0]])
    squared = e1[:, 0] * e1[:, 0] + e1[:, 1] * e1[:, 1] + e1[:, 2] * e1[:, 2]
    fallback = torch.sqrt(squared.double()).float() < 0.001
    alternative = cross(n, torch.stack((n[:, 1], -n[:, 2], n[:, 0]), dim=1))
    e1 = torch.where(fallback[:, None], alternative, e1)
    e2 = cross(n, e1)
    return normalize(e1), normalize(e2)


def _correlation_atan2(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ax, ay = x.abs(), y.abs()
    major_x = ax >= ay
    low = torch.where(major_x, ay, ax)
    high = torch.where(major_x, ax, ay)
    index = torch.where(high == 0, 0, (100000 * low / high).int()).clamp(0, 100000)
    table = _atan_table(str(x.device))[index]
    half_pi = torch.tensor(math.pi / 2, dtype=torch.float32, device=x.device)
    pi = torch.tensor(math.pi, dtype=torch.float32, device=x.device).double()
    angle = torch.where(major_x, table, half_pi - table).double()
    angle = torch.where((x >= 0) & (y < 0), -angle, angle)
    angle = torch.where((x < 0) & (y >= 0), pi - angle, angle)
    angle = torch.where((x < 0) & (y < 0), -pi + angle, angle)
    return angle.float()


@torch.no_grad()
def sample_correlation_atlas(frame: torch.Tensor,
                             points: torch.Tensor) -> torch.Tensor:
    """FreeSurfer MRISPfunctionVal including its off-sphere correction."""
    xyz = points.float()
    x, y, z = xyz.unbind(1)
    x2, y2, z2 = x * x, y * y, z * z
    radius = torch.sqrt((x2 + y2 + z2).double())
    outside = (radius - 100.0).abs() >= torch.finfo(torch.float32).eps
    f = x2.double() / 1e12 + y2.double() / 1e12 + z2.double() / 1e12
    g = 2 * (x2.double() / 1e8 + y2.double() / 1e8 + z2.double() / 1e8)
    h = x2.double() / 1e4 + y2.double() / 1e4 + z2.double() / 1e4 - 1
    correction = (-g + torch.sqrt((g * g - 4 * f * h).clamp_min(0)).float().double()) / (2 * f)
    projected = (xyz.double() + correction[:, None] * xyz.double() / 1e4).float()
    xyz = torch.where(outside[:, None], projected, xyz)
    radius = torch.where(outside, torch.full_like(radius, 100.0), radius).float()
    x, y, z = xyz.unbind(1)
    v_dim, u_dim = frame.shape
    d = (radius * radius - z * z).clamp_min(0)
    phi = torch.atan2(torch.sqrt(d.double()).float(), z)
    uf = ((u_dim * phi).double() / math.pi).float()
    u0, u1 = torch.floor(uf).long(), torch.ceil(uf).long()
    du = uf - u0.float()
    offset0 = torch.where((u0 < 0) | (u0 >= u_dim), v_dim // 2, 0)
    offset1 = torch.where((u1 < 0) | (u1 >= u_dim), v_dim // 2, 0)
    u0 = torch.where(u0 < 0, -u0, u0)
    u1 = torch.where(u1 < 0, -u1, u1)
    u0 = torch.where(u0 >= u_dim, u_dim - (u0 - u_dim + 1), u0)
    u1 = torch.where(u1 >= u_dim, u_dim - (u1 - u_dim + 1), u1)
    theta = _correlation_atan2(y, x)
    theta = torch.where(theta < 0, (theta.double() + 2 * math.pi).float(), theta)
    theta = torch.where(theta >= 2 * math.pi, (theta.double() - 2 * math.pi).float(), theta)
    vf = ((v_dim * theta).double() / (2 * math.pi)).float()
    v0, v1 = torch.floor(vf).long() % v_dim, torch.ceil(vf).long() % v_dim
    dv = vf - torch.floor(vf)
    return (du * dv * frame[(v1 + offset1) % v_dim, u1]
            + (1 - du) * dv * frame[(v1 + offset0) % v_dim, u0]
            + (1 - du) * (1 - dv) * frame[(v0 + offset0) % v_dim, u0]
            + du * (1 - dv) * frame[(v0 + offset1) % v_dim, u1])


@torch.no_grad()
def correlation_gradient_add(gradient: torch.Tensor,
                             positions: torch.Tensor, curvature: torch.Tensor,
                             e1: torch.Tensor, e2: torch.Tensor,
                             target_mean: torch.Tensor, target_variance: torch.Tensor,
                             avg_vertex_dist: float, *, l_corr: float = 1.0) -> torch.Tensor:
    """Scalar curvature-correlation force with the active registration weight."""
    d_dist = 0.1 * avg_vertex_dist
    target = sample_correlation_atlas(target_mean, positions)
    std = sample_correlation_atlas(target_variance, positions).double().sqrt().float()
    std = torch.where(std.abs() < torch.finfo(torch.float32).eps,
                      torch.full_like(std, 4.0), std)
    coef = (((target.double() - curvature.double()) * _float32(l_corr)) / std.double()).float()
    u = (e1.double() * d_dist).float()
    v = (e2.double() * d_dist).float()
    up = sample_correlation_atlas(target_mean, positions + u)
    um = sample_correlation_atlas(target_mean, positions - u)
    vp = sample_correlation_atlas(target_mean, positions + v)
    vm = sample_correlation_atlas(target_mean, positions - v)
    du = (up.double() - um.double()) / (2 * d_dist)
    dv = (vp.double() - vm.double()) / (2 * d_dist)
    change = coef.double()[:, None] * (du[:, None] * e1.double() + dv[:, None] * e2.double())
    return (gradient.double() - change).float()
