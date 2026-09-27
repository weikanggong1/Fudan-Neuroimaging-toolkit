"""Deterministic preflight for the FreeSurfer 8.2 topology correction stage.

The full ``mris_fix_topology -ga`` patch search is not implemented here.  This
module checks the ordered input mesh and reproduces its initial spherical
projection, five smoothing passes, and sphere centering.  It is intentionally
kept separate from the reconstruction entry point until defect discovery and
retessellation have matched the pinned native implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import time

import nibabel.freesurfer.io as fsio
import numpy as np
from numba import njit

from .inflate_center_python import center_vertices
from .smooth_surface_python import ordered_neighbors


@dataclass(frozen=True)
class TopologyCounts:
    vertices: int
    faces: int
    edges: int
    euler: int
    boundary_edges: int
    nonmanifold_edges: int


def topology_counts(faces: np.ndarray, nvertices: int) -> TopologyCounts:
    """Count the same V-E+F quantities used by ``MRIScomputeEulerNumber``.

    Native FreeSurfer counts all unripped vertices and faces and unique one-ring
    edges.  Surface files have no rip flags, so each vertex and face is active.
    The edge incidence counts expose open or nonmanifold inputs separately.
    """
    triangles = np.asarray(faces, dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("faces must have shape (n, 3)")
    if np.any(triangles < 0) or np.any(triangles >= nvertices):
        raise ValueError("face index outside vertex range")
    edge_pairs = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]],
                                 triangles[:, [2, 0]]), axis=0)
    edge_pairs.sort(axis=1)
    _, incidence = np.unique(edge_pairs, axis=0, return_counts=True)
    nedges = len(incidence)
    return TopologyCounts(nvertices, len(triangles), nedges,
                          nvertices - nedges + len(triangles),
                          int(np.count_nonzero(incidence == 1)),
                          int(np.count_nonzero(incidence > 2)))


@njit
def _project_sphere_point(x: np.float32, y: np.float32, z: np.float32):
    # mrisSphericalProjectXYZ performs two projections, rounding to float32
    # between them because the coordinates are stored in VERTEX fields.
    # Its float32 squared norm enters a double sqrt; Numba needs explicit widening.
    squared = np.float32(x * x)
    squared = np.float32(squared + np.float32(y * y))
    squared = np.float32(squared + np.float32(z * z))
    length = np.sqrt(np.float64(squared))
    scale = 100.0 / length
    x = np.float32(float(x) * scale)
    y = np.float32(float(y) * scale)
    z = np.float32(float(z) * scale)
    squared = np.float32(x * x)
    squared = np.float32(squared + np.float32(y * y))
    squared = np.float32(squared + np.float32(z * z))
    length = np.sqrt(np.float64(squared))
    scale = 100.0 / length
    return np.float32(float(x) * scale), np.float32(float(y) * scale), np.float32(float(z) * scale)


@njit
def _smooth_on_sphere(xyz: np.ndarray, indices: np.ndarray,
                      degree: np.ndarray, iterations: int) -> np.ndarray:
    current = xyz.copy()
    for _ in range(iterations):
        averaged = np.empty_like(current)
        for vertex in range(len(current)):
            x = np.float32(0)
            y = np.float32(0)
            z = np.float32(0)
            for slot in range(degree[vertex]):
                neighbor = indices[vertex, slot]
                x = np.float32(x + current[neighbor, 0])
                y = np.float32(y + current[neighbor, 1])
                z = np.float32(z + current[neighbor, 2])
            if degree[vertex] == 0:
                averaged[vertex] = current[vertex]
            else:
                denominator = np.float32(degree[vertex])
                averaged[vertex, 0] = np.float32(x / denominator)
                averaged[vertex, 1] = np.float32(y / denominator)
                averaged[vertex, 2] = np.float32(z / denominator)
        for vertex in range(len(current)):
            current[vertex, 0], current[vertex, 1], current[vertex, 2] = _project_sphere_point(
                averaged[vertex, 0], averaged[vertex, 1], averaged[vertex, 2])
    return current


def project_and_smooth_sphere(vertices: np.ndarray, faces: np.ndarray,
                              initial_center: bool = True) -> np.ndarray:
    """Reproduce the pre-defect canonical sphere from ``qsphere.nofix``."""
    xyz = np.asarray(vertices, dtype=np.float32)
    if initial_center:
        xyz = center_vertices(xyz)
    distance = np.sqrt(np.sum(xyz.astype(np.float64) ** 2, axis=1))
    ratio = np.divide(100.0, distance, out=np.zeros_like(distance), where=distance > 0)
    xyz = (xyz.astype(np.float64) * ratio[:, None]).astype(np.float32)
    neighbors = ordered_neighbors(faces, len(xyz))
    degrees = np.fromiter((len(row) for row in neighbors), np.int32, count=len(xyz))
    indices = np.zeros((len(xyz), int(degrees.max(initial=0))), np.int32)
    for vertex, row in enumerate(neighbors):
        indices[vertex, :len(row)] = row
    return _smooth_on_sphere(xyz, indices, degrees, 5)


@njit
def _radius_energy(xyz: np.ndarray, center: np.ndarray, radius_squared: float) -> float:
    result = 0.0
    for n in range(len(xyz)):
        dx = float(xyz[n, 0]) - center[0]
        dy = float(xyz[n, 1]) - center[1]
        dz = float(xyz[n, 2]) - center[2]
        delta = dx * dx + dy * dy + dz * dz - radius_squared
        result += delta * delta
    return result


@njit
def _radius_squared(xyz: np.ndarray, center: np.ndarray) -> float:
    result = 0.0
    for n in range(len(xyz)):
        dx = float(xyz[n, 0]) - center[0]
        dy = float(xyz[n, 1]) - center[1]
        dz = float(xyz[n, 2]) - center[2]
        result += dx * dx + dy * dy + dz * dz
    return result / len(xyz)


@njit
def _radius_gradient(xyz: np.ndarray, center: np.ndarray,
                     radius_squared: float) -> np.ndarray:
    gradient = np.zeros(3, np.float64)
    for n in range(len(xyz)):
        dx = float(xyz[n, 0]) - center[0]
        dy = float(xyz[n, 1]) - center[1]
        dz = float(xyz[n, 2]) - center[2]
        residual = dx * dx + dy * dy + dz * dz - radius_squared
        gradient[0] += residual * dx
        gradient[1] += residual * dy
        gradient[2] += residual * dz
    return gradient / len(xyz)


def center_sphere(vertices: np.ndarray) -> tuple[np.ndarray, int]:
    """Translate ``MRIScenterSphere_old`` through its double precision search."""
    xyz = np.asarray(vertices, np.float32)
    center = (xyz.min(axis=0).astype(np.float64)
              + xyz.max(axis=0).astype(np.float64)) / 2.0
    radius_squared = _radius_squared(xyz, center)
    energy = _radius_energy(xyz, center, radius_squared)
    zero = np.zeros(3, np.float64)
    if _radius_energy(xyz, zero, 10000.0) < energy:
        center = zero.copy()
        radius_squared = 10000.0
        energy = _radius_energy(xyz, center, radius_squared)
    zero_radius = _radius_squared(xyz, zero)
    if _radius_energy(xyz, zero, zero_radius) < energy:
        center = zero.copy()
        radius_squared = zero_radius
        energy = _radius_energy(xyz, center, radius_squared)
    previous = energy + 1.0
    iterations = 0
    while energy < previous and iterations < 100:
        iterations += 1
        previous = energy
        gradient = _radius_gradient(xyz, center, radius_squared)
        norm = np.linalg.norm(gradient)
        if norm > 1:
            gradient /= norm
            norm = 1.0
        step = 2.0
        while energy >= previous:
            step /= 2.0
            candidate = center + step * gradient
            energy = _radius_energy(xyz, candidate, _radius_squared(xyz, candidate))
            if step * norm < 1e-11:
                break
        if energy < previous:
            center += step * gradient
            radius_squared = _radius_squared(xyz, center)
        else:
            energy = _radius_energy(xyz, center, radius_squared)
    scaled = ((xyz.astype(np.float64) - center) * (100.0 / np.sqrt(radius_squared))).astype(np.float32)
    for _ in range(2):
        squared = scaled[:, 0] * scaled[:, 0]
        squared += scaled[:, 1] * scaled[:, 1]
        squared += scaled[:, 2] * scaled[:, 2]
        distance = np.sqrt(squared.astype(np.float64))
        scaled = (scaled.astype(np.float64) * (100.0 / distance)[:, None]).astype(np.float32)
    return scaled, iterations


def preflight_surface(path: str | Path) -> tuple[TopologyCounts, int]:
    vertices, faces = fsio.read_geometry(str(path))
    counts = topology_counts(faces, len(vertices))
    _, iterations = center_sphere(project_and_smooth_sphere(vertices, faces))
    return counts, iterations


def write_centered_topology_sphere(input_qsphere: str | Path,
                                   output_centered: str | Path) -> dict:
    """Write the exact post-centering sphere used by topology defect search.

    Read one qsphere.nofix triangle surface and write a surface with the same
    ordered faces and complete footer. The report gives input/output paths,
    mesh counts, centering iterations and elapsed seconds. No defect
    correction or additional file is produced.
    """
    started = time.perf_counter()
    source, output = Path(input_qsphere), Path(output_centered)
    if source.resolve() == output.resolve():
        raise ValueError("centered topology sphere must use a separate output")
    vertices, faces = fsio.read_geometry(str(source))
    centered, iterations = center_sphere(project_and_smooth_sphere(vertices, faces))
    raw = source.read_bytes()
    if raw[:3] != b"\xff\xff\xfe":
        raise ValueError("expected FreeSurfer triangular surface")
    start = raw.index(b"\n\n", 3) + 2
    nvertices = int.from_bytes(raw[start:start + 4], "big")
    if nvertices != len(centered):
        raise ValueError("surface vertex count changed during centering")
    xyz_start = start + 8
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(raw[:xyz_start] + centered.astype(">f4").tobytes()
                       + raw[xyz_start + 12 * nvertices:])
    return {"input": str(source), "output": str(output),
            "vertices": len(vertices), "faces": len(faces),
            "iterations": iterations, "seconds": time.perf_counter() - started}


@njit
def _edge_intersects(xyz: np.ndarray, a: int, b: int, c: int, d: int) -> bool:
    """Pinned ``edgesIntersect`` sphere predicate and near-zero fallback."""
    if a == c or a == d or b == c or b == d:
        return False
    u0 = xyz[a].astype(np.float64)
    u1 = xyz[b].astype(np.float64)
    u2 = xyz[c].astype(np.float64)
    u3 = xyz[d].astype(np.float64)
    n0 = np.cross(u0, u1)
    n1 = np.cross(u2, u3)
    a0 = np.dot(u0, n1)
    a1 = np.dot(u1, n1)
    a2 = np.dot(u2, n0)
    a3 = np.dot(u3, n0)
    if (abs(a0) < 1e-7 or abs(a1) < 1e-7
            or abs(a2) < 1e-7 or abs(a3) < 1e-7):
        origin0 = (u0 + u1) / 2.0
        origin1 = (u2 + u3) / 2.0
        n0 = np.cross(origin0, u1 - u0)
        n1 = np.cross(origin1, u3 - u2)
        a0 = np.dot(u0 - origin1, n1)
        a1 = np.dot(u1 - origin1, n1)
        a2 = np.dot(u2 - origin0, n0)
        a3 = np.dot(u3 - origin0, n0)
    product0 = a0 * a1
    product1 = a2 * a3
    if product0 > 0 or product1 > 0:
        return False
    if product0 < 0 or product1 < 0:
        return True
    midpoint_sum = u0 + u1 + u2 + u3
    n2 = np.cross(midpoint_sum, n0)
    projection0 = np.dot(u0, n2)
    projection1 = np.dot(u1, n2)
    projection2 = np.dot(u2, n2)
    projection3 = np.dot(u3, n2)
    return (max(projection2, projection3) >= min(projection0, projection1)
            and min(projection2, projection3) <= max(projection0, projection1))


@njit
def _crossing_edges(xyz: np.ndarray, edges: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    hit = np.zeros(len(edges), np.bool_)
    low = np.minimum(xyz[edges[:, 0]], xyz[edges[:, 1]])
    high = np.maximum(xyz[edges[:, 0]], xyz[edges[:, 1]])
    for pair in pairs:
        i, j = pair[0], pair[1]
        if i == j:
            continue
        if (low[i, 0] > high[j, 0] or low[j, 0] > high[i, 0]
                or low[i, 1] > high[j, 1] or low[j, 1] > high[i, 1]
                or low[i, 2] > high[j, 2] or low[j, 2] > high[i, 2]):
            continue
        if _edge_intersects(xyz, edges[i, 0], edges[i, 1],
                            edges[j, 0], edges[j, 1]):
            hit[i] = True
            hit[j] = True
    return hit


def ambiguous_faces(vertices: np.ndarray, faces: np.ndarray,
                    cell_width: float = 2.0) -> np.ndarray:
    """Identify faces participating in intersecting spherical edges.

    A spatial grid generates a superset of candidate edge pairs.  The same
    spherical predicate as the pinned source decides each candidate.  This is
    a defect-discovery probe; full FreeSurfer defect arbitration and its face
    hash traversal have not yet been reproduced.
    """
    xyz = np.asarray(vertices, np.float32)
    triangles = np.asarray(faces, np.int32)
    face_edges = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]],
                                 triangles[:, [2, 0]]), axis=0)
    face_edges.sort(axis=1)
    edges, inverse = np.unique(face_edges, axis=0, return_inverse=True)
    lower = np.floor((np.minimum(xyz[edges[:, 0]], xyz[edges[:, 1]]) + 100.0)
                     / cell_width).astype(np.int16)
    upper = np.floor((np.maximum(xyz[edges[:, 0]], xyz[edges[:, 1]]) + 100.0)
                     / cell_width).astype(np.int16)
    cells: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for edge in range(len(edges)):
        for x in range(lower[edge, 0], upper[edge, 0] + 1):
            for y in range(lower[edge, 1], upper[edge, 1] + 1):
                for z in range(lower[edge, 2], upper[edge, 2] + 1):
                    cells[x, y, z].append(edge)
    n_candidates = sum(len(row) * (len(row) - 1) // 2 for row in cells.values())
    pairs = np.empty((n_candidates, 2), np.int32)
    offset = 0
    for row in cells.values():
        if len(row) < 2:
            continue
        first, second = np.triu_indices(len(row), 1)
        ids = np.asarray(row, np.int32)
        size = len(first)
        pairs[offset:offset + size, 0] = ids[first]
        pairs[offset:offset + size, 1] = ids[second]
        offset += size
    edge_hit = _crossing_edges(xyz, edges, pairs)
    return np.any(edge_hit[inverse.reshape(3, -1).T], axis=1)


def defect_component_labels(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Assign source-order IDs to connected ambiguous-vertex components.

    The pinned build disables ``ADD_EXTRA_VERTICES`` and
    ``FIND_ENCLOSING_LOOP``, so ``MRISsegmentDefects`` starts one component
    at each lowest unmarked vertex and spreads over the face-order one-ring.
    This stops before the later retention, convex hull, and patch search.
    """
    triangles = np.asarray(faces, np.int32)
    marked = np.zeros(len(vertices), np.bool_)
    marked[triangles[ambiguous_faces(vertices, triangles)]] = True
    neighbors = ordered_neighbors(triangles, len(vertices))
    labels = np.zeros(len(vertices), np.int32)
    defect_number = 0
    for seed in np.flatnonzero(marked):
        if labels[seed]:
            continue
        defect_number += 1
        labels[seed] = defect_number
        queue = [int(seed)]
        for vertex in queue:
            for neighbor in neighbors[vertex]:
                if marked[neighbor] and labels[neighbor] == 0:
                    labels[neighbor] = defect_number
                    queue.append(neighbor)
    return labels


def defect_regions(labels: np.ndarray, faces: np.ndarray) -> list[tuple[list[int], list[int]]]:
    """Return source-order defect vertex and one-ring border lists."""
    component = np.asarray(labels, np.int32)
    neighbors = ordered_neighbors(np.asarray(faces, np.int32), len(component))
    regions = []
    for number in range(1, int(component.max(initial=0)) + 1):
        seed = int(np.flatnonzero(component == number)[0])
        vertices = [seed]
        found = {seed}
        for vertex in vertices:
            for neighbor in neighbors[vertex]:
                if component[neighbor] == number and neighbor not in found:
                    found.add(neighbor)
                    vertices.append(neighbor)
        border = []
        seen_border = set()
        for vertex in vertices:
            for neighbor in neighbors[vertex]:
                if component[neighbor] == 0 and neighbor not in seen_border:
                    seen_border.add(neighbor)
                    border.append(neighbor)
        regions.append((vertices, border))
    return regions


def defect_border_labels(labels: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Add each ordered border to its component's integer label map."""
    result = np.asarray(labels, np.int32).copy()
    for number, (_, border) in enumerate(defect_regions(labels, faces), start=1):
        result[border] = number
    return result


def defect_retention_status(canonical: np.ndarray, original: np.ndarray,
                            faces: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Mark initially retained (+1) or discarded (-1) defect vertices.

    This follows the pinned ``-ga`` pre-search checks: pairwise canonical
    distance, pairwise original distance, then inward-facing triangles.  The
    result corresponds to the native ``defect_status`` diagnostic and does not
    choose any replacement patch edges.
    """
    sphere = np.asarray(canonical, np.float32)
    orig = np.asarray(original, np.float32)
    triangles = np.asarray(faces, np.int32)
    points = sphere[triangles]
    normal = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    negative = np.sum(normal * np.sum(points, axis=1), axis=1, dtype=np.float32) < 0
    inward_vertex = np.zeros(len(sphere), np.bool_)
    inward_vertex[triangles[negative]] = True
    result = np.zeros(len(sphere), np.int8)
    for vertices, border in defect_regions(labels, triangles):
        discarded = np.zeros(len(vertices), np.bool_)
        all_vertices = vertices + border
        for coordinates, threshold in ((sphere, np.float32(0.01)),
                                       (orig, np.float32(0.75))):
            for i, first in enumerate(all_vertices):
                if i < len(vertices) and discarded[i]:
                    continue
                for j in range(i + 1, len(vertices)):
                    if discarded[j]:
                        continue
                    delta = coordinates[vertices[j]] - coordinates[first]
                    distance_squared = np.float32(delta[0] * delta[0]
                                                  + delta[1] * delta[1]
                                                  + delta[2] * delta[2])
                    if distance_squared < threshold:
                        discarded[j] = True
        indices = np.asarray(vertices, np.int32)
        discarded |= inward_vertex[indices]
        result[indices] = np.where(discarded, -1, 1)
    return result


def defect_hull_labels(labels: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Reproduce the pinned ``SMALL_CONVEX_HULL`` diagnostic labels."""
    component = np.asarray(labels, np.int32)
    neighbors = ordered_neighbors(np.asarray(faces, np.int32), len(component))
    result = defect_border_labels(component, faces)
    for number, (vertices, border) in enumerate(defect_regions(component, faces), start=1):
        occupied = set(vertices) | set(border)
        hull = list(border)
        for vertex in border:
            for neighbor in neighbors[vertex]:
                if neighbor not in occupied:
                    occupied.add(neighbor)
                    hull.append(neighbor)
        result[hull] = 2
        result[border] = -1
        result[vertices] = number
    return result


def genetic_base_translation(labels: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map source vertices/faces into the ordered pre-patch ``-ga`` surface.

    Source vertices outside defects come first in index order; defect vertices
    follow in component/BFS order, even when their search status is discard.
    Faces touching any defect vertex are removed before candidate patches.
    """
    component = np.asarray(labels, np.int32)
    triangles = np.asarray(faces, np.int32)
    order = list(np.flatnonzero(component == 0))
    for vertices, _ in defect_regions(component, triangles):
        order.extend(vertices)
    vertex_translation = np.empty(len(component), np.int32)
    vertex_translation[np.asarray(order, np.int32)] = np.arange(len(component), dtype=np.int32)
    face_translation = np.full(len(triangles), -1, np.int32)
    unaffected = np.all(component[triangles] == 0, axis=1)
    face_translation[unaffected] = np.arange(np.count_nonzero(unaffected), dtype=np.int32)
    return vertex_translation, face_translation


@njit
def _prune_against_base_edges(canonical: np.ndarray, edges: np.ndarray,
                              already_in_base: np.ndarray) -> np.ndarray:
    keep = np.ones(len(edges), np.bool_)
    for i in range(len(edges)):
        if not already_in_base[i]:
            continue
        for j in range(i + 1, len(edges)):
            if already_in_base[j] or not keep[j]:
                continue
            if _edge_intersects(canonical, edges[i, 0], edges[i, 1],
                                edges[j, 0], edges[j, 1]):
                keep[j] = False
    return keep


def genetic_candidate_edge_table(canonical: np.ndarray, faces: np.ndarray,
                                 labels: np.ndarray, status: np.ndarray,
                                 defect_number: int) -> tuple[np.ndarray, int]:
    """Return the ordered, unscored ``EDGE`` table before native ``qsort``.

    Columns are corrected-surface vertex 1, vertex 2, and ``used`` flag
    (0=new, 1=already in the base, 2=present in the original mesh).
    The MRI-derived ``len`` field and later patch search are not reproduced.
    """
    component = np.asarray(labels, np.int32)
    triangles = np.asarray(faces, np.int32)
    vertices, border = defect_regions(component, triangles)[defect_number]
    candidates = [vertex for vertex in vertices if status[vertex] == 1] + border
    first, second = np.triu_indices(len(candidates), 1)
    ids = np.asarray(candidates, np.int32)
    edges = np.column_stack((ids[first], ids[second])).astype(np.int32)
    all_pairs = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]],
                                triangles[:, [2, 0]]), axis=0)
    all_pairs.sort(axis=1)
    original_edges = set(map(tuple, all_pairs))
    base_pairs = all_pairs[np.all(component[all_pairs] == 0, axis=1)]
    base_edges = set(map(tuple, base_pairs))
    ordered = np.sort(edges, axis=1)
    already_in_base = np.fromiter((tuple(edge) in base_edges for edge in ordered),
                                  np.bool_, count=len(edges))
    in_original = np.fromiter((tuple(edge) in original_edges for edge in ordered),
                              np.bool_, count=len(edges))
    keep = _prune_against_base_edges(np.asarray(canonical, np.float32),
                                     edges, already_in_base)
    used = np.where(already_in_base, 1, np.where(in_original, 2, 0)).astype(np.int32)
    vertex_translation, _ = genetic_base_translation(component, triangles)
    translated = vertex_translation[edges[keep]]
    return np.column_stack((translated, used[keep])), len(edges)


def genetic_candidate_base_prune(canonical: np.ndarray, faces: np.ndarray,
                                 labels: np.ndarray, status: np.ndarray,
                                 defect_number: int) -> tuple[int, int]:
    """Count candidate edges removed by already present base-surface edges."""
    table, total = genetic_candidate_edge_table(canonical, faces, labels,
                                                status, defect_number)
    return total, total - len(table)
