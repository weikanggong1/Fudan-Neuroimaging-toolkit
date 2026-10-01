"""Radial newMSM sphere interpolation with cached static geometry.

The optimized execution keeps the reference containment tests, distance and face-order tie
break, unsigned areas and fallback candidates. It queues chunk work before one
host mask transfer; ``execution='reference'`` retains per-chunk transfers.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
import torch


def _dot(first, second):
    product = first*second
    return (product[..., 0]+product[..., 1])+product[..., 2]


def _cross(first, second):
    # Separate products/subtractions preserve Point's scalar operation order.
    return torch.stack((first[..., 1]*second[..., 2]-first[..., 2]*second[..., 1],
                        first[..., 2]*second[..., 0]-first[..., 0]*second[..., 2],
                        first[..., 0]*second[..., 1]-first[..., 1]*second[..., 0]), -1)


def _norm(vector):
    return torch.sqrt(_dot(vector, vector))


def _normalize(vector):
    length = _norm(vector)
    return vector/torch.where(length > 1e-8, length, torch.ones_like(length))[..., None]


def _projection_geometry(triangles):
    a, b, c = triangles.unbind(-2)
    first = _normalize(c-a)
    second = _normalize(b-a)
    normal = _normalize(_cross(first, second))
    edges = (_cross(b-a, c-a), _cross(c-b, a-b), _cross(a-c, b-c))
    return normal, _dot(normal, a), edges


def _edge_distance(points, triangles):
    """Triangle::dist_to_point's finite-edge and vertex distance rule."""
    a, b, c = triangles.unbind(-2)
    best = torch.full(points.shape[:-1], torch.inf, dtype=points.dtype, device=points.device)
    for first, second in ((a, b), (a, c), (b, c)):
        edge = second-first
        first_delta, second_delta = points-first, points-second
        valid = (_dot(first_delta, edge) > 0) & (_dot(second_delta, edge) < 0)
        distance = _norm(_cross(first_delta, second_delta))/_norm(edge)
        best = torch.where(valid & (distance < best), distance, best)
    for corner in (a, b, c):
        distance = _norm(points-corner)
        best = torch.where(distance < best, distance, best)
    return best


def _area_weights(triangles, points):
    """Unsigned area weights with the source's three-term arithmetic order."""
    a, b, c = triangles.unbind(-2)
    first = _norm(_cross(b-points, c-points))*0.5
    second = _norm(_cross(a-points, c-points))*0.5
    third = _norm(_cross(a-points, b-points))*0.5
    weights = torch.stack((first, second, third), dim=-1)
    total = (first+second)+third
    return weights/total[..., None]


class RadialSphereMap:
    def __init__(self, vertices, faces, device, *, execution='optimized'):
        if execution not in ('optimized', 'reference'):
            raise ValueError("sphere execution must be 'optimized' or 'reference'")
        self.execution = execution
        self.device = torch.device(device)
        self.vertices = torch.as_tensor(vertices, dtype=torch.float64, device=self.device)
        self.faces = torch.as_tensor(faces, dtype=torch.long, device=self.device)
        self.tree = cKDTree(np.asarray(vertices))
        incident = [[] for _ in range(len(vertices))]
        for face_id, triangle in enumerate(faces):
            for vertex in triangle: incident[vertex].append(face_id)
        width = max(map(len, incident))
        table = np.empty((len(vertices), width), np.int64)
        for vertex, items in enumerate(incident):
            table[vertex] = items+[items[0]]*(width-len(items))
        self.incident = torch.as_tensor(table, device=self.device)
        if execution == 'optimized':
            self.triangles = self.vertices[self.faces]
            self.normal, self.normal_dot_a, self.edge_normals = _projection_geometry(self.triangles)

    def _select(self, points, nearest):
        candidates = self.incident[nearest].reshape(len(points), -1)
        if self.execution == 'optimized':
            triangles = self.triangles[candidates]
            normal = self.normal[candidates]
            top = self.normal_dot_a[candidates]
            ab, bc, ca = (value[candidates] for value in self.edge_normals)
        else:
            triangles = self.vertices[self.faces[candidates]]
            normal, top, (ab, bc, ca) = _projection_geometry(triangles)
        a, b, c = triangles.unbind(-2)
        denominator = _dot(normal, points[:, None, :])
        projected = points[:, None, :]*(top/denominator)[:, :, None]
        inside_ab = _dot(_cross(b-a, projected-a), ab) > -1e-8
        inside_bc = _dot(_cross(c-b, projected-b), bc) > -1e-8
        inside_ca = _dot(_cross(a-c, projected-c), ca) > -1e-8
        inside = inside_ab & inside_bc & inside_ca & torch.isfinite(projected).all(-1)
        # Official Octree compares projected points' finite-edge/vertex
        # distances. Only exact distance ties keep the first mesh face.
        distance = torch.where(inside, _edge_distance(projected, triangles), torch.inf)
        minimum = distance.min(-1).values
        best = inside & (distance == minimum[:, None])
        sentinel = torch.full_like(candidates, len(self.faces))
        ordered = torch.where(best, candidates, sentinel)
        exists = torch.isfinite(minimum)
        choice = ordered.argmin(-1)
        row = torch.arange(len(points), device=self.device)
        return candidates[row, choice], projected[row, choice], exists

    def _fallback(self, points, query, face, projection, missing):
        if missing.any():
            missing_ids = np.flatnonzero(missing)
            k = min(32, len(self.vertices))
            expanded = self.tree.query(query[missing], k=k, workers=4)[1]
            expanded = np.asarray(expanded).reshape(len(missing_ids), k)
            ids = torch.as_tensor(missing_ids, device=self.device)
            better, q, found = self._select(points[ids], torch.as_tensor(expanded, device=self.device))
            if not bool(found.all()):
                raise RuntimeError('no containing radial sphere triangle; check folded input mesh')
            face[ids] = better
            projection[ids] = q
        return face, projection

    def weights(self, points, batch_size=None, *, project=True):
        if batch_size is None:
            batch_size = 32768 if self.execution == 'optimized' else 4096
        if batch_size <= 0:
            raise ValueError('sphere batch_size must be positive')
        points = points.to(dtype=torch.float64, device=self.device)
        if not len(points):
            empty_ids = torch.empty((0, 3), dtype=torch.long, device=self.device)
            empty_weights = torch.empty((0, 3), dtype=torch.float64, device=self.device)
            return empty_ids, empty_weights, empty_ids[:, 0]
        query = points.detach().cpu().numpy()
        nearest = self.tree.query(query, k=1, workers=4)[1][:, None]
        chosen_faces = []; chosen_weights = []; chosen_patches = []
        blocks = []
        for start in range(0, len(query), batch_size):
            stop = min(start+batch_size, len(query)); p = points[start:stop]
            near = torch.as_tensor(nearest[start:stop], device=self.device)
            face, projection, inside = self._select(p, near)
            if self.execution == 'reference':
                missing = (~inside).detach().cpu().numpy()
                face, projection = self._fallback(p, query[start:stop], face, projection, missing)
                w = _area_weights(self.vertices[self.faces[face]], projection if project else p)
                chosen_faces.append(self.faces[face]); chosen_weights.append(w); chosen_patches.append(face)
            else:
                blocks.append((start, stop, face, projection, inside))
        if self.execution == 'optimized':
            # The only containment-mask transfer for the entire call. All
            # chunks retain their original order and fallback candidate order.
            missing_all = (~torch.cat([item[4] for item in blocks])).detach().cpu().numpy()
            for start, stop, face, projection, _ in blocks:
                p = points[start:stop]
                face, projection = self._fallback(p, query[start:stop], face, projection, missing_all[start:stop])
                w = _area_weights(self.triangles[face], projection if project else p)
                chosen_faces.append(self.faces[face]); chosen_weights.append(w); chosen_patches.append(face)
        return torch.cat(chosen_faces), torch.cat(chosen_weights), torch.cat(chosen_patches)

    def sample(self, points, metric):
        # The likelihood samples unsigned areas at the original point, while
        # metric resampling uses its radial projection to the containing face.
        ids, weights, _ = self.weights(points, project=False)
        weighted = metric[ids]*weights
        # barycentric_interpolation accumulates in triangle corner order.
        return (weighted[:, 0]+weighted[:, 1])+weighted[:, 2]
