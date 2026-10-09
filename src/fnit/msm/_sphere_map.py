"""Radial newMSM sphere interpolation with cached static geometry.

The optimized execution keeps the reference containment tests, distance and face-order tie
break, unsigned areas and fallback candidates. It queues chunk work before one
host mask transfer; ``execution='reference'`` retains per-chunk transfers.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
import torch

from ._execution import cpu_workers, record_statistics
from ._spatial import ExactCellNearest
from . import _point_cpu


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
    if _point_cpu.enabled(vector):
        return _point_cpu.normalize(vector)
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
        numerator, denominator = _norm(_cross(first_delta, second_delta)), _norm(edge)
        distance = (_point_cpu.divide(numerator, denominator) if _point_cpu.enabled(numerator, denominator)
                    else numerator/denominator)
        best = torch.where(valid & (distance < best), distance, best)
    for corner in (a, b, c):
        distance = _norm(points-corner)
        best = torch.where(distance < best, distance, best)
    return best


def _area_weights(triangles, points):
    """Unsigned area weights with the source's three-term arithmetic order."""
    if _point_cpu.enabled(triangles, points):
        return _point_cpu.area_weights(triangles, points)
    a, b, c = triangles.unbind(-2)
    first = _norm(_cross(b-points, c-points))*0.5
    second = _norm(_cross(a-points, c-points))*0.5
    third = _norm(_cross(a-points, b-points))*0.5
    weights = torch.stack((first, second, third), dim=-1)
    total = (first+second)+third
    return weights/total[..., None]


class RadialSphereMap:
    def __init__(self, vertices, faces, device, *, execution='optimized', source_precision=False):
        if execution not in ('optimized', 'reference'):
            raise ValueError("sphere execution must be 'optimized' or 'reference'")
        self.execution = execution
        self.source_precision = source_precision
        self.device = torch.device(device)
        self.vertices = torch.as_tensor(vertices, dtype=torch.float64, device=self.device)
        self.faces = torch.as_tensor(faces, dtype=torch.long, device=self.device)
        self.tree = cKDTree(np.asarray(vertices))
        self.cpu_threads = cpu_workers()
        self.gpu_nearest = (ExactCellNearest(vertices, self.device)
                            if self.device.type == "cuda" and execution == "optimized" else None)
        incident = [[] for _ in range(len(vertices))]
        for face_id, triangle in enumerate(faces):
            for vertex in triangle: incident[vertex].append(face_id)
        width = max(map(len, incident))
        table = np.empty((len(vertices), width), np.int64)
        for vertex, items in enumerate(incident):
            table[vertex] = items+[items[0]]*(width-len(items))
        self.incident = torch.as_tensor(table, device=self.device)
        if source_precision:
            self.incident_cpu = table
            self.vertex_bytes = np.asarray(vertices, dtype=np.float64).tobytes()
            self.face_bytes = np.asarray(faces, dtype=np.int64).tobytes()
        if execution == 'optimized':
            self.triangles = self.vertices[self.faces]
            self.normal, self.normal_dot_a, self.edge_normals = _projection_geometry(self.triangles)

    def _select(self, points, nearest):
        if (self.device.type == 'cpu' and self.execution == 'optimized'
                and _point_cpu.enabled(
                    points, self.triangles, self.normal, self.normal_dot_a, *self.edge_normals)):
            # Independent CPU query rows avoid materializing point-by-face
            # triangle/cross-product tensors. Cached PyTorch geometry and
            # scalar operation order are retained. CUDA stays on its batched
            # tensor path; differentiable callers retain that path as well.
            from ._sphere_cpu import select_faces
            values = select_faces(
                points.numpy(), nearest.numpy(), self.incident.numpy(),
                self.triangles.numpy(), self.normal.numpy(), self.normal_dot_a.numpy(),
                *(edge.numpy() for edge in self.edge_normals), cpu_threads=self.cpu_threads)
            face, projected, exists, ambiguous = (torch.from_numpy(value) for value in values)
            return face, projected, exists, ambiguous if self.source_precision else None
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
        ratio = (_point_cpu.divide(top, denominator) if _point_cpu.enabled(top, denominator)
                 else top/denominator)
        projected = points[:, None, :]*ratio[:, :, None]
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
        ambiguous = inside.sum(-1) > 1 if self.source_precision else None
        return candidates[row, choice], projected[row, choice], exists, ambiguous

    def _source_select(self, query, nearest, face, projection, ambiguous):
        """Resolve only overlapping containing candidates in scalar order."""
        if ambiguous.any():
            from . import _fastpd_native
            selected = np.flatnonzero(ambiguous)
            candidates = self.incident_cpu[nearest[selected]].reshape(len(selected), -1)
            candidates = np.sort(candidates, axis=1).astype(np.int64, copy=False)
            values = np.frombuffer(_fastpd_native.source_radial_selection(
                self.vertex_bytes, self.face_bytes,
                np.asarray(query[selected], dtype=np.float64).tobytes(), candidates.tobytes(),
                len(self.vertices), len(self.faces), len(selected), candidates.shape[1]),
                dtype=np.float64).reshape(-1, 4)
            if np.any(values[:, 0] < 0):
                raise RuntimeError('no containing source-precision sphere candidate')
            ids = torch.as_tensor(selected, device=self.device)
            face[ids] = torch.as_tensor(values[:, 0].astype(np.int64), device=self.device)
            projection[ids] = torch.tensor(values[:, 1:], device=self.device)
        return face, projection

    def _fallback(self, points, query, face, projection, missing):
        if missing.any():
            if query is None:
                query = points.detach().cpu().numpy()
            missing_ids = np.flatnonzero(missing)
            record_statistics(tree_expanded_face_queries=len(missing_ids))
            k = min(32, len(self.vertices))
            expanded = self.tree.query(query[missing], k=k, workers=self.cpu_threads)[1]
            expanded = np.asarray(expanded).reshape(len(missing_ids), k)
            ids = torch.as_tensor(missing_ids, device=self.device)
            better, q, found, _ = self._select(points[ids], torch.as_tensor(expanded, device=self.device))
            if not bool(found.all()):
                raise RuntimeError('no containing radial sphere triangle; check folded input mesh')
            if self.source_precision:
                better, q = self._source_select(query[missing], expanded, better, q,
                                                np.ones(len(expanded), dtype=bool))
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
        # Source-precision native tie/fallback paths need host coordinates only
        # for ambiguous or missing GPU cells.  Keeping this lazy avoids a full
        # device-to-host copy for the common proven-containment path while
        # retaining the exact FP64/native arithmetic whenever a fallback is
        # actually required.  CPU-tree execution still needs the query up
        # front, as before.
        query = points.detach().cpu().numpy() if self.gpu_nearest is None else None
        if self.gpu_nearest is None:
            nearest_cpu = self.tree.query(query, k=1, workers=self.cpu_threads)[1][:, None]
            record_statistics(tree_nearest_queries=len(points))
            nearest = torch.as_tensor(nearest_cpu, device=self.device)
        else:
            nearest, uncertain = self.gpu_nearest.query(points)
            masks = torch.stack((uncertain, self.gpu_nearest.last_ties), -1).detach().cpu().numpy()
            uncertain_cpu = masks[:, 0]
            fallback_count = np.count_nonzero(uncertain_cpu)
            record_statistics(cuda_nearest_queries=len(points),
                              cuda_nearest_proved_queries=len(points)-fallback_count,
                              tree_nearest_queries=fallback_count,
                              nearest_tie_fallback_queries=np.count_nonzero(masks[:, 1]))
            if uncertain_cpu.any():
                ids = np.flatnonzero(uncertain_cpu)
                unresolved_query = points[torch.as_tensor(ids, device=self.device)].detach().cpu().numpy()
                fallback = self.tree.query(unresolved_query, k=1, workers=self.cpu_threads)[1]
                nearest[torch.as_tensor(ids, device=self.device)] = torch.as_tensor(fallback, device=self.device)
            nearest = nearest[:, None]
            nearest_cpu = None
        chosen_faces = []; chosen_weights = []; chosen_patches = []
        blocks = []
        for start in range(0, len(points), batch_size):
            stop = min(start+batch_size, len(points)); p = points[start:stop]
            near = nearest[start:stop]
            face, projection, inside, ambiguous = self._select(p, near)
            if self.execution == 'reference':
                if self.source_precision:
                    if query is None:
                        query = points.detach().cpu().numpy()
                    if nearest_cpu is None:
                        nearest_cpu = nearest.detach().cpu().numpy()
                    masks = torch.stack((~inside, ambiguous), -1).detach().cpu().numpy()
                    missing = masks[:, 0]
                    face, projection = self._source_select(query[start:stop], nearest_cpu[start:stop],
                                                          face, projection, masks[:, 1])
                else:
                    missing = (~inside).detach().cpu().numpy()
                face, projection = self._fallback(p, query[start:stop] if query is not None else None, face, projection, missing)
                w = _area_weights(self.vertices[self.faces[face]], projection if project else p)
                chosen_faces.append(self.faces[face]); chosen_weights.append(w); chosen_patches.append(face)
            else:
                blocks.append((start, stop, face, projection, inside, ambiguous))
        if self.execution == 'optimized':
            # The only containment-mask transfer for the entire call. All
            # chunks retain their original order and fallback candidate order.
            inside_all = torch.cat([item[4] for item in blocks])
            if self.source_precision:
                masks = torch.stack((~inside_all, torch.cat([item[5] for item in blocks])), -1).detach().cpu().numpy()
                missing_all = masks[:, 0]
                # Delay both host buffers until a native ambiguity or a tree
                # fallback is present.  This is the only path that consumes
                # source vertex IDs or query coordinates on the host.
                if missing_all.any() or masks[:, 1].any():
                    if query is None:
                        query = points.detach().cpu().numpy()
                    if nearest_cpu is None:
                        nearest_cpu = nearest.detach().cpu().numpy()
            else:
                missing_all = (~inside_all).detach().cpu().numpy()
            for start, stop, face, projection, _, _ in blocks:
                p = points[start:stop]
                if self.source_precision:
                    host_query = query[start:stop] if query is not None else None
                    host_nearest = nearest_cpu[start:stop] if nearest_cpu is not None else None
                    face, projection = self._source_select(host_query, host_nearest,
                                                          face, projection, masks[start:stop, 1])
                missing = missing_all[start:stop]
                fallback_query = (query[start:stop] if query is not None else
                                  p.detach().cpu().numpy() if missing.any() else np.empty((len(p), 3)))
                face, projection = self._fallback(p, fallback_query, face, projection, missing)
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
