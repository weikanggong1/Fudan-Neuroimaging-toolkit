"""Radial newMSM interpolation with ordered Octree candidate pools.

The leaf builder and native scalar fallback are part of FNIT's existing extension.
GPU containment and feature interpolation stay in PyTorch, with one instability
mask transfer per lookup and literal Point selection for shared boundaries.
"""
from __future__ import annotations

import numpy as np
import torch

from ._execution import cpu_workers, record_statistics
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
    """Pinned newMSM ordered leaf lookup with GPU interior containment.

    Candidate pools follow the original incremental Octree and its inclusive
    XYZ child order. Ordinary unique interiors stay on the device; overlapping
    or boundary containment uses literal scalar Point operations in the shared
    FNIT extension for both values of the historical ``source_precision`` flag.
    The caller still controls its subsequent warp arithmetic with that flag.
    """

    def __init__(self, vertices, faces, device, *, execution='optimized', source_precision=False):
        if execution not in ('optimized', 'reference'):
            raise ValueError("sphere execution must be 'optimized' or 'reference'")
        from . import _fastpd_native

        self.execution = execution
        self.source_precision = source_precision
        self.device = torch.device(device)
        # Native bytes, the Octree and tensor geometry must describe the same
        # snapshot even if the caller later mutates its NumPy arrays.
        vertex_array = np.array(vertices, dtype=np.float64, copy=True)
        face_array = np.array(faces, dtype=np.int64, copy=True)
        self.vertex_bytes = vertex_array.tobytes()
        self.face_bytes = face_array.tobytes()
        self.cpu_threads = cpu_workers()
        self.vertices = torch.as_tensor(vertex_array, dtype=torch.float64, device=self.device)
        self.faces = torch.as_tensor(face_array, dtype=torch.long, device=self.device)
        tree = _fastpd_native.build_ordered_face_octree(
            self.vertex_bytes, np.asarray(face_array, np.int32).tobytes(), len(vertices), len(faces))
        self.tree_buffers = tree
        count = tree['n_nodes']
        children = np.frombuffer(tree['children'], np.int32).reshape(count, 8)
        midpoint = np.frombuffer(tree['mid'], np.float64).reshape(count, 3)
        leaf_index = np.frombuffer(tree['leaf_index'], np.int32)
        parent = np.frombuffer(tree['parent'], np.int32)
        self.node_children = torch.tensor(children, dtype=torch.long, device=self.device)
        self.node_midpoints = torch.tensor(midpoint, dtype=torch.float64, device=self.device)
        self.node_is_leaf = torch.tensor(leaf_index >= 0, device=self.device)
        depth = np.zeros(count, np.int64)
        for node in range(1, count):
            depth[node] = depth[parent[node]] + 1
        self.max_depth = int(depth.max())
        offsets = np.frombuffer(tree['leaf_offsets'], np.int64)
        pools = np.frombuffer(tree['leaf_faces'], np.int32)
        width = max(1, int(np.max(offsets[1:] - offsets[:-1])))
        table = np.zeros((count, width), np.int64)
        valid = np.zeros_like(table, bool)
        for node in np.flatnonzero(leaf_index >= 0):
            pool = pools[offsets[node]:offsets[node+1]]
            if len(pool):
                table[node] = np.r_[pool, np.full(width-len(pool), pool[0], dtype=np.int32)]
                valid[node, :len(pool)] = True
        self.leaf_candidates = torch.tensor(table, device=self.device)
        self.leaf_candidate_valid = torch.tensor(valid, device=self.device)
        # Reference execution needs the cache for final interpolation as well
        # as diagnostic comparison; it can still recompute candidate normals.
        self.triangles = self.vertices[self.faces]
        self.normal, self.normal_dot_a, self.edge_normals = _projection_geometry(self.triangles)
        # Candidate geometry depends only on this mapper's frozen surface.
        # Stack the three edge tests so each CUDA operation handles all of
        # them together, preserving Point's scalar arithmetic order.
        if self.execution == 'optimized' and self.device.type == 'cuda':
            a, b, c = self.triangles.unbind(-2)
            self.containment_edges = torch.stack((b-a, c-b, a-c), -2)
            self.containment_normals = torch.stack(self.edge_normals, -2)
            self.edge_l1 = self.containment_edges.abs().sum(-1)
            self.normal_l1 = self.containment_normals.abs().sum(-1)

    def _leaf_nodes(self, points):
        node = torch.zeros(len(points), dtype=torch.long, device=self.device)
        for _ in range(self.max_depth):
            active = ~self.node_is_leaf[node]
            upper = points >= self.node_midpoints[node]
            child = upper[:, 0].long()*4 + upper[:, 1].long()*2 + upper[:, 2].long()
            # Bounds are inclusive upstream: exact split-plane ties keep the
            # last containing XYZ child, which is the upper half on each axis.
            node = torch.where(active, self.node_children[node, child], node)
        return node

    def _native_selection(self, points, node_ids):
        from . import _fastpd_native
        tree = self.tree_buffers
        query = points.detach().cpu().numpy()
        node_array = node_ids.detach().cpu().numpy().astype(np.int32, copy=False)
        raw = _fastpd_native.source_ordered_selection(
            self.vertex_bytes, self.face_bytes, query.tobytes(), node_array.tobytes(),
            tree['leaf_offsets'], tree['leaf_faces'], tree['fallback_offsets'], tree['fallback_faces'],
            len(self.vertices), len(self.faces), len(points), self.cpu_threads)
        value = np.frombuffer(raw, np.float64).reshape(-1, 4)
        patches = torch.tensor(value[:, 0].astype(np.int64), device=self.device)
        projection = torch.tensor(value[:, 1:], dtype=torch.float64, device=self.device)
        # Face selection is discrete. Preserve projection gradients for
        # differentiable queries/geometry without detaching the coordinates.
        if points.requires_grad or self.vertices.requires_grad:
            projection = self._project_selected(points, patches)
        return patches, projection

    def _project_selected(self, points, patches):
        if self.execution == 'reference':
            normal, top, _ = _projection_geometry(self.vertices[self.faces[patches]])
        else:
            normal, top = self.normal[patches], self.normal_dot_a[patches]
        denominator = _dot(normal, points)
        ratio = (_point_cpu.divide(top, denominator) if _point_cpu.enabled(top, denominator)
                 else top/denominator)
        return points*ratio[:, None]

    def _tensor_selection(self, points, nodes):
        candidates = self.leaf_candidates[nodes]
        valid = self.leaf_candidate_valid[nodes]
        triangles = self.triangles[candidates]
        if self.execution == 'reference' or self.device.type != 'cuda':
            normal, top, normals = _projection_geometry(triangles)
            a, b, c = triangles.unbind(-2)
            edges = torch.stack((b-a, c-b, a-c), -2)
            orthogonals = torch.stack(normals, -2)
            edge_l1 = edges.abs().sum(-1)
            normal_l1 = orthogonals.abs().sum(-1)
        else:
            normal, top = self.normal[candidates], self.normal_dot_a[candidates]
            edges = self.containment_edges[candidates]
            orthogonals = self.containment_normals[candidates]
            edge_l1 = self.edge_l1[candidates]
            normal_l1 = self.normal_l1[candidates]
        denominator = _dot(normal, points[:, None, :])
        ratio = (_point_cpu.divide(top, denominator) if _point_cpu.enabled(top, denominator)
                 else top/denominator)
        projected = points[:, None, :]*ratio[:, :, None]
        a, b, c = triangles.unbind(-2)
        delta = torch.stack((projected-a, projected-b, projected-c), -2)
        margins = _dot(_cross(edges, delta), orthogonals)
        inside = ((margins > -1e-8).all(-1) & valid
                  & torch.isfinite(projected).all(-1))
        unique = inside.sum(-1) == 1
        choice = inside.long().argmax(-1)
        row = torch.arange(len(points), device=self.device)
        patch = candidates[row, choice]
        projection = projected[row, choice]
        # Accepted epsilon plus a conservative FP64 operation-scale guard.
        # Every source-leaf candidate must have a stable interior/outside
        # classification. This rejects overlaps, source shared edges and
        # degenerate projections before using the GPU-only choice.
        scale = (edge_l1*delta.abs().sum(-1))*normal_l1
        guard = 1e-8 + (4096*np.finfo(np.float64).eps)*torch.clamp(scale, min=1.0)
        positive = (margins > guard).all(-1)
        outside = (margins < -guard).any(-1)
        classified = positive | outside | ~valid
        safe = (unique & positive[row, choice] & classified.all(-1)
                & (points >= -101.0).all(-1) & (points <= 101.0).all(-1)
                & torch.isfinite(points).all(-1))
        return patch, projection, safe

    def weights(self, points, batch_size=None, *, project=True):
        points = points.to(device=self.device, dtype=torch.float64)
        if batch_size is None:
            batch_size = 32768 if self.execution == 'optimized' else 4096
        if batch_size <= 0:
            raise ValueError('sphere batch_size must be positive')
        if not len(points):
            ids = torch.empty((0, 3), dtype=torch.long, device=self.device)
            return ids, torch.empty((0, 3), dtype=torch.float64, device=self.device), ids[:, 0]
        nodes = self._leaf_nodes(points)
        if self.device.type == 'cpu' and self.execution == 'optimized':
            patches, projection = self._native_selection(points, nodes)
            record_statistics(octree_native_queries=len(points))
        else:
            blocks = []
            for start in range(0, len(points), batch_size):
                stop = min(start+batch_size, len(points))
                patch, projection, safe = self._tensor_selection(points[start:stop], nodes[start:stop])
                blocks.append((patch, projection, safe))
            patches = torch.cat([block[0] for block in blocks])
            projection = torch.cat([block[1] for block in blocks])
            safe = torch.cat([block[2] for block in blocks])
            # One mask transfer per lookup. Only unstable rows move query
            # coordinates and node IDs to host, never all ordinary GPU points.
            mask = (~safe).detach().cpu().numpy()
            record_statistics(octree_device_containment_queries=int(np.count_nonzero(~mask)),
                              octree_native_queries=int(np.count_nonzero(mask)))
            if mask.any():
                ids = torch.as_tensor(np.flatnonzero(mask), device=self.device)
                chosen, literal = self._native_selection(points[ids], nodes[ids])
                patches[ids] = chosen
                projection[ids] = literal
        vertices = self.faces[patches]
        weights = _area_weights(self.vertices[vertices], projection if project else points)
        return vertices, weights, patches

    def sample(self, points, metric):
        ids, weights, _ = self.weights(points, project=False)
        weighted = metric[ids]*weights
        return (weighted[:, 0]+weighted[:, 1])+weighted[:, 2]
