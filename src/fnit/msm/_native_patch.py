"""Bounded native sphere patch proposals, verified at saved float32 precision.

These explicit output repairs change registration coordinates. Fixed-boundary
harmonic proposals and quantized joint neighbours preserve previously positive
faces; default source-compatible registration does not call these operations.
"""
from __future__ import annotations

import itertools
import time

import numpy as np

def signed(points, faces):
    t = np.asarray(points, dtype=np.float64)[faces]
    return (np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]) * t[:, 0]).sum(1)

def bad_components(faces, bad):
    remaining = set(map(int, bad))
    components = []
    while remaining:
        group = {min(remaining)}
        vertices = set(map(int, faces[list(group)].reshape(-1)))
        progress = True
        while progress:
            connected = {index for index in remaining if vertices.intersection(faces[index])}
            progress = bool(connected - group)
            group.update(connected)
            vertices.update(map(int, faces[list(connected)].reshape(-1)))
        remaining.difference_update(group)
        components.append(np.asarray(sorted(group), dtype=np.int64))
    return components

def topology_audit(points, faces):
    if faces.ndim != 2 or faces.shape[1] != 3 or not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("Integer triangular topology is required")
    if faces.min() < 0 or faces.max() >= len(points) or np.any(np.diff(np.sort(faces, axis=1), axis=1) == 0):
        raise ValueError("Invalid triangle indices")
    _, counts_faces = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True)
    directed = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    edges, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True)
    winding = np.bincount(inverse, weights=np.where(directed[:, 0] < directed[:, 1], 1, -1), minlength=len(edges))
    used = len(np.unique(faces))
    report = {"vertices": len(points), "faces": len(faces), "edges": len(edges),
              "used_vertices": used, "euler_characteristic": used - len(edges) + len(faces),
              "duplicate_faces": int(np.count_nonzero(counts_faces > 1)),
              "nonmanifold_or_boundary_edges": int(np.count_nonzero(counts != 2)),
              "inconsistent_winding_edges": int(np.count_nonzero((counts == 2) & (winding != 0)))}
    report["pass"] = (report["euler_characteristic"] == 2 and used == len(points) and
                      report["duplicate_faces"] == report["nonmanifold_or_boundary_edges"] == report["inconsistent_winding_edges"] == 0)
    return report

def disk_patch(faces, seed_faces, rings):
    variable = np.unique(faces[seed_faces])
    selected = np.flatnonzero(np.isin(faces, variable).any(1))
    for _ in range(rings - 1):
        variable = np.unique(faces[selected])
        selected = np.flatnonzero(np.isin(faces, variable).any(1))
    patch = faces[selected]
    vertices = np.unique(patch)
    directed = patch[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    edges, inverse, count = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True)
    boundary_edges = directed[count[inverse] == 1]
    boundary = []
    reason = None
    if (count > 2).any() or not len(boundary_edges): reason = "nonmanifold_or_closed_patch"
    elif len(vertices) - len(edges) + len(patch) != 1: reason = "patch_not_disk_euler"
    else:
        outgoing = {int(a): int(b) for a, b in boundary_edges}
        incoming = {int(b): int(a) for a, b in boundary_edges}
        if len(outgoing) != len(boundary_edges) or len(incoming) != len(boundary_edges): reason = "branching_boundary"
        else:
            start = min(outgoing)
            vertex = start
            while vertex not in boundary:
                boundary.append(vertex)
                vertex = outgoing.get(vertex, -1)
                if vertex == -1: break
            if vertex != start or len(boundary) != len(boundary_edges): reason = "multiple_or_open_boundary_loops"
    boundary = np.asarray(boundary, dtype=np.int64)
    interior = np.setdiff1d(vertices, boundary)
    return selected, vertices, boundary, interior, reason

def simple_polygon(polygon, tolerance):
    count = len(polygon)
    def orient(a, b, c):
        delta1, delta2 = b - a, c - a
        return delta1[0] * delta2[1] - delta1[1] * delta2[0]
    def on_segment(a, b, c):
        return abs(orient(a, b, c)) <= tolerance and np.all(c >= np.minimum(a, b) - np.sqrt(tolerance)) and np.all(c <= np.maximum(a, b) + np.sqrt(tolerance))
    for index in range(count):
        a, b = polygon[index], polygon[(index + 1) % count]
        if np.array_equal(a, b): return False
        for other in range(index + 1, count):
            if (other - index) % count in (1, count - 1): continue
            c, d = polygon[other], polygon[(other + 1) % count]
            values = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
            if values[0] * values[1] < -tolerance ** 2 and values[2] * values[3] < -tolerance ** 2: return False
            if on_segment(a, b, c) or on_segment(a, b, d) or on_segment(c, d, a) or on_segment(c, d, b): return False
    return True

def harmonic_target(points, faces, selected, vertices, boundary, interior, sign, *, allow_nonconvex=False):
    if not len(interior) or len(interior) > 2000:
        return None, {"reason": "no_interior_or_dense_solver_budget"}
    centre = points[boundary].mean(0)
    norm = np.linalg.norm(centre)
    if norm < 1e-12: return None, {"reason": "zero_patch_center"}
    normal = centre / norm
    axis = np.eye(3)[np.argmin(np.abs(normal))]
    e1 = np.cross(axis, normal)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    basis = np.column_stack((e1, e2))
    depth = points[vertices] @ normal
    if np.any(depth <= 0): return None, {"reason": "patch_outside_projection_hemisphere"}
    planar = (points[vertices] @ basis) * (100. / depth[:, None])
    lookup = np.full(len(points), -1, dtype=np.int64)
    lookup[vertices] = np.arange(len(vertices))
    origin = planar[lookup[boundary]].mean(0)
    planar -= origin
    polygon = planar[lookup[boundary]]
    edge = np.roll(polygon, -1, axis=0) - polygon
    turns = edge[:, 0] * np.roll(edge, -1, axis=0)[:, 1] - edge[:, 1] * np.roll(edge, -1, axis=0)[:, 0]
    scale = float(np.linalg.norm(edge, axis=1).max(initial=0))
    tolerance = max(1e-24, scale * scale * 1e-12)
    area = .5 * np.sum(polygon[:, 0] * np.roll(polygon, -1, axis=0)[:, 1] - polygon[:, 1] * np.roll(polygon, -1, axis=0)[:, 0])
    info = {"boundary_vertices": len(boundary), "interior_vertices": len(interior),
            "minimum_signed_boundary_turn": float((turns * sign).min(initial=np.inf)),
            "boundary_signed_planar_area_mm2": float(area), "boundary_convex": bool(np.all(turns * sign > tolerance)),
            "boundary_winding_matches_majority": bool(area * sign > 0),
            "boundary_simple": bool(simple_polygon(polygon, tolerance)),
            "nonconvex_experimental_allowed": allow_nonconvex,
            "convex_embedding_theorem_applies": bool(np.all(turns * sign > tolerance))}
    if not info["boundary_simple"] or not info["boundary_winding_matches_majority"] or (not allow_nonconvex and not info["boundary_convex"]):
        return None, {**info, "reason": "boundary_not_strict_convex_with_target_winding"}
    local_faces = faces[selected]
    neighbors = {int(vertex): set() for vertex in interior}
    for triangle in local_faces:
        for vertex in triangle:
            if int(vertex) in neighbors:
                neighbors[int(vertex)].update(int(other) for other in triangle if other != vertex)
    interior_lookup = {int(vertex): index for index, vertex in enumerate(interior)}
    matrix = np.zeros((len(interior), len(interior)), dtype=np.float64)
    rhs = np.zeros((len(interior), 2), dtype=np.float64)
    for vertex, adjacent in neighbors.items():
        index = interior_lookup[vertex]
        matrix[index, index] = len(adjacent)
        for other in adjacent:
            if other in interior_lookup: matrix[index, interior_lookup[other]] = -1
            else: rhs[index] += planar[lookup[other]]
    solved = np.linalg.solve(matrix, rhs)
    xy = solved + origin
    target = 100. * normal + xy @ basis.T
    target *= 100. / np.linalg.norm(target, axis=1, keepdims=True)
    info.update(reason="candidate", dirichlet_equation_residual=float(np.max(np.abs(matrix @ solved - rhs))))
    return target, info

def harmonic_repair(points, faces, original, *, maximum_rings=4, maximum_rounds=4, allow_nonconvex=False):
    initial = np.asarray(points, dtype=np.float32).astype(np.float64)
    if initial.shape != original.shape or not np.isfinite(initial).all() or not np.isfinite(original).all():
        raise ValueError("Finite matching native/reference geometry required")
    topo = topology_audit(initial, faces)
    if not topo["pass"]: raise ValueError("Topology audit failed; no geometric repair attempted")
    reference = signed(original, faces)
    if np.any(reference == 0) or not np.isfinite(reference).all() or np.count_nonzero(reference > 0) == np.count_nonzero(reference < 0):
        raise ValueError("Finite nondegenerate majority winding reference required")
    sign = 1 if np.count_nonzero(reference > 0) > np.count_nonzero(reference < 0) else -1
    baseline = sign * np.abs(reference)
    current = initial.copy()
    before = signed(current, faces) / baseline
    attempts = []
    updates = 0
    started = time.perf_counter()
    for round_index in range(maximum_rounds):
        bad = np.flatnonzero(signed(current, faces) / baseline <= 0)
        if not len(bad): break
        progress = False
        for component in bad_components(faces, bad):
            if np.all(signed(current, faces[component]) / baseline[component] > 0):
                continue
            for rings in range(1, maximum_rings + 1):
                selected, vertices, boundary, interior, reason = disk_patch(faces, component, rings)
                info = {"round": round_index, "rings": rings, "seed_faces": len(component),
                        "patch_faces": len(selected), "patch_vertices": len(vertices), "boundary_vertices": len(boundary), "interior_vertices": len(interior)}
                if reason:
                    attempts.append({**info, "reason": reason})
                    continue
                target, geometry_info = harmonic_target(current, faces, selected, vertices, boundary, interior, sign, allow_nonconvex=allow_nonconvex)
                info.update(geometry_info)
                if target is None:
                    attempts.append(info)
                    continue
                # Every incident face of an interior vertex lies within this complete disk.
                outside = np.flatnonzero(np.isin(faces, interior).any(1) & ~np.isin(np.arange(len(faces)), selected))
                if len(outside): raise AssertionError("Patch boundary does not contain all interior incident faces")
                prior = signed(current, faces[selected]) / baseline[selected]
                previous_good = prior > 0
                best_score = (int(np.count_nonzero(prior <= 0)), -float(prior.min()))
                best = None
                best_alpha = None
                # Increasing blend fractions select the first actually float32-passing candidate.
                fractions = sorted(set([2. ** power for power in range(-16, 1)] + [.001, .003, .01, .03, .1, .2, .3, .4, .6, .7, .8, .9, 1.]))
                for alpha in fractions:
                    trial = current[interior] + alpha * (target - current[interior])
                    trial *= 100. / np.linalg.norm(trial, axis=1, keepdims=True)
                    trial = trial.astype(np.float32).astype(np.float64)
                    local = current[vertices].copy()
                    lookup = np.full(len(current), -1, dtype=np.int64)
                    lookup[vertices] = np.arange(len(vertices))
                    local[lookup[interior]] = trial
                    ratios = signed(local, lookup[faces[selected]]) / baseline[selected]
                    if not np.isfinite(ratios).all() or np.any(ratios[previous_good] <= 0): continue
                    score = (int(np.count_nonzero(ratios <= 0)), -float(ratios.min()))
                    if score < best_score:
                        best, best_alpha, best_score = trial.copy(), alpha, score
                        if score[0] == 0: break
                if best is not None:
                    old = current[interior].copy()
                    current[interior] = best
                    updates += 1
                    progress = True
                    info.update(accepted=True, alpha=best_alpha, remaining_patch_folded_faces=best_score[0],
                                minimum_patch_orientation_ratio=-best_score[1],
                                maximum_joint_displacement_mm=float(np.linalg.norm(best - old, axis=1).max(initial=0)),
                                newly_folded_previously_positive_affected_faces=0)
                    attempts.append(info)
                    if best_score[0] == 0:
                        break
                    # A partial improvement cannot certify this component.
                    # Expand beyond thin nested triangles before another sweep.
                    continue
                info.update(accepted=False, reason="no_saved_float32_positive_preserving_improvement")
                attempts.append(info)
        if not progress: break
    after = signed(current, faces) / baseline
    displacement = np.linalg.norm(current - initial, axis=1)
    newly_bad = int(np.count_nonzero((before > 0) & (after <= 0)))
    if newly_bad: raise AssertionError("Harmonic accepted move folded an initially correct face")
    report = {"scope": "Explicit native-output harmonic patch correction",
              "success": not bool((after <= 0).any()), "absolute_folded_faces_before": int(np.count_nonzero(before <= 0)),
              "absolute_folded_faces_after": int(np.count_nonzero(after <= 0)),
              "minimum_absolute_orientation_ratio_before": float(before.min()), "minimum_absolute_orientation_ratio_after": float(after.min()),
              "reference_absolute_folded_faces": int(np.count_nonzero(reference * sign <= 0)),
              "reference_majority_sign": sign, "topology": topo, "accepted_joint_updates": updates,
              "moved_vertices": int(np.count_nonzero(displacement)), "maximum_vertex_displacement_mm": float(displacement.max(initial=0)),
              "newly_folded_previously_positive_faces": newly_bad, "attempts": attempts,
              "nonconvex_experimental_allowed": allow_nonconvex,
              "seconds": time.perf_counter() - started, "cpu_only": True,
              "saved_coordinates_precision": "float32"}
    return current, report

def neighbours(point, full):
    p = np.asarray(point, dtype=np.float32)
    directions = (list(itertools.product((-1, 0, 1), repeat=3)) if full
                  else [(0, 0, 0), (-1, 0, 0), (1, 0, 0),
                        (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)])
    options = []
    for direction in directions:
        if direction == (0, 0, 0):
            options.append(p.copy())
            continue
        trial = p.copy()
        for axis, value in enumerate(direction):
            if value:
                trial[axis] = np.nextafter(trial[axis], np.float32(np.inf * value))
        norm = np.linalg.norm(trial.astype(np.float64))
        if np.isfinite(norm) and norm > 0:
            options.append((trial.astype(np.float64) * (100. / norm)).astype(np.float32))
    return np.unique(np.asarray(options), axis=0).astype(np.float64)

def quantized_joint_repair(points, faces, reference, *, maximum_sweeps=8,
                           chunk_size=256):
    initial = np.asarray(points, dtype=np.float32).astype(np.float64)
    if not np.isfinite(initial).all():
        raise ValueError('Saved-float32 coordinates must be finite')
    base = signed(reference, faces)
    positive = np.count_nonzero(base > 0)
    negative = np.count_nonzero(base < 0)
    if not np.isfinite(base).all() or np.any(base == 0) or positive == negative:
        raise ValueError('Reference must be nondegenerate and have a majority winding')
    majority = 1 if positive > negative else -1
    denominator = majority * np.abs(base)
    before = signed(initial, faces) / denominator
    current = initial.copy()
    attempts, accepted, candidates = [], 0, 0
    started = time.perf_counter()
    for sweep in range(maximum_sweeps):
        bad = np.flatnonzero(signed(current, faces) / denominator <= 0)
        if not len(bad):
            break
        progress = False
        for face_index in bad:
            if signed(current, faces[face_index:face_index+1])[0] / denominator[face_index] > 0:
                continue
            variable = faces[face_index]
            affected = np.flatnonzero(np.isin(faces, variable).any(1))
            local_faces = faces[affected]
            prior = signed(current, local_faces) / denominator[affected]
            prior_good = prior > 0
            score_before = int(np.count_nonzero(prior <= 0)), -float(prior.min())
            best_score, best, best_displacement = score_before, None, None
            for full in (False, True):
                options = [neighbours(current[v], full) for v in variable]
                combinations = itertools.product(*[range(len(values)) for values in options])
                masks = [np.nonzero(local_faces == v) for v in variable]
                phase_count = 0
                while True:
                    keys = list(itertools.islice(combinations, chunk_size))
                    if not keys:
                        break
                    keys = np.asarray(keys, dtype=np.int64)
                    trial = np.stack([options[k][keys[:, k]] for k in range(3)], axis=1)
                    xyz = np.broadcast_to(current[local_faces],
                                          (len(keys), *current[local_faces].shape)).copy()
                    for k, (rows, corners) in enumerate(masks):
                        xyz[:, rows, corners, :] = trial[:, k, None, :]
                    determinant = (np.cross(xyz[:, :, 1] - xyz[:, :, 0],
                                            xyz[:, :, 2] - xyz[:, :, 0]) * xyz[:, :, 0]).sum(-1)
                    ratios = determinant / denominator[affected]
                    viable = np.isfinite(ratios).all(1) & (ratios[:, prior_good] > 0).all(1)
                    counts = np.count_nonzero(ratios <= 0, axis=1)
                    minimum = ratios.min(1)
                    offsets = np.linalg.norm(trial - current[variable], axis=-1)
                    for row in np.flatnonzero(viable):
                        score = int(counts[row]), -float(minimum[row])
                        distance = float(offsets[row].max()), float(offsets[row].sum())
                        improve = score < best_score
                        # For passing candidates select the smallest actual move.
                        if best is not None and score[0] == best_score[0] == 0:
                            improve = distance < best_displacement or (distance == best_displacement and score < best_score)
                        if improve:
                            best_score, best = score, trial[row].copy()
                            best_displacement = distance
                    phase_count += len(keys)
                    candidates += len(keys)
                if best is not None and best_score[0] == 0:
                    break
            if best is not None:
                current[variable] = best
                accepted += 1
                progress = True
                attempts.append({'sweep': sweep, 'affected_faces': len(affected),
                                 'remaining_affected_folded_faces': best_score[0],
                                 'minimum_affected_orientation_ratio': -best_score[1],
                                 'maximum_joint_displacement_mm': best_displacement[0]})
        if not progress:
            break
    after = signed(current, faces) / denominator
    newly_bad = int(np.count_nonzero((before > 0) & (after <= 0)))
    if newly_bad:
        raise AssertionError('A previously positive face was inverted')
    displacement = np.linalg.norm(current - initial, axis=1)
    report = {'scope': 'Explicit native-output quantized joint correction',
              'success': bool(np.all(after > 0)),
              'absolute_folded_faces_before': int(np.count_nonzero(before <= 0)),
              'absolute_folded_faces_after': int(np.count_nonzero(after <= 0)),
              'minimum_absolute_orientation_ratio_after': float(after.min()),
              'newly_folded_previously_positive_faces': newly_bad,
              'moved_vertices': int(np.count_nonzero(displacement)),
              'maximum_vertex_displacement_mm': float(displacement.max(initial=0)),
              'maximum_radius_error_mm': float(np.abs(np.linalg.norm(current, axis=1)-100).max(initial=0)),
              'accepted_joint_updates': accepted, 'evaluated_candidates': candidates,
              'seconds': time.perf_counter() - started,
              'attempts': attempts, 'cuda_used': False}
    return current, report

