"""Explicit native-output repair, checked at the precision written to GIFTI.

This operation changes registration coordinates. It is deliberately separate
from the source-compatible control/data mesh updates and the default report
policy. Every guarded move preserves the positive faces in its own star.
"""
from __future__ import annotations

import time

import numpy as np
import torch


def _signed_faces(points, faces):
    triangles = np.asarray(points, dtype=np.float64)[faces]
    return (np.cross(triangles[:, 1] - triangles[:, 0],
                     triangles[:, 2] - triangles[:, 0]) * triangles[:, 0]).sum(1)


def _validate_geometry(vertices, faces, original):
    points = np.asarray(vertices, dtype=np.float64)
    reference = np.asarray(original, dtype=np.float64)
    triangles = np.asarray(faces)
    if (points.ndim != 2 or points.shape[1] != 3 or reference.shape != points.shape
            or not len(points)):
        raise ValueError("native coordinates and orientation reference must be matching N x 3 arrays")
    if not np.isfinite(points).all() or not np.isfinite(reference).all():
        raise ValueError("native coordinates and orientation reference must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        saved = points.astype(np.float32)
    if not np.isfinite(saved).all():
        raise ValueError("native coordinates must remain finite in float32 GIFTI output")
    if np.any(np.linalg.norm(points, axis=1) <= 1e-12):
        raise ValueError("native sphere contains a zero-radius vertex")
    if (triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles)
            or not np.issubdtype(triangles.dtype, np.integer)):
        raise ValueError("native faces must be a nonempty integer M x 3 array")
    if triangles.min() < 0 or triangles.max() >= len(points):
        raise ValueError("native faces contain an out-of-range vertex index")
    triangles = triangles.astype(np.int64, copy=True)
    if np.any(np.diff(np.sort(triangles, axis=1), axis=1) == 0):
        raise ValueError("native faces contain repeated vertex indices")
    if len(np.unique(triangles)) != len(points):
        raise ValueError("each native vertex must belong to a face")
    baseline = _signed_faces(reference, triangles)
    if not np.isfinite(baseline).all() or np.any(baseline == 0):
        raise ValueError("native orientation reference must be finite and nondegenerate")
    if np.count_nonzero(baseline > 0) == np.count_nonzero(baseline < 0):
        raise ValueError("native orientation reference must have a majority winding")
    return points.copy(), triangles, reference.copy(), baseline


def _guarded_repair(vertices, faces, baseline, *, maximum_sweeps=64, audit=None,
                    face_gradients=False):
    """Improve folded vertex stars using bounded, float32-tested moves.

    Accept a move only when the local folded-face count decreases, or when
    that count is unchanged and the local minimum orientation ratio improves.
    A previously positive incident face must remain positive. ``audit`` is an
    internal testing hook receiving only the local before/after ratios.
    """
    if maximum_sweeps < 0:
        raise ValueError("maximum_sweeps must be nonnegative")
    points = np.asarray(vertices, dtype=np.float32).astype(np.float64)
    if not np.isfinite(points).all():
        raise ValueError("guarded native coordinates must remain finite in float32")
    incident = [[] for _ in points]
    for index, triangle in enumerate(faces):
        for vertex in triangle:
            incident[vertex].append(index)
    updates = 0
    sweeps = 0
    termination = "sweep_limit"
    for sweep in range(maximum_sweeps):
        ratios = _signed_faces(points, faces) / baseline
        bad = np.flatnonzero(ratios <= 0)
        if not len(bad):
            termination = "pass"
            break
        sweeps = sweep + 1
        progress = False
        for vertex in np.unique(faces[bad]):
            local_faces = np.asarray(incident[vertex], dtype=np.int64)
            local_triangles = faces[local_faces]
            neighbours = np.unique(local_triangles)
            neighbours = neighbours[neighbours != vertex]
            start = points[vertex].copy()
            before = _signed_faces(points, local_triangles) / baseline[local_faces]
            if not np.any(before <= 0):
                continue
            target = points[neighbours].mean(0)
            length = np.linalg.norm(target)
            directions = []
            if np.isfinite(length) and length > 1e-12:
                directions.append(target * (100.0 / length) - start)
            if face_gradients:
                gradients = []
                for row in np.flatnonzero(before <= 0):
                    triangle = local_triangles[row]
                    position = int(np.flatnonzero(triangle == vertex)[0])
                    first, second = triangle[(position + 1) % 3], triangle[(position + 2) % 3]
                    gradient = np.cross(points[first], points[second]) / baseline[local_faces[row]]
                    gradient -= start * (np.dot(gradient, start) / np.dot(start, start))
                    gradient_length = np.linalg.norm(gradient)
                    if np.isfinite(gradient_length) and gradient_length > 1e-12:
                        gradients.append(gradient)
                        directions.append(gradient * (100.0 / gradient_length))
                if gradients:
                    combined = np.sum(gradients, axis=0)
                    gradient_length = np.linalg.norm(combined)
                    if np.isfinite(gradient_length) and gradient_length > 1e-12:
                        directions.append(combined * (100.0 / gradient_length))
            best_score = (int(np.count_nonzero(before <= 0)), -float(before.min()))
            best = None
            steps = ((1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, .01, .05, .1, .25, .5, 1.)
                     if face_gradients else (1e-5, 1e-4, 1e-3, .01, .05, .1, .25, .5, 1.))
            for direction in directions:
                for step in steps:
                    trial = start + step * direction
                    trial_length = np.linalg.norm(trial)
                    if not np.isfinite(trial_length) or trial_length <= 1e-12:
                        continue
                    points[vertex] = (trial * (100.0 / trial_length)).astype(np.float32)
                    after = _signed_faces(points, local_triangles) / baseline[local_faces]
                    if not np.isfinite(after).all() or np.any(after[before > 0] <= 0):
                        continue
                    score = (int(np.count_nonzero(after <= 0)), -float(after.min()))
                    if score < best_score:
                        best_score = score
                        best = points[vertex].copy()
                        if score[0] == 0:
                            break
                if best_score[0] == 0:
                    break
            points[vertex] = start if best is None else best
            if best is not None:
                updates += 1
                progress = True
                if audit is not None:
                    audit(before.copy(), (_signed_faces(points, local_triangles)
                                          / baseline[local_faces]).copy())
        if not progress:
            termination = "stalled"
            break
    remaining = int(np.count_nonzero(_signed_faces(points, faces) / baseline <= 0))
    if remaining == 0:
        termination = "pass"
    return points, {"success": remaining == 0, "sweeps": sweeps, "updates": updates,
                    "maximum_sweeps": maximum_sweeps, "termination": termination,
                    "remaining_folded_faces": remaining,
                    "newly_folded_previously_positive_faces": 0}


def repair_native_sphere(vertices, faces, original, *, unfold_budgets=(128, 512, 1000),
                         maximum_guard_sweeps=64, maximum_patch_rings=16,
                         maximum_quantized_sweeps=8):
    """Return repaired coordinates and aggregate diagnostics without mutation.

    Try guarded local moves and quantized joint neighbours first. If necessary,
    restart the original saved coordinates for fixed-boundary harmonic patches
    and a final quantized correction. Source unfolding is a bounded fallback;
    the registration solver retains its original 1000-sweep budget.
    """
    from .msmsulc import _unfold

    if not isinstance(vertices, torch.Tensor) or vertices.dtype not in (torch.float32, torch.float64):
        raise TypeError("native vertices must be a float32 or float64 torch.Tensor")
    budgets = tuple(unfold_budgets)
    if (not budgets or any(not isinstance(value, int) or value < 0 for value in budgets)
            or not isinstance(maximum_guard_sweeps, int) or maximum_guard_sweeps < 0):
        raise ValueError("native repair budgets must be nonnegative integers")
    if any(not isinstance(value, int) or value < 0
           for value in (maximum_patch_rings, maximum_quantized_sweeps)):
        raise ValueError("native patch budgets must be nonnegative integers")
    started = time.perf_counter()
    points, triangles, reference, reference_signed = _validate_geometry(
        vertices.detach().cpu().numpy(), faces, original)
    sign = 1 if np.count_nonzero(reference_signed > 0) > np.count_nonzero(reference_signed < 0) else -1
    # An input native sphere can already contain inward faces. Repair toward
    # its majority winding, rather than preserving each defective face's sign.
    baseline = sign * np.abs(reference_signed)
    input_saved = points.astype(np.float32)
    before_ratios = _signed_faces(input_saved, triangles) / baseline
    before_bad = before_ratios <= 0
    report = {"applied": bool(before_bad.any()), "success": not bool(before_bad.any()),
              "moved_vertices": 0, "unfold_updates": 0, "guard_updates": 0,
              "harmonic_updates": 0, "quantized_updates": 0,
              "seconds": 0.0, "attempts": [],
              "orientation_reference_sign": sign,
              "reference_folded_faces": int(np.count_nonzero(reference_signed * sign <= 0)),
              "orientation_target": "absolute majority winding at written float32 precision",
              "maximum_vertex_displacement_mm": 0.0, "mean_vertex_displacement_mm": 0.0,
              "guard_preserves_positive_incident_faces": True}
    if not report["applied"]:
        report["seconds"] = time.perf_counter() - started
        return vertices.clone(), report
    best = input_saved.astype(np.float64)
    best_score = (int(before_bad.sum()), -float(before_ratios.min()))

    def consider(candidate):
        nonlocal best, best_score
        ratios = _signed_faces(candidate, triangles) / baseline
        score = (int(np.count_nonzero(ratios <= 0)), -float(ratios.min()))
        if np.isfinite(ratios).all() and score < best_score:
            best, best_score = candidate.copy(), score
        return score[0] == 0

    from ._native_patch import harmonic_repair, quantized_joint_repair
    if maximum_guard_sweeps:
        attempt_started = time.perf_counter()
        direct, guard = _guarded_repair(input_saved, triangles, baseline,
                                        maximum_sweeps=maximum_guard_sweeps,
                                        face_gradients=True)
        report["guard_updates"] += guard["updates"]
        report["attempts"].append({"stage": "direct_guard", "guard": guard,
                                   "seconds": time.perf_counter() - attempt_started})
        consider(direct)
    if best_score[0] and maximum_quantized_sweeps:
        corrected, quantized = quantized_joint_repair(best, triangles, reference,
                                                      maximum_sweeps=maximum_quantized_sweeps)
        report["quantized_updates"] += quantized["accepted_joint_updates"]
        report["attempts"].append({"stage": "direct_quantized", "quantized": quantized})
        consider(corrected)
    if best_score[0] and maximum_patch_rings:
        # Preserving faces made positive by an unsuccessful preliminary repair
        # can block a valid joint move. Restart the untouched input boundary.
        harmonic, patch = harmonic_repair(input_saved, triangles, reference,
                                           maximum_rings=maximum_patch_rings,
                                           allow_nonconvex=True)
        report["harmonic_updates"] += patch["accepted_joint_updates"]
        report["attempts"].append({"stage": "raw_harmonic_restart", "harmonic": patch})
        consider(harmonic)
        if not patch["success"] and maximum_quantized_sweeps:
            corrected, quantized = quantized_joint_repair(harmonic, triangles, reference,
                                                          maximum_sweeps=maximum_quantized_sweeps)
            report["quantized_updates"] += quantized["accepted_joint_updates"]
            report["attempts"].append({"stage": "harmonic_quantized", "quantized": quantized})
            consider(corrected)
    for budget in budgets if best_score[0] else ():
        attempt_started = time.perf_counter()
        coarse, moved = _unfold(torch.as_tensor(points, dtype=torch.float64,
                                               device=vertices.device), triangles,
                                maximum_sweeps=budget)
        coarse = coarse.detach().cpu().numpy()
        with np.errstate(over="ignore", invalid="ignore"):
            coarse_saved = coarse.astype(np.float32)
        if not np.isfinite(coarse_saved).all():
            report["unfold_updates"] += int(moved)
            report["attempts"].append({"unfold_budget": budget, "unfold_updates": int(moved),
                                       "termination": "nonfinite_unfold_output",
                                       "seconds": time.perf_counter() - attempt_started})
            continue
        coarse_ratios = _signed_faces(coarse_saved, triangles) / baseline
        guarded, guard = _guarded_repair(coarse, triangles, baseline,
                                         maximum_sweeps=maximum_guard_sweeps)
        ratios = _signed_faces(guarded, triangles) / baseline
        score = (int(np.count_nonzero(ratios <= 0)), -float(ratios.min()))
        report["unfold_updates"] += int(moved)
        report["guard_updates"] += guard["updates"]
        report["attempts"].append({"unfold_budget": budget, "unfold_updates": int(moved),
                                   "coarse_folded_faces": int(np.count_nonzero(coarse_ratios <= 0)),
                                   "coarse_newly_folded_faces": int(np.count_nonzero(
                                       (coarse_ratios <= 0) & ~before_bad)),
                                   "guard": guard, "seconds": time.perf_counter() - attempt_started})
        consider(guarded)
        if score[0] == 0:
            break
    displacement = np.linalg.norm(best - input_saved.astype(np.float64), axis=1)
    report.update(success=best_score[0] == 0, remaining_folded_faces=best_score[0],
                  newly_folded_previously_positive_faces=int(np.count_nonzero(
                      (_signed_faces(best, triangles) / baseline <= 0) & ~before_bad)),
                  moved_vertices=int(np.count_nonzero(displacement)),
                  maximum_vertex_displacement_mm=float(displacement.max()),
                  mean_vertex_displacement_mm=float(displacement.mean()),
                  seconds=time.perf_counter() - started)
    return torch.as_tensor(best, dtype=vertices.dtype, device=vertices.device), report
