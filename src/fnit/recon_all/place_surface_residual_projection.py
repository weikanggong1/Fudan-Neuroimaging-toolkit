"""显式逐面对分离候选；保持进入时的固定点及全部有序面。

独立实验算法，不等价于原 SOAP 清理，不改变默认表面放置路径。
"""
from __future__ import annotations

import time
import numpy as np
from scipy.optimize import minimize

from .place_surface_residual_cleanup import _checked, _face_normals, _pair_sets


def _skew(values):
    values = np.asarray(values)
    matrix = np.zeros(values.shape[:-1]+(3, 3), np.float64)
    matrix[..., 0, 1] = -values[..., 2]; matrix[..., 0, 2] = values[..., 1]
    matrix[..., 1, 0] = values[..., 2]; matrix[..., 1, 2] = -values[..., 0]
    matrix[..., 2, 0] = -values[..., 1]; matrix[..., 2, 1] = values[..., 0]
    return matrix


def _separating_planes(triangle):
    """按固定次序返回对方面的两个法向半空间及三个外侧边半空间。"""
    a, b, c = triangle.astype(np.float64)
    normal = np.cross(b-a, c-a)
    normal /= np.linalg.norm(normal)
    yield "normal_positive", a, normal
    yield "normal_negative", a, -normal
    for number, (start, end, inside) in enumerate(((a, b, c), (b, c, a), (c, a, b))):
        outward = np.cross(end-start, normal)
        outward /= np.linalg.norm(outward)
        if np.dot(inside-start, outward) > 0: outward = -outward
        yield "edge_outside_"+str(number), start, outward


def _constrained_projection(*, current, original, faces, initial_normals, initial_area2,
                            active, affected, moving_face, plane_origin, plane_axis,
                            max_displacement_mm, minimum_area_ratio, clearance_mm,
                            optimizer_iterations, opposing_face=None,
                            projection_mode="fixed_halfspaces"):
    """最小位移投影，解析约束Jacobian；接受仍由量化后的双检测决定。"""
    active = np.asarray(active, np.int64)
    local_slot = np.full(len(current), -1, np.int64); local_slot[active] = np.arange(len(active))
    local_faces = faces[affected]
    moving_ids = faces[moving_face]
    moving_slots = local_slot[moving_ids]
    signed = (current[moving_ids].astype(float)-plane_origin)@plane_axis
    free_plane = projection_mode == "free_separating_plane"
    if not free_plane and np.any(signed[moving_slots < 0] < clearance_mm):
        return None, {"decision": "fixed_corner_infeasible"}
    base = current.astype(np.float64)[local_faces]
    slots = local_slot[local_faces]
    target = current[active].astype(np.float64).copy()
    if not free_plane:
        for vertex, slot, distance in zip(moving_ids, moving_slots, signed):
            if slot >= 0: target[slot] += max(0., clearance_mm-distance)*plane_axis
    opponent = current[faces[opposing_face]].astype(float) if free_plane else None
    n0 = initial_normals[affected]; scale = initial_area2[affected]**2
    count = len(active)
    def geometry(flat):
        xyz = flat[:3*count].reshape(count, 3)
        tri = base.copy()
        valid = slots >= 0; tri[valid] = xyz[slots[valid]]
        u, v = tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]
        normals = np.cross(u, v)
        return xyz, normals, (_skew(v-u), -_skew(v), _skew(u))
    def constraints(flat):
        xyz, normal, _ = geometry(flat)
        orientation = np.sum(normal*n0, axis=1)/scale-1e-8
        area = np.sum(normal*normal, axis=1)/scale-minimum_area_ratio**2
        displacement = 1.-np.sum((xyz-original[active])**2, axis=1)/max_displacement_mm**2
        moving = current[moving_ids].astype(np.float64).copy()
        live = moving_slots >= 0; moving[live] = xyz[moving_slots[live]]
        if free_plane:
            axis, offset = flat[-4:-1], flat[-1]
            separation = moving@axis-offset-clearance_mm
            opponent_separation = offset-opponent@axis
            return np.concatenate((orientation, area, displacement, separation, opponent_separation))
        return np.concatenate((orientation, area, displacement, (moving-plane_origin)@plane_axis-clearance_mm))
    def jacobian(flat):
        xyz, normal, derivatives = geometry(flat)
        total = 2*len(affected)+count+3+(3 if free_plane else 0)
        matrix = np.zeros((total, 3*count+(4 if free_plane else 0)), np.float64)
        for corner, derivative in enumerate(derivatives):
            valid = slots[:, corner] >= 0
            row = np.flatnonzero(valid); col = slots[valid, corner]
            orient = np.einsum('fji,fj->fi', derivative[valid], n0[valid])/scale[valid, None]
            area = 2*np.einsum('fji,fj->fi', derivative[valid], normal[valid])/scale[valid, None]
            for axis in range(3):
                matrix[row, 3*col+axis] += orient[:, axis]
                matrix[len(affected)+row, 3*col+axis] += area[:, axis]
        for axis in range(3):
            matrix[2*len(affected)+np.arange(count), 3*np.arange(count)+axis] = -2*(xyz[:, axis]-original[active, axis])/max_displacement_mm**2
        plane_start = 2*len(affected)+count
        axis = flat[-4:-1] if free_plane else plane_axis
        for corner, slot in enumerate(moving_slots):
            if slot >= 0: matrix[plane_start+corner, 3*slot:3*slot+3] = axis
            if free_plane:
                moving_position = xyz[slot] if slot >= 0 else current[moving_ids[corner]]
                matrix[plane_start+corner, -4:-1] = moving_position
                matrix[plane_start+corner, -1] = -1.
        if free_plane:
            matrix[-3:, -4:-1] = -opponent
            matrix[-3:, -1] = 1.
        return matrix
    def objective(flat):
        value = np.sum((flat[:3*count].reshape(count, 3)-target)**2)
        if free_plane: value += 1e-6*np.sum((flat[-4:-1]-plane_axis)**2)
        return float(value)
    def gradient(flat):
        result = np.zeros(len(flat)); result[:3*count] = 2*(flat[:3*count]-target.ravel())
        if free_plane: result[-4:-1] = 2e-6*(flat[-4:-1]-plane_axis)
        return result
    initial = current[active].astype(float).ravel()
    optimizer_constraints = [{"type": "ineq", "fun": constraints, "jac": jacobian}]
    if free_plane:
        initial = np.concatenate((initial, plane_axis, [float(plane_axis@plane_origin)]))
        def unit_axis(flat): return np.sum(flat[-4:-1]**2)-1.
        def unit_axis_jac(flat):
            row = np.zeros(len(flat)); row[-4:-1] = 2*flat[-4:-1]; return row
        optimizer_constraints.append({"type": "eq", "fun": unit_axis, "jac": unit_axis_jac})
    tick = time.perf_counter()
    answer = minimize(objective, initial, jac=gradient,
                      method="SLSQP", constraints=optimizer_constraints,
                      options={"maxiter": optimizer_iterations, "ftol": 1e-9, "disp": False})
    values = constraints(answer.x)
    record = {"optimizer_success": bool(answer.success), "optimizer_status": int(answer.status),
              "optimizer_iterations": int(answer.nit), "minimum_optimizer_constraint": float(values.min()),
              "optimizer_seconds": time.perf_counter()-tick}
    if free_plane: record['unit_plane_constraint_error'] = float(abs(unit_axis(answer.x)))
    # SLSQP退出状态不替代真实质量检测；不把数值不可行候选写入坐标。
    if not np.isfinite(answer.x).all() or values.min() < -1e-9 or free_plane and abs(unit_axis(answer.x)) > 1e-9:
        return None, record | {"decision": "optimizer_infeasible"}
    candidate = current.copy(); candidate[active] = answer.x[:3*count].reshape(count, 3).astype(np.float32)
    return candidate, record


def resolve_projected_intersections(*, vertices: np.ndarray, faces: np.ndarray,
                                    fixed_vertices: np.ndarray, white_vertices: np.ndarray,
                                    max_displacement_mm: float = 1., minimum_area_ratio: float = .2,
                                    clearance_mm: float = .002, max_sweeps: int = 4,
                                    optimizer_iterations: int = 100,
                                    diagnostic_pair_limit: int | None = None,
                                    projection_mode: str = "fixed_halfspaces",
                                    crossing_guard: bool = False) -> tuple[np.ndarray, dict]:
    """逐相交面对的约束分离，返回坐标及全部接受/失败记录。

    vertices为有限float32(N,3)surface RAS/mm；faces为同索引整数(F,3)。
    fixed_vertices为bool(N,)，须含rip及cortex外顶点，固定在进入时坐标。
    white_vertices为有限自产同网格(N,3)，仅校验对应结构，不读取参考。
    最大累计位移默认1mm、面积比下限0.2保持上一协议；分离间距0.002mm、
    4轮及每次100次SLSQP为新候选参数，不是整体指标等效阈值。0/1/2环
    按固定顺序扩展，不动对方面顶点；两个法向侧与三个边外侧分别投影。
    diagnostic_pair_limit默认None完整处理，正整数冻结初始排序的前几个
    面对，仅用于最小定位；消除后不再补入其他面对。报告
    limited_diagnosis不能解释为完整清理。所有量化后受影响面需保朝向、
    面积、位移与固定点；source/F64两组阳性面对严格减少且不能新增。
    最终完整双谓词复检。无法消除返回failed_residual，输入不修改。
    非法参数/退化网格/全局与局部检测不一致抛异常；不调用原版程序、
    不启用GPU或改变TF32，解析Jacobian和SciPy求解采用F64。
    projection_mode默认fixed_halfspaces保持五个固定半空间；显式
    free_separating_plane以同五方向初始化可转动的单位法向/平面偏移，
    对方三个角点也全部约束在平面另一侧。它是独立新版候选族，顶点
    目标仍为最小位移、平面方向正则权重1e-6；质量门全部保持。
    crossing_guard默认False保留冻结v1候选；显式True缓存自产white并在
    每次候选完整受影响面检查无新增正长度内部穿越、无新增非内侧壁
    精确接触。近接触单独记录，不以源谓词容差抹掉短穿越。全固定且
    精确重合的内侧壁接触仍不能豁免穿越；初态已有穿越单列，本门
    通过不能解释为最终无穿越/整体质量通过。它不接入生产默认。
    """
    wall = time.perf_counter()
    original, triangles, fixed, _ = _checked(vertices, faces, fixed_vertices, white_vertices)
    for name, value in (("max_displacement_mm", max_displacement_mm), ("clearance_mm", clearance_mm)):
        if not np.isfinite(value) or value <= 0: raise ValueError(name+" must be positive finite")
    if not np.isfinite(minimum_area_ratio) or not 0 < minimum_area_ratio <= 1: raise ValueError("minimum_area_ratio must be in (0,1]")
    for name, value in (("max_sweeps", max_sweeps), ("optimizer_iterations", optimizer_iterations)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1: raise ValueError(name+" must be positive integer")
    if diagnostic_pair_limit is not None and (isinstance(diagnostic_pair_limit, (bool, np.bool_)) or not isinstance(diagnostic_pair_limit, (int, np.integer)) or diagnostic_pair_limit < 1):
        raise ValueError("diagnostic_pair_limit must be None or positive integer")
    if projection_mode not in ("fixed_halfspaces", "free_separating_plane"):
        raise ValueError("explicit supported projection_mode required")
    if not isinstance(crossing_guard, (bool, np.bool_)):
        raise ValueError("crossing_guard must be boolean")
    initial_normals = _face_normals(original, triangles); area2 = np.linalg.norm(initial_normals, axis=1)
    if np.any(area2 <= 0): raise ValueError("positive initial triangle areas required")
    result = original.copy(); source, independent = _pair_sets(result, triangles)
    initial_source, initial_independent = source.copy(), independent.copy()
    crossing_index = None
    crossing_pairs = nonmedial_contacts = set()
    initial_crossing_pairs = initial_nonmedial_contacts = set()
    crossing_qc_initial = None
    if crossing_guard:
        from .white_pial_crossing_python import WhitePialCrossingIndex
        crossing_index = WhitePialCrossingIndex(white_vertices=white_vertices, faces=triangles, fixed_vertices=fixed)
        crossing_pairs, contacts, legal, crossing_qc_initial = crossing_index.query(original)
        nonmedial_contacts = contacts-legal
        initial_crossing_pairs = crossing_pairs.copy(); initial_nonmedial_contacts = nonmedial_contacts.copy()
    selected_initial_pairs = None if diagnostic_pair_limit is None else set(
        sorted({tuple(sorted(pair)) for pair in source} | independent)[:diagnostic_pair_limit])
    adjacency = [set() for _ in range(len(result))]
    for a, b, c in triangles:
        adjacency[a].update((int(b), int(c))); adjacency[b].update((int(a), int(c))); adjacency[c].update((int(a), int(b)))
    report = {"schema_version": 1, "algorithm": "independent_projected_pair_solver_v1" if projection_mode == "fixed_halfspaces" else "independent_free_plane_pair_solver_v1",
              "parameters": {"max_displacement_mm": float(max_displacement_mm), "minimum_area_ratio": float(minimum_area_ratio),
                             "clearance_mm": float(clearance_mm), "max_sweeps": int(max_sweeps), "optimizer_iterations": int(optimizer_iterations),
                             "rings": [0, 1, 2], "optimizer_ftol": 1e-9, "optimizer_feasibility_tolerance": 1e-9,
                             "diagnostic_pair_limit": diagnostic_pair_limit, "projection_mode": projection_mode,
                             "free_plane_direction_regularization": 1e-6 if projection_mode == "free_separating_plane" else None,
                             "free_plane_unit_norm_tolerance": 1e-9 if projection_mode == "free_separating_plane" else None},
              "source_directed_pairs_before": len(source), "float64_pairs_before": len(independent),
              "fixed_vertices": int(fixed.sum()), "candidate_trace": [], "accepted": [],
              "limited_initial_pairs": None if selected_initial_pairs is None else [list(pair) for pair in sorted(selected_initial_pairs)],
              "strict_reproduction": "independent_algorithm_not_source_reproduction", "overall_metric_equivalence": "not_assessed"}
    if crossing_guard:
        report['algorithm'] = ('independent_free_plane_pair_solver_v2_crossing_guard' if projection_mode == 'free_separating_plane'
                               else 'independent_projected_pair_solver_v2_crossing_guard')
        report['parameters']['crossing_guard'] = True
        report['white_pial_guard'] = {'enabled': True, 'initial': crossing_qc_initial,
            'initial_nonmedial_exact_contacts': len(initial_nonmedial_contacts),
            'rule': 'no new proper interior crossing or nonmedial exact contact; not final-zero or whole-equivalence gate'}
    for sweep in range(max_sweeps):
        pairs = sorted({tuple(sorted(pair)) for pair in source} | independent)
        if selected_initial_pairs is not None: pairs = [pair for pair in pairs if pair in selected_initial_pairs]
        accepted_in_sweep = 0
        for first, second in pairs:
            if (first, second) not in independent and (first, second) not in source and (second, first) not in source: continue
            eligible = None
            for moving_face, opposing_face in ((first, second), (second, first)):
                active = set(map(int, triangles[moving_face])); active.difference_update(map(int, np.flatnonzero(fixed)))
                opposing_vertices = set(map(int, triangles[opposing_face])); active.difference_update(opposing_vertices)
                for ring in (0, 1, 2):
                    if ring: active.update({w for v in tuple(active) for w in adjacency[v] if not fixed[w] and w not in opposing_vertices})
                    ids = np.asarray(sorted(active), np.int64)
                    if not len(ids): continue
                    affected = np.flatnonzero(np.isin(triangles, ids).any(axis=1))
                    current_source, current_independent = _pair_sets(result, triangles, affected)
                    for name, origin, axis in _separating_planes(result[triangles[opposing_face]]):
                        candidate, record = _constrained_projection(current=result, original=original, faces=triangles,
                            initial_normals=initial_normals, initial_area2=area2, active=ids, affected=affected,
                            moving_face=moving_face, plane_origin=origin, plane_axis=axis,
                            max_displacement_mm=max_displacement_mm, minimum_area_ratio=minimum_area_ratio,
                            clearance_mm=clearance_mm, optimizer_iterations=optimizer_iterations,
                            opposing_face=opposing_face, projection_mode=projection_mode)
                        row = {"sweep": sweep, "pair": [first, second], "moving_face": moving_face,
                               "ring": ring, "plane": name, "moving_vertices": len(ids), "affected_faces": len(affected)} | record
                        if candidate is not None:
                            normals = _face_normals(candidate, triangles[affected])
                            displacement = np.linalg.norm(candidate[ids].astype(float)-original[ids], axis=1)
                            reason = None
                            if displacement.max(initial=0.) > max_displacement_mm: reason = "total_displacement_limit"
                            elif np.any(np.sum(normals*initial_normals[affected], axis=1) <= 0): reason = "face_orientation"
                            elif np.any(np.linalg.norm(normals, axis=1) < minimum_area_ratio*area2[affected]): reason = "face_area"
                            if reason is None:
                                after_source, after_independent = _pair_sets(candidate, triangles, affected)
                                row.update(source_directed_pairs=len(after_source), float64_pairs=len(after_independent))
                                if not after_source <= current_source or not after_independent <= current_independent: reason = "new_positive_pair"
                                elif len(after_source) >= len(current_source) or len(after_independent) >= len(current_independent): reason = "not_strictly_reduced_in_both_detectors"
                                else:
                                    next_crossings, next_contacts = crossing_pairs, nonmedial_contacts
                                    if crossing_index is not None:
                                        after_crossings, contacts, legal, crossing_qc = crossing_index.query(candidate, affected)
                                        after_contacts = contacts-legal; affected_set = set(map(int,affected))
                                        current_crossings = {pair for pair in crossing_pairs if pair[0] in affected_set}
                                        current_contacts = {pair for pair in nonmedial_contacts if pair[0] in affected_set}
                                        row['white_pial_crossing_qc'] = crossing_qc
                                        row['new_proper_crossings'] = len(after_crossings-current_crossings)
                                        row['new_nonmedial_exact_contacts'] = len(after_contacts-current_contacts)
                                        if not after_crossings <= current_crossings: reason = 'new_white_pial_proper_crossing'
                                        elif not after_contacts <= current_contacts: reason = 'new_white_pial_nonmedial_exact_contact'
                                        else:
                                            next_crossings = (crossing_pairs-current_crossings)|after_crossings
                                            next_contacts = (nonmedial_contacts-current_contacts)|after_contacts
                                    if reason is None:
                                        row['decision'] = 'eligible'
                                        eligible = (candidate, (source-current_source)|after_source, (independent-current_independent)|after_independent,
                                                    next_crossings, next_contacts, row.copy())
                            row['decision'] = reason or 'eligible'
                        report['candidate_trace'].append(row)
                        if eligible is not None: break
                    if eligible is not None: break
                if eligible is not None: break
            if eligible is not None:
                result, source, independent, crossing_pairs, nonmedial_contacts, row = eligible
                report['accepted'].append(row); accepted_in_sweep += 1
        if not accepted_in_sweep or not source and not independent: break
    final_source, final_independent = _pair_sets(result, triangles)
    guards = {"full_source_pair_set_matches_local_update": source == final_source,
              "full_float64_pair_set_matches_local_update": independent == final_independent,
              "no_new_source_directed_pairs": final_source <= initial_source,
              "no_new_float64_pairs": final_independent <= initial_independent}
    if not all(guards.values()) or not np.array_equal(result[fixed], original[fixed]): raise RuntimeError("global pair/fixed validation failed")
    if crossing_index is not None:
        full_crossings, contacts, legal, final_qc = crossing_index.query(result)
        full_contacts = contacts-legal
        crossing_checks = {'full_proper_pair_set_matches_local_update': full_crossings == crossing_pairs,
                           'full_nonmedial_contact_set_matches_local_update': full_contacts == nonmedial_contacts,
                           'no_new_proper_crossings': full_crossings <= initial_crossing_pairs,
                           'no_new_nonmedial_exact_contacts': full_contacts <= initial_nonmedial_contacts}
        if not all(crossing_checks.values()): raise RuntimeError('global white/pial crossing/contact validation failed')
        report['white_pial_guard'].update(final=final_qc, checks=crossing_checks,
            final_nonmedial_exact_contacts=len(full_contacts), no_new_gate='passed',
            final_proper_crossings_zero=not full_crossings,
            final_nonmedial_exact_contacts_zero=not full_contacts)
    displacement = np.linalg.norm(result.astype(float)-original, axis=1)
    report.update(status="passed_zero_intersections" if not source and not independent else "failed_residual",
                  scope="complete_residual_stage" if diagnostic_pair_limit is None else "limited_diagnosis",
                  source_directed_pairs_after=len(source), float64_pairs_after=len(independent),
                  source_intersecting_faces_after=len({a for a, b in source}),
                  float64_intersecting_faces_after=len({x for pair in independent for x in pair}),
                  fixed_different_elements=0, changed_vertices=int(np.count_nonzero(displacement)),
                  mean_displacement_mm=float(displacement.mean()), p99_displacement_mm=float(np.quantile(displacement, .99)),
                  max_displacement_mm=float(displacement.max(initial=0.)), global_checks=[guards],
                  full_resolver_seconds=time.perf_counter()-wall)
    return result, report
