"""独立 GA 最终缺陷写回；复用候选闭环和几何，不重新初始化曲率。"""
from __future__ import annotations

from collections.abc import Mapping
import time
import numpy as np

from .topology_candidate_cycle import evaluate_candidate_cycle, materialize_candidate_state, _one_ring_state
from .topology_candidate_geometry import candidate_geometry
from .topology_ga_ordering import _permutation


def finalize_vertex_usage(*, vertex_marks, vertex_rip, candidate_edges, edge_used,
        translated_defect_vertices, discarded_defect_vertices, translated_border_vertices):
    """最终边表→源 FINAL_VERTEX 标记、DISCARD/rip 和域内标记清理。

    marks:int32(N,)，rip:int32(N,) signed-char [-128,127]，edges:整数
    (E,2)，used:整数(E,)源0..6，defect:整数(K,)/discarded:bool(K,)，
    border:整数(B,)；全部索引为同版本corrected顶点序。3/4边端点先标2，
    按原缺陷顺序清mark，未标2的顶点设DISCARD/rip=1，最后清border。
    原本DISCARD但被使用的条目不擅自恢复；重复映射保留顺序语义。
    输入不修改。返回新marks/rip/DISCARD、新增剔除原条目int64(A,)和
    additional_discarded_count。负映射/越界/非法类型ValueError，不执行
    原版，不压缩顶点/重排面，不检查自相交，不改变精度和线程。
    全参必填；对应mris_fix_topology内部最终unused-vertex清理，无CLI。
    """
    marks, rip, edges, used, mapped, discarded, border = map(np.asarray,
        (vertex_marks, vertex_rip, candidate_edges, edge_used,
         translated_defect_vertices, discarded_defect_vertices, translated_border_vertices))
    if (marks.ndim != 1 or marks.dtype != np.int32 or rip.shape != marks.shape
            or rip.dtype != np.int32 or np.any(rip < -128) or np.any(rip > 127)):
        raise ValueError('same-order int32 marks and signed-char rip required')
    n = len(marks)
    for values, shape, name in [(edges, (None, 2), 'candidate_edges'),
            (mapped, (None,), 'translated_defect_vertices'), (border, (None,), 'translated_border_vertices')]:
        if (values.ndim != len(shape) or any(s is not None and values.shape[i] != s for i, s in enumerate(shape))
                or values.dtype.kind not in 'iu' or values.size and (values.min() < 0 or values.max() >= n)):
            raise ValueError(name+' requires complete in-range integer indices; negative mapping unsupported')
    if (used.shape != (len(edges),) or used.dtype.kind not in 'iu'
            or np.any(used < 0) or np.any(used > 6)
            or discarded.dtype != np.bool_ or discarded.shape != mapped.shape):
        raise ValueError('source edge states 0..6 and same-defect bool DISCARD required')
    new_marks, new_rip, new_discarded = marks.copy(), rip.copy(), discarded.copy()
    new_marks[edges[(used == 3) | (used == 4)].reshape(-1)] = 2
    newly = []
    for entry, vertex in enumerate(mapped):
        if new_marks[vertex] != 2:
            if not new_discarded[entry]:
                newly.append(entry)
            new_discarded[entry] = True
            new_rip[vertex] = 1
        new_marks[vertex] = 0
    new_marks[border] = 0
    return dict(vertex_marks=new_marks, vertex_rip=new_rip,
        discarded_defect_vertices=new_discarded,
        newly_discarded_entries=np.asarray(newly, np.int64), additional_discarded_count=len(newly))


def finalize_defect_patch(*, surface_state, snapshot, retessellation_inputs,
        candidate_inputs, best_ordering, best_discarded, best_fitness,
        use_counts, vertex_fitness, check_surface_intersection, intersection_checker):
    """GA搜索末态→最优状态回填→最终评分/恢复→重连/MRI匹配→清理。

    surface_state为完整自产一环/all候选末态，包含DVS及持久K/E/fx/
    curvbak/area等原字段；snapshot为同基底record_candidate_state的原
    快照。retessellation_inputs为完整candidate_geometry参数映射，
    inside_vertices全为有效corrected整数(K,)，不能用参考after补初态。
    candidate_inputs为evaluate_candidate完整自产MRI/矩阵/先验/权重。
    best_ordering为完整整数(E,)排列，best_discarded为原序bool(K,)，
    best_fitness为有限实际搜索double分数，RP计数int32(K,)非负、反馈
    float32(K,)有限。XYZ为float32 surface RAS/mm，曲率mm⁻¹，面面积mm²。

    check_surface_intersection为显式bool；为True时intersection_checker
    必须为具名回调(original_vertices,faces,patch_faces)→非负整数相交面
    数，为False时必须None。本标志只复现源可选检查，不替代生产质量门。
    全参必填。仅重设缺陷内rip为最优DISCARD的0/1，保留外部signed值；
    最终再评分一次，更新RP、选择性恢复和边复位后，完整重做几何。
    第二次几何不重复SFF/位移/评分/RP；保留最后评分真实K/E/fx/curvbak。
    最终分数与搜索最优不同只记录，不替换为历史分数。

    返回surface_state(完整最终面/邻接/坐标/法向)、edge_used、DISCARD、
    RP、fitness/fitness_matches_best、final_geometry、final_evaluation、
    cleanup、source_intersections或None、各步骤seconds；
    final_tessellation_complete=True仅表示这个缺陷最终写回，complete_ga
    仍False、final_mesh_quality=not_assessed，未压缩/修复全表面或写标准文件。
    非法/不完整状态ValueError，子函数/检查回调异常传播，输入不修改，
    失败不发布半成品。不改变CUDA/TF32/半精度/线程，不调用原版程序。
    """
    start = time.perf_counter()
    n, _, nf = _one_ring_state(surface_state)
    if not isinstance(retessellation_inputs, Mapping) or not isinstance(candidate_inputs, Mapping):
        raise ValueError('complete explicit retessellation and candidate parameter mappings required')
    if (not isinstance(check_surface_intersection, (bool, np.bool_))
            or check_surface_intersection and not callable(intersection_checker)
            or not check_surface_intersection and intersection_checker is not None):
        raise ValueError('explicit intersection policy and matching checker required')
    args = dict(retessellation_inputs)
    mapped = np.asarray(args['inside_vertices'])
    discarded = np.asarray(best_discarded)
    if (mapped.ndim != 1 or mapped.dtype.kind not in 'iu'
            or np.any(mapped < 0) or np.any(mapped >= n)
            or discarded.dtype != np.bool_ or discarded.shape != mapped.shape):
        raise ValueError('complete valid defect mapping and same-order best DISCARD required')
    ordering = _permutation(best_ordering, 'best_ordering')
    if len(ordering) != len(args['candidate_edges']):
        raise ValueError('best ordering must cover complete candidate edge table')
    best = np.asarray(best_fitness)
    if best.ndim or best.dtype.kind not in 'iuf' or not np.isfinite(best):
        raise ValueError('finite actual best_fitness required')
    rip = np.asarray(surface_state['vertex_rip'])
    if (rip.shape != (n,) or rip.dtype.kind not in 'biu'
            or np.any(rip < -128) or np.any(rip > 127)):
        raise ValueError('complete signed-char rip values required')
    state = dict(surface_state)
    rip = rip.astype(np.int32, copy=True)
    for entry, vertex in enumerate(mapped):
        rip[vertex] = int(discarded[entry])
    state['vertex_rip'] = rip
    args.update(ordering=ordering.copy(), inside_discarded=discarded.copy(), vertex_rip=rip != 0)
    if not np.array_equal(np.asarray(state['faces'])[:nf], args['base_faces']):
        raise ValueError('same original logical base faces required')
    tick = time.perf_counter()
    evaluation = evaluate_candidate_cycle(surface_state=state, snapshot=snapshot,
        retessellation_inputs=args, candidate_inputs=candidate_inputs,
        use_counts=use_counts, vertex_fitness=vertex_fitness)
    seconds = dict(final_score_restore=time.perf_counter()-tick)
    restored = evaluation['surface_state']
    args['edge_used'] = evaluation['edge_used']
    tick = time.perf_counter()
    local = candidate_inputs['local_priors']
    geometry = candidate_geometry(original_vertices=restored['original_vertices'],
        volume=candidate_inputs['volume'], ras_to_voxel=candidate_inputs['ras_to_voxel'],
        white_mean=local['white_mean'], gray_mean=local['gray_mean'],
        k1_mean=candidate_inputs['k1_mean'], k2_mean=candidate_inputs['k2_mean'],
        retessellation_inputs=args)
    normals = np.asarray(restored['vertex_normals']).copy()
    normals[geometry['selected_vertices']] = geometry['selected_vertex_normals']
    face_area = np.concatenate((np.asarray(restored['face_area'])[:nf],
        np.zeros(len(geometry['faces'])-nf, np.float32)))
    materialized = materialize_candidate_state(initial=restored, result=dict(geometry=geometry,
        vertex_normals=normals, face_area=face_area,
        flattened_vertices=restored['flattened_vertices'], vertex_curvbak=restored['curvbak'],
        sff={key:restored[key] for key in ('k1', 'k2', 'tangent_e1', 'tangent_e2')}))
    seconds['final_retessellation_and_match'] = time.perf_counter()-tick
    tick = time.perf_counter()
    intersections = None
    if check_surface_intersection:
        value = np.asarray(intersection_checker(original_vertices=geometry['original_vertices'],
            faces=geometry['faces'], patch_faces=geometry['selected_faces']))
        if value.ndim or value.dtype.kind not in 'iu' or int(value) < 0:
            raise ValueError('intersection checker must return nonnegative integer face count')
        intersections = int(value)
    seconds['optional_source_intersection_check'] = time.perf_counter()-tick
    tick = time.perf_counter()
    cleanup = finalize_vertex_usage(vertex_marks=np.asarray(materialized['vertex_marks'], np.int32),
        vertex_rip=np.asarray(materialized['vertex_rip'], np.int32),
        candidate_edges=args['candidate_edges'], edge_used=geometry['edge_used'],
        translated_defect_vertices=mapped, discarded_defect_vertices=discarded,
        translated_border_vertices=args['border_vertices'])
    materialized.update(vertex_marks=cleanup['vertex_marks'], vertex_rip=cleanup['vertex_rip'])
    seconds['final_usage_cleanup'] = time.perf_counter()-tick
    seconds['complete_finalization_api'] = time.perf_counter()-start
    actual_fitness = evaluation['fitness']
    return dict(surface_state=materialized, edge_used=geometry['edge_used'],
        discarded_defect_vertices=cleanup['discarded_defect_vertices'],
        use_counts=evaluation['use_counts'], vertex_fitness=evaluation['vertex_fitness'],
        fitness=actual_fitness, best_fitness=float(best), fitness_matches_best=actual_fitness == float(best),
        final_geometry=geometry, final_evaluation=evaluation, cleanup=cleanup,
        source_intersections=intersections, seconds=seconds,
        final_tessellation_complete=True, complete_ga=False,
        final_mesh_quality='not_assessed', overall_metric_equivalence='not_assessed')
