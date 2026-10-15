"""自产nofix/qsphere/MRI→按序完整GA、SOAP、压缩与两个坐标域质量检查。"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import time
import numpy as np

from .thread_budget import thread_budget
from .topology_genetic_base_prepare import prepare_nofix_genetic_base
from .topology_global_priors import global_topology_priors
from .topology_candidate_inputs import (prepare_candidate_input_context,
    prepare_defect_candidate_inputs)
from .topology_genetic_defect_inputs import prepare_ordered_genetic_defect_context
from .topology_genetic_search import (search_and_finalize_defect, TopologyRandomStream,
    source_population_limits)
from .topology_finish_geometry import finish_and_compact_topology
from .place_surface_final_cleanup import repair_intersections


def _next_inputs(**kwargs):
    # Separate current corrected adjacency from the immutable original cache.
    from .topology_genetic_next_defect_inputs import next_ordered_genetic_defect_inputs
    return next_ordered_genetic_defect_inputs(**kwargs)


def _ordered_csr(state):
    """只消费当前vt.v的逻辑vnum，不从面重建或丢弃悬空邻接。"""
    topology=np.asarray(state['vertex_topology']);n=len(topology)
    if topology.shape!=(n,7) or topology.dtype.kind not in 'iu':
        raise ValueError('complete current ordered vertex topology required')
    rows=state['neighbor_storage']
    if len(rows)!=n:raise ValueError('complete current neighbor storage required')
    packed=[]
    for vertex,row in enumerate(rows):
        a=np.asarray(row);degree=int(topology[vertex,3])
        if degree==0 and isinstance(row,(list,tuple)):a=np.empty(0,np.int32)
        if (a.ndim!=1 or a.dtype.kind not in 'iu' or not 0<=degree<=len(a)
                or np.any(a[:degree]<0) or np.any(a[:degree]>=n)):
            raise ValueError('logical current vnum exceeds valid ordered storage')
        packed.append(np.array(a[:degree],np.int32,copy=True))
    offsets=np.r_[0,np.cumsum([len(row)for row in packed],dtype=np.int64)]
    neighbors=np.concatenate(packed)if packed else np.empty(0,np.int32)
    return offsets,neighbors


def _quality(state, current_cleanup_vertices):
    """复用严格拓扑及双相交核；premesh的ORIGINAL交数留给后续标准清理。"""
    from .compare_subject import _topology
    from .topology_volume_python import _nonmanifold_vertex_links
    from .mris_remove_intersection_python import mark_intersections
    from .place_surface_intersection_marking import mark_source_intersections
    xyz=np.asarray(state['original_vertices']);faces=np.asarray(state['faces'])[:state['nfaces']]
    structural=_topology(xyz,faces)
    links=int(_nonmanifold_vertex_links(np.ascontiguousarray(faces,np.int32),len(xyz)))
    directed=np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]))
    _,group=np.unique(np.sort(directed,axis=1),axis=0,return_inverse=True)
    signs=np.where(directed[:,0]<directed[:,1],1,-1)
    conflicts=int(np.count_nonzero(np.bincount(group,weights=signs)))
    triangle=xyz[faces].astype(np.float64)
    area=np.linalg.norm(np.cross(triangle[:,1]-triangle[:,0],triangle[:,2]-triangle[:,0]),axis=1)/2
    finite=bool(np.isfinite(xyz).all()and np.isfinite(current_cleanup_vertices).all())
    source_original=int(mark_source_intersections(vertices=xyz,faces=faces,predicate_backend='numba')[1])
    source_current=int(mark_source_intersections(vertices=current_cleanup_vertices,faces=faces,predicate_backend='numba')[1])
    original_count=int(mark_intersections(xyz,faces)[1])
    current_count=int(mark_intersections(current_cleanup_vertices,faces)[1])
    zero_area=int(np.count_nonzero(area==0))
    # The source writes ORIGINAL after cleaning CURRENT/TMP. The standard
    # pipeline still remeshes and cleans that written geometry afterward.
    # Do not turn a diagnostic premesh intersection into an extra early gate.
    passed=bool(structural['valid']and finite and links==0 and conflicts==0
        and source_current==current_count==0)
    return dict(status='passed'if passed else'failed',topology=structural,
        finite_coordinates=finite,nonmanifold_vertices=links,edge_orientation_conflicts=conflicts,
        zero_area_faces=zero_area,original_source_intersecting_faces=source_original,
        current_source_intersecting_faces=source_current,original_geometric_intersecting_faces=original_count,
        current_geometric_intersecting_faces=current_count,
        original_geometry_requires_standard_remesh_and_intersection=True,
        final_mesh_quality='not_assessed',
        overall_metric_equivalence='not_assessed',strict_source_final_reproduction='not_assessed')


def _only_geometric_current_residual(quality):
    """仅当 CURRENT 的独立几何交数是唯一失败项时尝试二次清理。"""
    return bool(quality.get('status') == 'failed'
        and bool(quality.get('topology', {}).get('valid'))
        and bool(quality.get('finite_coordinates'))
        and quality.get('nonmanifold_vertices') == 0
        and quality.get('edge_orientation_conflicts') == 0
        and quality.get('current_source_intersecting_faces') == 0
        and quality.get('current_geometric_intersecting_faces', 0) > 0)


def repair_genetic_topology(*, vertices, faces, qsphere_vertices, volume,
        ras_to_voxel, source_metadata, threads=4, seed=1234,
        population_size=10, max_unchanged=10, niters=-1,
        initial_selection=True, vertex_eliminate=True, use_overlap=False,
        normal_status='surface', soap_iterations=5, progress_callback=None,
        face_detection_cache=False, blocker_scan_backend='python', distance_backend='serial', distance_device=None, state_backend="legacy", random_audit_callback=None,
        failure_state_callback=None):
    """同有序自身nofix/qsphere/MRI→全部缺陷完整自有搜索与premesh。

    vertices/qsphere_vertices为有限F32(N,3) surface RAS/mm，faces为共同
    整数(F,3)；MRI volume uint8(X,Y,Z)、4×4 surface RAS→体素变换及
    显式source_metadata与prepare_nofix_genetic_base一致。前六参必填。
    threads默认4，显式总CPU预算；seed默认1234，VNL流一个半球共享。
    population_size/max_unchanged/niters默认10/10/−1，initial_selection/
    vertex_eliminate默认True，use_overlap默认False，沿用原标准GA。
    finite niters或关闭策略按实际记录，不标标准政策。normal_status
    默认surface，本完整控制器只支持标准三角表面状态；soap_iterations默认5。
    random_audit_callback默认None，显式具名(index,low,high,value)观测完整半球
    连续随机流，返回值忽略，不改变区间/值或跨缺陷顺序；失败传播。
    progress_callback默认None，可传callable(*,defect_number,result)在每
    缺陷完整写回后观测自产结果；回调不接参考路径，异常传播，不隐藏。
    failure_state_callback默认None；仅最终双相交/结构质量门失败时，显式
    回调可收到compact ORIGINAL/CURRENT、面、rip、有序邻接、质量和清理诊断
    的独立副本。生产默认不保存影像或顶点；回调自行管理私有目录，异常传播。
    face_detection_cache默认False，仅接受bool/np.bool_；True时每缺陷开始根据
    当前保留面准备一次不可变索引，由该缺陷搜索和最终写回共同消费，
    下一个缺陷重新准备。动态邻接、几何及判定结果始终实时计算。
    blocker_scan_backend默认python，可显式numba；按原顺序扫描全部阻挡边，
    每缺陷搜索、最终评分和最终几何均传同一选择，无失败回退。
    state_backend默认legacy，显式packed需任务5完整状态API，搜索、最终评分与
    几何发布均传同一选择；缺实现明确失败，不退回legacy，不启用旧缓存。
    distance_device默认None；GPU后端必须显式可用cuda:N，CPU选择不接设备。
    GPU仅使用已验证的gpu_defect_signed_distance，失败不回退；原CPU路径保留。
    distance_backend默认serial，接受serial/parallel/gpu_defect_signed_distance；每缺陷候选和最终
    评分透传同一距离核选择。parallel仅在本threads预算内执行，未改变
    候选顺序或随机流。返回请求选择及各缺陷最终评分的实际执行/skip
    记录，局部任一轴<=5或unMRI权重为零不声明执行parallel。

    缓存原mris的准备/先验/边邻接；每缺陷used/prune读取前一缺陷末态
    corrected有序邻接。局部MRI/calibration使用当前corrected ORIGINAL。
    跨缺陷连续传递rip/K/E/fx/curvbak、删除计数/阈值与随机流，不将初态
    全25边表冒充连续输入。全部缺陷结束才metric/SOAP/save ORIGINAL/
    restore TMP/源序压缩；相交清理消费压缩CURRENT及真实有序一环，
    然后恢复ORIGINAL为premesh写出坐标。两槽都报告，CURRENT清理及
    拓扑结构严格检查；ORIGINAL的相交/零面积是中间态诊断，后续标准
    remesh及ORIGINAL相交清理不可省略，最终网格门另行检查。
    如果源标记为零而独立几何核仍见CURRENT相交，用同一rip和有序邻接
    对CURRENT再做一次几何标记SOAP；复检两核仍须同时为零，失败不发布。

    返回完整surface_state、metadata、original_domain_defect_labels、
    raw→base与base→compact映射、按序per_defect控制/耗时、finish报告、
    current_cleanup_vertices/diagnostics、quality与包括校验的seconds。
    不保留每候选大数组作为长期cache，不加载参考/文件/原版程序；外部
    读写不计API秒。CPU NumPy/Numba，GPU距离仅显式选择，不改精度默认或新增依赖。
    仅支持标准optimal_mapping=0；原始canonical缓存不可用于改变原
    球面坐标的其他mapping分支。此接口不是最终orig或完整recon-all。
    完整执行和质量通过分别记录；原完整GA数值/指标等效仍需独立对照。
    参数/域/不完整状态ValueError，原尚不支持的normal坐标变化明确报错，
    阶段异常/未解决相交或最终质量失败RuntimeError，不发布标准表面。
    输入不改；线程预算上下文恢复。对应mris_fix_topology -ga内部链。
    """
    started=time.perf_counter()
    from .topology_ga_backend_policy import validate_state_backend
    validate_state_backend(state_backend)
    if random_audit_callback is not None and not callable(random_audit_callback):
        raise ValueError("random_audit_callback must be callable or None")
    if progress_callback is not None and not callable(progress_callback):
        raise ValueError('progress_callback must be callable or None')
    if failure_state_callback is not None and not callable(failure_state_callback):
        raise ValueError('failure_state_callback must be callable or None')
    if not isinstance(use_overlap,(bool,np.bool_)):
        raise ValueError('explicit bool overlap policy required')
    if not isinstance(face_detection_cache,(bool,np.bool_)):
        raise ValueError('explicit bool face_detection_cache required')
    if not isinstance(blocker_scan_backend,str) or blocker_scan_backend not in ('python','numba'):
        raise ValueError('blocker_scan_backend must be python or numba')
    from .topology_distance_policy import validate_distance_policy
    distance_device=validate_distance_policy(distance_backend=distance_backend,distance_device=distance_device)
    source_population_limits(edge_count=0,population_size=population_size,max_unchanged=max_unchanged)
    if (isinstance(niters,(bool,np.bool_)) or not isinstance(niters,(int,np.integer)) or niters < -1
            or not isinstance(initial_selection,(bool,np.bool_))
            or not isinstance(vertex_eliminate,(bool,np.bool_))):
        raise ValueError('explicit source iteration and selection policies required')
    if (normal_status != 'surface'
            or isinstance(soap_iterations,(bool,np.bool_))
            or not isinstance(soap_iterations,(int,np.integer)) or soap_iterations < 0):
        raise ValueError('supported explicit final normal status and SOAP iterations required')
    if not isinstance(source_metadata,Mapping):
        raise ValueError('explicit source metadata mapping required')
    status=np.asarray(source_metadata.get('status'))
    if status.ndim or status.dtype.kind not in 'iu' or int(status)!=0:
        raise ValueError('complete controller requires triangular surface metadata.status=0')
    stages={};records=[];random=TopologyRandomStream(seed=seed,audit_callback=random_audit_callback)
    with thread_budget(threads=threads):
        tick=time.perf_counter()
        base=prepare_nofix_genetic_base(vertices=vertices,faces=faces,qsphere_vertices=qsphere_vertices,
            volume=volume,ras_to_voxel=ras_to_voxel,source_metadata=source_metadata)
        stages['base']=time.perf_counter()-tick
        state=base['surface_state'];labels=base['merge']['labels'];translation=base['vertex_translation']
        source=base['prepared']['base_arguments'];metric=base['metric']
        tick=time.perf_counter()
        priors=global_topology_priors(principal_curvature_fields=base['sff1'],
            smoothed_original_vertices=source['source_vertex_fields']['original_vertices'],faces=faces,
            neighbor_offsets=metric['one_ring_offsets'],neighbors=metric['one_ring_neighbors'],
            vertex_rip=metric['sff_arguments']['vertex_rip'],surface_status='sphere')
        context=prepare_candidate_input_context(vertices=vertices,faces=faces,defect_vertices=labels!=0,
            vertex_translation=translation,volume=volume,ras_to_voxel=ras_to_voxel,global_priors=priors)
        edge_context=prepare_ordered_genetic_defect_context(canonical_vertices=source['source_vertex_fields']['canonical_vertices'],
            faces=faces,defect_labels=labels,ordered_regions=base['merge']['regions'],vertex_translation=translation)
        stages['shared_inputs']=time.perf_counter()-tick
        p=context.preparation;deletion_count=1;threshold=np.float32(4)
        tick=time.perf_counter()
        for defect_number in range(len(base['merge']['regions'])):
            dt=time.perf_counter()
            edges=_next_inputs(canonical_vertices=source['source_vertex_fields']['canonical_vertices'],
                original_vertices=p['original_vertices'],original_normals=p['original_normals'],faces=faces,
                defect_labels=labels,ordered_regions=base['merge']['regions'],retention_status=base['retention_status'],
                vertex_translation=translation,white_values=p['white_values'],gray_values=p['gray_values'],
                volume=volume,ras_to_voxel=ras_to_voxel,defect_number=defect_number,
                current_surface_state=state,use_overlap=use_overlap,context=edge_context)
            candidate=prepare_defect_candidate_inputs(context=context,original_defect_hull=edges['original_hull_vertices'],
                candidate_edges=edges['edge_vertices'],current_corrected_original_vertices=state['original_vertices'])
            result=search_and_finalize_defect(surface_state=state,defect_inputs=edges,
                candidate_inputs=candidate['candidate_kwargs'],draw_random_number=random,
                population_size=population_size,max_unchanged=max_unchanged,niters=niters,
                initial_selection=initial_selection,vertex_eliminate=vertex_eliminate,
                deletion_count=deletion_count,elimination_threshold=threshold,
                check_surface_intersection=False,intersection_checker=None,
                face_detection_cache=face_detection_cache,blocker_scan_backend=blocker_scan_backend,
                distance_backend=distance_backend,distance_device=distance_device,state_backend=state_backend)
            state=result['surface_state'];deletion_count=result['deletion_count'];threshold=result['elimination_threshold']
            search=result['search'];final=result['finalization']
            records.append(dict(defect_number=defect_number,edges=len(edges['edge_vertices']),
                inside_vertices=len(edges['inside_vertices']),actual_policy=result['actual_policy'],
                counts=search['counts'],generations=search['generations'],termination=search['termination'],
                best_fitness=search['best_fitness'],actual_final_fitness=final['fitness'],
                final_matches_best=final['fitness_matches_best'],deletion_count=deletion_count,
                elimination_threshold=float(threshold),random_draws_after=random.draws,
                final_distance_policy=result.get('final_distance_policy'),
                seconds=dict(edges=edges['seconds'],local_prior=candidate['stage_seconds'],search=result['seconds'],
                    whole_defect=time.perf_counter()-dt)))
            if progress_callback is not None:progress_callback(defect_number=defect_number,result=result)
        stages['all_defects']=time.perf_counter()-tick
        tick=time.perf_counter()
        regions=base['merge']['regions']
        finished=finish_and_compact_topology(surface_state=state,metadata=base['metadata'],
            defect_vertices=[row['vertices']for row in regions],border_vertices=[row['border']for row in regions],
            vertex_translation=translation,normal_status=normal_status,iterations=soap_iterations)
        state=finished['surface_state'];stages['finish_and_compact']=time.perf_counter()-tick
        offsets,neighbors=_ordered_csr(state);tick=time.perf_counter()
        compact_faces=np.asarray(state['faces'])[:state['nfaces']]
        ripped=np.asarray(state['vertex_rip'])!=0
        cleaned,cleanup=repair_intersections(vertices=state['current_vertices'],faces=compact_faces,
            ripped=ripped,marking_backend='source_numba',device=None,
            ordered_neighbor_offsets=offsets,ordered_neighbors=neighbors)
        stages['current_intersection_cleanup']=time.perf_counter()-tick
        state=dict(state);state['current_vertices']=np.array(state['original_vertices'],copy=True)
        quality_started=time.perf_counter();quality=_quality(state,cleaned)
        quality_seconds=time.perf_counter()-quality_started
        cleanup=dict(cleanup)
        if _only_geometric_current_residual(quality):
            before=quality['current_geometric_intersecting_faces']
            tick=time.perf_counter()
            cleaned,geometric_cleanup=repair_intersections(vertices=cleaned,faces=compact_faces,
                ripped=ripped,marking_backend='legacy',device=None,
                ordered_neighbor_offsets=offsets,ordered_neighbors=neighbors)
            stages['current_geometric_fallback_cleanup']=time.perf_counter()-tick
            quality_started=time.perf_counter();quality=_quality(state,cleaned)
            quality_seconds+=time.perf_counter()-quality_started
            cleanup['independent_geometric_fallback']=dict(status='attempted',
                geometric_intersecting_faces_before=before,
                geometric_intersecting_faces_after=quality['current_geometric_intersecting_faces'],
                source_intersecting_faces_after=quality['current_source_intersecting_faces'],
                marking_backend='legacy',diagnostics=geometric_cleanup)
        else:
            stages['current_geometric_fallback_cleanup']=0.0
            cleanup['independent_geometric_fallback']=dict(status='not_needed')
        stages['quality']=quality_seconds
        if quality['status']!='passed':
            if failure_state_callback is not None:
                failure_state_callback(original_vertices=np.array(state['original_vertices'],copy=True),
                    current_cleanup_vertices=np.array(cleaned,copy=True),
                    faces=np.array(compact_faces,copy=True),vertex_rip=np.array(ripped,copy=True),
                    ordered_neighbor_offsets=np.array(offsets,copy=True),
                    ordered_neighbors=np.array(neighbors,copy=True),
                    quality=deepcopy(quality),current_cleanup=deepcopy(cleanup))
            raise RuntimeError('complete genetic topology failed premesh structure/current cleanup: '+repr(quality))
    return dict(surface_state=state,metadata=finished['metadata'],per_defect=records,random_draws=random.draws,
        original_domain_defect_labels=np.array(labels,copy=True),raw_to_base_vertex_translation=np.array(translation,copy=True),
        base_to_compact_vertex_translation=finished['vertex_translation'],
        finish=finished,current_cleanup_vertices=cleaned,current_cleanup=cleanup,quality=quality,
        seconds=dict(stages,complete_api_including_validation=time.perf_counter()-started),
        execution_status='complete',complete_ga=True,native_program_called=False,
        final_mesh_quality='not_assessed',required_next_steps=('standard_remesh','original_intersection_cleanup','final_mesh_quality'),
        strict_source_final_reproduction='not_assessed',overall_metric_equivalence='not_assessed',
        face_detection_cache=bool(face_detection_cache),blocker_scan_backend=blocker_scan_backend,
        distance_backend=distance_backend,distance_device=distance_device,state_backend=state_backend,
        standard_policy=bool(population_size==10 and max_unchanged==10 and niters==-1
            and initial_selection and vertex_eliminate and not use_overlap and soap_iterations==5
            and normal_status=='surface'))
