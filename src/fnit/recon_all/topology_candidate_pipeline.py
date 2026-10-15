"""一份完整候选初态→自产重连/匹配/SFF/五项评分；外层GA另行控制。

只复用已验证自有算子。完整初态及每缺陷校准先验均为显式输入，
不读参考后态或运行原版程序，不把一个候选当最终拓扑修复结果。
"""
from __future__ import annotations

from collections.abc import Mapping
import time
import numpy as np

from .topology_candidate_displacement import candidate_displacement
from .topology_candidate_geometry import candidate_geometry
from .topology_candidate_sff import candidate_second_fundamental_form
from .topology_patch_fitness import evaluate_prepared_patch_fitness


def _indices(value, upper, name):
    a = np.asarray(value)
    if (a.ndim != 1 or a.dtype.kind not in 'iu' or a.size and (a.min() < 0 or a.max() >= upper)):
        raise ValueError(name+' requires in-range ordered integer indices')
    return a


def candidate_face_check(*, faces, neighbor_offsets, ordered_neighbors,
        incident_offsets, incident_faces, patch_vertices):
    """源 mrisCheckDefectFaces：所选点每个一环共边必须恰有两个入射面。

    faces为整数(F,3)，两个offsets为整数(N+1,)有效CSR，flat neighbor/
    face为整数，patch_vertices为整数(P,)。全部编号属于同一版本，保留
    原行/面顺序及重复访问。仅检查原incidence定义，不检查几何相交、
    Euler、方向或解剖形状；返回source_result=1或-1及首次失败
    [vertex,neighbor,face_count]，不是最终网格放行。非法输入ValueError，
    输入不修改，无文件/GPU/原版程序调用、没有容差开关。
    """
    f = np.asarray(faces)
    no, io = np.asarray(neighbor_offsets), np.asarray(incident_offsets)
    neighbors, incident = np.asarray(ordered_neighbors), np.asarray(incident_faces)
    if (f.ndim != 2 or f.shape[1] != 3 or f.dtype.kind not in 'iu'
            or no.ndim != 1 or io.shape != no.shape or len(no) < 1
            or no.dtype.kind not in 'iu' or io.dtype.kind not in 'iu'):
        raise ValueError('ordered integer faces and same-size CSR offsets required')
    n = len(no)-1
    if f.size and (f.min() < 0 or f.max() >= n):
        raise ValueError('face vertex outside candidate')
    neighbors = _indices(neighbors,n,'ordered_neighbors')
    incident = _indices(incident,len(f),'incident_faces')
    selected = _indices(patch_vertices,n,'patch_vertices')
    for offsets, flat, name in [(no,neighbors,'neighbor'),(io,incident,'incident')]:
        if (offsets[0] != 0 or offsets[-1] != len(flat) or np.any(offsets[1:] < offsets[:-1])):
            raise ValueError('valid '+name+' CSR required')
    checked = 0
    for v in selected:
        for other in neighbors[no[v]:no[v+1]]:
            count = 0
            for face in incident[io[v]:io[v+1]]:
                # vertexInFace is bool: repeated corners in one face count once.
                if np.any(f[face] == other):
                    count += 1
            checked += 1
            if count != 2:
                return dict(source_result=-1,first_invalid=np.array([v,other,count],np.int64),checked_directed_edges=checked)
    return dict(source_result=1,first_invalid=None,checked_directed_edges=checked)


def _state(surface_state, retessellation_inputs):
    if not isinstance(surface_state,Mapping) or not isinstance(retessellation_inputs,Mapping):
        raise ValueError('complete explicit initial surface state and retessellation mappings required')
    names=('original_vertices','vertex_normals','tangent_e1','tangent_e2','flattened_vertices',
           'k1','k2','vertex_area','curvbak','faces','face_area')
    if any(name not in surface_state for name in names):
        raise ValueError('incomplete source candidate initial state')
    xyz=np.asarray(surface_state['original_vertices']);n=len(xyz)
    arrays={name:np.asarray(surface_state[name]) for name in names}
    for name in names[:5]:
        if arrays[name].shape!=(n,3) or arrays[name].dtype!=np.float32 or not np.isfinite(arrays[name]).all():
            raise ValueError(name+' requires finite original float32(N,3) state')
    for name in ('k1','k2','vertex_area','curvbak'):
        if arrays[name].shape!=(n,) or arrays[name].dtype!=np.float32 or not np.isfinite(arrays[name]).all():
            raise ValueError(name+' requires finite same-order float32(N,) state')
    faces=arrays['faces'];base=np.asarray(retessellation_inputs['base_faces'])
    if (faces.ndim!=2 or faces.shape[1]!=3 or faces.dtype.kind not in 'iu'
            or not np.array_equal(faces,base) or arrays['face_area'].dtype!=np.float32
            or arrays['face_area'].shape!=(len(faces),) or not np.isfinite(arrays['face_area']).all()):
        raise ValueError('complete logical base faces and corresponding source FACE.area required')
    return arrays


def evaluate_candidate(*, surface_state, retessellation_inputs, volume, ras_to_voxel,
        local_priors, gray_white_pdf, k1_mean, k2_mean, normal_dot_histogram,
        k1_histogram, k2_histogram, joint_curvature_pdf, joint_origin_mm_inv,
        joint_spacing_mm_inv, mri_weight=1., volume_unmri_weight=10.,
        normal_dot_weight=1., quadratic_curvature_weight=1., distance_backend='serial',
        distance_evaluator=None, distance_device=None):
    """同一完整初态及自产先验→一个完整候选的几何/SFF/五项/反馈副本。

    surface_state为create_genetic_base同语义完整状态：有限float32 orig/
    当前法向/切向/flattened(N,3)，k1/k2、vertex_area/curvbak(N,)，
    逻辑faces(F,3)及FACE.area(F,)；所有未改变项保持原值，不从参考填。
    retessellation_inputs是candidate_geometry的完整有序参数，base_faces
    必须等于初态完整逻辑面。volume:uint8(X,Y,Z)同主MRI；ras_to_voxel:
    4×4拓扑MRI-only tkregister RAS→连续体素，不用placement footer桥。
    local_priors为每defect一次自产defect_volume_priors返回值（局部uint8
    强度/白灰logP/shape/origin/scale及白灰mean/mean_log），不能读参考场。
    gray_white_pdf是自产校准joint uint8强度PDF；其形状/类型由MRI模块验。
    k1_mean/k2_mean非零mm⁻¹，来自首次全局SFF1 K-PDF的源1%-peak F32
    HISTO mean，供25轮smooth目标半径使用，不从SFF2顶点数组求均值。normal_dot_histogram
    与两个curvature histogram是原bins/counts/bin_size映射；joint_curvature
    PDF及origin/spacing mm⁻¹属于同一声明的全局曲率先验。全部这些参数
    必填，不隐式生成先验或初态；四个源double权重默认1/10/1/1。
    distance_evaluator默认None，可显式传入完整距离callable；实际后端
    从距离返回的actual_backend记录，缺省保持既有CPU接口和控制流。
    distance_device默认None；GPU后端必须显式可用cuda:N，CPU选择不接设备。
    GPU仅使用已验证的gpu_defect_signed_distance，失败不回退；原CPU路径保留。
    distance_backend默认serial，接受serial/parallel/gpu_defect_signed_distance；选择已有同序
    F32距离核。parallel按调用方当前Numba线程mask划分互不重叠x slab，
    本函数不更改mask；调用方用thread_budget显式限制总CPU预算。
    即使unMRI权重为零或局部任一轴<=5，仍先校验此参数；此时不执行
    距离核，distance_policy.actual_backend为None，不称已运行parallel。

    保存完整未DISCARD域的初态orig到本候选fx/fy/fz，再生成几何、位移，
    曲率权重非零时复用SFF，复用prepare5自产距离和反馈。新面FACE.area
    依源calloc置0，不用FaceNormCache面积替代；外部顶点属性/旧面面积
    保持。返回geometry/sff/fitness、完整新vertex_normals/curvbak/flattened
    state、源有向incidence检查和各阶段秒数，输入无别名、不修改。
    SFF作为评分准备先算，unMRI反馈不改变其输入；不是外层GA/DVS/RP/
    最终quality producer。非有限/跨版本/缺字段ValueError或KeyError，
    子函数失败原样抛出，面积加权未定义诊断保留IEEE NaN。中间候选
    源face_check失败仍按原fitness罚分，不输出最终标准表面、不降低QC。
    """
    start=time.perf_counter()
    from .topology_distance_policy import validate_distance_policy
    distance_device=validate_distance_policy(distance_backend=distance_backend,distance_device=distance_device)
    from .topology_distance_policy import resolve_distance_evaluator
    kernel_backend,distance_evaluator=resolve_distance_evaluator(distance_backend=distance_backend,
        distance_device=distance_device,distance_evaluator=distance_evaluator)
    initial=_state(surface_state,retessellation_inputs)
    if not isinstance(local_priors,Mapping):raise ValueError('explicit independently generated local_priors required')
    weight=np.asarray([mri_weight,volume_unmri_weight,normal_dot_weight,quadratic_curvature_weight])
    if weight.shape!=(4,) or not np.isfinite(weight).all():raise ValueError('finite actual source double weights required')
    mapped=np.asarray(retessellation_inputs['inside_vertices'])
    discarded=np.asarray(retessellation_inputs['inside_discarded'])
    if mapped.ndim!=1 or mapped.dtype.kind not in 'iu' or discarded.dtype!=bool or discarded.shape!=mapped.shape:
        raise ValueError('complete ordered defect mapping and bool DISCARD required')
    mapped=_indices(mapped,len(initial['original_vertices']),'inside_vertices')
    sampling=initial['flattened_vertices'].copy()
    for v in mapped[~discarded]:sampling[v]=initial['original_vertices'][v]
    tick=time.perf_counter()
    geometry=candidate_geometry(original_vertices=initial['original_vertices'],volume=volume,ras_to_voxel=ras_to_voxel,
        white_mean=local_priors['white_mean'],gray_mean=local_priors['gray_mean'],k1_mean=k1_mean,k2_mean=k2_mean,
        retessellation_inputs=dict(retessellation_inputs))
    seconds=dict(geometry_complete_api=time.perf_counter()-tick)
    selected=geometry['selected_vertices'];xyz=geometry['original_vertices']
    normal=initial['vertex_normals'].copy();normal[selected]=geometry['selected_vertex_normals']
    face_area=np.concatenate((initial['face_area'].copy(),np.zeros(len(geometry['faces'])-len(initial['faces']),np.float32)))
    tick=time.perf_counter()
    displacement=candidate_displacement(original_vertices=xyz,prematch_positions_mm=sampling,
        translated_defect_vertices=mapped,discarded_defect_vertices=discarded)
    curvbak=initial['curvbak'].copy()
    for v,value in zip(displacement['vertex_indices'],displacement['displacement_mm']):curvbak[v]=value
    seconds['displacement_complete_api']=time.perf_counter()-tick
    tick=time.perf_counter()
    if abs(float(quadratic_curvature_weight))>=float(np.finfo(np.float32).eps):
        sff=candidate_second_fundamental_form(original_vertices=xyz,vertex_normals=normal,
            neighbor_offsets=geometry['neighbor_offsets'],neighbors=geometry['ordered_neighbors'],patch_vertices=selected,
            initial_k1=initial['k1'],initial_k2=initial['k2'],initial_tangent_e1=initial['tangent_e1'],initial_tangent_e2=initial['tangent_e2'])
    else:
        sff={name:initial[field].copy() for name,field in [('k1','k1'),('k2','k2'),('tangent_e1','tangent_e1'),('tangent_e2','tangent_e2')]}
    seconds['sff_complete_api']=time.perf_counter()-tick
    face_check=candidate_face_check(faces=geometry['faces'],neighbor_offsets=geometry['neighbor_offsets'],
        ordered_neighbors=geometry['ordered_neighbors'],incident_offsets=geometry['incident_offsets'],
        incident_faces=geometry['incident_faces'],patch_vertices=selected)
    mri=dict(original_vertices=xyz,faces=geometry['faces'],vertex_normals=normal,face_normals=geometry['face_normals'],
        vertex_areas_mm2=initial['vertex_area'],face_areas_mm2=face_area,patch_vertices=selected,patch_faces=geometry['selected_faces'],
        volume=volume,ras_to_voxel=ras_to_voxel,gray_white_pdf=gray_white_pdf)
    normal_input=dict(original_vertices=xyz,vertex_normals=normal,vertex_areas_mm2=initial['vertex_area'],patch_vertices=selected,
        neighbor_offsets=geometry['neighbor_offsets'],ordered_neighbors=geometry['ordered_neighbors'],normal_dot_histogram=normal_dot_histogram)
    curvature=dict(principal_curvature_k1_mm_inv=sff['k1'],principal_curvature_k2_mm_inv=sff['k2'],vertex_areas_mm2=initial['vertex_area'],
        patch_vertices=selected,k1_histogram=k1_histogram,k2_histogram=k2_histogram,joint_curvature_pdf=joint_curvature_pdf,
        joint_origin_mm_inv=joint_origin_mm_inv,joint_spacing_mm_inv=joint_spacing_mm_inv)
    distance=dict(original_vertices=xyz,faces=geometry['faces'],face_normals=geometry['face_normals'],
        incident_offsets=geometry['incident_offsets'],incident_faces=geometry['incident_faces'],incident_corners=geometry['incident_corners'],
        volume_shape=local_priors['volume_shape'],volume_origin_mm=local_priors['volume_origin_mm'],volume_index_scale_per_mm=local_priors['volume_index_scale_per_mm'],
        backend=kernel_backend)
    volume_input={key:local_priors[key] for key in ['defect_intensity_volume','white_log_probability','gray_log_probability',
        'initial_white_mean_log_probability','initial_gray_mean_log_probability','volume_origin_mm','volume_index_scale_per_mm']}
    volume_input.update(initial_white_mean_intensity=local_priors['white_mean'],initial_gray_mean_intensity=local_priors['gray_mean'],
        vertex_sampling_positions_mm=sampling,initial_vertex_curvbak=curvbak,translated_defect_vertices=mapped,
        discarded_defect_vertices=discarded,distance_unset_sentinel_mm=np.float32(1000))
    tick=time.perf_counter()
    fitness=evaluate_prepared_patch_fitness(mri_inputs=mri,normal_inputs=normal_input,curvature_inputs=curvature,
        distance_inputs=distance,volume_inputs=volume_input,valid_faces=bool(face_check['source_result']>=0),
        mri_weight=mri_weight,volume_unmri_weight=volume_unmri_weight,normal_dot_weight=normal_dot_weight,
        quadratic_curvature_weight=quadratic_curvature_weight,
        distance_evaluator=distance_evaluator)
    seconds['complete_prepared_five_api']=time.perf_counter()-tick
    feedback=curvbak if fitness['vertex_curvbak'] is None else fitness['vertex_curvbak']
    executed=fitness['distance'] is not None
    mask=None;slabs=None
    if executed:
        if distance_backend=='parallel':
            from numba import get_num_threads
            mask=int(get_num_threads())
            slabs=min(mask,int(local_priors['volume_shape'][0]))
        elif distance_backend=='serial':slabs=1
    distance_policy=dict(requested_backend=distance_backend,executed=executed,
        actual_backend=fitness['distance'].get('actual_backend', distance_backend) if executed else None,skipped_unmri=not executed,
        caller_numba_thread_mask=mask,slab_count=slabs,
        effective_unmri_weight=float(fitness['composition']['effective_weights']['l_unmri']),
        worker_count_observed=False,requested_device=distance_device)
    seconds['complete_candidate_api_including_validation']=time.perf_counter()-start
    return dict(geometry=geometry,sff=sff,fitness=fitness,face_check=face_check,vertex_normals=normal,
        vertex_curvbak=feedback,flattened_vertices=sampling,face_area=face_area,displacement=displacement,seconds=seconds,
        distance_policy=distance_policy,
        scope='complete prepared initial-state candidate geometry and five likelihoods; not initial-state/global-prior producer or GA/final QC')
