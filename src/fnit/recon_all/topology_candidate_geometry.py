"""复用成熟有序重连、平滑、MRI匹配及法向，连续生成一个候选。

调用方提供真实基底状态、边排序及校准均值；本模块不创建这些上游，
不生成 GA 种群，也不把中间候选当最终可发布的拓扑修复结果。
"""
from __future__ import annotations

import time
import numpy as np

from .topology_candidate_normals import candidate_face_normals, candidate_vertex_normals
from .topology_defect_mri_match import defect_mri_match
from .topology_defect_smooth import defect_smooth_type2
from .place_surface_normals import ordered_face_csr, FaceNormalTopology
from .topology_retessellation_dynamic import (retessellate_and_detect_candidate,
    _check_prepared_base)


def candidate_incidents(*, faces, vertex_count, base_context=None, topology=None):
    """按完整有序面表构造入射面/角点CSR；不从参考恢复任何行。

    faces为合法整数(F,3)，vertex_count为整数N≥0。返回int32(N+1,)
    offsets及int32(3F,) face_ids/corners，重复面按原表保留。面序须为
    原基底相对顺序再追加候选面；不代表任意外来网格都采用此入射顺序。
    非整数、越界或尺寸非法抛ValueError；不修改输入，无文件读写。
    base_context默认None从全表生成；显式一次校验的不可变基底必须
    与本faces前缀完全相同，只追加本candidate新增面的入射和角点。
    复用成熟ordered_face_csr的整数稳定排序直接生成完整CSR，保留重复
    面/角点；缓存仍核验完整基底前缀，不缓存动态行或跳过原参数校验。
    三个返回数组独立分配，不跨坐标/面状态复用。
    topology默认None；显式FaceNormalTopology绑定同一完整有序面和N，
    校验后复用整数CSR并独立复制输出，不从参考或过期面版本恢复任何行。
    """
    triangles = np.asarray(faces)
    if (isinstance(vertex_count, (bool, np.bool_)) or not isinstance(vertex_count, (int, np.integer))
            or vertex_count < 0 or triangles.ndim != 2 or triangles.shape[1] != 3
            or not np.issubdtype(triangles.dtype, np.integer)
            or triangles.size and (triangles.min() < 0 or triangles.max() >= vertex_count)
            or 3*len(triangles) > np.iinfo(np.int32).max):
        raise ValueError('integer ordered faces and valid vertex_count required')
    if base_context is not None:
        _check_prepared_base(base_context)
        if (len(base_context.canonical)!=vertex_count or len(triangles)<len(base_context.base_faces)
                or not np.array_equal(triangles[:len(base_context.base_faces)],base_context.base_faces)):
            raise ValueError('candidate incident cache requires same immutable base faces prefix')
    if topology is None:
        offsets, ids, corners = ordered_face_csr(triangles, nvertices=vertex_count)
    else:
        if not isinstance(topology, FaceNormalTopology):
            raise ValueError('topology must be FaceNormalTopology or None')
        topology.validate(triangles, vertex_count)
        offsets, ids, corners = topology.offsets, topology.face_ids, topology.corners
    return dict(incident_offsets=offsets.astype(np.int32),
        incident_faces=ids.astype(np.int32),incident_corners=corners.astype(np.int32))


def candidate_geometry(*, original_vertices, volume, ras_to_voxel,
        white_mean, gray_mean, k1_mean, k2_mean, retessellation_inputs):
    """一个完整基底状态→重连→25次smooth2→40次MRI匹配→自产法向。

    original_vertices为float32(N,3)原几何surface RAS/mm；volume为uint8
    (X,Y,Z)归一化MRI，ras_to_voxel为MRI-only tkregister RAS→连续体素
    的有限4×4浮点矩阵。white_mean/gray_mean是无单位校准强度，k1_mean/
    k2_mean是有限非零mm⁻¹校准均值。retessellation_inputs字典为
    retessellate_and_detect_candidate全部具名参数：canonical/base_faces/
    neighbors/candidate_edges/edge_used/ordering/inside_vertices/
    inside_discarded/border_vertices/hull_vertices及可选rip/fixed-map/
    mode/overlaps字段。其canonical与original_vertices须同N同顶点顺序。

    返回原重连state，增加original_vertices匹配后float32(N,3)、
    smoothed_original_vertices、完整面序face_normals float32(F,3)、
    selected_vertex_normals float32(K,3)、cached_original_face_area_mm2、
    source incident/neighbor CSR及stage_seconds。面积仅为FaceNormCache
    original_area，不替代FACE.area或VERTEX.area。不自动更新外部顶点
    法向、K/E/curvbak、RP或DVS。未选择内点固定；所有原校准/候选参数
    来自调用者，不读取参考文件，不改变线程/精度/设备。非法输入明确
    ValueError；成熟算子失败直接抛出。没有可省略迭代开关，模式限
    当前成熟smooth2/match1。中间网格可能相交，最终质量仍由控制器
    独立检查；不是完整fitness/GA或原始T1整例接口。
    """
    start = time.perf_counter()
    xyz, data, matrix = map(np.asarray, (original_vertices, volume, ras_to_voxel))
    if xyz.dtype != np.float32 or xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError('finite float32 original_vertices (N,3) required')
    if data.dtype != np.uint8 or data.ndim != 3 or any(size == 0 for size in data.shape):
        raise ValueError('nonempty uint8 XYZ MRI required')
    if matrix.shape != (4, 4) or not np.issubdtype(matrix.dtype, np.floating) or not np.isfinite(matrix).all():
        raise ValueError('finite floating MRI-only 4x4 ras_to_voxel required')
    if not isinstance(retessellation_inputs, dict):
        raise ValueError('complete explicit retessellation_inputs dict required')
    if np.asarray(retessellation_inputs['canonical']).shape != xyz.shape:
        raise ValueError('original/canonical vertex correspondence required')
    means = np.asarray([white_mean, gray_mean, k1_mean, k2_mean], np.float32)
    if not np.isfinite(means).all() or np.any(means[2:] == 0):
        raise ValueError('finite calibration means; nonzero curvature means required')
    tick = time.perf_counter(); state = retessellate_and_detect_candidate(**retessellation_inputs)
    seconds = dict(retess_and_face_detection=time.perf_counter()-tick)
    inside = state['selected_vertices'][:state['ninside']]
    runtime = [state['neighbors'][vertex] for vertex in inside]
    # 只复用这一个实际候选面版本的整数CSR；下一候选重新构建。
    topology = FaceNormalTopology(triangles=state['faces'], nvertices=len(xyz))
    tick = time.perf_counter()
    smoothed = defect_smooth_type2(xyz=xyz, faces=state['faces'], inside_vertices=inside,
        runtime_neighbors=runtime, k1_mean=float(means[2]), k2_mean=float(means[3]), topology=topology)
    seconds['smooth_25'] = time.perf_counter()-tick
    tick = time.perf_counter()
    matched = defect_mri_match(xyz=smoothed, faces=state['faces'], inside_vertices=inside,
        runtime_neighbors=runtime, volume=data, ras_to_voxel=matrix,
        white_mean=float(means[0]), gray_mean=float(means[1]), iterations=40, topology=topology)
    seconds['match_40'] = time.perf_counter()-tick
    tick = time.perf_counter()
    incident = candidate_incidents(faces=state['faces'], vertex_count=len(xyz),
        base_context=retessellation_inputs.get('base_context'), topology=topology)
    geometry = candidate_face_normals(original_vertices=matched, faces=state['faces'],
        selected_faces=np.arange(len(state['faces']), dtype=np.int32))
    normal = candidate_vertex_normals(faces=state['faces'], face_normals=geometry['face_normals'],
        incident_offsets=incident['incident_offsets'], incident_faces=incident['incident_faces'],
        selected_vertices=state['selected_vertices'])
    seconds['complete_face_and_selected_vertex_normals'] = time.perf_counter()-tick
    offsets = np.zeros(len(xyz)+1, np.int32)
    offsets[1:] = np.cumsum([len(row) for row in state['neighbors']])
    flat = np.array([vertex for row in state['neighbors'] for vertex in row], np.int32)
    seconds['complete_api'] = time.perf_counter()-start
    return dict(state, **incident, original_vertices=matched, smoothed_original_vertices=smoothed,
        face_normals=geometry['face_normals'], selected_vertex_normals=normal,
        cached_original_face_area_mm2=geometry['cached_original_area_mm2'],
        neighbor_offsets=offsets, ordered_neighbors=flat, stage_seconds=seconds,
        scope='one prepared candidate continuous geometry only; not full fitness/GA/final QC')
