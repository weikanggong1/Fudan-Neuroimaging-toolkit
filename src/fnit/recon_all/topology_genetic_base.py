"""原拓扑修复的有序初始网格 producer；不从参考 after 状态填补字段。"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import time
import numpy as np

from .topology_preflight_python import genetic_base_translation
from .topology_retessellation_python import source_base_neighbors
from .smooth_surface_python import ordered_neighbors


COPIED_VECTOR_FIELDS = ('current_vertices', 'original_vertices', 'temporary_vertices',
                        'canonical_vertices', 'vertex_normals')
COPIED_SCALAR_FIELDS = ('val', 'val2', 'valbak', 'val2bak', 'imag_val', 'curv',
                        'curvbak', 'stat', 'mean', 'mean_imag', 'std_error',
                        'H', 'K', 'k1', 'k2')
TOPOLOGY_COLUMNS = ('nsize_cur', 'nsize_max', 'vtotal', 'vnum', 'v2num', 'v3num', 'num')


def _integer_array(value, shape, name):
    a = np.asarray(value)
    if a.shape != shape or a.dtype.kind not in 'iu':
        raise ValueError(name+' requires integer shape '+str(shape))
    if a.size and (a.min() < np.iinfo(np.int32).min or a.max() > np.iinfo(np.int32).max):
        raise ValueError(name+' exceeds source int32')
    return np.array(a, np.int32, copy=True)


def create_genetic_base(*, faces, defect_labels, defect_status,
        source_vertex_fields, source_scalar_fields, source_neighbors,
        source_incident_faces, source_incident_corners, source_vertex_rip,
        source_metadata, vertex_translation=None, face_translation=None,
        ordered_defect_vertices=None) -> dict:
    """完整原序初态→独立corrected base、原→新映射与源零分配属性。

    faces为原整数(F,3)，defect_labels为非负整数(N,)（0未标、1…D连通
    缺陷），defect_status为整数(N,)（0 KEEP/1 DISCARD，外部值不参与）。
    source_vertex_fields要求5个float32(N,3)：current_vertices=raw TMP，
    original_vertices=两轮平均orig，temporary_vertices=tx，canonical_vertices
    为已中心化球面，vertex_normals为restore TMP后重新计算的法向；不能
    将smoothedorig的original_normals代替它。坐标surface RAS/mm，法向
    无量纲。source_scalar_fields要求COPIED_SCALAR_FIELDS的15个float32
    (N,)字段；只复制原bits，包括已有NaN，不以参考after或缺省0补齐。

    source_neighbors为原有序一环list[N]；source_incident_faces/corners
    为原有序list[N]，每行相同长度，角点0…2与faces对应。必须由自有
    原网格拓扑产生，保留重复角点。source_vertex_rip为原整数/bool(N,)，
    输入只供明确状态；源构建不复制rip，只有inside DISCARD设置新rip1。
    source_metadata要求use_real_ras/hemisphere/volume_geometry/status/
    origxyz_status，复制到输出。可选vertex/face_translation默认为None：
    复用成熟genetic_base_translation；若显式提供必须与自产映射相同。
    ordered_defect_vertices默认None，要求每标签为连通组件，沿用原BFS。
    源merge将多个通过border相邻的组件concat后，可显式传D个有序整数
    inside数组；每行须完整无重复覆盖该标签，此时严格保留concat顺序，
    不重做BFS、不要求merged inside直接连通，也不读取参考translation。

    返回 {surface_state, vertex_translation, face_translation, source_order,
    metadata, seconds, scope}。surface_state含logical faces与nfaces、五
    copied向量、15标量、vertex_rip/marks/border、7列topology、有序邻接/
    incident rows；未被原构建赋值的area/e1/e2/fx/undef/面法向等明确按
    原calloc/bzero为0。outside一环保留双端未标的悬空边，不从剩余面
    重建；inside邻接/incident为空、nsize0；outside nsizeCur/Max1。
    source_order为新→原全排列，容量metadata保留N+10/2F但不发布未用
    存储。输入不修改/不别名，不改线程/GPU/TF32；非法或不完整状态
    ValueError，无显式顺序时缺陷不连通/映射错误明确失败。只覆盖GA模式初始
    base，不覆盖greedy删顶点、metric重算、候选变形或完整GA。
    """
    started = time.perf_counter()
    labels = np.asarray(defect_labels)
    if labels.ndim != 1 or labels.dtype.kind not in 'iu' or len(labels) == 0:
        raise ValueError('nonempty integer defect_labels (N,) required')
    n = len(labels)
    labels = _integer_array(labels, (n,), 'defect_labels')
    if (np.any(labels < 0) or labels.max() > n
            or not np.array_equal(np.unique(labels[labels > 0]),np.arange(1,int(labels.max())+1))):
        raise ValueError('positive defect labels must be contiguous 1..D')
    triangles = np.asarray(faces)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError('original ordered faces (F,3) required')
    triangles = _integer_array(triangles, triangles.shape, 'faces')
    if triangles.size and (triangles.min() < 0 or triangles.max() >= n):
        raise ValueError('faces outside original vertex storage')
    status = _integer_array(defect_status, (n,), 'defect_status')
    if np.any(status[labels != 0] < 0) or np.any(status[labels != 0] > 1):
        raise ValueError('inside defect_status must be KEEP0 or DISCARD1')
    rip = np.asarray(source_vertex_rip)
    if rip.shape != (n,) or rip.dtype.kind not in 'biu':
        raise ValueError('source_vertex_rip requires explicit original (N,) flags')
    if not isinstance(source_vertex_fields, Mapping) or not isinstance(source_scalar_fields, Mapping):
        raise ValueError('complete explicit source vector and scalar mappings required')
    required_meta = ('use_real_ras', 'hemisphere', 'volume_geometry', 'status', 'origxyz_status')
    if not isinstance(source_metadata, Mapping) or any(k not in source_metadata for k in required_meta):
        raise ValueError('source geometry/status metadata must be explicit')
    vectors, scalars = {}, {}
    for name in COPIED_VECTOR_FIELDS:
        if name not in source_vertex_fields:
            raise ValueError('missing copied source vector '+name)
        a = np.asarray(source_vertex_fields[name])
        if a.shape != (n,3) or a.dtype != np.float32 or not np.isfinite(a).all():
            raise ValueError('finite float32(N,3) copied vector required: '+name)
        vectors[name] = a
    for name in COPIED_SCALAR_FIELDS:
        if name not in source_scalar_fields:
            raise ValueError('missing copied source scalar '+name)
        a = np.asarray(source_scalar_fields[name])
        if a.shape != (n,) or a.dtype != np.float32:
            raise ValueError('float32(N,) copied scalar required: '+name)
        scalars[name] = a
    for name, rows in [('source_neighbors', source_neighbors),
                       ('source_incident_faces', source_incident_faces),
                       ('source_incident_corners', source_incident_corners)]:
        if not isinstance(rows, Sequence) or len(rows) != n:
            raise ValueError(name+' requires one ordered row per source vertex')
    expected_neighbors = ordered_neighbors(triangles,n)
    if any(list(a) != b for a,b in zip(source_neighbors,expected_neighbors)):
        raise ValueError('initial source one-ring order must match the original face-order topology')
    if ordered_defect_vertices is None:
        # The mature BFS only visits connected labels. Merged source domains
        # have a separate explicit order path; never leave mapping slots empty.
        for number in range(1, int(labels.max())+1):
            selected = np.flatnonzero(labels == number)
            found = {int(selected[0])}; queue = [int(selected[0])]
            for v in queue:
                for other in source_neighbors[v]:
                    if not isinstance(other, (int,np.integer)) or other < 0 or other >= n:
                        raise ValueError('invalid source neighbor index')
                    if labels[other] == number and int(other) not in found:
                        found.add(int(other));queue.append(int(other))
            if len(found) != len(selected):
                raise ValueError('each defect label must be one connected source component unless ordered merge rows are explicit')
        vtrans, ftrans = genetic_base_translation(labels, triangles)
    else:
        if not isinstance(ordered_defect_vertices,Sequence) or len(ordered_defect_vertices)!=int(labels.max()):
            raise ValueError('one complete ordered inside row per defect required')
        rows=[]
        for number,value in enumerate(ordered_defect_vertices,1):
            value=np.asarray(value)
            if value.ndim!=1:
                raise ValueError('ordered inside row must be one-dimensional')
            row=_integer_array(value,value.shape,'ordered_defect_vertices')
            if not np.array_equal(np.sort(row),np.flatnonzero(labels==number)):
                raise ValueError('ordered inside rows must exactly cover labels without duplicates')
            rows.append(row)
        order=np.concatenate([np.flatnonzero(labels==0),*rows]).astype(np.int32)
        vtrans=np.empty(n,np.int32);vtrans[order]=np.arange(n,dtype=np.int32)
        ftrans=np.full(len(triangles),-1,np.int32)
        unchanged=np.all(labels[triangles]==0,axis=1)
        ftrans[unchanged]=np.arange(int(unchanged.sum()),dtype=np.int32)
    if not np.array_equal(np.sort(vtrans), np.arange(n)):
        raise ValueError('mature original-to-corrected mapping is not a complete permutation')
    for supplied, actual, name in [(vertex_translation,vtrans,'vertex_translation'),
                                   (face_translation,ftrans,'face_translation')]:
        if supplied is not None and not np.array_equal(_integer_array(supplied,actual.shape,name),actual):
            raise ValueError(name+' must equal independently generated original ordered base mapping')
    order = np.argsort(vtrans).astype(np.int32)
    neighbors = source_base_neighbors(source_neighbors=source_neighbors, labels=labels, translation=vtrans)
    kept = ftrans >= 0
    corrected_faces = np.ascontiguousarray(vtrans[triangles[kept]], np.int32)
    face_rows, corner_rows = [], []
    incident_seen = np.zeros((len(triangles),3),np.bool_)
    # Filter the original source incident rows rather than rebuilding them from
    # corrected triangles. Both tables have independent source ordering.
    for old in order:
        f = np.asarray(source_incident_faces[old]); c = np.asarray(source_incident_corners[old])
        if f.size == 0 and isinstance(source_incident_faces[old], (list,tuple)):
            f = f.astype(np.int32)
        if c.size == 0 and isinstance(source_incident_corners[old], (list,tuple)):
            c = c.astype(np.int32)
        if f.ndim != 1 or c.shape != f.shape or f.dtype.kind not in 'iu' or c.dtype.kind not in 'iu':
            raise ValueError('same-length original ordered incident integer rows required')
        if f.size and (f.min() < 0 or f.max() >= len(triangles) or c.min() < 0 or c.max() > 2
                or not np.all(triangles[f,c] == old)):
            raise ValueError('original incident face/corner does not belong to its vertex')
        packed = f.astype(np.int64)*3+c
        if len(np.unique(packed)) != len(packed) or np.any(incident_seen[f,c]):
            raise ValueError('duplicated original incident face/corner entry')
        incident_seen[f,c] = True
        if labels[old] == 0:
            selected = ftrans[f] >= 0
            face_rows.append(np.array(ftrans[f[selected]],np.int32,copy=True))
            corner_rows.append(np.array(c[selected],np.uint8,copy=True))
        else:
            face_rows.append(np.empty(0,np.int32));corner_rows.append(np.empty(0,np.uint8))
    if not incident_seen.all():
        raise ValueError('complete original face/corner incidence required')
    top = np.zeros((n,7),np.int32)
    outside = labels[order] == 0
    degrees = np.asarray([len(row) for row in neighbors],np.int32)
    top[outside,0:2] = 1
    top[:,2] = top[:,3] = degrees
    top[:,6] = np.asarray([len(row) for row in face_rows],np.int32)
    state = {name:np.array(a[order],np.float32,copy=True) for name,a in vectors.items()}
    state.update({name:np.array(a[order],np.float32,copy=True) for name,a in scalars.items()})
    state.update(nfaces=len(corrected_faces),faces=corrected_faces,
        face_rip=np.zeros(len(corrected_faces),np.int32),
        face_normals=np.zeros((len(corrected_faces),3),np.float32),face_area=np.zeros(len(corrected_faces),np.float32),
        flattened_vertices=np.zeros((n,3),np.float32),tangent_e1=np.zeros((n,3),np.float32),
        tangent_e2=np.zeros((n,3),np.float32),vertex_area=np.zeros(n,np.float32),
        undefval=np.zeros(n,np.float32),old_undefval=np.zeros(n,np.float32),
        vertex_marks=np.zeros(n,np.int32),vertex_border=np.zeros(n,np.int32),
        vertex_rip=((labels[order] != 0)&(status[order] == 1)).astype(np.int32),
        vertex_topology=top,neighbor_storage=[np.array(row,np.int32,copy=True) for row in neighbors],
        incident_faces=face_rows,incident_corners=corner_rows)
    metadata=copy.deepcopy(dict(source_metadata))
    # VOL_GEOM assignment copies spatial fields/fname and its two transform
    # caches, but register_mat is not assigned into the fresh null destination.
    if isinstance(metadata['volume_geometry'],Mapping) and 'register_mat' in metadata['volume_geometry']:
        metadata['volume_geometry']['register_mat']=None
    metadata.update(surface_type='MRIS_TRIANGULAR_SURFACE',surface_type_code=3,nsize=1,max_nsize=1,
                    vertex_capacity=n+10,face_capacity=2*len(triangles))
    return dict(surface_state=state,vertex_translation=vtrans.copy(),face_translation=ftrans.copy(),
        source_order=order,metadata=metadata,seconds=time.perf_counter()-started,
        inside_order_source='component_BFS' if ordered_defect_vertices is None else 'explicit_source_merge_order',
        scope='complete ordered GA initial base; not greedy search, metric recomputation, candidate or full GA')
