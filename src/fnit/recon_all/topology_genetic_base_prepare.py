"""从原 nofix 几何/MRI生成 base 前字段，保留显式曲率与缺陷输入边界。"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import time
import numpy as np
from numba import njit

from .place_surface_normals import initial_vertex_normals, ordered_face_csr
from .smooth_surface_python import ordered_neighbors, average_positions
from .topology_edge_mri_score import _sample, _median_twice, _validated_volume

# Fixed triangular MRISreadOverAlloc -> MRIScorrectTopology GA path: these
# calloc/bzero fields are never assigned before the base-copy loop. They are
# not loaded by the geometry, original-property or vertex-position readers.
ZERO_BEFORE_BASE = ('valbak', 'imag_val', 'curvbak', 'stat', 'mean',
                    'mean_imag', 'std_error')


@njit(cache=True, fastmath=False)
def _sample_border_fields(xyz, normal, ripped, volume, matrix, initial_border):
    white = np.zeros(len(xyz), np.float32)
    gray = np.zeros(len(xyz), np.float32)
    border = initial_border.copy()
    for v in range(len(xyz)):
        if ripped[v]:
            continue
        # The source has double white_val/gray_val. val receives the average
        # BEFORE either sample is rounded to its separate float32 field.
        w = _sample(volume, matrix,
            np.float64(xyz[v,0])-.5*np.float64(normal[v,0]),
            np.float64(xyz[v,1])-.5*np.float64(normal[v,1]),
            np.float64(xyz[v,2])-.5*np.float64(normal[v,2]))
        g = _sample(volume, matrix,
            np.float64(xyz[v,0])+.5*np.float64(normal[v,0]),
            np.float64(xyz[v,1])+.5*np.float64(normal[v,1]),
            np.float64(xyz[v,2])+.5*np.float64(normal[v,2]))
        white[v], gray[v] = np.float32(w), np.float32(g)
        border[v] = np.float32((w+g)/2.)
    return border, white, gray


def prepare_nofix_border_fields(*, vertices, faces, defect_vertices,
                               initial_border_values, volume, ras_to_voxel) -> dict:
    """raw nofix/MRI→独立两轮median后的val/val2/val2bak与raw法向。

    vertices为有限float32(N,3) surface RAS/mm；faces为原有序整数(F,3)。
    defect_vertices是bool(N,)全部缺陷标记，临时rip；initial_border_values
    是有限float32(N,)源采样前val（缺陷处此前写defect.area，非缺陷将被
    覆盖）。volume为uint8(X,Y,Z)归一化MRI，ras_to_voxel为有限(4,4)
    surface RAS→连续体素矩阵。每参数必填，无设备/迭代/精度开关。

    复用原有序法向、MRI采样及两轮中值；val由双精度白灰采样先平均再
    写float32，独立于val2/val2bak滤波。缺陷val保留输入，缺陷白灰为原
    fresh三角面状态的零初始化，不跳过外部邻域。返回val/val2/val2bak
    float32(N,)、raw_vertex_normals float32(N,3)、有序neighbors及seconds。
    输入不变，不改线程/精度/CUDA。格式/非有限/非法索引报ValueError；
    有关联面的raw法向全抵消报NotImplementedError，须由上游完整metric
    pending-repair先处理，不能静默以零法向代替。没有原软件独立CLI。
    """
    started=time.perf_counter()
    xyz=np.asarray(vertices);triangles=np.asarray(faces);marked=np.asarray(defect_vertices)
    border=np.asarray(initial_border_values)
    if (xyz.ndim!=2 or xyz.shape[1:]!=(3,) or xyz.dtype!=np.float32 or len(xyz)==0
            or not np.isfinite(xyz).all() or triangles.ndim!=2 or triangles.shape[1:]!=(3,)
            or triangles.dtype.kind not in 'iu' or marked.shape!=(len(xyz),) or marked.dtype!=np.bool_
            or border.shape!=(len(xyz),) or border.dtype!=np.float32 or not np.isfinite(border).all()):
        raise ValueError('finite raw float32 geometry/border, integer faces and bool all-defect mask required')
    if triangles.size and (triangles.min()<0 or triangles.max()>=len(xyz)):
        raise ValueError('original face index outside vertex range')
    image,matrix=_validated_volume(volume,ras_to_voxel)
    triangles=np.ascontiguousarray(triangles,np.int32)
    neighbors=ordered_neighbors(triangles,len(xyz))
    normal=initial_vertex_normals(vertices=xyz,triangles=triangles)
    offsets,_,_=ordered_face_csr(triangles,len(xyz))
    if np.any((np.diff(offsets)>0)&np.all(normal==0,axis=1)):
        raise NotImplementedError('raw metric normal-pending repair must precede base preparation')
    fields=_sample_border_fields(np.ascontiguousarray(xyz),normal,marked,image,matrix,border)
    degree=np.asarray([len(row) for row in neighbors],np.int32)
    rings=np.zeros((len(xyz),int(degree.max(initial=0))),np.int32)
    for v,row in enumerate(neighbors):rings[v,:len(row)]=row
    outputs=[_median_twice(value,marked,rings,degree) for value in fields]
    return dict(zip(('val','val2','val2bak'),outputs),raw_vertex_normals=normal,
        neighbors=neighbors,seconds=time.perf_counter()-started,
        scope='fresh triangular nofix MRI border fields; not complete pre-base metric or GA')


def prepare_genetic_base_sff1_inputs(*, vertices, faces, canonical_vertices,
                                     defect_vertices, precenter_vertices=None) -> dict:
    """生成首次 global SFF 的原序几何/法向/二环字段，不读取参考曲率。

    vertices/canonical_vertices为同序有限float32(N,3)，分别raw nofix和
    自有已中心化球面，坐标surface RAS/mm；faces为原序整数(F,3)，
    defect_vertices为bool(N,)全部缺陷。四参数必填，不做额外投影/平滑。
    先复用完整五轮normal pending核计算canonical metric，再按源rip规则
    计算raw法向：inside保留canonical法向/面积，outside重算raw。首次
    SFF前解除rip，不把后续base前的全raw法向当作SFF输入。

    复用成熟inflation一环/二环原序表，返回sff_arguments供
    global_second_fundamental_form_state展开：current_vertices、mixed
    vertex_normals、二环CSR、全False vertex_rip、fresh初始化为0的
    initial_k1/initial_k2/initial_h/initial_gaussian_curvature、vertex_area。
    另返回canonical_normals/raw_all_vertex_normals/raw_original_areas、normal阶段诊断、
    一环CSR和seconds。面积mm²，初始曲率各为mm^-1/mm^-2。
    precenter_vertices默认None，兼容旧SFF阶段；给定时须为实际
    project_and_smooth_sphere输出的有限float32(N,3)/mm，另自产
    initial_segment_vertex_normals（原startup、中心前metric法向）。
    未提供返回None，禁止以中心后canonical法向假称此初始字段。

    本接口不将pending导致的共享坐标变更静默丢弃：若任一阶段真实
    扰动坐标或仍未解决，明确NotImplementedError，调用方须在更早
    preflight传播完整metric状态；正常零pending路径保留原算法。
    输入格式/索引非法报ValueError，不改全局线程/精度，不调用原版
    程序；二环/SFF1真实门与完整base门分别报告，不是完整GA。
    """
    from .normal_pending_python import repair_vertex_normals
    from .place_surface_normals import FaceNormalTopology
    from .inflate_topology import inflate_neighbor_tables
    started=time.perf_counter()
    raw=np.asarray(vertices);canonical=np.asarray(canonical_vertices)
    triangles=np.asarray(faces);ripped=np.asarray(defect_vertices)
    if (raw.ndim!=2 or raw.shape[1:]!=(3,) or raw.dtype!=np.float32 or len(raw)==0
            or canonical.shape!=raw.shape or canonical.dtype!=np.float32
            or not np.isfinite(raw).all() or not np.isfinite(canonical).all()
            or ripped.shape!=(len(raw),) or ripped.dtype!=np.bool_):
        raise ValueError('same-order finite raw/canonical float32 and bool all-defect mask required')
    topology=FaceNormalTopology(triangles=triangles,nvertices=len(raw))
    def normal_stage(xyz, *, mask=None, initial=None, areas=None):
        result=repair_vertex_normals(positions=xyz,faces=triangles,topology=topology,
            status='sphere',unitize=True,fix_vertex_area=True,ripped=mask,
            initial_normals=initial,initial_areas=areas)
        if len(result.unresolved_vertices) or result.changed_vertices.any():
            raise NotImplementedError('preflight must propagate normal pending coordinate changes before SFF1')
        return result
    initial_segment_state=None
    if precenter_vertices is not None:
        precenter=np.asarray(precenter_vertices)
        if precenter.shape!=raw.shape or precenter.dtype!=np.float32 or not np.isfinite(precenter).all():
            raise ValueError('explicit actual precenter sphere float32(N,3) required')
        initial_segment_state=normal_stage(precenter)
    canonical_state=normal_stage(canonical)
    raw_state=normal_stage(raw,mask=ripped,initial=canonical_state.normals,
        areas=canonical_state.areas)
    final_raw=normal_stage(raw)
    one,n1,two,n2=inflate_neighbor_tables(normal_topology=topology)
    def csr(table,degree):
        offsets=np.empty(len(raw)+1,np.int32);offsets[0]=0
        totals=np.cumsum(degree,dtype=np.int64)
        if totals[-1]>np.iinfo(np.int32).max:raise ValueError('source neighbor storage exceeds int32')
        offsets[1:]=totals
        neighbors=table[np.arange(table.shape[1])[None,:]<degree[:,None]].copy()
        return offsets,neighbors
    offsets,neighbors=csr(two,n2);one_offsets,one_neighbors=csr(one,n1)
    zero=np.zeros(len(raw),np.float32)
    args=dict(current_vertices=raw.copy(),vertex_normals=raw_state.normals,
        neighbor_offsets=offsets,neighbors=neighbors,vertex_rip=np.zeros(len(raw),bool),
        initial_k1=zero.copy(),initial_k2=zero.copy(),initial_h=zero.copy(),
        initial_gaussian_curvature=zero.copy(),vertex_area=raw_state.areas)
    return dict(sff_arguments=args,canonical_normals=canonical_state.normals,
        initial_segment_vertex_normals=None if initial_segment_state is None else initial_segment_state.normals,
        raw_all_vertex_normals=final_raw.normals,raw_original_areas=final_raw.areas,
        one_ring_offsets=one_offsets,
        one_ring_neighbors=one_neighbors,seconds=time.perf_counter()-started,
        normal_pending_counts=dict(initial_segment=None if initial_segment_state is None else initial_segment_state.pending_counts.tolist(),
            canonical=canonical_state.pending_counts.tolist(),
            ripped_raw=raw_state.pending_counts.tolist(),all_raw=final_raw.pending_counts.tolist()),
        scope='self-generated prepared SFF1 fields; complete raw-to-base validation pending')


def prepare_genetic_base_field_inputs(*, vertices, faces, canonical_vertices,
        defect_labels, defect_status, defect_areas_mm2, ordered_defect_regions,
        principal_curvature_fields, volume, ras_to_voxel, source_metadata,
        initial_border_values=None) -> dict:
    """自产几何/MRI/已计算原序曲率→create_genetic_base的全部字段参数。

    vertices/faces为fresh原三角面nofix，canonical_vertices为自有preflight
    中心化球面float32(N,3)mm。defect_labels整数(N,)为0或1…D；status
    整数(N,)为inside KEEP0/DISCARD1；defect_areas_mm2 float32(D,)为原
    缺陷面积。ordered_defect_regions为D个dict，每个显式inside/border/
    hull有序原顶点整数数组，inside必须对应标签，重叠hull/border保留。
    initial_border_values默认None，兼容未合并component以其原面积填val；
    对经过source merge的domain必须显式传float32(N,)原component val，
    因为merge发生在首次val赋值之后，并不将新area回写到原顶点val。

    principal_curvature_fields要求上游自有global SFF完整float32(N,)
    k1/k2/H/K；单位依次mm^-1/mm^-1/mm^-1/mm^-2，不能用参考after填入。
    未更新K/H可保留原IEEE位；本函数不重算/猜测SFF或有序vtotal。volume/
    ras_to_voxel同border接口。source_metadata为create_genetic_base要求
    的显式头/空间/status属性，不从参考几何补值。所有参数必填。

    返回{base_arguments,field_provenance,stage_seconds,validation_scope}。
    base_arguments可具名展开到create_genetic_base；5向量、15标量全
    完整。raw TMP current/temporary与两轮平均orig分开；最终curv按源
    labels→borders→逐defect(hull2,border-1,inside编号)覆盖。7个未写入
    标量从fresh读取的原bzero规则自产零，而非冻结参考。incidents复用
    原face/corner CSR。非法/不完整输入报ValueError；raw pending分支
    见border接口。无原FS调用、输入别名、线程/精度更改；global SFF/
    缺陷面积/完整前metric仍属于显式上游，不宣称rawMRI→base全链验收。
    """
    # The standalone border stage does not construct a genetic base. Load
    # its schema only here so that this independently usable stage does not
    # require later retessellation/controller modules at import time.
    from .topology_genetic_base import COPIED_SCALAR_FIELDS
    started=time.perf_counter()
    xyz=np.asarray(vertices);triangles=np.asarray(faces);canonical=np.asarray(canonical_vertices)
    labels=np.asarray(defect_labels);status=np.asarray(defect_status)
    if (xyz.ndim!=2 or xyz.shape[1:]!=(3,) or xyz.dtype!=np.float32 or len(xyz)==0
            or canonical.shape!=xyz.shape or canonical.dtype!=np.float32
            or not np.isfinite(xyz).all() or not np.isfinite(canonical).all()
            or labels.shape!=(len(xyz),) or labels.dtype.kind not in 'iu' or np.any(labels<0)
            or status.shape!=labels.shape or status.dtype.kind not in 'iu'):
        raise ValueError('same-order raw/canonical float32 and original integer labels/status required')
    d=int(labels.max(initial=0));areas=np.asarray(defect_areas_mm2)
    if (areas.shape!=(d,) or areas.dtype!=np.float32 or not np.isfinite(areas).all()
            or np.any(areas<0) or not np.array_equal(np.unique(labels[labels>0]),np.arange(1,d+1))
            or not isinstance(ordered_defect_regions,Sequence) or len(ordered_defect_regions)!=d
            or np.any((status[labels>0]<0)|(status[labels>0]>1))):
        raise ValueError('complete contiguous defect labels, areas and ordered region records required')
    if not isinstance(principal_curvature_fields,Mapping) or not isinstance(source_metadata,Mapping):
        raise ValueError('explicit upstream curvature fields and metadata required')
    curvature={}
    for name in ('k1','k2','H','K'):
        a=np.asarray(principal_curvature_fields.get(name))
        if a.shape!=(len(xyz),) or a.dtype!=np.float32:
            raise ValueError('complete float32 upstream SFF field required: '+name)
        curvature[name]=a.copy()
    regions=[]
    for i,record in enumerate(ordered_defect_regions,1):
        if not isinstance(record,Mapping):raise ValueError('explicit ordered defect record required')
        row={}
        for name in ('inside','border','hull'):
            a=np.asarray(record.get(name))
            if (a.ndim!=1 or a.dtype.kind not in 'iu' or (a.size and (a.min()<0 or a.max()>=len(xyz)))):
                raise ValueError('valid ordered original-index '+name+' required')
            row[name]=a.astype(np.int32,copy=True)
        if not np.array_equal(np.sort(row['inside']),np.flatnonzero(labels==i)):
            raise ValueError('ordered inside must match its complete defect label')
        regions.append(row)
    if initial_border_values is None:
        initial=np.zeros(len(xyz),np.float32)
        for number,area in enumerate(areas,1):initial[labels==number]=area
    else:
        initial=np.asarray(initial_border_values)
        if initial.shape!=(len(xyz),) or initial.dtype!=np.float32 or not np.isfinite(initial).all():
            raise ValueError('finite explicit original-component initial_border_values float32(N,) required')
        initial=initial.copy()
    border=prepare_nofix_border_fields(vertices=xyz,faces=triangles,defect_vertices=labels!=0,
        initial_border_values=initial,volume=volume,ras_to_voxel=ras_to_voxel)
    tick=time.perf_counter()
    original=average_positions(vertices=xyz,faces=triangles,iterations=2,device='cpu')
    averaged_seconds=time.perf_counter()-tick
    scalar={name:np.zeros(len(xyz),np.float32) for name in ZERO_BEFORE_BASE}
    scalar.update({name:border[name] for name in ('val','val2','val2bak')},**curvature)
    curv=np.zeros(len(xyz),np.float32)
    for i,row in enumerate(regions,1):curv[row['inside']]=np.float32(i)
    for i,row in enumerate(regions,1):curv[row['border']]=np.float32(i)
    for i,row in enumerate(regions,1):
        curv[row['hull']]=np.float32(2);curv[row['border']]=np.float32(-1);curv[row['inside']]=np.float32(i)
    scalar['curv']=curv
    if set(scalar)!=set(COPIED_SCALAR_FIELDS):raise RuntimeError('incomplete copied scalar schema')
    offsets,ids,corners=ordered_face_csr(triangles,len(xyz))
    args=dict(faces=np.array(triangles,np.int32,copy=True),defect_labels=labels.copy(),defect_status=status.copy(),
        source_vertex_fields=dict(current_vertices=xyz.copy(),original_vertices=original,
            temporary_vertices=xyz.copy(),canonical_vertices=canonical.copy(),vertex_normals=border['raw_vertex_normals']),
        source_scalar_fields=scalar,source_neighbors=border['neighbors'],
        source_incident_faces=[ids[offsets[v]:offsets[v+1]].astype(np.int32) for v in range(len(xyz))],
        source_incident_corners=[corners[offsets[v]:offsets[v+1]].astype(np.int32) for v in range(len(xyz))],
        source_vertex_rip=np.zeros(len(xyz),np.int32),source_metadata=dict(source_metadata),
        ordered_defect_vertices=[row['inside'].copy() for row in regions])
    provenance={name:'fresh triangular allocation; no pre-base assignment' for name in ZERO_BEFORE_BASE}
    provenance.update(val='double raw sample mean -> F32 -> independent two medians; defect.area retained',
        val2='white raw sample -> F32 -> two medians',val2bak='gray raw sample -> F32 -> two medians',
        curv='ordered source diagnostic labels/borders/hull overlay',
        k1='explicit self-generated global SFF',k2='explicit self-generated global SFF',
        H='explicit self-generated global SFF',K='explicit self-generated global SFF')
    return dict(base_arguments=args,field_provenance=provenance,
        stage_seconds=dict(border=border['seconds'],two_round_orig_average=averaged_seconds,
            complete_api=time.perf_counter()-started),
        validation_scope=dict(raw_mri_to_base_fully_validated=False,global_sff_explicit_upstream=True,
            defect_area_explicit_upstream=True,before_base_source_capture='pending authorization',
            production_default_changed=False,overall_metric_equivalence='not_assessed'))


def prepare_nofix_genetic_base(*, vertices, faces, qsphere_vertices, volume,
                               ras_to_voxel, source_metadata) -> dict:
    """fresh nofix/qsphere/MRI→独立完整初始base候选，不读取参考属性。

    vertices/qsphere_vertices为同有序有限float32(N,3) surface RAS/mm，
    faces为共同整数(F,3)；volume uint8(X,Y,Z)、ras_to_voxel(4,4)
    surface RAS→MRI连续体素，source_metadata为显式自身头/空间属性。
    六参必填，无设备或精度默认，不修改输入或全局线程/CUDA/TF32。

    复用原球面中心化/组件发现、canonical→ripped raw→raw metric、
    全局SFF1、源ordered merge、原raw retention、ordered小hull、三个
    median字段及两轮orig平均，最后调用create_genetic_base。合并前
    初始val与合并后area分开；K/H来自raw mixed-normal SFF1，base法向
    则来自最后raw metric，不使用原mris后续SFF2或参考DVS的任何字段。

    返回base的完整字段外附sff1、merge、retention_status、prepared、
    center_iterations与seconds；这些中间数组均自产。异常沿用各成熟
    子函数，特别保留pending几何变化的明确未支持失败，不静默弃掉。
    此API是初始producer候选，不执行GA；完整真实字段/网格/整例及
    指标等效状态需由调用者另外验证，不切换生产默认。
    """
    from .topology_preflight_python import (center_sphere, project_and_smooth_sphere,
        defect_component_labels, defect_retention_status)
    from .topology_global_sff import global_second_fundamental_form_state
    from .topology_defect_merge import merge_neighboring_defects
    from .topology_genetic_base import create_genetic_base
    started=time.perf_counter()
    raw,triangles,sphere=np.asarray(vertices),np.asarray(faces),np.asarray(qsphere_vertices)
    if (raw.ndim!=2 or raw.shape[1:]!=(3,) or raw.dtype!=np.float32
            or sphere.shape!=raw.shape or sphere.dtype!=np.float32
            or not np.isfinite(raw).all() or not np.isfinite(sphere).all()):
        raise ValueError('same-order finite float32 nofix/qsphere coordinates required')
    precenter=project_and_smooth_sphere(sphere,triangles)
    canonical,iterations=center_sphere(precenter)
    component=defect_component_labels(canonical,triangles)
    metric=prepare_genetic_base_sff1_inputs(vertices=raw,faces=triangles,
        canonical_vertices=canonical,defect_vertices=component!=0,precenter_vertices=precenter)
    sff=global_second_fundamental_form_state(**metric['sff_arguments'])
    merge=merge_neighboring_defects(canonical_vertices=canonical,
        initial_vertex_normals=metric['initial_segment_vertex_normals'],
        vertex_normals=metric['canonical_normals'],original_vertex_area=metric['raw_original_areas'],
        faces=triangles,component_labels=component)
    # The source retention pass precedes averaging orig, and iterates the
    # merged concatenated inside order. Outside0/KEEP+1/DISCARD-1 is converted
    # only at the source base's explicit KEEP0/DISCARD1 boundary.
    retained=defect_retention_status(canonical,raw,triangles,merge['labels'],
        ordered_regions=merge['regions'])
    status=np.asarray(retained==-1,np.int32)
    neighbors=ordered_neighbors(triangles,len(raw))
    regions=[]
    for row in merge['regions']:
        inside,border=row['vertices'],row['border']
        occupied=set(inside.tolist())|set(border.tolist());hull=border.tolist()
        for vertex in border:
            for adjacent in neighbors[vertex]:
                if adjacent not in occupied:
                    occupied.add(adjacent);hull.append(adjacent)
        regions.append(dict(inside=inside,border=border,hull=np.asarray(hull,np.int32)))
    fields=prepare_genetic_base_field_inputs(vertices=raw,faces=triangles,
        canonical_vertices=canonical,defect_labels=merge['labels'],defect_status=status,
        defect_areas_mm2=np.asarray([row['area'] for row in merge['regions']],np.float32),
        ordered_defect_regions=regions,
        principal_curvature_fields=dict(k1=sff['k1'],k2=sff['k2'],H=sff['h'],K=sff['gaussian_curvature']),
        volume=volume,ras_to_voxel=ras_to_voxel,source_metadata=source_metadata,
        initial_border_values=merge['initial_vertex_val'])
    base=create_genetic_base(**fields['base_arguments'])
    if not np.array_equal(base['vertex_translation'],merge['vertex_translation']):
        raise RuntimeError('self-generated merge/base translations disagree')
    return dict(base,prepared=fields,sff1=sff,merge=merge,retention_status=retained,
        metric=metric,center_iterations=iterations,seconds=time.perf_counter()-started,
        complete_ga=False,overall_metric_equivalence='not_assessed',production_default_changed=False)
