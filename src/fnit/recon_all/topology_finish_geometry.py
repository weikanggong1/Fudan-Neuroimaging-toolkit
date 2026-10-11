"""全部缺陷结束后的源五轮 SOAP、坐标域保存和有序压缩组合。"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import time

import numpy as np
from numba import njit

from .topology_compact import _integers, _flags, compact_ripped_surface_elements


@njit(cache=True, fastmath=False)
def _soap(xyz, offsets, neighbors, ripped, fixed, iterations):
    current = xyz.copy()
    for _ in range(iterations):
        target = current.copy()
        for vertex in range(len(xyz)):
            if ripped[vertex] or fixed[vertex]:
                continue
            x,y,z = current[vertex]
            num = np.float32(1)
            for slot in range(offsets[vertex], offsets[vertex+1]):
                other = neighbors[slot]
                if ripped[other]:
                    continue
                x = np.float32(x+current[other,0])
                y = np.float32(y+current[other,1])
                z = np.float32(z+current[other,2])
                num = np.float32(num+np.float32(1))
            target[vertex,0] = np.float32(x/num)
            target[vertex,1] = np.float32(y/num)
            target[vertex,2] = np.float32(z/num)
        current = target
    return current


def soap_bubble_vertex_positions(*, vertices, neighbor_offsets, neighbors,
        vertex_rip, vertex_marks, iterations=5) -> dict:
    """显式有序一环/F32坐标→源双pass SOAP，新坐标及原marks副本。

    vertices 为 float32(N,3) surface RAS/mm，neighbor_offsets 整数
    (N+1,)、neighbors 整数(E,) 为源 vt.v[0:vnum] CSR，vertex_rip
    为源 signed-char/bool(N)，vertex_marks 整数(N)。marks==1 固定；
    所有其他非 rip 顶点按原一环顺序把自身和非 rip 邻点累加 F32后
    除以 F32 count，每轮先全体求目标再全体写回。不是 Gauss-Seidel
    或不含中心点的average_positions。iterations 默认5，整数≥0；
    源函数保存/恢复 marked，故本返回保持原marks，不留下内部2/3。

    返回 vertices float32(N,3)、vertex_marks int32(N)副本、moving
    顶点数、iterations、seconds（含校验/JIT，外部读写不计）。无
    输入修改/文件/GPU/线程/精度设置，非法类型/CSR/索引/非有限报
    ValueError，结果非有限报FloatingPointError。属于源
    MRISsoapBubbleVertexPositions 内部步骤，没有独立原软件CLI；
    不执行metric、拓扑搜索、相交清理或最终质量验收。
    """
    started = time.perf_counter();xyz = np.asarray(vertices)
    if xyz.dtype != np.float32 or xyz.ndim != 2 or xyz.shape[1:] != (3,) or not np.isfinite(xyz).all():
        raise ValueError('finite float32 vertices (N,3) required')
    n = len(xyz)
    raw = np.asarray(neighbors)
    if raw.ndim != 1:
        raise ValueError('one-dimensional ordered neighbors required')
    adjacent = _integers(raw,raw.shape,'neighbors',maximum=n)
    offsets = _integers(neighbor_offsets,(n+1,),'neighbor_offsets')
    if offsets[0]!=0 or offsets[-1]!=len(adjacent) or np.any(np.diff(offsets.astype(np.int64))<0):
        raise ValueError('complete ordered one-ring CSR required')
    marks = _integers(vertex_marks,(n,),'vertex_marks')
    ripped = _flags(vertex_rip,n,'vertex_rip')
    if isinstance(iterations,(bool,np.bool_)) or not isinstance(iterations,(int,np.integer)) or iterations<0:
        raise ValueError('iterations must be a nonnegative integer')
    result = _soap(np.ascontiguousarray(xyz),offsets,adjacent,ripped,marks==1,int(iterations))
    if not np.isfinite(result).all():
        raise FloatingPointError('nonfinite SOAP geometry')
    return dict(vertices=result,vertex_marks=marks,moving_vertices=int((~ripped&(marks!=1)).sum()),
        iterations=int(iterations),seconds=time.perf_counter()-started)


def _logical_rows(state, n, triangles):
    top = _integers(state['vertex_topology'],(n,7),'vertex_topology')
    if np.any(top<0) or np.any(top[:,0]>top[:,1]) or np.any(top[:,:2]>3):
        raise ValueError('source topology 0..3 rings and nonnegative counts required')
    rows = {}
    for name,col,limit in (('neighbor_storage',3,n),('incident_faces',6,len(triangles)),('incident_corners',6,3)):
        value=state[name]
        if not isinstance(value,Sequence) or len(value)!=n:
            raise ValueError(name+' requires complete ordered storage')
        converted=[]
        for vertex,source in enumerate(value):
            a=np.asarray(source);count=int(top[vertex,col])
            if a.ndim!=1 or len(a)<count:
                raise ValueError(name+' logical count exceeds storage')
            if count==0 and isinstance(source,(list,tuple)):a=np.empty(0,np.int32)
            converted.append(_integers(a[:count],(count,),name,maximum=limit))
        rows[name]=converted
    incident=rows['incident_faces'];corners=rows['incident_corners']
    seen=np.zeros((len(triangles),3),np.bool_)
    for vertex,(f,c) in enumerate(zip(incident,corners)):
        if len(f) and not np.all(triangles[f,c]==vertex):
            raise ValueError('ordered incidence does not belong to its vertex')
        packed=f.astype(np.int64)*3+c
        if len(np.unique(packed))!=len(packed) or np.any(seen[f,c]):
            raise ValueError('duplicate logical face/corner incidence')
        seen[f,c]=True
    if not seen.all():raise ValueError('complete logical face/corner incidence required')
    offsets=np.r_[0,np.cumsum([len(row) for row in incident],dtype=np.int64)]
    flat=np.concatenate(incident) if incident else np.empty(0,np.int32)
    cor=np.concatenate(corners) if corners else np.empty(0,np.int32)
    return top,rows,offsets,flat,cor


def finish_topology_geometry(*, surface_state, defect_vertices,
        border_vertices, vertex_translation, normal_status, iterations=5) -> dict:
    """全部GA缺陷完成状态→源metric几何子步/五輪SOAP/save orig/restore TMP。

    surface_state 为完整自有基底/候选状态，需 original/current/
    temporary_vertices、vertex_normals float32(N,3)，vertex_area
    float32(N)、rip/marks/topology、有序neighbor/incident/face和logical
    nfaces。defect_vertices/border_vertices 为所有源缺陷的有序原编号
    rows，vertex_translation 为完整原→当前int映射（−1删除，源越界
    mapped编号按原循环跳过）。normal_status 必须显式surface/sphere/
    spherical_patch/parameterized_sphere；不支持plane/cut的额外面朝向
    语义。iterations 默认5，对应主程序 nsmooth=5，可显式0。

    忘记多环缓存，仅消费原vnum一环；复用经过验证的normal-pending
    五轮内核，传入真正原序邻接/incidence和现法向/面积/rip，不从面
    重建或跳过退化处理。这里覆盖metric会改变且最终保留的XYZ/normal
    子步，不生成不被此终段消费的distance/angle/area统计，不能当完整
    MRIScomputeMetricProperties接口。 unresolved pending按严格门报错。
    marks全部1，再按每缺陷inside/border映射置0，调用同模块完整SOAP。
    原始 ORIGINAL 更新为 SOAP后几何；current 恢复显式temporary/raw
    坐标，marks清零。法向保持源SOAP之前metric状态，不暗中再重算。

    返回 surface_state、metric_geometry（退化/坐标变更诊断）、soap、
    seconds。输入不修改，不调用原程序，不清理相交/不执行GA或发布。
    非法或缺失状态ValueError，未支持status/未解决法向RuntimeError。
    source_final_phase_complete=False表示完整全25原版参考/QC尚未验证，
    不能把当前组装候选当整例验收。另可调用finish_and_compact_topology
    完成源序元素压缩，之后源主命令的current清理与ORIGINAL写出仍分开。
    """
    from .normal_pending_python import _repair
    start=time.perf_counter()
    if not isinstance(surface_state,Mapping):raise ValueError('complete explicit surface_state required')
    state=dict(surface_state)
    required=('original_vertices','current_vertices','temporary_vertices','vertex_normals','vertex_area',
        'vertex_rip','vertex_topology','neighbor_storage','incident_faces','incident_corners','faces','nfaces','face_rip')
    if any(key not in state for key in required):raise ValueError('complete final geometry/topology state required')
    xyz=np.asarray(state['original_vertices']);n=len(xyz)
    for key in ('original_vertices','current_vertices','temporary_vertices','vertex_normals'):
        a=np.asarray(state[key])
        if a.shape!=(n,3) or a.dtype!=np.float32 or not np.isfinite(a).all():
            raise ValueError('finite source float32(N,3) field required: '+key)
    area=np.asarray(state['vertex_area'])
    if area.shape!=(n,) or area.dtype!=np.float32 or not np.isfinite(area).all():
        raise ValueError('explicit finite float32 vertex_area required')
    faces=np.asarray(state['faces']);nf=state['nfaces']
    if (isinstance(nf,(bool,np.bool_)) or not isinstance(nf,(int,np.integer))
            or faces.ndim!=2 or faces.shape[1:]!=(3,) or not 0<=nf<=len(faces)):
        raise ValueError('explicit logical nfaces within source face capacity required')
    triangles=_integers(faces[:nf],(int(nf),3),'faces',maximum=n)
    rip=_flags(state['vertex_rip'],n,'vertex_rip')
    flags=np.asarray(state['face_rip'])
    if flags.shape!=(len(faces),):raise ValueError('full face capacity rip flags required')
    frip=_flags(flags,len(faces),'face_rip')[:nf]
    if normal_status not in ('surface','sphere','spherical_patch','parameterized_sphere'):
        raise ValueError('explicit supported final normal_status required')
    top,rows,incident_offsets,incident,corner=_logical_rows(state,n,triangles)
    if isinstance(iterations,(bool,np.bool_)) or not isinstance(iterations,(int,np.integer)) or iterations<0:
        raise ValueError('iterations must be a nonnegative integer')
    translation=np.asarray(vertex_translation)
    if translation.ndim!=1:raise ValueError('complete original-domain vertex_translation required')
    translation=_integers(translation,translation.shape,'vertex_translation')
    if not isinstance(defect_vertices,Sequence) or not isinstance(border_vertices,Sequence) or len(defect_vertices)!=len(border_vertices):
        raise ValueError('all ordered defects and corresponding borders required')
    marks=np.ones(n,np.int32)
    for inside,border in zip(defect_vertices,border_vertices):
        for row in (inside,border):
            raw=np.asarray(row)
            if raw.ndim!=1:raise ValueError('ordered original-domain defect/border row required')
            values=_integers(raw,raw.shape,'defect/border',maximum=len(translation))
            mapped=translation[values];valid=(mapped>=0)&(mapped<n)
            marks[mapped[valid]]=0
    # Preserve source ordered vt.vnum even for unsupported face edges. The
    # mature pending normal core uses these rows to perturb failed neighbors.
    degrees=top[:,3].copy();width=int(degrees.max(initial=0))
    neighbors=np.zeros((n,width),np.int64)
    for vertex,row in enumerate(rows['neighbor_storage']):neighbors[vertex,:len(row)]=row
    positions=np.array(xyz,copy=True);normals=np.array(state['vertex_normals'],copy=True)
    areas=area.copy();origarea=np.zeros(n,np.float32);rawface=np.zeros((nf,3),np.float32);border=np.zeros(n,np.bool_)
    tick=time.perf_counter()
    pending,counts,history,changed,updates=_repair(positions,triangles.astype(np.int64),
        incident_offsets,incident,corner,neighbors,degrees,rip,frip,normals,areas,origarea,rawface,border,
        False,True,True)
    metric_time=time.perf_counter()-tick
    if len(pending):raise RuntimeError('final metric normal-pending remains unresolved')
    neighbor_offsets=np.r_[0,np.cumsum(degrees,dtype=np.int64)]
    flat=np.concatenate(rows['neighbor_storage']) if n else np.empty(0,np.int32)
    soap=soap_bubble_vertex_positions(vertices=positions,neighbor_offsets=neighbor_offsets,
        neighbors=flat,vertex_rip=rip,vertex_marks=marks,iterations=iterations)
    top=top.copy();top[:,0:2]=1;top[:,2]=top[:,3];top[:,4:6]=0
    state.update(original_vertices=soap['vertices'],current_vertices=np.array(state['temporary_vertices'],copy=True),
        vertex_normals=normals,vertex_area=areas,vertex_marks=np.zeros(n,np.int32),vertex_topology=top,
        neighbor_storage=[row.copy() for row in rows['neighbor_storage']])
    return dict(surface_state=state,metric_geometry=dict(pending_counts=counts,
        changed_vertices=np.flatnonzero(changed),coordinate_update_count=int(updates),
        seconds=metric_time,retained_state_scope='XYZ and vertex normals; not complete metric statistics'),
        soap=soap,seconds=time.perf_counter()-start,iterations=int(iterations),
        source_final_phase_complete=False,complete_ga=False,final_mesh_quality='not_assessed')


def finish_and_compact_topology(*, surface_state, metadata, defect_vertices,
        border_vertices, vertex_translation, normal_status, iterations=5) -> dict:
    """复用finish→compact；两个坐标域和阶段报告均保留，尚未final QC。

    参数/单位/失败行为分别见上述两接口；不添加新的默认或近似。
    返回compact字段加finish报告和含两次校验的total_seconds。调用方
    仍须选择current/TMP清理和ORIGINAL写出、WM几何头及remesh/相交门。
    """
    start=time.perf_counter()
    result=finish_topology_geometry(surface_state=surface_state,defect_vertices=defect_vertices,
        border_vertices=border_vertices,vertex_translation=vertex_translation,
        normal_status=normal_status,iterations=iterations)
    compact=compact_ripped_surface_elements(surface_state=result['surface_state'],metadata=metadata)
    compact.update(finish=result,total_seconds=time.perf_counter()-start)
    return compact
