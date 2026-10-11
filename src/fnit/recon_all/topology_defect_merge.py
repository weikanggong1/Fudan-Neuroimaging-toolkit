"""独立 NumPy 邻近缺陷合并；保留有序列表及合并前顶点强度字段。"""
from __future__ import annotations
import math
import time
import numpy as np
from .smooth_surface_python import ordered_neighbors
from .topology_preflight_python import defect_regions


def _sum_rows(array, indices):
    total=np.zeros(array.shape[1:],np.float32)
    for vertex in indices:total=np.asarray(total+array[vertex],np.float32)
    return total


def merge_neighboring_defects(*, canonical_vertices, initial_vertex_normals, vertex_normals,
                              original_vertex_area, faces, component_labels):
    """完整premerge组件/metric → 原顺序merged regions、标签与复制字段。

    canonical_vertices为有限float32(N,3)，自产已中心化球面surface RAS/mm。
    initial_vertex_normals为同序有限float32(N,3)，进入correctTopology前
    未中心化canonical metric保留的法向；vertex_normals为中心化后、
    实际合并之前canonical metric的法向。两个相位须独立自产，不能
    用新的法向覆盖未合并组件的初始border法向。
    original_vertex_area为有限非负float32(N,) raw原表面origarea/mm²，
    不是canonical面积或后续基底calloc面积。faces为原有序整数(F,3)；
    component_labels为整数(N,)的0及连续1..D，每个正标签对应完整
    premerge连通组件。六参数必填，无阈值/设备/线程/精度默认。

    复用原面顺序一环和组件BFS。按旧组件顺序反复计算border一环与
    他组件的接触次数，>=2时按编号顺序并入，vertices直接concat不重做
    BFS，border按动态mark重建。合并后的area是源F32累计origarea除
    顶点数（mm²字段），不改成sum；normal为F32内部法向均值，不再
    单位化。未合并组件保留原sum面积和单位border法向。

    返回dict：regions为按未清空旧编号排序的tuple，每项vertices及
    border=int32向量、status=int32全KEEP0、area=float32、centroid/
    normal=float32(3,)、source_component=int旧0起始编号；labels为
    compact int32(N,)；source_order为outside升序接各merged有序inside
    的int32(N,)；vertex_translation为该顺序逆映射；source_marks为
    原未压缩mark int32(N,)；initial_vertex_val为float32(N,)（原组件
    sum面积在merge之前已写val，合并不回写）；merge_events=int32(M,3)
    每行[目标旧编号,并入旧编号,接触数]、seconds及phase说明。

    输入不变，不读取参考或执行原版；只覆盖固定ADD_EXTRA_VERTICES=0、
    FIND_ENCLOSING_LOOP=0、MERGE_NEIGHBORING_DEFECTS=1。不是缺陷发现、
    retention、SFF或完整GA。非法类型/域/断连标签抛ValueError，F32统计
    溢出抛FloatingPointError。属于mris_fix_topology内部
    mrisMergeNeighboringDefects，没有独立原CLI。
    """
    started=time.perf_counter()
    xyz,initial_normals,normals,areas,triangles,labels=map(np.asarray,
        (canonical_vertices,initial_vertex_normals,vertex_normals,original_vertex_area,faces,component_labels))
    n=len(xyz) if xyz.ndim else 0
    if (xyz.shape!=(n,3) or n==0 or xyz.dtype!=np.float32 or normals.shape!=xyz.shape
            or normals.dtype!=np.float32 or initial_normals.shape!=xyz.shape
            or initial_normals.dtype!=np.float32 or not np.isfinite(initial_normals).all()
            or not np.isfinite(xyz).all()
            or not np.isfinite(normals).all() or areas.shape!=(n,) or areas.dtype!=np.float32
            or not np.isfinite(areas).all() or np.any(areas<0)
            or triangles.ndim!=2 or triangles.shape[1:]!=(3,) or triangles.dtype.kind not in 'iu'
            or labels.shape!=(n,) or labels.dtype.kind not in 'iu' or np.any(labels<0)):
        raise ValueError('same-order finite F32 metric and integer faces/component labels required')
    if triangles.size and (triangles.min()<0 or triangles.max()>=n):
        raise ValueError('face vertex outside raw domain')
    count=int(labels.max(initial=0))
    if count>n or not np.array_equal(np.unique(labels[labels>0]),np.arange(1,count+1)):
        raise ValueError('contiguous positive premerge component IDs required')
    neighbors=ordered_neighbors(triangles.astype(np.int32),n)
    specs=defect_regions(labels,triangles)
    original_val=np.zeros(n,np.float32)
    marks=labels.astype(np.int32,copy=True)
    groups=[]
    for index,(inside,border) in enumerate(specs):
        if len(inside)!=np.count_nonzero(labels==index+1):
            raise ValueError('premerge positive label must be a connected source component')
        area=np.float32(_sum_rows(areas,inside))
        center=np.asarray(_sum_rows(xyz,inside)/np.float32(len(inside)),np.float32)
        normal=_sum_rows(initial_normals,border)
        square=np.float32(np.float32(normal[0]*normal[0]+normal[1]*normal[1])+normal[2]*normal[2])
        length=np.float32(math.sqrt(float(square)))
        if abs(float(length))<float(np.finfo(np.float32).eps):length=np.float32(1)
        normal=np.asarray(normal/length,np.float32)
        original_val[inside]=area
        groups.append(dict(vertices=list(inside),border=list(border),area=area,
            centroid=center,normal=normal,source_component=index))
    events=[];index=0
    while index<count:
        group=groups[index]
        contacts=np.zeros(count,np.int32)
        for border in group['border']:
            for vertex in neighbors[border]:
                mark=int(marks[vertex])
                if mark and mark!=index+1:contacts[mark-1]+=1
        changed=False
        for other in range(count):
            if contacts[other]<2:continue
            # 同一轮contact表不会因并入而重算；清空条目依源仍可命中。
            changed=True
            events.append((index,other,int(contacts[other])))
            group['vertices'].extend(groups[other]['vertices'])
            groups[other]['vertices']=[];groups[other]['border']=[]
            inside=group['vertices'];denominator=np.float32(len(inside))
            group['area']=np.float32(_sum_rows(areas,inside)/denominator)
            group['centroid']=np.asarray(_sum_rows(xyz,inside)/denominator,np.float32)
            group['normal']=np.asarray(_sum_rows(normals,inside)/denominator,np.float32)
        if changed:
            marks[group['vertices']]=index+1
            border=[]
            for vertex in group['vertices']:
                for adjacent in neighbors[vertex]:
                    if marks[adjacent]:continue
                    marks[adjacent]=-1;border.append(adjacent)
            marks[border]=0;group['border']=border
            # 源for-loop i-- 后重跑同组件，新的contact表依新border计算。
        else:index+=1
    retained=[group for group in groups if group['vertices']]
    compact=np.zeros(n,np.int32);regions=[];inside_order=[]
    for number,group in enumerate(retained,1):
        vertices=np.asarray(group['vertices'],np.int32)
        compact[vertices]=number;inside_order.extend(group['vertices'])
        if not (np.isfinite(group['area']) and np.isfinite(group['centroid']).all()
                and np.isfinite(group['normal']).all()):
            raise FloatingPointError('source F32 defect statistics overflow')
        regions.append(dict(group,vertices=vertices,border=np.asarray(group['border'],np.int32),
                            status=np.zeros(len(vertices),np.int32)))
    order=np.asarray(np.flatnonzero(labels==0).tolist()+inside_order,np.int32)
    if len(order)!=n or len(np.unique(order))!=n:raise RuntimeError('merge lost the raw vertex domain')
    translation=np.empty(n,np.int32);translation[order]=np.arange(n,dtype=np.int32)
    return dict(regions=tuple(regions),labels=compact,source_order=order,
        vertex_translation=translation,source_marks=marks,initial_vertex_val=original_val,
        merge_events=np.asarray(events,np.int32).reshape(-1,3),seconds=time.perf_counter()-started,
        source_macros=dict(ADD_EXTRA_VERTICES=0,FIND_ENCLOSING_LOOP=0,MERGE_NEIGHBORING_DEFECTS=1),
        phase='premerge val and precenter border normals retained; postcenter normals only for merged means, before retention and global SFF')
