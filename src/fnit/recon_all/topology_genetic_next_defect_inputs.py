"""当前 corrected 邻接/球面→下一缺陷边表；原 MRI 准备按版本复用。"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import hashlib
import time
import numpy as np

from .topology_genetic_defect_inputs import (
    GeneticDefectInputContext, _region_signature, _signature,
    ordered_genetic_defect_inputs, prepare_ordered_genetic_defect_context,
)


def _current_candidate_base(*, surface_state, vertex_count, candidates,
                            vertex_translation):
    """读取被消费行的有向 vnum 前缀；不重建面边或扩为全部 base 剪枝。"""
    if not isinstance(surface_state, Mapping):
        raise ValueError('explicit current_surface_state field mapping required')
    required=('canonical_vertices','vertex_topology','neighbor_storage')
    if any(key not in surface_state for key in required):
        raise ValueError('current corrected canonical_vertices/topology/storage required')
    canonical=np.asarray(surface_state['canonical_vertices'])
    topology=np.asarray(surface_state['vertex_topology'])
    rows=surface_state['neighbor_storage']
    if (canonical.shape!=(vertex_count,3) or canonical.dtype!=np.float32
            or not np.isfinite(canonical).all()):
        raise ValueError('finite corrected-domain float32(N,3) canonical_vertices required')
    if (topology.shape!=(vertex_count,7) or topology.dtype.kind not in 'iu'
            or np.any(topology[:,3]<0)):
        raise ValueError('integer current vertex_topology(N,7) with nonnegative vnum required')
    if not isinstance(rows, Sequence) or len(rows)!=vertex_count:
        raise ValueError('current neighbor_storage must contain N ordered rows')
    # Source edgeExists reads only vt[vno1].v[:vnum].  A reverse-only edge,
    # rip flag, or stale face list must not change this directed predicate.
    # Only candidate pairs become ETable rows; source forward-prune does not
    # enumerate other accepted base edges elsewhere in the current surface.
    base=set();digest=hashlib.sha256();queried=0;used_rows=0
    translated=np.asarray(vertex_translation)[candidates]
    for i,raw_first in enumerate(candidates[:-1]):
        first=int(translated[i]);size=int(topology[first,3])
        row=np.asarray(rows[first])
        if row.ndim!=1 or row.dtype.kind not in 'iu' or size>len(row):
            raise ValueError('consumed neighbor row must contain integer vnum prefix')
        row=row[:size]
        if row.size and (row.min()<0 or row.max()>=vertex_count):
            raise ValueError('consumed current neighbor index outside corrected vertex domain')
        digest.update(np.asarray([first,size],dtype='<i8').tobytes())
        digest.update(np.asarray(row,dtype='<i8').tobytes())
        neighbors=set(map(int,row));used_rows+=1
        for j in range(i+1,len(candidates)):
            queried+=1
            if int(translated[j]) in neighbors:
                # The old core uses undirected membership keys.  Each raw pair
                # is enumerated once in this exact candidate order, so populate
                # a key only when the corresponding directed source query hit.
                base.add(tuple(sorted((int(raw_first),int(candidates[j])))))
    return canonical,frozenset(base),dict(
        current_candidate_queries=queried,consumed_neighbor_rows=used_rows,
        consumed_neighbor_prefix_sha256=digest.hexdigest(),
        current_canonical_sha256=hashlib.sha256(canonical.tobytes(order='C')).hexdigest(),
        used_base_candidate_pairs=len(base),
        directed_edge_exists=True,rip_flags_consumed=False,
        current_faces_used_to_rebuild_neighbors=False,
        prune_scope='current ETable used1 rows; only forward candidate pairs',
        outside_candidate_base_edges_scanned=False,
    )


def next_ordered_genetic_defect_inputs(*, canonical_vertices, original_vertices,
        original_normals, faces, defect_labels, ordered_regions,
        retention_status, vertex_translation, white_values, gray_values,
        volume, ras_to_voxel, defect_number, current_surface_state,
        use_overlap=False, context=None) -> dict:
    """静态原网格/MRI准备+当前 corrected 状态→下一缺陷完整边输入。

    canonical_vertices 是原 mris 顶点域的 float32(N,3) 球面坐标（mm）；
    faces/defect_labels/ordered_regions/vertex_translation 绑定同一原网格
    与合并顺序。original_vertices/original_normals、white_values/gray_values、
    volume/ras_to_voxel、retention_status 和 defect_number 的结构与成熟
    ordered_genetic_defect_inputs 相同，MRI评分与原边100惩罚保持原规则。

    current_surface_state 必填，消费 corrected 顶点域 canonical_vertices
    float32(N,3)、vertex_topology 整数(N,7) 的第3列 vnum（从0计）与
    neighbor_storage 的 N 个有序整数行。仅原候选 vno1 的 [:vnum] 被
    查询；保留有向、不对称、重复邻接和未 rip 过滤语义。不从当前
    faces 重建邻接，不以 ORIGINAL/当前形状坐标代替 canonical 槽。
    其他持久 state 字段原样留给外层控制器，不在本函数修改或重置。

    context 默认 None；可传 prepare_ordered_genetic_defect_context 的
    原网格版本缓存。原 canonical/faces/labels/映射或源域顺序变更后
    必须重建缓存。当前 canonical/邻接每次重新消费，不跨缺陷缓存。
    use_overlap 默认 False，源标准 edge_table=0；True 复用完整原重叠
    计算/1000上限。当前 used1 优先于原 used2；剪枝仅针对本 ETable
    中 used1 行与后续候选，保持 source forward 顺序，不扩大检查域。

    返回成熟边表/ES/segment/overlap 全部结构，并增加 dynamic_base
    元数据及 seconds.dynamic_state_prepare/complete_next_api（含版本
    校验、当前状态读取和计算）。无全局线程/精度/GPU设置、文件或
    原版程序调用。缺字段、非法计数/被访问行、非有限球面、错误域
    或缓存版本抛 ValueError。输入不修改，GA/QC/整例仍由外层完成。
    对应固定 mrisTessellateDefect 内部 edgeExists/current canonical
    剪枝及旧边分段，没有独立原软件 CLI。
    """
    started=time.perf_counter()
    original_canonical=np.asarray(canonical_vertices)
    original_faces=np.asarray(faces);labels=np.asarray(defect_labels)
    translation=np.asarray(vertex_translation);status=np.asarray(retention_status)
    if context is None:
        context=prepare_ordered_genetic_defect_context(canonical_vertices=original_canonical,
            faces=original_faces,defect_labels=labels,ordered_regions=ordered_regions,
            vertex_translation=translation)
    elif (not isinstance(context,GeneticDefectInputContext)
            or context.signatures!=tuple(_signature(a) for a in
                (original_canonical,original_faces,labels,translation))
            or context.region_signature!=_region_signature(ordered_regions)):
        raise ValueError('same original surface version and ordered regions required for context')
    if (isinstance(defect_number,(bool,np.bool_))
            or not isinstance(defect_number,(int,np.integer))
            or not 0<=defect_number<len(context.regions)):
        raise ValueError('defect_number must select one complete ordered region')
    if (status.shape!=labels.shape or status.dtype.kind not in 'iu'
            or np.any((labels==0)&(status!=0))
            or np.any((labels>0)&(status!=1)&(status!=-1))):
        raise ValueError('retention_status requires outside0/KEEP+1/DISCARD-1')
    inside,border=context.regions[int(defect_number)]
    candidates=np.asarray([v for v in inside if status[v]==1]+list(border),np.int32)
    current,base,dynamic=_current_candidate_base(surface_state=current_surface_state,
        vertex_count=len(labels),candidates=candidates,vertex_translation=translation)
    # Reindex the current corrected sphere to original numbering, then reuse
    # the already validated scoring/prune/ES/overlap implementation unchanged.
    # source_order transforms it back for the core's corrected-domain queries.
    current_original=np.ascontiguousarray(current[translation])
    transient=replace(context,signatures=(_signature(current_original),*context.signatures[1:]),
        base_pairs=base)
    prepare_seconds=time.perf_counter()-started
    result=ordered_genetic_defect_inputs(canonical_vertices=current_original,
        original_vertices=original_vertices,original_normals=original_normals,
        faces=original_faces,defect_labels=labels,ordered_regions=ordered_regions,
        retention_status=status,vertex_translation=translation,white_values=white_values,
        gray_values=gray_values,volume=volume,ras_to_voxel=ras_to_voxel,
        defect_number=defect_number,use_overlap=use_overlap,context=transient)
    result['dynamic_base']=dynamic
    result['seconds']['dynamic_state_prepare']=prepare_seconds
    result['seconds']['complete_next_api']=time.perf_counter()-started
    result['static_context_seconds']=context.seconds
    return result
