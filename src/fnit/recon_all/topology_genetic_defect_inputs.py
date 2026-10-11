"""显式源有序缺陷区域→MRI候选边表、旧边分段及真实重叠列表。"""
from __future__ import annotations

import time
import hashlib
from dataclasses import dataclass
from collections.abc import Mapping
import numpy as np
from numba import njit

from .topology_preflight_python import (_retention_regions,
    _prune_against_base_edges, _edge_intersects)
from .topology_edge_mri_score import source_mri_edge_scores
from .topology_edge_segments import segment_intersecting_edges
from .smooth_surface_python import ordered_neighbors

SOURCE_MAX_OVERLAP_EDGES = 1000


def _signature(array):
    a=np.asarray(array)
    return (a.shape,a.dtype.str,hashlib.sha256(a.tobytes(order='C')).hexdigest())


def _region_signature(regions):
    return tuple((tuple(map(int,r['vertices'])),tuple(map(int,r['border'])))
        if isinstance(r,Mapping) else (tuple(map(int,r[0])),tuple(map(int,r[1]))) for r in regions)


@dataclass(frozen=True)
class GeneticDefectInputContext:
    """同一原表面版本的只读区域、邻接与边集合；不缓存候选可变几何。"""
    signatures: tuple
    region_signature: tuple
    regions: tuple
    neighbors: tuple
    original_pairs: frozenset
    base_pairs: frozenset
    source_order: np.ndarray
    seconds: float


def prepare_ordered_genetic_defect_context(*,canonical_vertices,faces,
        defect_labels,ordered_regions,vertex_translation) -> GeneticDefectInputContext:
    """原有序网格/区域/映射→一次性静态准备，不计算MRI分数或GA。

    五个必填参数的形状/空间同ordered_genetic_defect_inputs。返回只读
    GeneticDefectInputContext，绑定全部数组原bytes与显式region顺序；
    邻接、原边/base边和corrected→原排列每表面只建立一次。seconds
    包含校验/hash/建立，CPU，无全局状态改变。非法域/映射抛ValueError。
    不能跨网格版本或替换候选当前坐标；调用时校验版本，不读取参考。
    """
    started=time.perf_counter()
    if ordered_regions is None:
        raise ValueError('complete explicit ordered regions required; merged order must not be guessed')
    canonical=np.asarray(canonical_vertices);triangles=np.asarray(faces)
    labels=np.asarray(defect_labels);translation=np.asarray(vertex_translation)
    if (canonical.ndim!=2 or canonical.shape[1:]!=(3,) or not len(canonical)
            or canonical.dtype!=np.float32 or not np.isfinite(canonical).all()):
        raise ValueError('finite original-domain float32(N,3) canonical vertices required')
    n=len(canonical)
    if (triangles.ndim!=2 or triangles.shape[1:]!=(3,) or triangles.dtype.kind not in 'iu'
            or labels.shape!=(n,) or labels.dtype.kind not in 'iu'
            or translation.shape!=(n,) or translation.dtype.kind not in 'iu'
            or (triangles.size and (triangles.min()<0 or triangles.max()>=n))):
        raise ValueError('valid original integer faces/labels/mapping required')
    regions=_retention_regions(labels,triangles,ordered_regions)
    source_order=np.asarray([*np.flatnonzero(labels==0),*[v for inside,_ in regions for v in inside]],np.int32)
    expected=np.empty(n,np.int32);expected[source_order]=np.arange(n,dtype=np.int32)
    if not np.array_equal(translation,expected):
        raise ValueError('independent ordered-region original-to-corrected mapping required')
    rows=ordered_neighbors(triangles,n)
    pairs=np.concatenate((triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]]))
    pairs.sort(axis=1)
    source_order.flags.writeable=False
    return GeneticDefectInputContext(
        signatures=tuple(_signature(a) for a in (canonical,triangles,labels,translation)),
        region_signature=_region_signature(ordered_regions),
        regions=tuple((tuple(a),tuple(b)) for a,b in regions),
        neighbors=tuple(tuple(row) for row in rows),original_pairs=frozenset(map(tuple,pairs)),
        base_pairs=frozenset(tuple(p) for p in pairs if labels[p[0]]==0 and labels[p[1]]==0),
        source_order=source_order,seconds=time.perf_counter()-started)


@njit(cache=True, fastmath=False)
def _source_overlap_rows(canonical, edges, limit):
    rows=[]
    complete=np.ones(len(edges),np.bool_)
    for i in range(len(edges)):
        saved=np.empty(min(limit,max(0,len(edges)-1)),np.int32)
        count=0
        for j in range(len(edges)):
            if j==i:
                continue
            if _edge_intersects(canonical,edges[i,0],edges[i,1],edges[j,0],edges[j,1]):
                if count<limit:
                    saved[count]=j
                count+=1
                # The source discovers the 1001st match before marking the
                # saved first 1000 entries as incomplete. Exact 1000 is complete.
                if count>limit:
                    complete[i]=False
                    break
        rows.append(saved[:min(count,limit)].copy())
    return rows,complete


def ordered_genetic_defect_inputs(*,canonical_vertices,original_vertices,
        original_normals,faces,defect_labels,ordered_regions,
        retention_status,vertex_translation,white_values,gray_values,
        volume,ras_to_voxel,defect_number,use_overlap=False,context=None) -> dict:
    """原有序区域和自产MRI准备字段→完整单缺陷GA边输入，不执行GA。

    canonical_vertices/original_vertices/original_normals均为float32(N,3)
    原顶点域；前两者为surface RAS/mm，分别中心化球面与两轮平均
    ORIGINAL。original_normals须由平均ORIGINAL自产，不是raw/base法向。
    faces为原有序整数(F,3)，defect_labels为完整0或1…D整数(N,)；
    ordered_regions为D个显式源合并顺序的vertices/border字典（或二元组），
    允许合并inside不连通，完整域与零标签边界集合必须匹配。
    retention_status为outside0/KEEP+1/DISCARD−1整数(N,)，仅inside消费；
    vertex_translation为完整原→corrected整数(N,)排列，必须与outside
    原编号顺序+regions inside concat一致，不能从参考after反推。
    white_values/gray_values为自产raw采样与两轮median的float32(N,)；
    volume为uint8(X,Y,Z)，ras_to_voxel为surface RAS→该MRI的(4,4)。
    defect_number为从0开始的源域编号；上述参数必填。use_overlap默认False，
    对应原标准GA的edge_table=0。True显式计算重叠；False不以空列表
    假称完整重叠计算。
    context默认None；可传prepare_ordered_genetic_defect_context的同版本
    只读缓存，共享原邻接/边集合。全部绑定数组及域顺序修改后拒绝复用。

    复用源交叉、base边删去、完整MRI评分、旧边分段。返回edge_vertices
    int32(E,2) corrected编号、edge_used int32(E,)（0新/1base/2原边）、
    edge_scores float32(E,)、ordering int32(E,)初始顺序，以及原枚举行号。
    old_edge_table_indices是inside-first ES行→完整ETable索引；segment_rows
    是源max_segments顺序的ETable行号列表，保留丢弃后的空槽；
    old_edge_segments为同ES顺序int32数组。真实overlaps是E个升序index
    数组，overlap_complete bool(E,)；超过源1000上限保留首1000且标False。
    另返回CSR、原/新域顺序、每步seconds。关闭overlap时相关项为None。

    不修改输入/全局线程/精度，CPU NumPy/Numba，无原版程序或参考读取。
    缺/重复域、错误排列、非法类型/形状/非有限或越界报ValueError。
    当前每个区域单独生成；未完成种群、metric/QC或recon-all整例验收。
    对应mrisTessellateDefect内部EDGE表/segmentIntersectingEdges与
    mrisComputeOptimalRetessellation的重叠准备，无独立原软件CLI。
    """
    start=time.perf_counter()
    canonical=np.asarray(canonical_vertices);original=np.asarray(original_vertices)
    normals=np.asarray(original_normals);triangles=np.asarray(faces)
    labels=np.asarray(defect_labels);status=np.asarray(retention_status)
    translation=np.asarray(vertex_translation)
    white=np.asarray(white_values);gray=np.asarray(gray_values)
    if (canonical.ndim!=2 or canonical.shape[1:]!=(3,) or not len(canonical)
            or any(a.shape!=canonical.shape or a.dtype!=np.float32 or not np.isfinite(a).all()
                   for a in (canonical,original,normals))):
        raise ValueError('same original-domain finite float32(N,3) geometry/normals required')
    n=len(canonical)
    if any(a.shape!=(n,) or a.dtype!=np.float32 or not np.isfinite(a).all()
           for a in (white,gray)):
        raise ValueError('same original-domain finite float32(N,) white/gray values required')
    if (triangles.ndim!=2 or triangles.shape[1:]!=(3,) or triangles.dtype.kind not in 'iu'
            or labels.shape!=(n,) or labels.dtype.kind not in 'iu'
            or status.shape!=(n,) or status.dtype.kind not in 'iu'
            or translation.shape!=(n,) or translation.dtype.kind not in 'iu'
            or not isinstance(use_overlap,(bool,np.bool_))):
        raise ValueError('explicit original integer topology/status/mapping and bool overlap policy required')
    if context is None:
        context=prepare_ordered_genetic_defect_context(canonical_vertices=canonical,
            faces=triangles,defect_labels=labels,ordered_regions=ordered_regions,
            vertex_translation=translation)
    elif (not isinstance(context,GeneticDefectInputContext)
            or context.signatures!=tuple(_signature(a) for a in (canonical,triangles,labels,translation))
            or context.region_signature!=_region_signature(ordered_regions)):
        raise ValueError('same original surface version and ordered regions required for context')
    regions=context.regions
    if (isinstance(defect_number,(bool,np.bool_)) or not isinstance(defect_number,(int,np.integer))
            or not 0<=defect_number<len(regions)):
        raise ValueError('defect_number must select one complete ordered region')
    if np.any((labels==0)&(status!=0)) or np.any((labels>0)&(status!=1)&(status!=-1)):
        raise ValueError('retention_status requires outside0/KEEP+1/DISCARD-1')
    source_order=context.source_order
    inside,border=regions[int(defect_number)]
    neighbor_rows=context.neighbors
    occupied=set(inside)|set(border);hull=list(border)
    for vertex in border:
        for other in neighbor_rows[vertex]:
            if other not in occupied:
                occupied.add(other);hull.append(other)
    candidates=np.asarray([v for v in inside if status[v]==1]+list(border),np.int32)
    a,b=np.triu_indices(len(candidates),1)
    raw_edges=np.ascontiguousarray(np.column_stack((candidates[a],candidates[b])),np.int32)
    undirected=np.sort(raw_edges,axis=1)
    in_original=np.fromiter((tuple(p) in context.original_pairs for p in undirected),bool,count=len(raw_edges))
    in_base=np.fromiter((tuple(p) in context.base_pairs for p in undirected),bool,count=len(raw_edges))
    used=np.where(in_base,1,np.where(in_original,2,0)).astype(np.int32)
    tick=time.perf_counter()
    scores=source_mri_edge_scores(original_vertices=original,original_normals=normals,
        edge_vertices=raw_edges,white_values=white,gray_values=gray,
        original_edge_present=in_original,volume=volume,ras_to_voxel=ras_to_voxel)
    times=dict(mri_scoring=time.perf_counter()-tick)
    tick=time.perf_counter()
    keep=_prune_against_base_edges(np.ascontiguousarray(canonical),raw_edges,in_base)
    kept=np.flatnonzero(keep)
    translated=translation[raw_edges[keep]].astype(np.int32)
    scores=scores[keep];used=used[keep]
    # Preserve the existing tested FNIT ordering: EDGE.len, vno1, enumeration
    # index. Fixed upstream comparator has no valid equal-key return branch.
    permutation=np.lexsort((kept,translated[:,0],scores))
    translated=translated[permutation];scores=scores[permutation];used=used[permutation]
    kept=kept[permutation]
    times['prune_and_sort']=time.perf_counter()-tick
    tick=time.perf_counter()
    old=np.flatnonzero(used==2).astype(np.int32)
    both=np.isin(translated[old,0],translation[np.asarray(inside,np.int32)])&np.isin(
        translated[old,1],translation[np.asarray(inside,np.int32)])
    old_indices=np.concatenate((old[both],old[~both])).astype(np.int32)
    if len(old):
        es,groups,assigned=segment_intersecting_edges(canonical[source_order],translated[old],
            translation[np.asarray(inside,np.int32)])
        if not np.array_equal(es,translated[old_indices]):
            raise RuntimeError('mature inside-first ES order changed')
    else:
        es=np.empty((0,2),np.int32);groups=[];assigned=np.empty(0,np.int32)
    segments=[old_indices[np.asarray(row,np.int32)].copy() for row in groups]
    times['old_edge_segmentation']=time.perf_counter()-tick
    tick=time.perf_counter()
    overlaps=complete=offsets=values=None
    if use_overlap:
        rows,complete=_source_overlap_rows(np.ascontiguousarray(canonical[source_order]),
            np.ascontiguousarray(translated),SOURCE_MAX_OVERLAP_EDGES)
        overlaps=[np.asarray(row,np.int32) for row in rows]
        offsets=np.r_[0,np.cumsum([len(row) for row in overlaps],dtype=np.int64)]
        values=np.concatenate(overlaps).astype(np.int32) if overlaps else np.empty(0,np.int32)
    times['edge_table_overlap']=time.perf_counter()-tick
    times['complete_api']=time.perf_counter()-start
    return dict(edge_vertices=translated,edge_used=used,edge_scores=scores,
        ordering=np.arange(len(translated),dtype=np.int32),source_edge_indices=kept.astype(np.int32),
        unpruned_edges=len(raw_edges),candidate_source_vertices=candidates,
        original_inside_vertices=np.asarray(inside,np.int32),original_border_vertices=np.asarray(border,np.int32),
        inside_vertices=translation[np.asarray(inside,np.int32)].astype(np.int32),
        inside_discarded=(status[np.asarray(inside,np.int32)]==-1),
        inside_status_source=(status[np.asarray(inside,np.int32)]==-1).astype(np.int32),
        border_vertices=translation[np.asarray(border,np.int32)].astype(np.int32),
        hull_vertices=translation[np.asarray(hull,np.int32)].astype(np.int32),
        original_hull_vertices=np.asarray(hull,np.int32),
        old_edge_vertices=es,old_edge_table_indices=old_indices,old_edge_segments=assigned,
        segment_rows=segments,overlaps=overlaps,overlap_complete=complete,
        overlap_offsets=offsets,overlap_values=values,source_order=source_order,
        use_overlap=bool(use_overlap),max_overlap_edges=SOURCE_MAX_OVERLAP_EDGES,
        seconds=times,complete_ga=False,final_mesh_quality='not_assessed')


def all_ordered_genetic_defect_inputs(*,canonical_vertices,original_vertices,
        original_normals,faces,defect_labels,ordered_regions,retention_status,
        vertex_translation,white_values,gray_values,volume,ras_to_voxel,
        use_overlap=False) -> dict:
    """同一原表面全部显式有序区域→ETable列表，静态邻接/边集只准备一次。

    输入同单区域函数，去掉defect_number；use_overlap默认False。返回
    {defects,context,context_seconds,complete_api_seconds,complete_ga=False}，
    defects按源区域顺序。所有MRI准备必须由上游一次完成；本函数不
    重读MRI，不执行搜索或将初始base冒充前一缺陷修复后的当前状态。
    版本guard/失败语义与单区域相同，不修改输入、线程或GPU策略。
    """
    started=time.perf_counter()
    context=prepare_ordered_genetic_defect_context(canonical_vertices=canonical_vertices,
        faces=faces,defect_labels=defect_labels,ordered_regions=ordered_regions,
        vertex_translation=vertex_translation)
    values=[ordered_genetic_defect_inputs(canonical_vertices=canonical_vertices,
        original_vertices=original_vertices,original_normals=original_normals,
        faces=faces,defect_labels=defect_labels,ordered_regions=ordered_regions,
        retention_status=retention_status,vertex_translation=vertex_translation,
        white_values=white_values,gray_values=gray_values,volume=volume,
        ras_to_voxel=ras_to_voxel,defect_number=i,use_overlap=use_overlap,context=context)
        for i in range(len(context.regions))]
    return dict(defects=values,context=context,context_seconds=context.seconds,
        complete_api_seconds=time.perf_counter()-started,complete_ga=False,
        final_mesh_quality='not_assessed')
