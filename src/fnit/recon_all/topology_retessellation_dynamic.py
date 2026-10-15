"""有序缺陷边动态重连；复用原有球面相交谓词，不运行原软件程序。

覆盖 retessellateDefect_wkr 的边/标记更新和重试控制。调用方须自产完整
基底、有序边表、overlap表及缺陷映射；本函数不创建MRI fitness/GA种群。
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import time
import numpy as np
from .topology_preflight_python import _edge_intersects
from .topology_retessellation_python import _contains_neighbor_on_sphere, detect_patch_faces

OUTSIDE, INSIDE, EDGE, TRIANGLE, DISCARDED = 0, 1, 2, 3, 4
NOT_USED, FIXED, ORIGINAL, NEW, BOTH, TEMP_NEW, TEMP_BOTH = range(7)


def _validated_base(canonical, base_faces, neighbors):
    """普通接口与一次准备接口共用同一全网格校验，保持源行顺序。"""
    xyz=np.asarray(canonical,np.float32);faces=np.asarray(base_faces)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or not np.isfinite(xyz).all() or np.any(np.all(xyz==0,axis=1)):
        raise ValueError('finite nonzero canonical (N,3) required')
    n=len(xyz)
    if faces.ndim!=2 or faces.shape[1]!=3 or (faces.size and
            (not np.issubdtype(faces.dtype,np.integer) or faces.min()<0 or faces.max()>=n)):
        raise ValueError('valid integer base faces required')
    if len(neighbors)!=n:raise ValueError('one ordered neighbor row per vertex required')
    rows=[]
    for v,row in enumerate(neighbors):
        a=np.asarray(row)
        if a.size and (not np.issubdtype(a.dtype,np.integer) or a.min()<0 or a.max()>=n):
            raise ValueError('neighbor invalid vertex indices')
        a=a.astype(np.int32,copy=True)
        if a.ndim!=1 or len(np.unique(a))!=len(a) or v in a:raise ValueError('distinct one-ring neighbors required')
        rows.append(a.tolist())
    links=[set(row) for row in rows]
    for v,row in enumerate(rows):
        if any(v not in links[w] for w in row):raise ValueError('source base adjacency must be symmetric')
    return xyz,faces.astype(np.int32,copy=True),rows,links


@dataclass(frozen=True)
class PreparedRetessellationBase:
    """一次校验的不可变基底；必须由prepare_retessellation_base创建。"""
    canonical: np.ndarray
    base_faces: np.ndarray
    neighbors: tuple
    linked_neighbors: tuple
    face_rows: tuple
    incident_faces: tuple
    incident_corners: tuple
    face_identities: frozenset
    state_sha256: str
    preparation_seconds: float
    _token: object


_PREPARED_TOKEN=object()


def _check_prepared_base(base):
    """值缓冲只读，另检查调用者可改变的 ndarray 元数据。"""
    if (not isinstance(base,PreparedRetessellationBase) or base._token is not _PREPARED_TOKEN
            or base.canonical.dtype!=np.float32 or base.base_faces.dtype!=np.int32
            or base.canonical.shape!=(len(base.neighbors),3)
            or base.base_faces.shape!=(len(base.face_rows),3)
            or base.canonical.flags.writeable or base.base_faces.flags.writeable
            or not base.canonical.flags.c_contiguous or not base.base_faces.flags.c_contiguous):
        raise ValueError('unchanged immutable prepared base metadata required')


def prepare_retessellation_base(*, canonical, base_faces, neighbors):
    """同序canonical/基底面/源邻接→一次校验的不可变可复用基底。

    canonical为有限非零float32(N,3)球面坐标；base_faces为整数(F,3)，
    neighbors为完整N行源有序一环（可包含未由面支撑的外部边）。全
    整数/范围/重复/对称校验与普通重连共用，不从面重建邻接。复制
    坐标/面为bytes-backed只读数组和tuple/frozenset的行、入射与面集；
    调用者改变原输入不影响缓存，不能重新打开writeable。返回本类，
    含全部初态SHA及含校验/复制/索引的preparation_seconds。

    后续重连显式传context.canonical/base_faces/neighbors和base_context；
    必须为这三个对象本身，混入另一状态直接ValueError，不偷偷复用
    失效缓存。仅固定基底可跨candidate复用；边used/ordering、status、
    rip和标记仍每候选处理。没有默认坐标变换/阈值，不改线程/精度。
    非法输入抛ValueError，输入不变；不执行GA/DVS或生产最终QC。
    """
    tick=time.perf_counter();xyz,faces,rows,links=_validated_base(canonical,base_faces,neighbors)
    xyz=np.frombuffer(np.ascontiguousarray(xyz).tobytes(),np.float32).reshape(xyz.shape)
    faces=np.frombuffer(faces.tobytes(),np.int32).reshape(faces.shape)
    face_rows=tuple(tuple(map(int,face)) for face in faces)
    incident=[[] for _ in range(len(xyz))];corners=[[] for _ in range(len(xyz))]
    for face_id,face in enumerate(face_rows):
        for corner,v in enumerate(face):incident[v].append(face_id);corners[v].append(corner)
    rows=tuple(tuple(row) for row in rows)
    offsets=np.zeros(len(xyz)+1,np.int64);offsets[1:]=np.cumsum([len(row) for row in rows])
    flat=np.fromiter((v for row in rows for v in row),dtype=np.int32,count=int(offsets[-1]))
    digest=hashlib.sha256(xyz.tobytes()+faces.tobytes()+offsets.tobytes()+flat.tobytes()).hexdigest()
    return PreparedRetessellationBase(xyz,faces,rows,tuple(frozenset(row) for row in links),
        face_rows,tuple(tuple(row) for row in incident),tuple(tuple(row) for row in corners),
        frozenset(tuple(sorted(face)) for face in face_rows),
        digest,time.perf_counter()-tick,_PREPARED_TOKEN)


def retessellate_ordered_edges(*, canonical, base_faces, neighbors, candidate_edges,
        edge_used, ordering, inside_vertices, inside_discarded, border_vertices,
        hull_vertices, vertex_rip=None, fixed_mapping=None, mode='all',
        overlaps=None, overlap_complete=None, base_context=None,
        blocker_scan_backend='python', blocker_scan_audit=False):
    """对一次真实缺陷候选运行有序动态边控制，返回新状态和完整决策。

    canonical 是 float32 (N,3) 非零有限球面点，单位为原 canonical RAS；
    base_faces 为保留 (F,3) int32 面；neighbors 为源一环的有序 N 行，
    不从面重新构造。candidate_edges 是 MRI 排序后的 (K,2) int32 原表；
    edge_used 是原 0…6 状态；ordering 是完整 K 个源边编号排列。inside/
    border/hull 都采用当前 corrected 网格索引；inside_discarded 是源
    status==DISCARD 的 bool(Kinside)，保留其顺序。vertex_rip 默认为全
    false；fixed_mapping 默认为全true，false 表示实验 optimal-map 无效。
    mode='all' 保留全部可加入顶点；'some' 执行球面面包含、顺序删邻接及
    被移除阻挡边的重试。overlaps 可提供每条边的有序预计算候选行；其
    complete=true 时仅需再检查 hull[nborder:]；每行保留源顺序和重复项，
    不做去重。缺省无表时检查全部
    inside+hull 边，完整求解仍遵循顺序。表必须由调用方自产。
    base_context默认None走原完整校验；显式prepared基底只copy被修改行，
    固定基底入射/面索引复用。必须同时传缓存本身三个只读对象，其他
    状态报错；不缓存候选坐标、动态标记/边或改变接受/拒绝顺序。
    blocker_scan_backend默认python；显式numba将同一完整首阻挡扫描
    和已有谓词放在编译循环内。只复制实际query邻接行，每次源加边/
    删边同步对应行，返回的全N邻接copy仍保留；不缩减源扫描范围。
    prepared不可变canonical内复用原F64边端点/叉积，改变行立即重新准备；
    不跨候选保存scanner，无prepared基底时仍逐查询读取当前canonical。
    blocker_scan_audit默认False；显式True逐条记录实际谓词调用的
    (a,b,v,w,hit)有序little-endian int32 SHA-256与总数，只用于验证。
    该审计含临时缓冲/哈希开销，不能当未观测吞吐；不改变判定。
    返回 dict 的 faces/neighbor rows/edge_used/marks/selected_vertices/
    selected_edges 是独立状态；临时面只用于 some 的分类，最终恢复原
    base_faces 后由 detect_patch_faces 发现最终面。retessellation_trace
    记录各轮每条实际测试边的接受、阻挡、标记；无MRI/GA/finalQC声明。
    非法参数或非对称基底邻接报 ValueError；输入不修改，不写标准表面。
    """
    if blocker_scan_backend not in ('python','numba'):
        raise ValueError('blocker_scan_backend must be python or numba')
    if not isinstance(blocker_scan_audit,(bool,np.bool_)):
        raise ValueError('blocker_scan_audit must be bool or np.bool_')
    if base_context is None:
        xyz,faces,rows,links=_validated_base(canonical,base_faces,neighbors)
    else:
        _check_prepared_base(base_context)
        if (canonical is not base_context.canonical
                or base_faces is not base_context.base_faces
                or neighbors is not base_context.neighbors):
            raise ValueError('prepared base requires its own immutable canonical/faces/neighbor objects')
        xyz,faces=base_context.canonical,base_context.base_faces
        rows,links=list(base_context.neighbors),list(base_context.linked_neighbors)
    edges=np.asarray(candidate_edges);used=np.asarray(edge_used)
    n=len(xyz);k=len(edges)
    def ids(a,name):
        a=np.asarray(a)
        if a.size==0:return a.astype(np.int32)
        if not np.issubdtype(a.dtype,np.integer) or (a.size and (a.min()<0 or a.max()>=n)):raise ValueError(name+' invalid vertex indices')
        return a.astype(np.int32,copy=True)
    if edges.ndim!=2 or edges.shape[1]!=2:raise ValueError('candidate edges (K,2) required')
    edges=ids(edges,'candidate edges')
    if np.any(edges[:,0]==edges[:,1]):raise ValueError('self candidate edge')
    if used.shape!=(k,) or not np.issubdtype(used.dtype,np.integer) or np.any(used<0) or np.any(used>6):raise ValueError('source edge states 0…6 required')
    used=used.astype(np.int32,copy=True)
    order=np.asarray(ordering)
    if order.shape!=(k,) or not np.issubdtype(order.dtype,np.integer) or not np.array_equal(np.sort(order),np.arange(k)):raise ValueError('complete edge ordering permutation required')
    inside=ids(inside_vertices,'inside');border=ids(border_vertices,'border');hull=ids(hull_vertices,'hull')
    if any(a.ndim!=1 or len(np.unique(a))!=len(a) for a in [inside,border,hull]):raise ValueError('distinct ordered one-dimensional domain lists required')
    discarded=np.asarray(inside_discarded)
    if discarded.dtype!=np.bool_ or discarded.shape!=inside.shape:raise ValueError('bool source inside_discarded required')
    if len(border)>len(hull) or not np.array_equal(hull[:len(border)],border):raise ValueError('source hull must begin with full ordered border')
    if set(inside)&set(border):raise ValueError('inside and border domains overlap')
    if mode not in ('all','some'):raise ValueError('mode must be all or some')
    rip=np.zeros(n,np.bool_) if vertex_rip is None else np.asarray(vertex_rip)
    valid=np.ones(n,np.bool_) if fixed_mapping is None else np.asarray(fixed_mapping)
    if rip.dtype!=np.bool_ or valid.dtype!=np.bool_ or rip.shape!=(n,) or valid.shape!=(n,):raise ValueError('bool rip/fixed-mapping arrays required')
    if overlaps is not None:
        if len(overlaps)!=k:raise ValueError('one overlap list per edge required')
        normalized_overlaps=[]
        for row in overlaps:
            a=np.asarray(row)
            if a.ndim!=1 or (a.size and (not np.issubdtype(a.dtype,np.integer) or a.min()<0 or a.max()>=k)):raise ValueError('ordered overlap edge ID row required')
            normalized_overlaps.append(a.astype(np.int32,copy=True))
        overlaps=normalized_overlaps
        complete=np.asarray(overlap_complete)
        if complete.dtype!=np.bool_ or complete.shape!=(k,):raise ValueError('bool overlap completeness required')
    elif overlap_complete is not None:raise ValueError('overlap completeness requires overlap rows')
    marks=np.zeros(n,np.int32);marks[inside[~discarded]]=INSIDE;marks[border]=TRIANGLE
    # Source temporary faces participate only in vertex classification; outside
    # face ordering and final neighbor updates are preserved independently.
    if base_context is None:
        tempfaces=[list(map(int,f)) for f in faces];incidents=[[] for _ in range(n)]
        for fi,face in enumerate(tempfaces):
            for v in face:incidents[v].append(fi)
        face_set={tuple(sorted(f)) for f in tempfaces}
    elif mode=='all':
        # All-vertices mode never calls contains/add_faces or reads temporary
        # incidence. It only appends ordered edges and marks their endpoints.
        tempfaces=[];incidents=None;face_set=None
    else:
        tempfaces=list(base_context.face_rows);incidents=list(base_context.incident_faces)
        face_set=set(base_context.face_identities)
    copied_rows=set()
    full_scan=np.concatenate((inside,hull)).astype(np.int32,copy=False)
    extra_scan=hull[len(border):]
    scanner=None
    predicate_count=0;scan_calls=0
    predicate_digest=hashlib.sha256() if blocker_scan_audit else None
    if blocker_scan_backend=='numba':
        from .topology_edge_blocker_scan import OrderedBlockerScan
        scanner=OrderedBlockerScan(scan_vertices=full_scan,neighbor_rows=rows,audit=blocker_scan_audit,
            canonical=xyz if base_context is not None else None)
    def writable_row(v):
        if isinstance(rows[v],tuple):
            rows[v]=list(rows[v]);links[v]=set(links[v]);copied_rows.add(v)
    def writable_incident(v):
        if isinstance(incidents[v],tuple):incidents[v]=list(incidents[v])
    def add_edge(a,b):
        writable_row(a);writable_row(b)
        if b not in links[a]:rows[a].append(b);links[a].add(b)
        if a not in links[b]:rows[b].append(a);links[b].add(a)
        if scanner is not None:
            scanner.sync_vertex(a,rows[a]);scanner.sync_vertex(b,rows[b])
    def contains(v,fi):
        tri=tempfaces[fi]
        if any(marks[w]==OUTSIDE for w in tri):return False
        return bool(_contains_neighbor_on_sphere(xyz,np.asarray(tri,np.int32),np.asarray([v],np.int32)))
    removed=[]
    def remove_vertex(v):
        if marks[v]==OUTSIDE:raise ValueError('source removeVertex cannot remove outside vertex')
        for w in list(rows[v]):
            writable_row(w)
            rows[w].remove(v);links[w].remove(v)
            if scanner is not None:scanner.sync_vertex(w,rows[w])
            if not rows[w]:marks[w]=INSIDE
        writable_row(v);rows[v]=[];links[v].clear();marks[v]=DISCARDED;removed.append(v)
        if scanner is not None:scanner.sync_vertex(v,rows[v])
    def update_vertex(v,fi):
        count=0;pos=0
        while pos<len(rows[v]):
            w=rows[v][pos]
            if marks[w]!=TRIANGLE and contains(w,fi):remove_vertex(w);count+=1
            else:pos+=1
        return count
    def add_faces(a,b):
        pos=0
        while pos<len(rows[a]):
            c=rows[a][pos]
            if c!=b and c in links[b] and marks[c]!=OUTSIDE and tuple(sorted((a,b,c))) not in face_set:
                fi=len(tempfaces);tempfaces.append([a,b,c]);face_set.add(tuple(sorted((a,b,c))))
                for v in (a,b,c):writable_incident(v);incidents[v].append(fi);marks[v]=TRIANGLE
                gone=update_vertex(a,fi);update_vertex(c,fi);update_vertex(b,fi);pos-=gone
            pos+=1
    def possibly_add(a,b):
        if mode=='all':add_edge(a,b);marks[a]=marks[b]=TRIANGLE;return True
        if marks[b]<marks[a]:a,b=b,a
        ma,mb=marks[a],marks[b]
        if ma==INSIDE:
            if mb==TRIANGLE and any(contains(a,f) for f in incidents[b]):marks[a]=DISCARDED;removed.append(a);return False
            add_edge(a,b);marks[a]=EDGE
            if mb!=TRIANGLE:marks[b]=EDGE
            return True
        if ma==EDGE:
            if mb!=EDGE and any(contains(a,f) for f in incidents[b]):remove_vertex(a);return False
            add_edge(a,b);add_faces(a,b);return True
        if ma==TRIANGLE:add_edge(a,b);add_faces(a,b);return True
        return False
    intersected=np.zeros(k,np.int32);blocker=np.full((k,2),-1,np.int32);trace=[];rounds=0
    while True:
        rounds+=1;modified=False
        if rounds>len(inside)+2:raise RuntimeError('retessellation retries exceeded removed inside domain')
        for ei in order:
            ei=int(ei);a,b=map(int,edges[ei])
            if intersected[ei] or (used[ei] and used[ei]!=ORIGINAL) or marks[a]==DISCARDED or marks[b]==DISCARDED or rip[a] or rip[b]:continue
            hit=None
            if overlaps is not None:
                for other in overlaps[ei]:
                    if used[other] and used[other]!=ORIGINAL:hit=tuple(map(int,edges[other]));break
            if hit is None:
                scan=extra_scan if overlaps is not None and complete[ei] else full_scan
                if scanner is not None:
                    hit=scanner.first_blocker(xyz=xyz,a=a,b=b,scan_vertices=scan,valid=valid)
                else:
                    scan_calls+=1;sequence=[] if blocker_scan_audit else None
                    for v in scan:
                        v=int(v)
                        if v in (a,b):continue
                        for w in rows[v]:
                            if w in (a,b) or not valid[w]:continue
                            intersects=_edge_intersects(xyz,a,b,v,w)
                            if blocker_scan_audit:
                                sequence.append((a,b,v,w,int(intersects)));predicate_count+=1
                            if intersects:hit=(v,w);break
                        if hit is not None:break
                    if blocker_scan_audit:
                        predicate_digest.update(np.asarray(sequence,dtype='<i4').reshape(-1,5).tobytes())
            if hit is None:
                accepted=possibly_add(a,b)
                if accepted:used[ei]=TEMP_BOTH if used[ei] else TEMP_NEW
                trace.append(dict(round=rounds,edge=ei,endpoints=[a,b],decision='accepted' if accepted else 'classification_discard',post_marks=[int(marks[a]),int(marks[b])]))
            else:
                intersected[ei]=2;blocker[ei]=hit
                trace.append(dict(round=rounds,edge=ei,endpoints=[a,b],decision='intersection',blocker=list(hit)))
        for ei in order:
            ei=int(ei);a,b=map(int,edges[ei])
            if used[ei] in (TEMP_NEW,TEMP_BOTH):
                was=used[ei];used[ei]=NOT_USED if was==TEMP_NEW else ORIGINAL
                # 固定源此处 continue；非triangle临时边不参与本轮blocker更新。
                if marks[a]!=TRIANGLE or marks[b]!=TRIANGLE:continue
                used[ei]=NEW if was==TEMP_NEW else BOTH
            if intersected[ei]==2 and (marks[blocker[ei,0]]==DISCARDED or marks[blocker[ei,1]]==DISCARDED):intersected[ei]=0;modified=True
            else:intersected[ei]=1
        if not modified:break
    selected_edges=np.asarray([int(i) for i in order if used[i] in (NEW,BOTH)],np.int32)
    for ei in selected_edges:marks[edges[ei]]=5  # USED_VERTEX, temporary collection mark
    selected_inside=[int(v) for v,drop in zip(inside,discarded) if not drop and marks[v]==5]
    for v,drop in zip(inside,discarded):
        if not drop:marks[v]=OUTSIDE
    for v in border:marks[v]=OUTSIDE
    # Public neighbor rows stay independent mutable lists, including untouched
    # outside rows. No returned row aliases immutable prepared storage.
    if base_context is not None:rows=[list(row) if isinstance(row,tuple) else row for row in rows]
    return dict(faces=faces.copy(),neighbors=rows,edge_used=used,marks=marks,
        selected_edges=selected_edges,selected_vertices=np.asarray(selected_inside+border.tolist(),np.int32),
        ninside=len(selected_inside),ndiscarded=int(np.count_nonzero(~discarded)-len(selected_inside)),
        removed_vertices=np.asarray(removed,np.int32),temporary_faces=np.asarray(tempfaces[len(faces):],np.int32).reshape(-1,3),
        retessellation_trace=trace,rounds=rounds,complete_ga=False,
        prepared_base_used=base_context is not None,prepared_copied_rows=len(copied_rows),
        blocker_scan_backend=blocker_scan_backend,
        blocker_scan_preparation_seconds=None if scanner is None else scanner.preparation_seconds,
        blocker_scan_calls=scan_calls if scanner is None else scanner.calls,
        blocker_predicate_calls=(predicate_count if blocker_scan_audit else None) if scanner is None else scanner.predicate_calls,
        blocker_predicate_sequence_sha256=(predicate_digest.hexdigest() if blocker_scan_audit else None)
            if scanner is None else (scanner.sequence_digest.hexdigest() if blocker_scan_audit else None))


def retessellate_and_detect_candidate(*, canonical, face_detection_context=None, **edge_inputs):
    """有序边候选→完整保留面和自产局部新面；不执行MRI匹配或GA。

    canonical 与 edge_inputs 使用 retessellate_ordered_edges 同一输入
    定义。先完成动态边控制，再复用既有源面发现/attachment/球面定向。
    返回相同状态字典；faces为全网格，patch_faces为新增有序面，
    selected_faces为原面表后追加的连续int32编号，face_decisions为
    完整自产判定记录。既有外部面与顶点映射保持，质量验收由调用方
    另执行；中间candidate的相交不等于最终producer通过。
    face_detection_context默认None使用成熟面发现；显式不可变保留面
    上下文复用面身份与边计数，但仍完整消费当前邻接及按序更新新增面。
    缓存不匹配完整保留面时报错，不跨动态面版本复用。
    """
    start=time.perf_counter()
    result=retessellate_ordered_edges(canonical=canonical,**edge_inputs)
    retessellation_seconds=time.perf_counter()-start;start=time.perf_counter()
    base=result['faces']
    if face_detection_context is None:
        patch,report=detect_patch_faces(canonical=canonical,
            vertices=result['selected_vertices'],neighbors=result['neighbors'],
            external_faces=base)
    else:
        from .topology_face_detection_cache import detect_patch_faces_cached
        patch,report=detect_patch_faces_cached(canonical=canonical,
            vertices=result['selected_vertices'],neighbors=result['neighbors'],
            external_faces=base,context=face_detection_context)
    result['faces']=np.concatenate((base,patch)).astype(np.int32,copy=False)
    result['patch_faces']=patch
    result['selected_faces']=np.arange(len(base),len(base)+len(patch),dtype=np.int32)
    result['face_decisions']=report
    result['retessellation_seconds']=retessellation_seconds
    result['face_detection_seconds']=time.perf_counter()-start
    return result
