"""只读表面放置质量审计：完整几何候选、严格proper交线与网格结构。"""
from __future__ import annotations
import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from .white_pial_crossing_python import _strict_relations, _rational_triangle, _exact_relation


def _possible_open_crossing(first, second):
    """只剔除已证明不可能开平面两侧的面对，不按近接触长度豁免。

    完全相同角点的行列式精确为0。其余角点用既有64*epsilon的
    permanent误差界证明符号；不确定符号同时保留两种可能。proper
    需要两个不同角点分别为正、负，双方均须满足。大量合法共享边/点
    因此无需Fraction裁剪；所有可能proper仍交既有完整精确谓词。
    """
    shared=np.all(first[:,:,None,:]==second[:,None,:,:],axis=3)
    def possible(points, plane, known):
        vector=points-plane[:,None,0,:]
        edge1=plane[:,1]-plane[:,0];edge2=plane[:,2]-plane[:,0]
        normal=np.cross(edge1,edge2)
        distance=np.sum(vector*normal[:,None,:],axis=2)
        permanent=np.zeros(distance.shape,np.float64)
        for axis in range(3):
            j,k=(axis+1)%3,(axis+2)%3
            permanent+=np.abs(vector[:,:,axis])*(np.abs(edge1[:,j]*edge2[:,k])+np.abs(edge1[:,k]*edge2[:,j]))[:,None]
        error=64*np.finfo(np.float64).eps*(permanent+1.)
        positive=(distance>=-error)&~known;negative=(distance<=error)&~known
        result=np.zeros(len(points),bool)
        for i in range(3):
            for j in range(3):
                if i!=j:result|=positive[:,i]&negative[:,j]
        return result
    return possible(first,second,shared.any(axis=2))&possible(second,first,shared.any(axis=1))


def mesh_structure(*, vertices, faces):
    """有限surface RAS/mm坐标(N,3)、整数有序面(F,3)→结构/面积/体积。

    返回顶点/面/边数量、Euler、面支撑连接组件、边界/非流形边、退化面、
    总面积mm²和有向闭网格体积mm³。开放/非流形网格体积仅为有向积分，
    volume_semantics明确标注；不拿它当脑区/no-th3体积。输入不被修改。
    """
    xyz=np.asarray(vertices,np.float64);tri=np.asarray(faces)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or not np.isfinite(xyz).all():raise ValueError('finite (N,3) coordinates required')
    if tri.ndim!=2 or tri.shape[1]!=3 or not np.issubdtype(tri.dtype,np.integer):raise ValueError('ordered integer (F,3) faces required')
    if tri.size and (tri.min()<0 or tri.max()>=len(xyz)):raise ValueError('face outside vertex domain')
    edges=np.sort(np.concatenate((tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]])),axis=1)
    unique,counts=np.unique(edges,axis=0,return_counts=True)
    graph=coo_matrix((np.ones(2*len(unique),np.int8),(np.r_[unique[:,0],unique[:,1]],np.r_[unique[:,1],unique[:,0]])),shape=(len(xyz),len(xyz))).tocsr()
    components=int(connected_components(graph,directed=False,return_labels=False))
    points=xyz[tri];area=np.linalg.norm(np.cross(points[:,1]-points[:,0],points[:,2]-points[:,0]),axis=1)/2
    signed_volume=float(np.einsum('ij,ij->i',points[:,0],np.cross(points[:,1],points[:,2])).sum()/6)
    closed=bool(np.all(counts==2))
    return dict(vertices=len(xyz),faces=len(tri),edges=len(unique),euler=len(xyz)-len(unique)+len(tri),connected_components=components,
        boundary_edges=int(np.count_nonzero(counts==1)),nonmanifold_edges=int(np.count_nonzero(counts>2)),
        isolated_vertices=len(xyz)-len(np.unique(tri)),degenerate_faces=int(np.count_nonzero(area==0)),
        total_area_mm2=float(area.sum()),minimum_face_area_mm2=float(area.min()) if len(area) else None,
        signed_volume_mm3=signed_volume,volume_semantics='closed_oriented_mesh_integral' if closed else 'open_or_nonmanifold_integral_not_anatomical_volume')


def placement_structure_passed(structure):
    """单半球闭表面结构门：零退化/边界/非流形面边、单连接组件。

    structure为mesh_structure返回统计；缺少任一项不能声称通过。
    本门独立于source marker、严格self/cross与固定坐标门。
    """
    return all(structure.get(key) == expected for key, expected in {
        'degenerate_faces': 0, 'boundary_edges': 0,
        'nonmanifold_edges': 0, 'connected_components': 1}.items())


def proper_pairs(*, first_vertices, faces, second_vertices=None, chunk_faces: int = 256,
                 exclude_shared_vertices: bool = False):
    """枚举全部严格proper面对；同面网格self或有序white/pial跨表面。

    first_vertices/second_vertices为有限浮点(N,3)surface RAS/mm；faces为
    同域整数有序(F,3)。second_vertices默认None表示自相交；self默认
    仅排除同一面，真实proper即使共享角点仍计入。显式
    exclude_shared_vertices=True单列原MHT式非邻接计数，不充当完整质量。
    跨表面不排除同索引/共享坐标。chunk_faces默认256仅
    控制峰值候选内存，没有候选上限或距离接受阈值。cKDTree包围球和
    AABB保守筛选，F64误差界内使用既有IEEE Fraction精确谓词。
    返回面对集合与数值报告；proper要求双方开平面两侧且交线长度>0，
    点/边/共面接触不冒充proper。退化面对单列，不能因此宣称质量通过。
    不改坐标/标签/rip，不调用原版程序，不参与优化接受。
    """
    if type(chunk_faces) is not int or chunk_faces<1:raise ValueError('positive integer chunk_faces required')
    if type(exclude_shared_vertices) is not bool:raise ValueError('boolean exclude_shared_vertices required')
    if second_vertices is not None and exclude_shared_vertices:raise ValueError('cross-surface queries cannot exempt shared vertices')
    tick=time.perf_counter();first=np.asarray(first_vertices);tri=np.asarray(faces)
    mesh_structure(vertices=first,faces=tri)
    self_query=second_vertices is None
    second=first if self_query else np.asarray(second_vertices)
    if second.shape!=first.shape or not np.isfinite(second).all():raise ValueError('same ordered finite vertex domain required')
    a=first.astype(np.float64)[tri];b=second.astype(np.float64)[tri]
    centers_a=a.mean(axis=1);centers_b=b.mean(axis=1)
    radii_a=np.nextafter(np.linalg.norm(a-centers_a[:,None],axis=2).max(axis=1),np.inf)
    radii_b=np.nextafter(np.linalg.norm(b-centers_b[:,None],axis=2).max(axis=1),np.inf)
    tree=cKDTree(centers_b);maximum=float(radii_b.max()) if len(b) else 0.
    low_a,high_a=a.min(axis=1),a.max(axis=1);low_b,high_b=b.min(axis=1),b.max(axis=1)
    degenerate_a=np.linalg.norm(np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0]),axis=1)==0
    degenerate_b=np.linalg.norm(np.cross(b[:,1]-b[:,0],b[:,2]-b[:,0]),axis=1)==0
    pairs=set();broad=tested=exact_count=degenerate_pairs=certified_rejections=0
    rational_a={};rational_b={}
    for start in range(0,len(a),chunk_faces):
        ids=np.arange(start,min(start+chunk_faces,len(a)))
        rows=tree.query_ball_point(centers_a[ids],np.nextafter(radii_a[ids]+maximum+1e-8,np.inf))
        left=np.repeat(ids,[len(row) for row in rows]);right=np.concatenate(rows).astype(np.int64) if any(len(row) for row in rows) else np.empty(0,np.int64)
        broad+=len(left)
        if not len(left):continue
        keep=np.all(low_a[left]<=high_b[right]+1e-8,axis=1)&np.all(low_b[right]<=high_a[left]+1e-8,axis=1)
        if self_query:
            keep &= left<right
            if exclude_shared_vertices:keep &= np.all(tri[left,:,None]!=tri[right,None,:],axis=(1,2))
        left,right=left[keep],right[keep]
        invalid=degenerate_a[left]|degenerate_b[right];degenerate_pairs+=int(invalid.sum());left,right=left[~invalid],right[~invalid]
        if not len(left):continue
        possible=_possible_open_crossing(a[left],b[right]);certified_rejections+=int(np.count_nonzero(~possible))
        left,right=left[possible],right[possible]
        if not len(left):continue
        relation,overlap,_,_,ambiguous,_=_strict_relations(a[left],b[right]);tested+=len(left)
        for slot in np.flatnonzero(ambiguous):
            fa,fb=int(left[slot]),int(right[slot]);exact_count+=1
            if fa not in rational_a:rational_a[fa]=_rational_triangle(a[fa])
            if fb not in rational_b:rational_b[fb]=_rational_triangle(b[fb])
            relation[slot],overlap[slot],_,_=_exact_relation(rational_a[fa],rational_b[fb])
        pairs.update((int(left[slot]),int(right[slot])) for slot in np.flatnonzero(relation==1))
    shared_proper=sum(bool(np.any(tri[first,:,None]==tri[second,None,:])) for first,second in pairs) if self_query else None
    return pairs,dict(proper_pairs=len(pairs),self_query=self_query,broad_phase_pairs=broad,tested_pairs=tested,
        exclude_shared_vertices=exclude_shared_vertices,shared_vertex_proper_pairs=shared_proper,
        nonshared_vertex_proper_pairs=len(pairs)-shared_proper if self_query else None,
        exact_fallback_pairs=exact_count,degenerate_candidate_pairs=degenerate_pairs,
        certified_open_plane_rejections=certified_rejections,
        enumeration_seconds=time.perf_counter()-tick,chunk_faces=chunk_faces,
        proper_minimum_intersection_length_mm=0.,broad_phase_roundoff_padding_mm=1e-8,
        predicate='existing_F64_with_IEEE_fraction_fallback',candidate_truncation=False)
