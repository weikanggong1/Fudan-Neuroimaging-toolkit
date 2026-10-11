"""缺陷 unMRI 的完整有向距离前置：源 bbox、三角面/边/角点及归并。

独立 Python/Numba 实现固定 d932 的 active new_loop3，复用本项目
角度加权伪法向和原 incident 邻接。无参考读写、原版程序、GPU或近似EDT。
"""
from __future__ import annotations
import numpy as np
from numba import get_num_threads, njit, prange
from .topology_angle_pseudonormals import (
    angle_weighted_vertex_pseudonormals, other_edge_faces,
)

_UNSET = np.float32(1000)


@njit(cache=True, fastmath=False)
def _dot(a, b):
    return np.float32(np.float32(a[0]*b[0] + a[1]*b[1]) + a[2]*b[2])


@njit(cache=True, fastmath=False)
def _cross(a, b):
    return np.array([np.float32(a[1]*b[2]-b[1]*a[2]),
                     np.float32(a[2]*b[0]-b[2]*a[0]),
                     np.float32(a[0]*b[1]-b[0]*a[1])], np.float32)


@njit(cache=True, fastmath=False)
def _index(value, origin, scale):
    return int(np.float64(np.float32(scale*np.float32(value-origin))) + .5)


@njit(cache=True, fastmath=False)
def _prepare_buffers(xyz, faces, selected_faces, shape, origin, scale):
    bounds = np.empty((len(selected_faces), 7), np.int64)
    coordinates = np.empty((len(selected_faces), 9), np.float32)
    used = 0
    delta = int(np.float64(2)*np.float64(scale[0]))
    for face_id in selected_faces:
        ids = faces[face_id]
        lower = np.empty(3, np.int64); upper = np.empty(3, np.int64)
        for axis in range(3):
            lo = min(xyz[ids[0],axis],xyz[ids[1],axis],xyz[ids[2],axis])
            hi = max(xyz[ids[0],axis],xyz[ids[1],axis],xyz[ids[2],axis])
            lower[axis] = _index(lo,origin[axis],scale[axis])-delta
            upper[axis] = _index(hi,origin[axis],scale[axis])+delta
        if (lower[0] > shape[0]-1 or lower[1] > shape[1]-1 or lower[2] > shape[2]-1
                or upper[0] < 0 or upper[1] < 0 or upper[2] < 0):
            continue
        bounds[used,0]=face_id
        for axis in range(3):
            bounds[used,1+2*axis]=lower[axis];bounds[used,2+2*axis]=upper[axis]
            for corner in range(3):
                # ENTRY_MEMBERS grouping: x0/x1/x2 then y0/y1/y2 then z.
                coordinates[used,axis*3+corner]=xyz[ids[corner],axis]
        used+=1
    return bounds[:used].copy(),coordinates[:used].copy()


@njit(cache=True, fastmath=False)
def _merge(current, candidate):
    if abs(candidate) < abs(current):
        return candidate
    if abs(candidate) == abs(current) and candidate > np.float32(0):
        return candidate
    return current


@njit(cache=True, fastmath=False, error_model='numpy')
def _edge_candidate(vec_a, vec_b, edge, length_squared, vertex_a_normal,
                    vertex_b_normal, edge_normal):
    projection = _dot(vec_a,edge)
    if projection < 0:
        return _dot(vec_a,vec_a),_dot(vertex_a_normal,vec_a)
    if projection < length_squared:
        sq = np.float32(_dot(vec_a,vec_a)-np.float32(np.float32(projection*projection)/length_squared))
        return max(np.float32(0),sq),_dot(edge_normal,vec_a)
    return _dot(vec_b,vec_b),_dot(vertex_b_normal,vec_b)


@njit(cache=True, fastmath=False, error_model='numpy')
def _distance_fill(xyz,faces,normals,pseudo,bounds,other,shape,origin,scale,
                   distance,x_start,x_stop):
    # Each voxel consumes the same complete ordered face sequence. Only the
    # disjoint x write domain changes; face tests and float32 operations do not.
    for row in range(len(bounds)):
        if bounds[row,2]<x_start or bounds[row,1]>=x_stop:
            continue
        fno=bounds[row,0];ids=faces[fno]
        nf=normals[fno]
        vertex=xyz[ids]
        edge=np.empty((3,3),np.float32);edge_half=np.empty((3,3),np.float32)
        edge_pseudo=np.empty((3,3),np.float32);edge_sq=np.empty(3,np.float32)
        for index in range(3):
            for axis in range(3):
                edge[index,axis]=np.float32(vertex[(index+1)%3,axis]-vertex[index,axis])
                edge_pseudo[index,axis]=np.float32(nf[axis]+normals[other[row,index],axis])
            edge_half[index]=_cross(nf,edge[index]);edge_sq[index]=_dot(edge[index],edge[index])
        center=np.zeros(3,np.float32)
        for corner in range(3):
            for axis in range(3):center[axis]=np.float32(center[axis]+vertex[corner,axis])
        for axis in range(3):center[axis]=np.float32(center[axis]*np.float32(.33))
        far_sq=np.float32(0)
        for corner in range(3):
            diff=np.empty(3,np.float32)
            for axis in range(3):diff[axis]=np.float32(center[axis]-vertex[corner,axis])
            far_sq=max(far_sq,_dot(diff,diff))
        far=np.float32(np.float32(np.sqrt(far_sq))*np.float32(1.01))
        lo=np.empty(3,np.int64);hi=np.empty(3,np.int64)
        for axis in range(3):
            lo[axis]=max(0,bounds[row,1+2*axis]);hi[axis]=min(shape[axis]-1,bounds[row,2+2*axis])
        lo[0]=max(lo[0],x_start);hi[0]=min(hi[0],x_stop-1)
        vec=np.empty((3,3),np.float32);mean_vec=np.empty(3,np.float32)
        point=np.empty(3,np.float32)
        for i in range(lo[0],hi[0]+1):
            point[0]=np.float32(origin[0]+np.float32(np.float32(i)/scale[0]))
            for j in range(lo[1],hi[1]+1):
                point[1]=np.float32(origin[1]+np.float32(np.float32(j)/scale[1]))
                dx=np.float32(point[0]-center[0]);dy=np.float32(point[1]-center[1])
                partial=np.float32(np.float32(dx*dx)+np.float32(dy*dy))
                for k in range(lo[2],hi[2]+1):
                    point[2]=np.float32(origin[2]+np.float32(np.float32(k)/scale[2]))
                    dz=np.float32(point[2]-center[2])
                    center_sq=np.float32(partial+np.float32(dz*dz))
                    current=distance[i,j,k]
                    bound=np.float32(abs(current)+far)
                    if np.float32(center_sq*np.float32(.98)) > np.float32(bound*bound):continue
                    for corner in range(3):
                        for axis in range(3):vec[corner,axis]=np.float32(point[axis]-vertex[corner,axis])
                    for axis in range(3):
                        mean_vec[axis]=np.float32(np.float32(np.float32(vec[0,axis]+vec[1,axis])+vec[2,axis])*np.float32(.3333333))
                    val0,val1,val2=_dot(vec[0],edge_half[0]),_dot(vec[1],edge_half[1]),_dot(vec[2],edge_half[2])
                    if val0>=0 and val1>=0 and val2>=0:
                        distance[i,j,k]=_merge(current,_dot(nf,mean_vec))
                    else:
                        least=np.float32(_UNSET*_UNSET);sign=np.float32(0)
                        if val0<=0:
                            least,sign=_edge_candidate(vec[0],vec[1],edge[0],edge_sq[0],pseudo[ids[0]],pseudo[ids[1]],edge_pseudo[0])
                        if val1<=0:
                            sq,sg=_edge_candidate(vec[1],vec[2],edge[1],edge_sq[1],pseudo[ids[1]],pseudo[ids[2]],edge_pseudo[1])
                            if least>sq:least,sign=sq,sg
                        if val2<=0:
                            sq,sg=_edge_candidate(vec[2],vec[0],edge[2],edge_sq[2],pseudo[ids[2]],pseudo[ids[0]],edge_pseudo[2])
                            if least>sq:least,sign=sq,sg
                        if np.float32(np.float32(current*current)*np.float32(1.01))<least:continue
                        positive=np.float32(np.sqrt(least))
                        if positive>abs(current):continue
                        # Source SIGN(0)=-1.0; equal abs prefers strictly positive.
                        candidate=np.float32(positive if sign>0 else -positive)
                        distance[i,j,k]=_merge(current,candidate)


@njit(cache=True, fastmath=False, error_model='numpy')
def _distance(xyz,faces,normals,pseudo,bounds,other,shape,origin,scale):
    distance=np.full((shape[0],shape[1],shape[2]),_UNSET,np.float32)
    _distance_fill(xyz,faces,normals,pseudo,bounds,other,shape,origin,scale,
                   distance,0,shape[0])
    return distance


@njit(cache=True, fastmath=False, error_model='numpy', parallel=True)
def _distance_parallel_kernel(xyz,faces,normals,pseudo,bounds,other,shape,origin,scale,
                              slab_count):
    distance=np.full((shape[0],shape[1],shape[2]),_UNSET,np.float32)
    for slab in prange(slab_count):
        slab_index=np.int64(slab)
        x_start=slab_index*shape[0]//slab_count
        x_stop=(slab_index+1)*shape[0]//slab_count
        _distance_fill(xyz,faces,normals,pseudo,bounds,other,shape,origin,scale,
                       distance,x_start,x_stop)
    return distance


def _distance_parallel(xyz,faces,normals,pseudo,bounds,other,shape,origin,scale):
    # Keep the thread query outside the compiled function: calling it inside
    # introduces dynamic globals and prevents Numba's on-disk JIT cache.
    slab_count=min(get_num_threads(),int(shape[0]))
    return _distance_parallel_kernel(xyz,faces,normals,pseudo,bounds,other,shape,
                                      origin,scale,slab_count)


def defect_signed_distance(*,original_vertices,faces,face_normals,
        incident_offsets,incident_faces,incident_corners,volume_shape,
        volume_origin_mm,volume_index_scale_per_mm,ordered_candidate_faces=None,
        backend='serial')->dict:
    """完整当前候选几何→局部 XYZ float32/mm 有向最小距离和内部字段。

    original_vertices(N,3)为source orig surface RAS/mm；faces(F,3)与
    face_normals(F,3)为同几何源面法向。三个incident参数保留vt.f/n原顺序。
    volume_shape为3正整数XYZ；origin/scale分别3个float32起点mm及坐标
    到索引乘数mm^-1，后者不是MRIspacing的除数。候选面默认遍历全部面；
    可传同源空间索引的保守有序面列表，不得遗漏任何与扩展bbox相交面。
    本函数计算自己的bbox/角度与边伪法向、投影和距离，不消费参考距离。
    返回distance_mm(X,Y,Z)、buffer_bounds(B,7)、buffer_coordinates(B,9)、
    other_edge_faces(B,3)、query_vertices(Q,)、query_pseudonormals(Q,3)。
    未处理值固定1000mm，不将其当真实脑距离。backend默认serial；显式
    parallel仅将X轴写域分成不相交slab，每个体素仍按原面序归并，使用
    调用者当前Numba线程预算，不改变全局线程或输入。几何/伪法向每次
    重算，不跨候选缓存；原源码同绝对距离严格正值优先。
    所有数据需有限，几何/CSR非法ValueError，非有限距离FloatingPointError。
    原内部步骤无独立CLI；只完整距离前置，后续体积评分/curvbak另调用。
    """
    if not isinstance(backend,str) or backend not in ('serial','parallel'):
        raise ValueError('backend must be serial or parallel')
    xyz,normals=np.asarray(original_vertices,np.float32),np.asarray(face_normals,np.float32)
    triangles=np.asarray(faces)
    shape=np.asarray(volume_shape)
    origin,scale=np.asarray(volume_origin_mm,np.float32),np.asarray(volume_index_scale_per_mm,np.float32)
    if (xyz.ndim!=2 or xyz.shape[1]!=3 or triangles.ndim!=2 or triangles.shape[1]!=3
            or not np.issubdtype(triangles.dtype,np.integer) or normals.shape!=(len(triangles),3)
            or shape.shape!=(3,) or not np.issubdtype(shape.dtype,np.integer) or np.any(shape<1)
            or origin.shape!=(3,) or scale.shape!=(3,) or np.any(scale<=0)
            or not all(np.isfinite(a).all() for a in [xyz,normals,origin,scale])):
        raise ValueError('finite current mesh and positive integer XYZ volume geometry required')
    if triangles.size and (triangles.min()<0 or triangles.max()>=len(xyz)):
        raise ValueError('face vertex outside current geometry')
    selected=np.arange(len(triangles),dtype=np.int64) if ordered_candidate_faces is None else np.asarray(ordered_candidate_faces)
    if (selected.ndim!=1 or not np.issubdtype(selected.dtype,np.integer)
            or selected.size and (selected.min()<0 or selected.max()>=len(triangles))):
        raise ValueError('ordered candidate face IDs within current mesh required')
    xyz,normals,triangles=np.ascontiguousarray(xyz),np.ascontiguousarray(normals),np.ascontiguousarray(triangles,np.int64)
    bounds,coordinates=_prepare_buffers(xyz,triangles,np.ascontiguousarray(selected,np.int64),shape.astype(np.int64),origin,scale)
    edge_faces=other_edge_faces(faces=triangles,incident_offsets=incident_offsets,incident_faces=incident_faces,selected_faces=bounds[:,0])
    queries=np.unique(triangles[bounds[:,0]])
    queried=angle_weighted_vertex_pseudonormals(original_vertices=xyz,faces=triangles,face_normals=normals,
        incident_offsets=incident_offsets,incident_faces=incident_faces,incident_corners=incident_corners,selected_vertices=queries)
    pseudonormals=np.zeros(xyz.shape,np.float32);pseudonormals[queries]=queried
    distance_kernel=_distance if backend=='serial' else _distance_parallel
    distances=distance_kernel(xyz,triangles,normals,pseudonormals,bounds,edge_faces,shape.astype(np.int64),origin,scale)
    if not np.isfinite(distances).all():raise FloatingPointError('nonfinite source signed distance')
    return {'distance_mm':distances,'buffer_bounds':bounds,'buffer_coordinates':coordinates,
            'other_edge_faces':edge_faces,'query_vertices':queries,'query_pseudonormals':queried,
            'scope':'complete source active signed-distance prerequisite; not full unMRI/GA'}
