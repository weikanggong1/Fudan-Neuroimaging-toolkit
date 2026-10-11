"""完整源距离场的独立GPU候选；每体素按源面序更新，跨体素并行。"""
from __future__ import annotations
import numpy as np
import torch
from numba import njit
from .topology_defect_signed_distance import _prepare_buffers,_dot,_cross
from .topology_angle_pseudonormals import other_edge_faces,angle_weighted_vertex_pseudonormals


@njit(cache=True,fastmath=False)
def _face_fields(xyz,faces,normals,pseudo,bounds,other):
    data=np.empty((len(bounds),55),np.float32)
    for row in range(len(bounds)):
        ids=faces[bounds[row,0]];vertex=xyz[ids];nf=normals[bounds[row,0]];center=np.zeros(3,np.float32)
        for c in range(3):
            edge=vertex[(c+1)%3]-vertex[c];half=_cross(nf,edge)
            for axis in range(3):
                data[row,c*3+axis]=vertex[c,axis];data[row,12+c*3+axis]=pseudo[ids[c],axis]
                data[row,21+c*3+axis]=edge[axis];data[row,30+c*3+axis]=half[axis]
                data[row,39+c*3+axis]=np.float32(nf[axis]+normals[other[row,c],axis])
                center[axis]=np.float32(center[axis]+vertex[c,axis])
            data[row,48+c]=_dot(edge,edge)
        center=center*np.float32(.33);far=np.float32(0)
        for c in range(3):
            delta=center-vertex[c];far=max(far,_dot(delta,delta))
        for axis in range(3):data[row,9+axis]=nf[axis];data[row,51+axis]=center[axis]
        data[row,54]=np.float32(np.float32(np.sqrt(far))*np.float32(1.01))
    return data


@njit(cache=True,fastmath=False)
def _block_rows(bounds,shape,block):
    total=shape[0]*shape[1]*shape[2];count=(total+block-1)//block
    offsets=np.zeros(count+1,np.int64);rows=[]
    for b in range(count):
        lo=np.full(3,np.iinfo(np.int64).max,np.int64);hi=np.zeros(3,np.int64)
        for index in range(b*block,min((b+1)*block,total)):
            i=index//(shape[1]*shape[2]);j=(index//shape[2])%shape[1];k=index%shape[2]
            for axis,value in enumerate((i,j,k)):lo[axis]=min(lo[axis],value);hi[axis]=max(hi[axis],value)
        for row in range(len(bounds)):
            overlap=True
            for axis in range(3):
                if bounds[row,1+axis*2]>hi[axis]or bounds[row,2+axis*2]<lo[axis]:overlap=False
            if overlap:rows.append(row)
        offsets[b+1]=len(rows)
    return offsets,np.asarray(rows,np.int64)


def gpu_defect_signed_distance(*,original_vertices,faces,face_normals,incident_offsets,incident_faces,incident_corners,volume_shape,volume_origin_mm,volume_index_scale_per_mm,ordered_candidate_faces=None,backend='serial',device,block_voxels=64):
    """完整当前几何→F32(X,Y,Z)mm距离及原字段，显式CUDA，失败不回退。

    输入与成熟defect_signed_distance一致；backend仅保留serial/parallel
    输入兼容，实际后端始终gpu_ordered_voxels。每个体素完整按源面序
    消费全部扩展bbox候选，原当前距离剪枝、正值同分优先和1000哨兵
    保持；只并行不同体素。block_voxels只取32/64/128，控制内存块。
    CPU复用完整角点伪法向/邻面/bbox准备，动态几何不跨候选缓存。
    新块CSR只保守排除不与整块包围盒相交的面，保持原序及重复项。
    """
    if backend not in ('serial','parallel'):raise ValueError('source backend must be serial or parallel')
    if isinstance(block_voxels,(bool,np.bool_))or not isinstance(block_voxels,(int,np.integer))or block_voxels not in (32,64,128):raise ValueError('block_voxels must be an integer 32,64 or 128')
    block_voxels=int(block_voxels)
    target=torch.device(device)
    if target.type!='cuda'or target.index is None:raise ValueError('explicit cuda:N required')
    xyz=np.asarray(original_vertices,np.float32);normal=np.asarray(face_normals,np.float32);tri=np.asarray(faces);shape=np.asarray(volume_shape);origin=np.asarray(volume_origin_mm,np.float32);scale=np.asarray(volume_index_scale_per_mm,np.float32)
    if xyz.ndim!=2 or xyz.shape[1:]!=(3,)or tri.ndim!=2 or tri.shape[1:]!=(3,)or tri.dtype.kind not in 'iu'or normal.shape!=tri.shape or shape.shape!=(3,)or shape.dtype.kind not in 'iu'or np.any(shape<1)or origin.shape!=(3,)or scale.shape!=(3,)or np.any(scale<=0)or any(not np.isfinite(x).all()for x in [xyz,normal,origin,scale]):raise ValueError('finite current geometry and positive XYZ dimensions required')
    if int(shape[0])*int(shape[1])*int(shape[2])>np.iinfo(np.int32).max:
        raise ValueError('GPU grid exceeds signed 32-bit linear indexing; no truncation or fallback')
    if tri.size and(tri.min()<0 or tri.max()>=len(xyz)):raise ValueError('face vertex outside mesh')
    selected=np.arange(len(tri),dtype=np.int64)if ordered_candidate_faces is None else np.asarray(ordered_candidate_faces)
    if selected.ndim!=1 or selected.dtype.kind not in 'iu'or selected.size and(selected.min()<0 or selected.max()>=len(tri)):raise ValueError('ordered candidate face indices required')
    xyz=np.ascontiguousarray(xyz);normal=np.ascontiguousarray(normal);tri=np.ascontiguousarray(tri,np.int64);shape=shape.astype(np.int64)
    bounds,coordinates=_prepare_buffers(xyz,tri,np.ascontiguousarray(selected,np.int64),shape,origin,scale)
    other=other_edge_faces(faces=tri,incident_offsets=incident_offsets,incident_faces=incident_faces,selected_faces=bounds[:,0])
    queries=np.unique(tri[bounds[:,0]])
    queried=angle_weighted_vertex_pseudonormals(original_vertices=xyz,faces=tri,face_normals=normal,incident_offsets=incident_offsets,incident_faces=incident_faces,incident_corners=incident_corners,selected_vertices=queries)
    pseudo=np.zeros(xyz.shape,np.float32);pseudo[queries]=queried
    fields=_face_fields(xyz,tri,normal,pseudo,bounds,other);offsets,rows=_block_rows(bounds,shape,block_voxels)
    required_bytes=sum(x.nbytes for x in [fields,bounds,offsets,rows,origin,scale])+int(np.prod(shape))*4
    if required_bytes>20_000_000_000:raise MemoryError("distance tensors exceed 20 GB budget")
    from .topology_gpu_distance_kernel import distance_kernel
    with torch.cuda.device(target):
        data=[torch.tensor(np.ascontiguousarray(x),device=target)for x in [fields,bounds,offsets,rows,origin,scale]]
        output=torch.empty(tuple(map(int,shape)),device=target,dtype=torch.float32)
        distance_kernel[(len(offsets)-1,)](*data,output,*map(int,shape),BLOCK=block_voxels,enable_fp_fusion=False,num_warps=4)
        distance=output.cpu().numpy()
    if not np.isfinite(distance).all():raise FloatingPointError('nonfinite GPU source distance')
    return {'distance_mm':distance,'buffer_bounds':bounds,'buffer_coordinates':coordinates,'other_edge_faces':other,'query_vertices':queries,'query_pseudonormals':queried,'scope':'complete source active signed-distance prerequisite; not full unMRI/GA','actual_backend':'gpu_ordered_voxels','block_voxels':block_voxels,'complete_face_rows':len(bounds),'block_face_entries':len(rows)}
