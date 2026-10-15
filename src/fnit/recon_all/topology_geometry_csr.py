"""完整当前面序→源序CSR；显式计数/填充，保留重复项和空行。"""
import numpy as np
from numba import njit

@njit(cache=True, fastmath=False)
def _counted_csr(faces, nvertices):
    offsets=np.zeros(nvertices+1,np.int64)
    for face in range(len(faces)):
        for corner in range(3):offsets[faces[face,corner]+1]+=1
    for vertex in range(nvertices):offsets[vertex+1]+=offsets[vertex]
    cursor=offsets[:-1].copy()
    face_ids=np.empty(3*len(faces),np.int64)
    corners=np.empty(3*len(faces),np.int64)
    for face in range(len(faces)):
        for corner in range(3):
            vertex=faces[face,corner];entry=cursor[vertex]
            face_ids[entry]=face;corners[entry]=corner;cursor[vertex]+=1
    return offsets,face_ids,corners


def counted_face_csr(faces, nvertices):
    """完整整数CSR，保留每个顶点的源face/corner次序及重复项。

    faces:当前候选有序整数(F,3)，nvertices:非负Python/NumPy整数N。
    输出独立int64 offsets(N+1)、face_ids(3F)、corners(3F)，同成熟接口。
    每次读取当前faces，输入不变；空行/空面合法；非法输入分配前报错。
    独立显式调用，不更改全局函数；由几何调用方选择是否使用。
    """
    if not isinstance(nvertices,(int,np.integer))or isinstance(nvertices,(bool,np.bool_))or nvertices<0:
        raise ValueError('nvertices must be a nonnegative integer')
    tri=np.asarray(faces)
    if tri.ndim!=2 or tri.shape[1]!=3 or not np.issubdtype(tri.dtype,np.integer):
        raise ValueError('faces must be integer (F,3)')
    if np.any(tri<0)or np.any(tri>=nvertices):raise ValueError('face index outside vertex domain')
    if nvertices>np.iinfo(np.int64).max-1 or 3*len(tri)>np.iinfo(np.int64).max:
        raise ValueError('CSR exceeds int64 capacity')
    return _counted_csr(np.ascontiguousarray(tri,np.int64),int(nvertices))
