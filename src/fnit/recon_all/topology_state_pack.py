"""当前调用完整候选状态打包；无持久缓存、全局替换或动态字段快照。

每次读取当前邻接与属性，复用成熟整数前缀核和属性发布函数。
"""
import numpy as np
from . import topology_candidate_cycle as cycle
from .topology_candidate_state_cache import _pack, _csr, _append_prefix


def _pack_storage_prefix(rows, counts):
    """全量检查当前有效前缀；长度匹配的 ndarray 行直接参与复制。

    不保存行、值或验证结果。容量行仍切到当前计数，整型自身保证有限
    和整数性质，但全部有效值仍检查 int32 范围；浮点行仍逐值检查。
    Python 行复用成熟打包路径，返回的扁平值始终与输入独立。
    """
    if not len(rows) or not all(isinstance(row, np.ndarray) for row in rows):
        return _pack(rows, counts)
    counts = np.asarray(counts)
    if (counts.shape != (len(rows),) or counts.dtype.kind not in 'iu'
            or np.any(counts < 0)):
        raise ValueError('complete nonnegative integer row counts required')
    prefixes = []
    for row, count in zip(rows, counts):
        count = int(count)
        if row.ndim != 1 or len(row) < count:
            raise ValueError('complete one-dimensional numeric storage rows required')
        prefixes.append(row if len(row) == count else row[:count])
    offsets = np.zeros(len(counts)+1, np.int64)
    offsets[1:] = np.cumsum(counts, dtype=np.int64)
    values = np.concatenate(prefixes)
    if len(values) != int(offsets[-1]) or values.dtype.kind not in 'iuf':
        raise ValueError('complete one-dimensional numeric storage rows required')
    if values.dtype.kind in 'iu':
        invalid = values.size and (int(values.min()) < np.iinfo(np.int32).min
                                   or int(values.max()) > np.iinfo(np.int32).max)
    else:
        invalid = (not np.isfinite(values).all() or values.size and
                   (values.min() < np.iinfo(np.int32).min
                    or values.max() > np.iinfo(np.int32).max)
                   or np.any(values != np.trunc(values)))
    if invalid:
        raise ValueError('ordered row values must be exact int32 integers')
    # concatenate 已产生独立当前缓冲；int32 无需再复制一次。
    return offsets, values.astype(np.int32, copy=False)


def materialize_candidate_state_packed(*,initial,result,validated=None):
    """当前初态与完整候选→完整状态，整数全域验证后复用源属性发布。

    validated仅供同一次cycle使用其刚完成的一环校验；独立调用默认重验。
    不保存跨候选上下文。原容量尾部及未改变邻接对象保留，新增邻接复制；
    所有incident/corner先复制全缓冲再分成独立非重叠行视图。
    完整有序prefix、冗余邻接CSR和角点域每次验证，失败不发布。
    只支持成熟producer的合法整数拓扑；损坏冗余CSR额外明确拒绝。
    坐标、法线、K/E/H/curvbak/RP及浮点更新全部复用成熟发布函数。
    """
    n,old,nf=cycle._one_ring_state(initial)if validated is None else validated
    g=result['geometry'];faces=np.asarray(g['faces']);rows=g['neighbors']
    if (len(rows)!=n or faces.ndim!=2 or faces.shape[1:]!=(3,)or faces.dtype.kind not in 'iu'
            or faces.size and(faces.min()<0 or faces.max()>=n)
            or not np.array_equal(faces[:nf],np.asarray(initial['faces'])[:nf])or len(g['removed_vertices'])):
        raise ValueError('complete ordered base face prefix and all vertices required')
    no,nv=_csr(g['neighbor_offsets'],g['ordered_neighbors'],n,n,'current neighbor')
    io,iv=_csr(g['incident_offsets'],g['incident_faces'],n,len(faces),'current incident')
    _,cv=_csr(io,g['incident_corners'],n,3,'current corner')
    ro,rv=_pack(rows,np.fromiter((len(row)for row in rows),np.int64,count=n))
    if not np.array_equal(ro,no)or not np.array_equal(rv,nv):raise ValueError('complete candidate rows must equal current CSR')
    for name,counts,new_offsets,new_values in [
        ('neighbor_storage',old[:,3],no,nv),
        ('incident_faces',old[:,6],io,iv),
        ('incident_corners',old[:,6],io,cv),
    ]:
        old_offsets,old_values=_pack_storage_prefix(initial[name],counts)
        if _append_prefix(old_offsets,old_values,new_offsets,new_values)>=0:
            raise ValueError('complete source append-only prefix required: '+name)
    top=old.copy();degree=np.diff(no);changed=np.flatnonzero(degree!=old[:,3])
    neighbors=list(initial['neighbor_storage'])
    for v in changed:
        top[v,0:2]=1;top[v,2:4]=degree[v];neighbors[v]=nv[no[v]:no[v+1]].copy()
    top[:,6]=np.diff(io)
    ids=iv.copy();corner=cv.astype(np.uint8,copy=True)
    incidents=[ids[a:b]for a,b in zip(io[:-1],io[1:])]
    corners=[corner[a:b]for a,b in zip(io[:-1],io[1:])]
    return cycle._publish_candidate_attributes(initial=initial,result=result,top=top,
        neighbors=neighbors,incidents=incidents,corners=corners,n=n,nf=nf)
