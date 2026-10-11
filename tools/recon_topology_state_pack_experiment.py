"""独立实验：当前完整候选状态装配，复用成熟打包/前缀/发布函数。

无持久缓存，不改变生产默认；仅同源合法一环/all候选。每次检查全部当前行。
"""
from contextlib import contextmanager
import numpy as np
from fnit.recon_all import topology_candidate_cycle as cycle
from fnit.recon_all.topology_candidate_state_cache import _pack, _csr, _append_prefix


def materialize_packed(*,initial,result,validated=None):
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
        old_offsets,old_values=_pack(initial[name],counts)
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


@contextmanager
def state_pack_scope(enabled):
    """仅串行实验作用域替换局部装配回调；成功/异常均恢复函数身份。"""
    if not isinstance(enabled,bool):raise ValueError('explicit bool enabled required')
    original=cycle._candidate_state_validated
    if enabled:cycle._candidate_state_validated=materialize_packed
    try:yield
    finally:cycle._candidate_state_validated=original
