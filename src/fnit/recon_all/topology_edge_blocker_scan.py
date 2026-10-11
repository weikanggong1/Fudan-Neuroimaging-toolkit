"""将有序首阻挡扫描放入Numba；复用同一球面边谓词，不缩减候选域。"""
from __future__ import annotations
import time
import hashlib
import numpy as np
from numba import njit
from numba.typed import List
from .topology_preflight_python import _edge_intersects


@njit(cache=True, fastmath=False)
def _edge_intersects_query_half(xyz, a, b, c, d, u0, u1, query_normal):
    """复用当前查询边的double向量/叉积；退化分支仍执行成熟完整谓词。"""
    if a == c or a == d or b == c or b == d:
        return False
    u2 = xyz[c].astype(np.float64)
    u3 = xyz[d].astype(np.float64)
    n1 = np.cross(u2, u3)
    a0 = np.dot(u0, n1)
    a1 = np.dot(u1, n1)
    a2 = np.dot(u2, query_normal)
    a3 = np.dot(u3, query_normal)
    # 保留原near-zero分支的完整运算与最终共线projection，不拆写它。
    if (abs(a0) < 1e-7 or abs(a1) < 1e-7
            or abs(a2) < 1e-7 or abs(a3) < 1e-7):
        return _edge_intersects(xyz, a, b, c, d)
    product0 = a0 * a1
    product1 = a2 * a3
    if product0 > 0 or product1 > 0:
        return False
    if product0 < 0 or product1 < 0:
        return True
    return _edge_intersects(xyz, a, b, c, d)


@njit(cache=True, fastmath=False)
def _scan(xyz, a, b, scan_vertices, row_mapping, rows, valid, audit):
    capacity=0
    if audit:
        for v in scan_vertices:capacity+=len(rows[row_mapping[v]])
    sequence=np.empty((capacity,5),np.int32)
    calls=0
    # 仅在这一完整scan中复用；下一次调用重新读取当前canonical。
    u0=xyz[a].astype(np.float64)
    u1=xyz[b].astype(np.float64)
    query_normal=np.cross(u0,u1)
    for v in scan_vertices:
        if v==a or v==b:continue
        for w in rows[row_mapping[v]]:
            if w==a or w==b or not valid[w]:continue
            hit=_edge_intersects_query_half(xyz,a,b,v,w,u0,u1,query_normal)
            if audit:
                sequence[calls,0]=a;sequence[calls,1]=b
                sequence[calls,2]=v;sequence[calls,3]=w
                sequence[calls,4]=int(hit)
            calls+=1
            if hit:return int(v),int(w),calls,sequence[:calls]
    return -1,-1,calls,sequence[:calls] if audit else sequence


class OrderedBlockerScan:
    """一个candidate的动态有序扫描缓冲，不跨候选或几何状态缓存。

    scan_vertices为源完整inside+hull有序int32行；可重复，原循环每个
    重复行仍执行。neighbor_rows为完整N行源邻接，只有实际query行被
    复制到typed数组；不从面重建、去重、排序或丢弃未支撑边。外部控制
    每次修改邻接后必须sync_vertex(vertex, row)，更新该顶点完整当前行。
    rows属性的每行是独立buffer，不与输入、prepared基底或返回状态共享。
    first_blocker显式传当前canonical与固定mapping有效域，沿扫描顺序
    使用原谓词规则并返回原首(v,w)或None。仅将本scan不变的查询边
    两个F32→F64向量及其cross提到循环外，其他dot/product运算顺序
    保持；nearzero及最终退化分支调用已有_edge_intersects完整求值。
    下一次scan重新读取canonical，不跨坐标/候选缓存或保存hit；未选择
    近似候选，fastmath=False，不改变坐标/精度/线程。

    query_mask记录缓冲覆盖域；扫描额外hull子域时必须属于该域。
    sync只更新有query的行；其他行从未被扫描。调用者负责动态邻接
    修改次序；本类不执行GA、分类/删除/接受或生产质量检查。
    audit默认False；True返回有序实际谓词序列的SHA-256诊断，包含
    临时缓冲与哈希开销。canonical有限性由成熟重连接口一次校验。
    """
    def __init__(self, *, scan_vertices, neighbor_rows, audit=False):
        start=time.perf_counter();scan=np.asarray(scan_vertices)
        n=len(neighbor_rows)
        if not isinstance(audit,(bool,np.bool_)):raise ValueError('audit must be bool or np.bool_')
        self.audit=bool(audit);self.sequence_digest=hashlib.sha256() if audit else None
        if (scan.ndim!=1 or scan.dtype.kind not in 'iu'
                or scan.size and (scan.min()<0 or scan.max()>=n)):
            raise ValueError('complete integer source query vertex sequence required')
        self.mapping=np.full(n,-1,np.int32)
        # Typed lists need an explicit first array even for an empty query domain.
        self.rows=List();self.rows.append(np.empty(0,np.int32))
        for vertex in scan:
            vertex=int(vertex)
            if self.mapping[vertex]<0:
                row=self._row(neighbor_rows[vertex],n)
                self.mapping[vertex]=len(self.rows)
                self.rows.append(row)
        self.preparation_seconds=time.perf_counter()-start
        self.calls=0;self.predicate_calls=0

    @staticmethod
    def _row(row,n):
        value=np.asarray(row)
        if value.size==0 and isinstance(row,(tuple,list)):value=np.empty(0,np.int32)
        if (value.ndim!=1 or value.dtype.kind not in 'iu'
                or value.size and (value.min()<0 or value.max()>=n)):
            raise ValueError('full ordered integer source neighbor row required')
        return np.array(value,np.int32,copy=True)

    def sync_vertex(self, vertex, row):
        if (isinstance(vertex,(bool,np.bool_)) or not isinstance(vertex,(int,np.integer))
                or vertex<0 or vertex>=len(self.mapping)):
            raise ValueError('valid integer query vertex required')
        slot=self.mapping[vertex]
        if slot>=0:self.rows[int(slot)]=self._row(row,len(self.mapping))

    def first_blocker(self, *, xyz, a, b, scan_vertices, valid):
        scan=np.asarray(scan_vertices)
        if (scan.ndim!=1 or scan.dtype!=np.int32 or scan.size and
                (scan.min()<0 or scan.max()>=len(self.mapping) or np.any(self.mapping[scan]<0))):
            raise ValueError('ordered scan must stay inside prepared query rows')
        v,w,count,sequence=_scan(xyz,a,b,scan,self.mapping,self.rows,valid,self.audit)
        self.calls+=1;self.predicate_calls+=count
        if self.audit:self.sequence_digest.update(sequence.astype('<i4',copy=False).tobytes())
        return None if v<0 else (v,w)
