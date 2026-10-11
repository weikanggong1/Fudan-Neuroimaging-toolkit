"""显式不可变基底状态→完整候选装配；全行验证，非局部截断。"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
import time
import numpy as np
from numba import njit

from .topology_candidate_cycle import _one_ring_state, _publish_candidate_attributes

_TOKEN = object()


def _frozen(array):
    value = np.ascontiguousarray(array)
    return np.frombuffer(value.tobytes(), value.dtype).reshape(value.shape)


def _pack(rows, counts):
    """完整有效前缀，保留行序；容量尾部由初态独立持有。"""
    offsets = np.zeros(len(counts)+1, np.int64)
    offsets[1:] = np.cumsum(counts, dtype=np.int64)
    # Source restores/storage are normally ndarray rows. concatenate copies
    # their ordered prefixes in native NumPy; fromiter would visit every scalar
    # in Python. Retessellation returns Python list rows, where fromiter is
    # faster and avoids creating N small arrays. Both paths validate all values.
    if len(rows) and all(isinstance(row, np.ndarray) for row in rows):
        values = np.concatenate([row[:int(count)] for row,count in zip(rows,counts)])
        if values.ndim != 1 or len(values) != int(offsets[-1]) or values.dtype.kind not in 'iuf':
            raise ValueError('complete one-dimensional numeric storage rows required')
    else:
        values = np.fromiter(chain.from_iterable(row[:int(count)] for row, count in zip(rows, counts)),
                             dtype=np.float64, count=int(offsets[-1]))
    if (not np.isfinite(values).all() or values.size and (values.min()<np.iinfo(np.int32).min
            or values.max()>np.iinfo(np.int32).max) or np.any(values!=np.trunc(values))):
        raise ValueError('ordered row values must be exact int32 integers')
    return offsets, values.astype(np.int32)


@dataclass(frozen=True)
class CandidateStateContext:
    """仅拓扑基底有效前缀的 bytes-backed 快照；不保存动态属性。"""
    topology: np.ndarray
    faces: np.ndarray
    neighbor_offsets: np.ndarray
    neighbors: np.ndarray
    incident_offsets: np.ndarray
    incidents: np.ndarray
    corners: np.ndarray
    preparation_seconds: float
    _token: object


def prepare_candidate_state_context(*, surface_state):
    """完整一环/all 初态→显式可复用上下文，首次校验/打包另计。

    surface_state 由现有完整 producer/DVS 生成，N 顶点，整数一环
    topology(N,7)、有序容量行、逻辑 faces(F,3)。有效前缀必须与原
    num/vnum 相符；未使用容量尾部不裁掉、不用面重建邻接。返回只读
    CandidateStateContext；原输入后续改变不能改快照。没有坐标计算/
    单位转换、线程/精度切换、文件/原程序调用。非法一环/缺字段/
    越界有效索引或面/角点不匹配报 ValueError；多环另用原 DVS API。
    """
    started = time.perf_counter()
    n, top, nf = _one_ring_state(surface_state)
    no, nv = _pack(surface_state['neighbor_storage'], top[:,3])
    io, iv = _pack(surface_state['incident_faces'], top[:,6])
    co, cv = _pack(surface_state['incident_corners'], top[:,6])
    faces = np.asarray(surface_state['faces'])[:nf]
    if (faces.size and (faces.min() < 0 or faces.max() >= n)
            or nv.size and (nv.min() < 0 or nv.max() >= n)
            or iv.size and (iv.min() < 0 or iv.max() >= nf)
            or cv.size and (cv.min() < 0 or cv.max() > 2)
            or not np.array_equal(io, co)):
        raise ValueError('valid complete ordered base topology required')
    return CandidateStateContext(*map(_frozen,(top, faces, no, nv, io, iv, cv)),
                                 time.perf_counter()-started, _TOKEN)


@njit(cache=True, fastmath=False)
def _append_prefix(old_offsets, old_values, new_offsets, new_values):
    """完整 N 行有序前缀逐元素校验；未排序/去重或跳过不变域。"""
    for vertex in range(len(old_offsets)-1):
        old_a, old_b = old_offsets[vertex], old_offsets[vertex+1]
        new_a, new_b = new_offsets[vertex], new_offsets[vertex+1]
        if new_b-new_a < old_b-old_a:
            return vertex
        for entry in range(old_b-old_a):
            if old_values[old_a+entry] != new_values[new_a+entry]:
                return vertex
    return -1


def _csr(offsets, values, n, upper, name):
    o, v = np.asarray(offsets), np.asarray(values)
    if (o.shape != (n+1,) or o.dtype.kind not in 'iu' or v.ndim != 1
            or v.dtype.kind not in 'iu' or o[0] != 0 or o[-1] != len(v)
            or np.any(o[1:] < o[:-1]) or v.size and (v.min() < 0 or v.max() >= upper)):
        raise ValueError('complete valid '+name+' CSR required')
    return o.astype(np.int64, copy=False), v.astype(np.int32, copy=False)


def materialize_candidate_state_cached(*, initial, result, context):
    """显式上下文与完整候选→完整新状态；每调用全域检查有效前缀。

    initial 为当前完整 state：拓扑基底须与 context 相同，容量尾部可
    保留当前值；坐标、rip/marks、K/E/H、面积、fx/curvbak/RP 不缓存。
    result 是现有 evaluate_candidate 的完整返回；冗余邻接行与 CSR
    必须相同，完整有序面前缀、append-only 邻接/面/角点全部验证。
    返回与旧装配相同字段和逐元素值：不变邻接复用 initial 容量行；
    新增邻接独立复制；incident/corner 为新独立 flat buffer 的互不
    重叠行视图（旧容量尾部按原装配移除）。修改输入候选不改输出，
    行间不别名。源已存在 H/K 仅非空 fit 写回，动态更新复用原 helper。
    context 非本函数生成/跨基底/损坏 metadata/多环/删点/前缀改变
    抛 ValueError，失败不发布部分结果。无 GPU/IO/精度/最终 QC 判定。
    """
    n, old, nf = _one_ring_state(initial)
    if not isinstance(context, CandidateStateContext) or context._token is not _TOKEN:
        raise ValueError('explicit prepared CandidateStateContext required')
    arrays=(context.topology, context.faces, context.neighbor_offsets, context.neighbors,
            context.incident_offsets, context.incidents, context.corners)
    if (any(a.flags.writeable or not a.flags.c_contiguous for a in arrays)
            or context.topology.shape != old.shape or context.faces.shape != (nf,3)
            or not np.array_equal(context.topology, old)
            or not np.array_equal(context.faces, np.asarray(initial['faces'])[:nf])):
        raise ValueError('unchanged complete topology/counts/base faces required')
    _csr(context.neighbor_offsets,context.neighbors,n,n,'prepared neighbor')
    _csr(context.incident_offsets,context.incidents,n,nf,'prepared incident')
    _csr(context.incident_offsets,context.corners,n,3,'prepared corner')
    if (not np.array_equal(np.diff(context.neighbor_offsets),old[:,3])
            or not np.array_equal(np.diff(context.incident_offsets),old[:,6])):
        raise ValueError('prepared CSR counts differ from full topology')
    # Mutable caller rows are checked by value every call; no object-id trust.
    for name, counts, offsets, values in (
        ('neighbor_storage',old[:,3],context.neighbor_offsets,context.neighbors),
        ('incident_faces',old[:,6],context.incident_offsets,context.incidents),
        ('incident_corners',old[:,6],context.incident_offsets,context.corners)):
        o, v = _pack(initial[name], counts)
        if not np.array_equal(o, offsets) or not np.array_equal(v, values):
            raise ValueError('current ordered base '+name+' differs from prepared context')
    g = result['geometry'];faces=np.asarray(g['faces'])
    if (faces.ndim != 2 or faces.shape[1:] != (3,) or faces.dtype.kind not in 'iu'
            or faces.size and (faces.min() < 0 or faces.max() >= n)
            or not np.array_equal(faces[:nf], context.faces) or len(g['removed_vertices'])):
        raise ValueError('candidate must preserve complete ordered base face prefix and all vertices')
    no, nv = _csr(g['neighbor_offsets'],g['ordered_neighbors'],n,n,'neighbor')
    io, iv = _csr(g['incident_offsets'],g['incident_faces'],n,len(faces),'incident')
    _, cv = _csr(io,g['incident_corners'],n,3,'corner')
    rows=g['neighbors']
    if len(rows)!=n:
        raise ValueError('complete candidate neighbor rows required')
    ro, rv = _pack(rows, np.fromiter((len(row) for row in rows),np.int64,count=n))
    if not np.array_equal(ro,no) or not np.array_equal(rv,nv):
        raise ValueError('complete candidate neighbor rows must equal its CSR')
    for old_o, old_v, new_o, new_v, name in (
        (context.neighbor_offsets,context.neighbors,no,nv,'adjacency'),
        (context.incident_offsets,context.incidents,io,iv,'face incidence'),
        (context.incident_offsets,context.corners,io,cv,'corner incidence')):
        if _append_prefix(old_o,old_v,new_o,new_v)>=0:
            raise ValueError('source append-only ordered '+name+' required')
    top=old.copy();degrees=np.diff(no);changed=np.flatnonzero(degrees!=old[:,3])
    neighbors=list(initial['neighbor_storage'])
    for vertex in changed:
        top[vertex,0:2]=1;top[vertex,2:4]=degrees[vertex]
        neighbors[vertex]=nv[no[vertex]:no[vertex+1]].copy()
    top[:,6]=np.diff(io)
    ids=iv.copy();corner=cv.astype(np.uint8,copy=True)
    incidents=[ids[a:b] for a,b in zip(io[:-1],io[1:])]
    corners=[corner[a:b] for a,b in zip(io[:-1],io[1:])]
    return _publish_candidate_attributes(initial=initial,result=result,top=top,
        neighbors=neighbors,incidents=incidents,corners=corners,n=n,nf=nf)
