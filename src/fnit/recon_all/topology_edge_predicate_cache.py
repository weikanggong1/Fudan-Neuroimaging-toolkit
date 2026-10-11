"""显式缓存同一不可变 canonical 的有向四顶点布尔谓词；不缓存动态阻挡。"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
import threading
import time

import numpy as np
from numba import njit, types
from numba.typed import Dict

from .topology_edge_blocker_scan import OrderedBlockerScan, _edge_intersects_query_half


def _immutable_layout(xyz):
    """检查完整 F32 C 数组和其整个只读 base 链；最终值缓冲必须是 bytes。"""
    if (not isinstance(xyz, np.ndarray) or xyz.dtype != np.float32
            or xyz.ndim != 2 or xyz.shape[1] != 3
            or not xyz.flags.c_contiguous or xyz.flags.writeable):
        raise ValueError("complete bytes-backed immutable float32 (N,3) canonical required")
    chain = []
    value = xyz
    while isinstance(value, np.ndarray):
        if value.flags.writeable:
            raise ValueError("mutable canonical base buffer is not cacheable")
        chain.append((value, value.dtype.str, value.shape, value.strides,
                      value.__array_interface__["data"][0]))
        value = value.base
    if not isinstance(value, bytes):
        raise ValueError("immutable canonical must terminate in a bytes value buffer")
    # A partial view cannot silently declare a different complete vertex domain.
    if xyz.nbytes != len(value) or xyz.__array_interface__["data"][0] != chain[-1][4]:
        raise ValueError("complete canonical value buffer required; partial views are unsupported")
    return tuple(chain), value


@njit(cache=True, fastmath=False)
def _scan_memoized(xyz, a, b, scan_vertices, row_mapping, rows, valid,
                   audit, memo, maximum_entries, counters):
    """保持成熟 scan 顺序和查询运算；只有纯几何 bool 读取/写入稀疏 memo。"""
    capacity = 0
    if audit:
        for v in scan_vertices:
            capacity += len(rows[row_mapping[v]])
    sequence = np.empty((capacity, 5), np.int32)
    calls = 0
    u0 = xyz[a].astype(np.float64)
    u1 = xyz[b].astype(np.float64)
    query_normal = np.cross(u0, u1)
    for v in scan_vertices:
        if v == a or v == b:
            continue
        for w in rows[row_mapping[v]]:
            if w == a or w == b or not valid[w]:
                continue
            # Direction and all four IDs stay in their original order.
            key = (np.int64(a), np.int64(b), np.int64(v), np.int64(w))
            if key in memo:
                hit = memo[key]
                counters[0] += 1
            else:
                hit = _edge_intersects_query_half(xyz, a, b, v, w, u0, u1, query_normal)
                counters[1] += 1
                if len(memo) < maximum_entries:
                    memo[key] = hit
                    counters[2] += 1
                else:
                    counters[3] += 1
            if audit:
                sequence[calls, 0] = a
                sequence[calls, 1] = b
                sequence[calls, 2] = v
                sequence[calls, 3] = w
                sequence[calls, 4] = int(hit)
            calls += 1
            if hit:
                return int(v), int(w), calls, sequence[:calls]
    return -1, -1, calls, sequence[:calls] if audit else sequence


class EdgePredicateMemo:
    """一缺陷独占的可释放几何缓存；拒绝可变/不同坐标域和并发写入。

    canonical 为完整 bytes-backed、C 连续、有限非零 float32 (N,3)，
    使用 PreparedRetessellationBase.canonical。maximum_entries 默认250000，
    为非负整数；满后不新增，但每个 miss 仍调用完整成熟谓词。键为有序
    (a,b,v,w)，值仅bool，无邻接、valid、marks、首阻挡或候选反馈缓存。
    坐标采用原 canonical RAS，坐标单位及 signed zero 不变。

    只绑定此数组本身及不可变 bytes 和元信息；调用者修改原准备输入不会
    影响它。不同 canonical、shape/dtype/stride/base 元信息变化、已关闭、
    fork 后复用或同时写入均抛 ValueError/RuntimeError。无全局状态、随机
    调用、线程设置、CUDA或精度修改。close()/with 退出清除字典；每个缺陷
    必须新建并释放。report() 返回累计计数、缓存上限和输入SHA，不是计时
    benchmark，也不包含未受上限约束的全量独特key估计。
    """
    def __init__(self, *, canonical, maximum_entries=250_000):
        tick = time.perf_counter()
        if (isinstance(maximum_entries, (bool, np.bool_))
                or not isinstance(maximum_entries, (int, np.integer))
                or maximum_entries < 0 or maximum_entries > np.iinfo(np.int64).max):
            raise ValueError("maximum_entries must be a nonnegative int64-range integer")
        self._layout, self._buffer = _immutable_layout(canonical)
        if (not np.isfinite(canonical).all()
                or np.any(np.all(canonical == 0, axis=1))
                or len(canonical) > np.iinfo(np.int32).max):
            raise ValueError("finite nonzero canonical and int32-range vertex domain required")
        self.canonical = canonical
        self.maximum_entries = int(maximum_entries)
        self.canonical_sha256 = hashlib.sha256(canonical.tobytes()).hexdigest()
        self._memo = Dict.empty(key_type=types.UniTuple(types.int64, 4), value_type=types.boolean)
        self._counters = np.zeros(4, np.int64)
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._closed = False
        self.preparation_seconds = time.perf_counter() - tick

    def _check(self, xyz):
        if self._closed or os.getpid() != self._pid:
            raise RuntimeError("memo is closed or belongs to a different process")
        if xyz is not self.canonical:
            raise ValueError("memo requires the exact immutable canonical domain; create a new context")
        layout, buffer = _immutable_layout(xyz)
        if (buffer is not self._buffer or len(layout) != len(self._layout)
                or any(a[0] is not b[0] or a[1:] != b[1:]
                       for a, b in zip(layout, self._layout))):
            raise ValueError("immutable canonical metadata changed; create a new context")

    @contextmanager
    def _exclusive(self, xyz):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("concurrent or reentrant writes to one memo are prohibited")
        try:
            self._check(xyz)
            yield
        finally:
            self._lock.release()

    def report(self):
        """返回typed累计计数；cap满后的独特key总数未知，明确记None。"""
        with self._exclusive(self.canonical):
            hits, misses, inserts, bypass = map(int, self._counters)
            return dict(canonical_sha256=self.canonical_sha256,
                        vertex_count=len(self.canonical), maximum_entries=self.maximum_entries,
                        entries=len(self._memo), logical_predicate_calls=hits + misses,
                        cache_hits=hits, actual_predicate_evaluations=misses,
                        inserted_unique_keys=inserts, capacity_bypass_evaluations=bypass,
                        hit_rate=hits / (hits + misses) if hits + misses else 0.0,
                        complete_unique_key_count=inserts if not bypass else None,
                        preparation_seconds=self.preparation_seconds,
                        geometry_only=True, concurrent_writes_allowed=False)

    def close(self):
        """释放本缺陷的稀疏字典；重复关闭无操作，活动调用中关闭报错。"""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("cannot close a memo during an active call")
        try:
            if os.getpid() != self._pid:
                raise RuntimeError("memo belongs to a different process")
            if not self._closed:
                self._memo.clear()
                # Dict.clear() may keep its hash-table allocation. Drop the
                # owned reference as well so a completed defect releases it.
                self._memo = None
                self._closed = True
        finally:
            self._lock.release()

    def __enter__(self):
        with self._exclusive(self.canonical):
            pass
        return self

    def __exit__(self, *_):
        self.close()


class MemoizedOrderedBlockerScan(OrderedBlockerScan):
    """显式memo加上成熟动态行缓冲；默认 OrderedBlockerScan 不变。

    scan_vertices、neighbor_rows、audit与成熟构造接口相同；memo必填，
    且其完整坐标数须等于邻接行数。sync_vertex继续使用成熟复制与更新。
    first_blocker的xyz必须是memo.canonical；a/b为域内整数，valid为完整
    bool(N,)，scan为已准备域内的同序int32行。返回首(v,w)或None；审计
    counters和SHA包含每次logical谓词（命中也记录），不会重排/省略动态
    行、valid或首次命中。失败时抛异常；对象不执行GA或生产默认接线。
    """
    def __init__(self, *, scan_vertices, neighbor_rows, memo, audit=False):
        if not isinstance(memo, EdgePredicateMemo):
            raise ValueError("explicit EdgePredicateMemo required")
        with memo._exclusive(memo.canonical):
            if len(neighbor_rows) != len(memo.canonical):
                raise ValueError("memo and full neighbor vertex domains must match")
            super().__init__(scan_vertices=scan_vertices, neighbor_rows=neighbor_rows, audit=audit)
        self.memo = memo

    def sync_vertex(self, vertex, row):
        """同步完整动态行；同一memo活动调用中禁止改行，不缓存邻接。"""
        with self.memo._exclusive(self.memo.canonical):
            super().sync_vertex(vertex, row)

    def first_blocker(self, *, xyz, a, b, scan_vertices, valid):
        scan = np.asarray(scan_vertices)
        valid = np.asarray(valid)
        n = len(self.mapping)
        if (scan.ndim != 1 or scan.dtype != np.int32 or scan.size and
                (scan.min() < 0 or scan.max() >= n or np.any(self.mapping[scan] < 0))):
            raise ValueError("ordered scan must stay inside prepared query rows")
        if valid.dtype != np.bool_ or valid.shape != (n,):
            raise ValueError("complete bool fixed-mapping domain required")
        for value in (a, b):
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                    or value < 0 or value >= n):
                raise ValueError("integer query vertices must stay in the canonical domain")
        with self.memo._exclusive(xyz):
            v, w, count, sequence = _scan_memoized(
                xyz, int(a), int(b), scan, self.mapping, self.rows, valid, self.audit,
                self.memo._memo, self.memo.maximum_entries, self.memo._counters)
            self.calls += 1
            self.predicate_calls += count
            if self.audit:
                self.sequence_digest.update(sequence.astype("<i4", copy=False).tobytes())
        return None if v < 0 else (v, w)
