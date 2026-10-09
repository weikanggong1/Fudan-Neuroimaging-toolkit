"""完整球候选与有界运动AABB的PyTorch空间索引，不执行顶点更新。

source/query中心为float64(P,3)、RAS/mm，radii为float64(Q,)/mm。
返回int64(Q+1) CSR偏移、int32(M)源面编号与诊断。候选完整无截断；
边界保护只增加候选，实际碰撞仍用原规则判断。mris_place_surface内部
辅助步骤，无独立官方CLI。无新依赖，不调用外部软件。
"""
from __future__ import annotations

import math
import numpy as np
import torch


@torch.no_grad()
def conservative_face_candidates_torch(
    source_centers, query_centers, radii, *, device: str = "cuda:0",
    query_chunk_size: int = 2048, maximum_chunk_candidates: int = 4000000,
    source_low=None, source_high=None, query_low=None, query_high=None,
    motion_bound: float | None = None, source_faces=None, query_faces=None,
):
    """以完整网格单元查询构建球候选CSR，可安全排除不可能接触的面。

    source_centers(F,3)、query_centers(Q,3)、radii(Q,)均须float64有限值，
    半径非负。device明确CUDA编号，cpu仅算法回归。query_chunk_size为
    初始查询分块；maximum_chunk_candidates控制单块临时候选数量，超限
    自动减小查询块，不删候选；单查询仍超限时抛MemoryError。
    可选low/high为对应初始float32/float64面bbox，四项必须同时给出。
    motion_bound为每个顶点相对初始位置的最大欧氏位移(mm)，需要调用方
    运行时检查；仅此时可用2*bound的bbox间隔排除未来不可能相交的面。
    可选source_faces(F,3)/query_faces(Q,3)必须同时给出，排除共享顶点的
    面对，复用源placement固定拓扑语义。输出候选顺序可以与KD树不同；
    首试步相交bool不依赖候选顺序，retained-MHT重试不得使用本入口。
    CUDA/OOM及输入/坐标编码异常传播，不静默切CPU或截断查询。
    """
    target = torch.device(device)
    if target.type not in ("cpu", "cuda") or target.type == "cuda" and target.index is None:
        raise ValueError("device must be cpu or an explicitly indexed CUDA target")
    if query_chunk_size < 1 or maximum_chunk_candidates < 1:
        raise ValueError("chunk sizes must be positive")
    def prepare(value, name, dtype=torch.float64):
        tensor = value.to(target) if isinstance(value, torch.Tensor) else torch.as_tensor(np.asarray(value), device=target)
        if tensor.dtype != dtype:
            raise TypeError(f"{name} must be {dtype}")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must be finite")
        return tensor
    sources, queries = prepare(source_centers, "source_centers"), prepare(query_centers, "query_centers")
    radius = prepare(radii, "radii")
    if sources.ndim != 2 or sources.shape[1] != 3 or queries.ndim != 2 or queries.shape[1] != 3 or radius.shape != (len(queries),):
        raise ValueError("centers must be (F/Q,3) and radii (Q,)")
    if bool((radius < 0).any()):
        raise ValueError("radii must be nonnegative")
    if len(sources) > np.iinfo(np.int32).max:
        raise ValueError("source face ids exceed int32")
    supplied_boxes = (source_low, source_high, query_low, query_high)
    use_boxes = any(value is not None for value in supplied_boxes)
    sl = sh = ql = qh = None
    if use_boxes:
        if any(value is None for value in supplied_boxes) or motion_bound is None:
            raise ValueError("four bbox arrays and motion_bound are required together")
        if not math.isfinite(motion_bound) or motion_bound < 0:
            raise ValueError("motion_bound must be finite and nonnegative")
        def box(value, shape):
            result = torch.as_tensor(value, device=target)
            if result.dtype not in (torch.float32, torch.float64) or result.shape != shape or not bool(torch.isfinite(result).all()):
                raise ValueError("bbox must be finite float32/float64 arrays of matching shape")
            return result.to(torch.float64)
        sl, sh, ql, qh = (box(value, sources.shape if index < 2 else queries.shape) for index, value in enumerate(supplied_boxes))
        if bool((sl > sh).any()) or bool((ql > qh).any()):
            raise ValueError("bbox low cannot exceed high")
    if (source_faces is None) != (query_faces is None):
        raise ValueError("source_faces and query_faces must be supplied together")
    sf = qf = None
    if source_faces is not None:
        def faces(value, rows):
            result = torch.as_tensor(value, device=target)
            if result.dtype not in (torch.int32, torch.int64) or result.shape != (rows, 3) or bool((result < 0).any()):
                raise ValueError("faces must be nonnegative integer (F/Q,3)")
            return result
        sf, qf = faces(source_faces, len(sources)), faces(query_faces, len(queries))
    info = {"source_faces": len(sources), "query_faces": len(queries), "device": str(target),
            "query_chunk_size": query_chunk_size, "maximum_chunk_candidates": maximum_chunk_candidates,
            "bbox_motion_bound_mm": motion_bound if use_boxes else None,
            "excludes_shared_vertices": sf is not None, "candidate_truncation": False,
            "grid_candidates": 0, "sphere_candidates": 0, "retained_candidates": 0,
            "adaptive_chunk_reductions": 0}
    if not len(sources) or not len(queries):
        return np.zeros(len(queries)+1, np.int64), np.empty(0, np.int32), info
    eps = torch.finfo(torch.float64).eps
    guard = 64 * eps * (queries.abs() + radius[:, None] + 1)
    maximum_radius, maximum_guard = float(radius.max()), float(guard.max())
    width = max(1.0, 2 * (maximum_radius + maximum_guard) + 1e-8)
    if not math.isfinite(width) or width <= 0:
        raise ValueError("query geometry exceeds finite grid encoding")
    info["grid_cell_width_mm"] = width
    source_float_cells = torch.floor(sources / width)
    low_float_cells = torch.floor((queries - radius[:, None] - guard) / width)
    high_float_cells = torch.floor((queries + radius[:, None] + guard) / width)
    if any(bool((cells.abs() >= 2**61).any()) for cells in (source_float_cells, low_float_cells, high_float_cells)):
        raise ValueError("grid coordinates exceed safe int64 encoding")
    source_cells, low_cells, high_cells = source_float_cells.to(torch.int64), low_float_cells.to(torch.int64), high_float_cells.to(torch.int64)
    if bool((high_cells-low_cells > 1).any()):
        raise RuntimeError("complete query needs more than two cells per axis")
    origin = torch.minimum(source_cells.amin(0), low_cells.amin(0))
    last = torch.maximum(source_cells.amax(0), high_cells.amax(0))
    dimensions = (last-origin+1).cpu().tolist()
    if math.prod(dimensions) >= 2**63:
        raise ValueError("packed grid key exceeds int64")
    def encode(cells):
        shifted = cells-origin
        return (shifted[..., 0]*dimensions[1]+shifted[..., 1])*dimensions[2]+shifted[..., 2]
    source_keys = encode(source_cells)
    order = torch.argsort(source_keys, stable=True)
    sorted_keys = source_keys[order]
    shifts = torch.tensor([[x,y,z] for x in (0,1) for y in (0,1) for z in (0,1)], dtype=torch.int64, device=target)
    offsets, parts, start = [0], [], 0
    while start < len(queries):
        stop = min(start+query_chunk_size, len(queries))
        while True:
            cells = low_cells[start:stop, None, :]+shifts[None]
            valid = (cells <= high_cells[start:stop, None, :]).all(2)
            keys = encode(cells).flatten()
            begin = torch.searchsorted(sorted_keys, keys, right=False)
            end = torch.searchsorted(sorted_keys, keys, right=True)
            counts = torch.where(valid.flatten(), end-begin, torch.zeros_like(begin))
            total = int(counts.sum())
            if total <= maximum_chunk_candidates:
                break
            if stop-start == 1:
                raise MemoryError("single complete query exceeds maximum_chunk_candidates; increase its budget")
            stop = start+max(1, (stop-start)//2)
            info["adaptive_chunk_reductions"] += 1
        info["grid_candidates"] += total
        if total:
            prefix = torch.cumsum(counts, 0)-counts
            query_ids = torch.repeat_interleave(torch.arange(stop-start, device=target).repeat_interleave(8), counts, output_size=total)
            locations = torch.repeat_interleave(begin, counts, output_size=total)+torch.arange(total, device=target)-torch.repeat_interleave(prefix, counts, output_size=total)
            ids = order[locations]
            delta = sources[ids]-queries[start:stop][query_ids]
            squared = (delta[:,0]*delta[:,0]+delta[:,1]*delta[:,1])+delta[:,2]*delta[:,2]
            rr = radius[start:stop][query_ids]
            squared_radius = rr*rr
            keep = squared <= squared_radius+64*eps*(squared.abs()+squared_radius.abs()+1)
            info["sphere_candidates"] += int(keep.sum())
            if use_boxes:
                # Both triangles can move by bound. The checked runtime bound
                # makes disjoint expanded initial boxes impossible to intersect.
                margin = 2*motion_bound
                qids = query_ids+start
                keep &= ((sh[ids]+margin >= ql[qids]) & (sl[ids]-margin <= qh[qids])).all(1)
            if sf is not None:
                shared = (sf[ids, :, None] == qf[query_ids+start, None, :]).any(2).any(1)
                keep &= ~shared
            ids, query_ids = ids[keep], query_ids[keep]
            row_counts = torch.bincount(query_ids, minlength=stop-start).cpu().numpy()
            part = ids.to(torch.int32).cpu().numpy()
        else:
            row_counts, part = np.zeros(stop-start, np.int64), np.empty(0, np.int32)
        prefix_cpu = np.cumsum(row_counts, dtype=np.int64)+offsets[-1]
        offsets.extend(prefix_cpu.tolist())
        parts.append(part)
        info["retained_candidates"] += len(part)
        start = stop
    return np.asarray(offsets, np.int64), np.concatenate(parts), info
