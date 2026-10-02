"""PyTorch implementation of the fixed FreeSurfer 8.2 surface thickness step."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import nibabel.freesurfer as fs
import numpy as np
from numba import njit
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
import torch


def _normals(pial: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(pial)
    for corner in range(3):
        index = faces[:, corner]
        edge0 = pial[index] - pial[faces[:, (corner - 1) % 3]]
        edge1 = pial[faces[:, (corner + 1) % 3]] - pial[index]
        edge0 = torch.nn.functional.normalize(edge0, dim=1)
        edge1 = torch.nn.functional.normalize(edge1, dim=1)
        face_normal = torch.nn.functional.normalize(torch.cross(edge0, edge1, dim=1), dim=1)
        result.index_add_(0, index, face_normal)
    return torch.nn.functional.normalize(result, dim=1)


def _adjacency_csr(faces: np.ndarray, nvertices: int):
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.concatenate((edges, edges[:, ::-1]))
    graph = coo_matrix((np.ones(len(edges), dtype=bool), (edges[:, 0], edges[:, 1])),
                       shape=(nvertices, nvertices)).tocsr()
    graph.sort_indices()
    return graph


@njit(cache=True)
def _reachable_distances(first, direct, graph_offsets, graph_indices,
                         a_offsets, a_indices, a_accepted, a_distances,
                         b_offsets, b_indices, b_accepted, b_distances,
                         seen=None, queue=None):
    """一次 BFS 选择两个方向的最近合法候选；保留含自身的 20 跳范围。"""
    if seen is None:
        seen = np.zeros(len(graph_offsets) - 1, dtype=np.int32)
    if queue is None:
        queue = np.empty(len(seen), dtype=np.int32)
    result = np.empty((len(direct), 2), dtype=np.float32)
    for row in range(len(direct)):
        vertex = first + row
        stamp = first + row + 1
        a_nearest, b_nearest = -1, -1
        a_best, b_best = direct[row], direct[row]
        for i in range(a_offsets[row], a_offsets[row + 1]):
            if a_accepted[i] and a_distances[i] < a_best:
                a_nearest, a_best = a_indices[i], a_distances[i]
        for i in range(b_offsets[row], b_offsets[row + 1]):
            if b_accepted[i] and b_distances[i] < b_best:
                b_nearest, b_best = b_indices[i], b_distances[i]
        seen[vertex] = stamp
        queue[0] = vertex
        head, tail = 0, 1
        for _ in range(20):
            if (a_nearest < 0 or seen[a_nearest] == stamp) and \
                    (b_nearest < 0 or seen[b_nearest] == stamp):
                break
            end = tail
            while head < end:
                current = queue[head]
                head += 1
                for i in range(graph_offsets[current], graph_offsets[current + 1]):
                    neighbor = graph_indices[i]
                    if seen[neighbor] != stamp:
                        seen[neighbor] = stamp
                        queue[tail] = neighbor
                        tail += 1
            if tail == end:
                break
        a_best, b_best = direct[row], direct[row]
        for i in range(a_offsets[row], a_offsets[row + 1]):
            if a_accepted[i] and seen[a_indices[i]] == stamp and a_distances[i] < a_best:
                a_best = a_distances[i]
        for i in range(b_offsets[row], b_offsets[row + 1]):
            if b_accepted[i] and seen[b_indices[i]] == stamp and b_distances[i] < b_best:
                b_best = b_distances[i]
        result[row, 0], result[row, 1] = a_best, b_best
    return result


@njit(cache=True)
def _clip_mean(distances):
    """保留旧函数 float32 求和及截断后 float64 求和的转换顺序。"""
    result = np.empty(len(distances), dtype=np.float32)
    for row in range(len(distances)):
        a, b = distances[row, 0], distances[row, 1]
        if a <= 5.0 and b <= 5.0:
            result[row] = np.float32(a + b) / 2.0
        else:
            result[row] = (min(float(a), 5.0) + min(float(b), 5.0)) / 2.0
    return result


def _radius_candidates(query_np, source, target, normal, base_normal,
                       direct, tree, *, reverse=False):
    """完整空间候选；CPU 查询最多使用调用者 Torch 线程预算内的四个线程。"""
    radius = np.minimum(direct.cpu().numpy().astype(np.float64), 5.0) + 1e-4
    nearby = tree.query_ball_point(query_np, radius,
                                   workers=min(4, torch.get_num_threads()),
                                   return_sorted=True)
    counts = np.fromiter((len(row) for row in nearby), dtype=np.int64,
                         count=len(nearby))
    offsets = np.empty(len(counts) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    indices = np.fromiter((index for row in nearby for index in row),
                          dtype=np.int32, count=int(offsets[-1]))
    rows = np.repeat(np.arange(len(counts)), counts)
    index_gpu = torch.as_tensor(indices.astype(np.int64), device=source.device)
    row_gpu = torch.as_tensor(rows, device=source.device)
    delta = source[row_gpu] - target[index_gpu] if reverse else target[index_gpu] - source[row_gpu]
    distance = torch.linalg.vector_norm(delta, dim=1)
    accepted = ((delta * base_normal[row_gpu]).sum(1) >= 0) & \
               ((normal[index_gpu] * base_normal[row_gpu]).sum(1) >= 0) & \
               (distance < direct[row_gpu])
    return offsets, indices, accepted.cpu().numpy(), distance.cpu().numpy()


@torch.no_grad()
def thickness_map(white_file: str | Path, pial_file: str | Path,
                  output_file: str | Path, *, device: str = "cuda:0") -> dict:
    """完整空间候选和编译可达性计算厚度，供 recon-all 默认调用。

    white_file、pial_file 是相同顶点顺序和有序面的三角表面，坐标为
    surface RAS、单位 mm；output_file 写同顺序 float32 morph 厚度（mm）。
    device 默认 cuda:0，也支持 cpu。每方向保留法向条件、20 跳和 5 mm
    截断，完整查询能影响截断后输出的候选，没有固定最近点数量上限。
    CPU 空间查询使用 min(4, torch.get_num_threads()) 个线程；调用期间
    保持调用者线程设置不变。不改变 TF32 设置或启用半精度。返回顶点数、
    分步/总秒数、候选数和实际空间查询线程预算 kdtree_workers。
    输入不匹配、非有限值、I/O 或设备错误抛异常，不生成近似结果。
    官方命令：mris_place_surface --thickness white pial 20 5 thickness。
    """
    gpu = torch.device(device).type == "cuda"
    if gpu:
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    kdtree_workers = min(4, torch.get_num_threads())
    white_np, white_faces = fs.read_geometry(str(white_file))
    pial_np, pial_faces = fs.read_geometry(str(pial_file))
    if white_np.shape != pial_np.shape or not np.array_equal(white_faces, pial_faces) \
            or not np.isfinite(white_np).all() or not np.isfinite(pial_np).all():
        raise ValueError("white and pial must have finite vertices and identical topology")
    white = torch.as_tensor(white_np, dtype=torch.float32, device=device)
    pial = torch.as_tensor(pial_np, dtype=torch.float32, device=device)
    faces = torch.as_tensor(white_faces.astype(np.int64), device=device)
    normal = _normals(pial, faces)
    graph = _adjacency_csr(white_faces, len(white_np))
    tree_pial, tree_white = cKDTree(pial_np), cKDTree(white_np)
    if gpu:
        torch.cuda.synchronize(device)
    setup_seconds = time.perf_counter() - start
    result = np.empty(len(white_np), dtype=np.float32)
    seen = np.zeros(len(white_np), dtype=np.int32)
    queue = np.empty(len(white_np), dtype=np.int32)
    candidate_pairs, peak_candidate_pairs = 0, 0
    for first in range(0, len(white_np), 1024):
        stop = min(first + 1024, len(white_np))
        w, p, n = white[first:stop], pial[first:stop], normal[first:stop]
        direct = torch.linalg.vector_norm(p - w, dim=1)
        a = _radius_candidates(white_np[first:stop], w, pial, normal, n,
                               direct, tree_pial)
        b = _radius_candidates(pial_np[first:stop], p, white, normal, n,
                               direct, tree_white, reverse=True)
        pairs = len(a[1]) + len(b[1])
        candidate_pairs += pairs
        peak_candidate_pairs = max(peak_candidate_pairs, pairs)
        distances = _reachable_distances(first, direct.cpu().numpy(),
                                         graph.indptr, graph.indices, *a, *b, seen, queue)
        result[first:stop] = _clip_mean(distances)
    if gpu:
        torch.cuda.synchronize(device)
    compute_seconds = time.perf_counter() - start - setup_seconds
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    fs.write_morph_data(str(output), result)
    return {"vertices": len(white_np), "device": str(white.device),
            "setup_seconds": setup_seconds, "compute_seconds": compute_seconds,
            "total_seconds": time.perf_counter() - start,
            "candidate_pairs": candidate_pairs, "peak_candidate_pairs": peak_candidate_pairs,
            "kdtree_workers": kdtree_workers,
            "candidate_search": "complete-radius", "radius_guard_mm": 1e-4,
            "maximum_hops": 20, "maximum_direction_thickness_mm": 5.0}


# 冻结阶段脚本继续使用此名称；生产与诊断共享同一个算法函数。
thickness_map_indexed = thickness_map


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("white")
    parser.add_argument("pial")
    parser.add_argument("output")
    args = parser.parse_args(argv)
    print(thickness_map(args.white, args.pial, args.output, device=args.device))


if __name__ == "__main__":
    main()
