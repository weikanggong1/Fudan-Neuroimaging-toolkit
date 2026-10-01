"""PyTorch implementation of the fixed FreeSurfer 8.2 surface thickness step."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import nibabel.freesurfer as fs
import numpy as np
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


def _adjacency(faces: np.ndarray, nvertices: int) -> list[list[int]]:
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.concatenate((edges, edges[:, ::-1]))
    graph = coo_matrix((np.ones(len(edges), dtype=bool), (edges[:, 0], edges[:, 1])),
                       shape=(nvertices, nvertices)).tocsr()
    graph.sort_indices()
    return [graph.indices[graph.indptr[v]:graph.indptr[v + 1]].tolist()
            for v in range(nvertices)]


def _nearest_reachable(vertex: int, candidates: list[int], graph: list[list[int]]) -> int | None:
    if not candidates:
        return None
    if candidates[0] == vertex:
        return 0
    seen = {vertex}
    frontier = [vertex]
    for _ in range(20):
        nxt = []
        for v in frontier:
            for neighbour in graph[v]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    nxt.append(neighbour)
        if candidates[0] in seen:
            return 0
        frontier = nxt
        if not frontier:
            break
    return next((i for i, v in enumerate(candidates) if v in seen), None)


def _knn(query: torch.Tensor, target: torch.Tensor, count: int,
         tree: cKDTree | None) -> torch.Tensor:
    if tree is None:
        distances = torch.cdist(query, target, compute_mode="use_mm_for_euclid_dist")
        return distances.topk(count, largest=False, sorted=True).indices
    _, indices = tree.query(query.cpu().numpy(), k=count, workers=4)
    if count == 1:
        indices = indices[:, None]
    return torch.as_tensor(indices, dtype=torch.long, device=query.device)


def _candidates(query: torch.Tensor, target: torch.Tensor, normal: torch.Tensor,
                base_normal: torch.Tensor, source: torch.Tensor, direct: torch.Tensor,
                count: int, tree: cKDTree | None,
                reverse: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = _knn(query, target, count, tree)
    delta = source[:, None, :] - target[indices] if reverse else target[indices] - source[:, None, :]
    distance = torch.linalg.vector_norm(delta, dim=2)
    accepted = ((delta * base_normal[:, None, :]).sum(2) >= 0) & \
               ((normal[indices] * base_normal[:, None, :]).sum(2) >= 0) & \
               (distance < direct[:, None])
    order = distance.argsort(dim=1)
    return (indices.gather(1, order).cpu().numpy(),
            accepted.gather(1, order).cpu().numpy(),
            distance.gather(1, order).cpu().numpy())


def thickness_map(white_file: str | Path, pial_file: str | Path,
                  output_file: str | Path, *, device: str = "cuda:0") -> dict:
    """Write thickness using FreeSurfer's 20-hop rule.

    CUDA computes normals, nearest-neighbour distances and orientation tests.
    The sparse graph reachability check runs in Python on CPU. Candidate searches
    expand beyond 256 when a closer eligible vertex may remain unseen.
    """
    start = time.perf_counter()
    white_np, white_faces = fs.read_geometry(str(white_file))
    pial_np, pial_faces = fs.read_geometry(str(pial_file))
    if not np.array_equal(white_faces, pial_faces) or not np.isfinite(white_np).all() \
            or not np.isfinite(pial_np).all():
        raise ValueError("white and pial must have finite vertices and identical topology")
    white = torch.as_tensor(white_np, dtype=torch.float32, device=device)
    pial = torch.as_tensor(pial_np, dtype=torch.float32, device=device)
    faces = torch.as_tensor(white_faces.astype(np.int64), device=device)
    normal = _normals(pial, faces)
    graph = _adjacency(white_faces, len(white_np))
    tree_pial = cKDTree(pial_np) if white.device.type == "cpu" else None
    tree_white = cKDTree(white_np) if white.device.type == "cpu" else None
    setup_seconds = time.perf_counter() - start
    result = np.empty(len(white_np), dtype=np.float32)
    count = min(256, len(white_np))
    expanded_searches = 0
    for first in range(0, len(white_np), 1024):
        stop = min(first + 1024, len(white_np))
        w, p, n = white[first:stop], pial[first:stop], normal[first:stop]
        direct = torch.linalg.vector_norm(p - w, dim=1)
        ai, ak, ad = _candidates(w, pial, normal, n, w, direct, count, tree_pial)
        bi, bk, bd = _candidates(p, white, normal, n, p, direct, count, tree_white, reverse=True)
        direct_np = direct.cpu().numpy()
        for row, vertex in enumerate(range(first, stop)):
            def best_distance(indices, accepted, distances, query, target, tree, reverse):
                nonlocal expanded_searches
                search_count = count
                while True:
                    selected = _nearest_reachable(vertex, indices[accepted].tolist(), graph)
                    best = direct_np[row] if selected is None else distances[accepted][selected]
                    if search_count == len(white_np) or distances[-1] > min(best, 5.0) + 1e-4:
                        return best
                    search_count = min(search_count * 2, len(white_np))
                    expanded_searches += 1
                    more_indices, more_accepted, more_distances = _candidates(
                        query[row:row + 1], target, normal, n[row:row + 1],
                        query[row:row + 1], direct[row:row + 1], search_count,
                        tree, reverse=reverse)
                    indices, accepted, distances = (more_indices[0], more_accepted[0],
                                                    more_distances[0])

            da = best_distance(ai[row], ak[row], ad[row], w, pial, tree_pial, False)
            db = best_distance(bi[row], bk[row], bd[row], p, white, tree_white, True)
            result[vertex] = (min(da, 5.0) + min(db, 5.0)) / 2
    compute_seconds = time.perf_counter() - start - setup_seconds
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    fs.write_morph_data(str(output), result)
    return {"vertices": len(white_np), "device": str(white.device),
            "setup_seconds": setup_seconds, "compute_seconds": compute_seconds,
            "total_seconds": time.perf_counter() - start,
            "expanded_searches": expanded_searches}


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
