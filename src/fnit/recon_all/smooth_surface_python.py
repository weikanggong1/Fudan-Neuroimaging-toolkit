"""Ordered vertex averaging for recon-all's ``mris_smooth -nw``."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np
from numba import njit

from .place_surface_normals import ordered_face_csr


@njit(cache=True, fastmath=False)
def _ordered_neighbor_storage(faces, offsets, face_ids, corners):
    """由成熟有序入射 CSR 按 previous→following 首次出现去重。"""
    counts = np.zeros(len(offsets)-1, np.int64)
    storage = np.empty(2*len(face_ids), np.int64)
    for vertex in range(len(counts)):
        begin = 2*offsets[vertex]
        count = 0
        for entry in range(offsets[vertex], offsets[vertex+1]):
            face = faces[face_ids[entry]]
            corner = corners[entry]
            for slot in range(2):
                neighbor = face[(corner+2-slot) % 3]
                present = False
                for previous in range(count):
                    if storage[begin+previous] == neighbor:
                        present = True
                        break
                if not present:
                    storage[begin+count] = neighbor
                    count += 1
        counts[vertex] = count
    return storage, counts


def ordered_neighbors(faces: np.ndarray, nvertices: int) -> list[list[int]]:
    """完整有序三角面→按原面/角点次序的一环邻接，每行独立可写。

    faces为整数(F,3)，nvertices为非负整数N，包含孤立顶点。返回N行
    list[int]：每个角点先previous再following，重复项只保留首次出现；
    不排序邻居、不删除自环/重复角点，也不从候选外部边重造动态邻接。
    复用ordered_face_csr的完整形状/范围校验及稳定面序；整数去重由
    Numba执行，首次JIT另计，无新依赖或精度策略。没有全局缓存，每次
    faces改变均重新构造。无坐标/单位/文件读写；非法类型、shape、N或
    索引抛ValueError。对应mrisCompleteTopology内部步骤，无独立原CLI。
    """
    offsets, face_ids, corners = ordered_face_csr(faces, nvertices)
    triangles = np.asarray(faces, dtype=np.int64)
    storage, counts = _ordered_neighbor_storage(triangles, offsets, face_ids, corners)
    return [storage[2*offsets[vertex]:2*offsets[vertex]+counts[vertex]].tolist()
            for vertex in range(nvertices)]


def average_positions(vertices: np.ndarray, faces: np.ndarray,
                      iterations: int = 10, device: str = "cpu") -> np.ndarray:
    xyz = np.asarray(vertices, dtype=np.float32).copy()
    neighbors = ordered_neighbors(faces, len(xyz))
    degrees = np.fromiter((len(row) for row in neighbors), dtype=np.int32,
                          count=len(neighbors))
    maximum = int(degrees.max(initial=0))
    indices = np.zeros((len(xyz), maximum), dtype=np.int32)
    for vertex, row in enumerate(neighbors):
        indices[vertex, :len(row)] = row
    if device.startswith("cuda"):
        import torch

        current = torch.as_tensor(xyz, device=device)
        ring = torch.as_tensor(indices, dtype=torch.int64, device=device)
        degree = torch.as_tensor(degrees, device=device)
        for _ in range(iterations):
            updated = current.clone()
            for slot in range(maximum):
                valid = degree > slot
                updated[valid] += current[ring[valid, slot]]
            current = updated / (degree[:, None] + 1)
        return current.cpu().numpy()
    if device != "cpu":
        raise ValueError(f"unsupported device: {device}")
    for _ in range(iterations):
        updated = xyz.copy()
        for slot in range(maximum):
            valid = degrees > slot
            updated[valid] += xyz[indices[valid, slot]]
        xyz = updated / (degrees[:, None] + 1).astype(np.float32)
    return xyz


def smooth_surface(input_path: str | Path, output_path: str | Path,
                   iterations: int = 10, device: str = "cpu") -> None:
    raw = Path(input_path).read_bytes()
    if raw[:3] != b"\xff\xff\xfe":
        raise ValueError("expected FreeSurfer triangular surface")
    start = raw.index(b"\n\n", 3) + 2
    nvertices = int.from_bytes(raw[start:start + 4], "big")
    nfaces = int.from_bytes(raw[start + 4:start + 8], "big")
    coordinates_start = start + 8
    coordinates_end = coordinates_start + 12 * nvertices
    faces_end = coordinates_end + 12 * nfaces
    if len(raw) < faces_end:
        raise ValueError("truncated triangular surface")
    vertices, faces = fsio.read_geometry(str(input_path))
    smoothed = average_positions(vertices, faces, iterations, device)
    with open(output_path, "wb") as stream:
        stream.write(b"\xff\xff\xfecreated by fnit\n\n")
        stream.write(raw[start:coordinates_start])
        stream.write(np.asarray(smoothed, dtype=">f4").tobytes())
        stream.write(raw[coordinates_end:])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args(argv)
    smooth_surface(args.input, args.output, iterations=args.iterations,
                   device=args.device)


if __name__ == "__main__":
    main()
