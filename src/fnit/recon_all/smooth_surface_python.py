"""Ordered vertex averaging for recon-all's ``mris_smooth -nw``."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np


def ordered_neighbors(faces: np.ndarray, nvertices: int) -> list[list[int]]:
    """Recreate mrisCompleteTopology_old's face-order one-ring lists."""
    neighbors: list[list[int]] = [[] for _ in range(nvertices)]
    for a, b, c in faces:
        for vertex, previous, following in ((a, c, b), (b, a, c), (c, b, a)):
            linked = neighbors[vertex]
            if previous not in linked:
                linked.append(int(previous))
            if following not in linked:
                linked.append(int(following))
    return neighbors


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
