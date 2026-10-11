"""Pre-unfold projection and sampled metrics for FreeSurfer 8.2 ``mris_sphere``.

The default recon-all command reads ``inflated`` and the matching ``smoothwm``
metric surface. This module implements projection, surface output, and
single-vertex metric constraints; it does not implement ``MRISunfold`` or
fold/overlap removal.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np

from .sphere_python import initial_scale, project_radially
from .smooth_surface_python import ordered_neighbors


def project_before_standard_unfold(vertices: np.ndarray, *,
                                   bbox_mode: str = "source_float") -> np.ndarray:
    """返回标准球面优化前float32(N,3)有序坐标。

    vertices为非空有限(N,3) surface RAS/mm。显式复用initial_scale的
    bbox_mode='source_float'默认使用固定Conda编译单元的float跨度；
    'legacy_integer'仅供历史截断规则对照。随后居中并投影到100mm球面。
    返回不改顶点顺序的NumPy float32数组，不读写文件。非法shape/非有限
    输入抛ValueError；原点投影沿已有project_radially零半径处理。
    属于mris_sphere内部初始化，无独立CLI；真实相位检查点见说明页。
    两组完整同输入逐步坐标与固定Conda参考一致；参考自产网格仍有
    残余翻折，调用者必须另行检查质量，不能由一致性推断质量通过。
    旧quick球面保留initial_scale默认legacy_integer，不受本调用影响。
    """
    return project_radially(initial_scale(vertices, bbox_mode=bbox_mode))



def write_standard_sphere_surface(output: str | Path, vertices: np.ndarray,
                                  faces: np.ndarray, source_surface: str | Path,
                                  *, create_stamp: str = "created by Python conventional sphere") -> None:
    """Write sphere coordinates while preserving input volume geometry bytes.

    FreeSurfer stores volume geometry as text after the triangle arrays.
    Nibabel's formatter rounds that text, so retain the original block.
    Later FreeSurfer provenance tags are deliberately not copied.
    """
    with Path(source_surface).open("rb") as stream:
        if stream.read(3) != b"\xff\xff\xfe":
            raise ValueError("source surface is not a triangle surface")
        stream.readline()
        stream.readline()
        nvertices, nfaces = struct.unpack(">ii", stream.read(8))
        if nvertices != len(vertices) or nfaces != len(faces):
            raise ValueError("source surface topology differs")
        stream.seek(12 * (nvertices + nfaces), 1)
        footer = stream.read()
    if footer:
        cras = footer.find(b"cras   =")
        if cras < 0:
            raise ValueError("source surface has an incomplete volume geometry block")
        footer = footer[:footer.index(b"\n", cras) + 1]
    fsio.write_geometry(str(output), np.asarray(vertices, np.float32),
                        np.asarray(faces, np.int32),
                        create_stamp=create_stamp)
    with Path(output).open("ab") as stream:
        stream.write(footer)


def original_metric_distance_rings(vertices: np.ndarray, faces: np.ndarray,
                                   vertex: int, max_ring: int = 7,
                                   neighbors: list[list[int]] | None = None
                                   ) -> tuple[list[list[int]], list[np.ndarray]]:
    """Source-order candidate rings and metric distances before sparse sampling.

    ``MRISsampleDistances`` measures on ``smoothwm``. Its first two rings are
    seeded with direct Euclidean distance; outer rings use corrected shortest
    paths. Each ring receives one same-ring relaxation. Random selection and
    pairwise averaging occur afterward.
    """
    xyz = np.asarray(vertices, dtype=np.float32)
    if neighbors is None:
        neighbors = ordered_neighbors(faces, len(xyz))
    marked = {vertex: 1}
    distance = {vertex: np.float32(0)}
    rings = [[vertex]]
    values: list[np.ndarray] = []

    def euclidean(first: int, second: int) -> np.float32:
        delta = xyz[first] - xyz[second]
        squared = np.float32(delta[0] * delta[0] + delta[1] * delta[1])
        squared = np.float32(squared + delta[2] * delta[2])
        return np.float32(np.sqrt(squared))

    for depth in range(1, max_ring + 1):
        ring = []
        for prior in rings[-1]:
            for other in neighbors[prior]:
                if other not in marked:
                    marked[other] = depth
                    ring.append(other)
        for current in ring:
            best = np.float32(1e10)
            for other in neighbors[current]:
                if other not in marked or marked[other] == depth:
                    continue
                step = euclidean(current, other)
                if depth > 1:
                    step = np.float32(step / np.float32(1.09))
                candidate = np.float32(step + distance[other])
                if candidate < best:
                    best = candidate
            distance[current] = euclidean(current, vertex) if depth <= 2 else best
        for current in ring:
            best = distance[current]
            for other in neighbors[current]:
                if marked.get(other) != depth:
                    continue
                step = euclidean(current, other)
                if depth > 1:
                    step = np.float32(step / np.float32(1.09))
                candidate = np.float32(distance[other] + step)
                if candidate < best:
                    best = candidate
            distance[current] = best
        rings.append(ring)
        values.append(np.asarray([distance[current] for current in ring],
                                 dtype=np.float32))
    return rings[1:], values


class FreeSurferSphereRandom:
    """FreeSurfer's seeded VNL stream, including ``setRandomSeed``'s first draw."""

    def __init__(self, seed: int):
        self.position = 0
        self.borrow = 0
        current = seed
        self.values = []
        for _ in range(37):
            current = (current * 1664525 + 1) & 0xffffffff
            self.values.append(current)
        for _ in range(1000):
            self._lrand32()
        self.random_number(1)

    def _lrand32(self) -> int:
        p1 = self.values[(37 + self.position - 24) % 37]
        result = (p1 - self.values[self.position] - self.borrow) & 0xffffffff
        if result < p1:
            self.borrow = 0
        if result > p1:
            self.borrow = 1
        self.values[self.position] = result
        self.position = (self.position + 1) % 37
        return result

    def random_number(self, maximum: int) -> float:
        first = self._lrand32()
        second = self._lrand32()
        value = first / 0xffffffff + second / (0xffffffff * 0xffffffff)
        return float(np.float32(value)) * maximum


def sample_standard_metric_neighbors(vertices: np.ndarray, faces: np.ndarray,
                                     vertex: int, rng: FreeSurferSphereRandom,
                                     neighbors: list[list[int]] | None = None
                                     ) -> tuple[list[int], np.ndarray]:
    """Return a vertex's pre-symmetry three-ring and sampled 3–7-ring distances.

    The caller must pass the same ``rng`` while iterating vertices in ascending
    order. The first three rings are kept in full; rings 3–7 add up to eight
    sampled points each, including repeats from the complete third ring.
    """
    xyz = np.asarray(vertices, np.float32)
    rings, values = original_metric_distance_rings(xyz, faces, vertex,
                                                    neighbors=neighbors)
    selected = [other for ring in rings[:3] for other in ring]
    distances = []
    for other in selected:
        delta = xyz[vertex] - xyz[other]
        squared = np.float32(delta[0] * delta[0] + delta[1] * delta[1])
        distances.append(np.float32(np.sqrt(np.float32(
            squared + delta[2] * delta[2]))))
    for ring, raw in zip(rings[2:], values[2:]):
        if len(ring) <= 8:
            selected.extend(ring)
            distances.extend(raw)
            continue
        available = ring.copy()
        previous = []
        min_angle = np.float32(.9 * 2 * math.pi / 8)
        for _ in range(8):
            attempts = 0
            while True:
                index = int(rng.random_number(len(ring) - 1) + .5)
                if available[index] < 0:
                    continue
                other = available[index]
                vector = xyz[other] - xyz[vertex]
                accepted = True
                for prior_id in previous:
                    prior = xyz[prior_id] - xyz[vertex]
                    length = math.sqrt(sum(float(x) * float(x) for x in vector))
                    prior_length = math.sqrt(sum(float(x) * float(x) for x in prior))
                    dot = np.float32(vector[0] * prior[0] + vector[1] * prior[1]
                                     + vector[2] * prior[2])
                    normalizer = max(length * prior_length, abs(float(dot)))
                    angle = (np.float32(math.acos(float(dot) / normalizer))
                             if normalizer else np.float32(0))
                    if angle < min_angle:
                        accepted = False
                        break
                attempts += 1
                if attempts > len(ring):
                    min_angle = np.float32(min_angle * np.float32(.75))
                    attempts = 0
                if accepted or min_angle == 0:
                    selected.append(other)
                    distances.append(raw[index])
                    previous.append(other)
                    available[index] = -1
                    break
    return selected, np.asarray(distances, dtype=np.float32)
