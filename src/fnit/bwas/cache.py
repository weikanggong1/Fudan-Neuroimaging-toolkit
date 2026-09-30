"""Packed voxel-major BWAS cache shared by subject blocks."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np


def build_packed_cache(matrices: list[np.ndarray], directory: Path,
                       subject_block_size: int, voxel_chunk: int = 512) -> float:
    """Write [voxel, subject-in-block, time] shards, preserving subject order."""
    if directory.exists():
        raise FileExistsError(f"packed cache already exists: {directory}")
    directory.mkdir(mode=0o700, parents=True)
    nvox = matrices[0].shape[0]
    if any(matrix.ndim != 2 or matrix.shape[0] != nvox for matrix in matrices):
        raise ValueError("all subject caches must have the same voxel count")
    start = perf_counter()
    shards = []
    for st in range(0, len(matrices), subject_block_size):
        group = matrices[st:st+subject_block_size]
        lengths = [int(matrix.shape[1]) for matrix in group]
        file = f"subjects-{st:05d}.npy"
        packed = np.lib.format.open_memmap(
            directory / file, mode="w+", dtype=np.float32,
            shape=(nvox, len(group), max(lengths)),
        )
        for lo in range(0, nvox, voxel_chunk):
            hi = min(lo+voxel_chunk, nvox)
            block = np.zeros((hi-lo, len(group), max(lengths)), dtype=np.float32)
            for row, matrix in enumerate(group):
                block[:, row, :lengths[row]] = matrix[lo:hi]
            packed[lo:hi] = block
        packed.flush()
        del packed
        shards.append({"file": file, "lengths": lengths})
    manifest = {"subjects": len(matrices), "voxels": nvox,
                "subject_block_size": subject_block_size, "shards": shards}
    (directory / "manifest.json").write_text(json.dumps(manifest) + "\n")
    return perf_counter()-start


def open_packed_cache(directory: Path, subjects: int, voxels: int,
                      subject_block_size: int):
    """Return memory maps and scan lengths in original subject order."""
    manifest = json.loads((directory / "manifest.json").read_text())
    expected = (subjects, voxels, subject_block_size)
    found = tuple(manifest[key] for key in
                  ("subjects", "voxels", "subject_block_size"))
    if found != expected:
        raise ValueError(f"packed cache shape/block mismatch: {found} != {expected}")
    shards = []
    for entry in manifest["shards"]:
        lengths = np.asarray(entry["lengths"], dtype=np.int32)
        matrix = np.load(directory / entry["file"], mmap_mode="r")
        if matrix.shape != (voxels, len(lengths), int(lengths.max())):
            raise ValueError(f"invalid packed shard: {entry['file']}")
        shards.append((matrix, lengths))
    if sum(len(lengths) for _, lengths in shards) != subjects:
        raise ValueError("packed cache subject count mismatch")
    return shards
