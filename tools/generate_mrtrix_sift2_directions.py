"""Generate the fixed MRtrix 026e850d SIFT2 direction grid asset."""

from __future__ import annotations

import argparse
import hashlib
import math
import re
from pathlib import Path

import numpy as np
from scipy.linalg import qr, solve_triangular
from scipy.spatial import ConvexHull
from scipy.special import sph_harm_y


PREDEFINED_SHA256 = "176bd8f40f35fd229633fdd90be28c36400f94ae105c38513ae7f96f8ed574ad"
SET_SHA256 = "609e092f82064eab70521d95c171d9f392d41153a95651e678eee6134b0585f1"


def _checked_text(path: Path, expected_hash: str) -> str:
    contents = path.read_bytes()
    if hashlib.sha256(contents).hexdigest() != expected_hash:
        raise ValueError(f"{path} does not match MRtrix commit 026e850d")
    return contents.decode()


def generate(predefined_cpp: Path, set_cpp: Path, output: Path) -> None:
    text = _checked_text(predefined_cpp, PREDEFINED_SHA256)
    _checked_text(set_cpp, SET_SHA256)
    match = re.search(r"tesselation_1281_data\[\]\s*=\s*\{([^}]+)\}", text, re.S)
    if match is None:
        raise ValueError("tesselation_1281_data is absent")
    angles = np.array([float(value) for value in match.group(1).split(",") if value.strip()]).reshape(1281, 2)
    azimuth, elevation = angles.T
    directions = np.column_stack(
        (np.cos(azimuth) * np.sin(elevation), np.sin(azimuth) * np.sin(elevation), np.cos(elevation))
    )

    # set.cpp creates a convex hull of each direction and its antipode, then
    # stores the undirected hull edges with the antipodal indices identified.
    hull = ConvexHull(np.concatenate((directions, -directions)))
    adjacent = [set() for _ in range(1281)]
    for triangle in hull.simplices:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            i, j = int(triangle[a] % 1281), int(triangle[b] % 1281)
            if i != j:
                adjacent[i].add(j)
                adjacent[j].add(i)
    indptr = np.zeros(1282, dtype=np.int32)
    for i, neighbors in enumerate(adjacent):
        indptr[i + 1] = indptr[i] + len(neighbors)
    neighbors = np.concatenate([np.array(sorted(row), dtype=np.int16) for row in adjacent])

    # fmls.cpp calibrates 1281 integration weights by Householder QR so that
    # the even real SH integral is correct through LforN(1281)+2 = 50.
    sh = np.empty((1281, 1326), dtype=np.float64)
    column = 0
    for degree in range(0, 51, 2):
        for order in range(-degree, degree + 1):
            value = sph_harm_y(degree, abs(order), elevation, azimuth)
            sh[:, column] = (
                math.sqrt(2.0) * value.imag if order < 0 else
                math.sqrt(2.0) * value.real if order > 0 else value.real
            )
            column += 1
    target = np.zeros(column)
    target[0] = 2.0 * math.sqrt(math.pi)
    q, r = qr(sh.T, mode="economic", pivoting=False)
    weights = solve_triangular(r, q.T @ target)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        directions=directions.astype(np.float64),
        adjacency_indptr=indptr,
        adjacency_indices=neighbors,
        integration_weights=weights.astype(np.float64),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predefined_cpp", type=Path)
    parser.add_argument("set_cpp", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    generate(args.predefined_cpp, args.set_cpp, args.output)
