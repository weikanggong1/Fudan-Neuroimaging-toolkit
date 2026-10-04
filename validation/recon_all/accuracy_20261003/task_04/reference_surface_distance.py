"""定位真实 T1 表面链的首次网格差异；拓扑不同则计算双向点到三角面距离。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


STAGES = ("orig.nofix", "orig.premesh", "orig", "white.preaparc",
          "white", "pial", "sphere", "sphere.reg")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quality(vertices: np.ndarray, faces: np.ndarray, sphere: bool) -> dict:
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]],
                            faces[:, [2, 0]]))
    edges.sort(axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    graph = coo_matrix((np.ones(len(unique) * 2, dtype=np.uint8),
                        (np.r_[unique[:, 0], unique[:, 1]],
                         np.r_[unique[:, 1], unique[:, 0]])),
                       shape=(len(vertices), len(vertices))).tocsr()
    result = {"components": int(connected_components(graph, directed=False)[0]),
              "boundary_edges": int(np.count_nonzero(counts == 1)),
              "nonmanifold_edges": int(np.count_nonzero(counts > 2)),
              "euler": int(len(vertices) - len(unique) + len(faces))}
    if sphere:
        triangles = vertices[faces]
        normal = np.cross(triangles[:, 1] - triangles[:, 0],
                          triangles[:, 2] - triangles[:, 0])
        radial = triangles.mean(axis=1) - vertices.mean(axis=0)
        signed = np.einsum("ij,ij->i", normal, radial)
        result["inward_or_degenerate_faces"] = int(np.count_nonzero(signed <= 0))
    return result


def _point_triangle_distance(point: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """返回一个 surface RAS 点到每个闭合三角面的精确欧氏距离，单位 mm。"""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac, bc = b - a, c - a, c - b
    ap = point - a
    normal = np.cross(ab, ac)
    normal_sq = np.einsum("ij,ij->i", normal, normal)
    plane_dot = np.einsum("ij,ij->i", ap, normal)
    projection = point - normal * (plane_dot / np.maximum(normal_sq, 1e-30))[:, None]
    projected = projection - a
    d00 = np.einsum("ij,ij->i", ab, ab)
    d01 = np.einsum("ij,ij->i", ab, ac)
    d11 = np.einsum("ij,ij->i", ac, ac)
    d20 = np.einsum("ij,ij->i", projected, ab)
    d21 = np.einsum("ij,ij->i", projected, ac)
    denominator = d00 * d11 - d01 * d01
    u = (d11 * d20 - d01 * d21) / np.maximum(denominator, 1e-30)
    v = (d00 * d21 - d01 * d20) / np.maximum(denominator, 1e-30)
    inside = (normal_sq > 1e-20) & (u >= -1e-12) & (v >= -1e-12) & (u + v <= 1 + 1e-12)

    def segment(start: np.ndarray, vector: np.ndarray) -> np.ndarray:
        fraction = np.einsum("ij,ij->i", point - start, vector)
        fraction /= np.maximum(np.einsum("ij,ij->i", vector, vector), 1e-30)
        closest = start + np.clip(fraction, 0, 1)[:, None] * vector
        return np.linalg.norm(point - closest, axis=1)

    boundary = np.minimum.reduce((segment(a, ab), segment(b, bc),
                                  segment(c, a - c)))
    plane = np.abs(plane_dot) / np.sqrt(np.maximum(normal_sq, 1e-30))
    return np.where(inside, plane, boundary)


def _point_to_mesh(source: np.ndarray, target: np.ndarray,
                   faces: np.ndarray) -> np.ndarray:
    """利用顶点上界和三角面外接半径筛选，仍逐点精确求全网格最近三角面。"""
    triangles = target[faces].astype(np.float64)
    centers = triangles.mean(axis=1)
    radius = float(np.linalg.norm(triangles - centers[:, None], axis=2).max())
    used_vertices = target[np.unique(faces)]
    vertex_tree, face_tree = cKDTree(used_vertices), cKDTree(centers)
    distances = np.empty(len(source), dtype=np.float64)
    for start in range(0, len(source), 1024):
        points = source[start:start + 1024].astype(np.float64)
        upper = vertex_tree.query(points, workers=4)[0]
        nearby = face_tree.query_ball_point(points, upper + radius + 1e-7,
                                            workers=4)
        for offset, ids in enumerate(nearby):
            if not ids:
                raise ValueError("point-to-triangle search found no candidate face")
            distances[start + offset] = _point_triangle_distance(
                points[offset], triangles[ids]).min()
    return distances


def _summary(values: np.ndarray) -> dict:
    return {"mean_mm": float(values.mean()), "p99_mm": float(np.quantile(values, .99)),
            "max_mm": float(values.max()), "over_0_1_mm": int(np.count_nonzero(values > .1))}


def compare(reference: Path, candidate: Path) -> dict:
    result = {"reference": str(reference), "candidate": str(candidate),
              "unit": "surface RAS mm", "stages": {}}
    for hemi in ("lh", "rh"):
        result["stages"][hemi] = {}
        for stage in STAGES:
            left, right = (root / "surf" / f"{hemi}.{stage}"
                           for root in (reference, candidate))
            if not left.is_file() or not right.is_file():
                result["stages"][hemi][stage] = {
                    "reference_exists": left.is_file(), "candidate_exists": right.is_file()}
                continue
            a, fa = fsio.read_geometry(str(left))
            b, fb = fsio.read_geometry(str(right))
            aligned = fa.shape == fb.shape and np.array_equal(fa, fb)
            row = {"reference_sha256": _sha256(left), "candidate_sha256": _sha256(right),
                   "reference_vertices": len(a), "candidate_vertices": len(b),
                   "reference_faces": len(fa), "candidate_faces": len(fb),
                   "ordered_faces_equal": aligned,
                   "reference_quality": _quality(a, fa, stage.startswith("sphere")),
                   "candidate_quality": _quality(b, fb, stage.startswith("sphere"))}
            if aligned and a.shape == b.shape:
                row["indexed_vertex_distance"] = _summary(np.linalg.norm(a - b, axis=1))
            else:
                row["indexed_vertex_distance"] = None
                row["candidate_to_reference_triangle"] = _summary(_point_to_mesh(b, a, fa))
                row["reference_to_candidate_triangle"] = _summary(_point_to_mesh(a, b, fb))
            result["stages"][hemi][stage] = row
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.reference, args.candidate)
    report["code_commit"] = args.code_commit
    report["comparator_sha256"] = _sha256(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
