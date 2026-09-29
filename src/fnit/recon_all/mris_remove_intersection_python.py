"""检测表面自相交；非零相交交由 Conda 源码构建的迭代修复程序处理。"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import nibabel.freesurfer as fs
import numpy as np
from scipy.spatial import cKDTree


def _interval(p: np.ndarray, d: np.ndarray) -> tuple[float, float] | None:
    """Project a triangle's intersection with a plane onto one coordinate."""
    if d[0] * d[1] > 0:
        order = (2, 0, 1)
    elif d[0] * d[2] > 0:
        order = (1, 0, 2)
    elif d[1] * d[2] > 0 or d[0] != 0:
        order = (0, 1, 2)
    elif d[1] != 0:
        order = (1, 0, 2)
    elif d[2] != 0:
        order = (2, 0, 1)
    else:
        return None
    a, b, c = order
    x = p[a] + (p[b] - p[a]) * d[a] / (d[a] - d[b])
    y = p[a] + (p[c] - p[a]) * d[a] / (d[a] - d[c])
    return (min(x, y), max(x, y))


def _coplanar_overlap(v: np.ndarray, u: np.ndarray, normal: np.ndarray) -> bool:
    axis = int(np.argmax(np.abs(normal)))
    v = np.delete(v, axis, axis=1)
    u = np.delete(u, axis, axis=1)

    def orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    for tri_a, tri_b in ((v, u), (u, v)):
        for a in range(3):
            p, q = tri_a[a], tri_a[(a + 1) % 3]
            for b in range(3):
                r, s = tri_b[b], tri_b[(b + 1) % 3]
                o1, o2 = orient(p, q, r), orient(p, q, s)
                o3, o4 = orient(r, s, p), orient(r, s, q)
                if o1 * o2 < 0 and o3 * o4 < 0:
                    return True
                for x, y, point, turn in ((p, q, r, o1), (p, q, s, o2),
                                          (r, s, p, o3), (r, s, q, o4)):
                    if turn == 0 and np.all(point >= np.minimum(x, y)) and np.all(point <= np.maximum(x, y)):
                        return True
        signs = [orient(tri_b[i], tri_b[(i + 1) % 3], tri_a[0])
                 for i in range(3)]
        if min(signs) >= 0 or max(signs) <= 0:
            return True
    return False


def _triangles_intersect(v: np.ndarray, u: np.ndarray) -> bool:
    """Triangle-plane interval test, using the native 1e-6 plane tolerance."""
    n1 = np.cross(v[1] - v[0], v[2] - v[0])
    du = (u - v[0]) @ n1
    abs_du = np.abs(du)
    if np.any(abs_du > 1e-6) and du[0] * du[1] > 0 and du[0] * du[2] > 0:
        return False
    du[abs_du < 1e-6] = 0.0
    if abs_du[0] < 1e-5 and abs_du[1] < 1e-5 and abs_du[2] < 1e-6:
        du[:] = 0.0

    n2 = np.cross(u[1] - u[0], u[2] - u[0])
    dv = (v - u[0]) @ n2
    abs_dv = np.abs(dv)
    if np.any(abs_dv > 1e-6) and dv[0] * dv[1] > 0 and dv[0] * dv[2] > 0:
        return False
    dv[abs_dv < 1e-6] = 0.0
    if abs_dv[0] < 1e-5 and abs_dv[1] < 1e-5 and abs_dv[2] < 1e-6:
        dv[:] = 0.0

    axis = int(np.argmax(np.abs(np.cross(n1, n2))))
    iv = _interval(v[:, axis], dv)
    iu = _interval(u[:, axis], du)
    if iv is None or iu is None:
        return _coplanar_overlap(v, u, n1)
    return iv[0] <= iu[1] and iu[0] <= iv[1]


def mark_intersections(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, int]:
    """Return native-style marked vertices and intersecting face count.

    The bounding-sphere and axis-aligned box filters cannot discard a true
    triangle intersection. Faces sharing a vertex are excluded as in FreeSurfer.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    triangles = vertices[faces]
    marks = np.zeros(len(vertices), dtype=bool)
    if len(faces) < 2:
        return marks, 0
    centers = triangles.mean(axis=1)
    radii = np.linalg.norm(triangles - centers[:, None], axis=2).max(axis=1)
    pairs = cKDTree(centers).query_pairs(2 * float(radii.max()) + 1e-5,
                                                output_type="ndarray")
    if len(pairs) == 0:
        return marks, 0
    a, b = pairs.T
    near = np.sum((centers[a] - centers[b]) ** 2, axis=1) <= (radii[a] + radii[b] + 1e-5) ** 2
    pairs = pairs[near]
    a, b = pairs.T
    low, high = triangles.min(axis=1), triangles.max(axis=1)
    overlap = np.all(low[a] <= high[b] + 1e-5, axis=1) & np.all(low[b] <= high[a] + 1e-5, axis=1)
    pairs = pairs[overlap]
    a, b = pairs.T
    disjoint = np.all(faces[a, :, None] != faces[b, None, :], axis=(1, 2))
    intersecting_faces = np.zeros(len(faces), dtype=bool)
    for fa, fb in pairs[disjoint]:
        if _triangles_intersect(triangles[fa], triangles[fb]):
            intersecting_faces[fa] = intersecting_faces[fb] = True
    marks[faces[intersecting_faces].ravel()] = True
    return marks, int(intersecting_faces.sum())


def remove_intersection_surface(input_path: str | Path, output_path: str | Path,
                                *, binary: str | Path | None = None,
                                assets_dir: str | Path | None = None) -> tuple[int, int]:
    """写出修复表面；返回输入相交面数和受影响顶点数。"""
    vertices, faces = fs.read_geometry(str(input_path))
    marks, count = mark_intersections(vertices, faces)
    if count:
        executable = Path(binary or Path(sys.prefix) / "bin/mris_remove_intersection")
        if not executable.is_file():
            raise FileNotFoundError(f"intersection repair binary not found: {executable}")
        output = Path(output_path)
        temporary = output.with_name(output.name + ".fnit-intersection")
        try:
            env = dict(os.environ)
            if assets_dir is not None:
                env["FREESURFER_HOME"] = str(Path(assets_dir).resolve())
            subprocess.run([str(executable), str(input_path), str(temporary)],
                           env=env, check=True)
            corrected, corrected_faces = fs.read_geometry(str(temporary))
            _, remaining = mark_intersections(corrected, corrected_faces)
            if remaining:
                raise RuntimeError(f"intersection repair left {remaining} intersecting faces")
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
        return count, int(marks.sum())
    if Path(input_path).resolve() != Path(output_path).resolve():
        shutil.copyfile(input_path, output_path)
    return count, int(marks.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--binary", type=Path)
    args = parser.parse_args()
    faces, vertices = remove_intersection_surface(args.input, args.output,
                                                  binary=args.binary)
    print(f"Found {faces} intersecting faces; marked {vertices} vertices")


if __name__ == "__main__":
    main()
