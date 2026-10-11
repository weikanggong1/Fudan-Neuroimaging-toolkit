"""自有有向相交检测与完整SOAP清理文件接口，不执行原生程序。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import tempfile
import time
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


def mark_intersections(vertices: np.ndarray, faces: np.ndarray, *,
                       separation_backend: str = "python", device: str | None = None,
                       chunk_size: int = 65536) -> tuple[np.ndarray, int]:
    """Return native-style marked vertices and intersecting face count.

    The bounding-sphere and axis-aligned box filters cannot discard a true
    triangle intersection. Faces sharing a vertex are excluded as in FreeSurfer.

    separation_backend默认python，保留全部原逐对谓词；numba/torch用
    F64误差区间排除确定平面分离的对，其余仍按原顺序执行原谓词。
    torch要求显式cuda:N device，chunk_size是正整数分块预算；不改变
    原候选球/盒筛选、1e-6平面容差、共面规则或返回的标记/面数。
    """
    if separation_backend not in ("python", "numba", "torch"):
        raise ValueError("separation_backend must be python, numba or torch")
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
    pairs = pairs[disjoint]
    if separation_backend != "python" and len(pairs):
        from .topology_triangle_plane_filter import separated_triangle_pairs
        keep = ~separated_triangle_pairs(triangles_a=triangles[pairs[:, 0]],
            triangles_b=triangles[pairs[:, 1]], backend=separation_backend,
            device=device, chunk_size=chunk_size)
        pairs = pairs[keep]
    intersecting_faces = np.zeros(len(faces), dtype=bool)
    for fa, fb in pairs:
        if _triangles_intersect(triangles[fa], triangles[fb]):
            intersecting_faces[fa] = intersecting_faces[fb] = True
    marks[faces[intersecting_faces].ravel()] = True
    return marks, int(intersecting_faces.sum())


def _write_preserving_surface(input_path: Path, output_path: Path, vertices: np.ndarray) -> None:
    """只替换坐标字节，保留nibabel已读取的有序面、几何头和所有尾标记。"""
    with input_path.open("rb") as stream:
        if stream.read(3) != b"\xff\xff\xfe":
            raise ValueError("expected FreeSurfer triangle surface")
        stream.readline()
        stream.readline()
        counts = stream.read(8)
        if len(counts) != 8:
            raise ValueError("truncated triangle surface counts")
        count, _ = struct.unpack(">ii", counts)
        start = stream.tell()
    if vertices.shape != (count, 3):
        raise ValueError("intersection cleanup changed vertex count")
    original = input_path.read_bytes()
    if len(original) < start + 12 * count:
        raise ValueError("truncated triangle surface coordinates")
    output_path.write_bytes(original[:start] + np.asarray(vertices, dtype=">f4").tobytes()
                            + original[start + 12 * count:])


def run_remove_intersection(*, input_path: str | Path, output_path: str | Path,
                            marking_backend: str = "source_numba", device: str | None = None,
                            candidate_grid_cells_per_axis: int = 2, require_zero: bool = True,
                            ordered_neighbor_offsets: np.ndarray | None = None,
                            ordered_neighbors: np.ndarray | None = None,
                            report_path: str | Path | None = None) -> dict:
    """完整读取→自有有向标记→100次SOAP/轮→停止/复检→原子写出。

    input_path/output_path为三角表面，可同路径；坐标为surface RAS/mm。
    marking_backend默认source_numba，source_torch需明确device(cpu/cuda:N)，
    都复用既有完整清理器；不调用原版C++。candidate_grid_cells_per_axis
    默认2，3仅供source_torch完整候选；require_zero默认True，停滞残余
    非零时抛RuntimeError并保留输入/已有输出。False仅供与原命令停止
    语义诊断，不应绕过生产质量门。零相交保持源文件字节；修复只替换
    坐标，不改变面/顶点编号、几何头或尾标记。report_path默认None，给出
    写JSON；报告含输入/输出SHA、读/计算/写/总秒、完整清理轨迹与复检。
    ordered_neighbor_offsets/ordered_neighbors默认同时None，完整保持既有
    文件路径；压缩后源vt一环可显式给整数(N+1,)/(E,)CSR，两者须同时
    传入并与文件顶点编号一致。按原序保留非面支持边，不从面重建，
    非法CSR在写出前ValueError；不更改100SOAP、停滞或严格双重复检。
    非法输入、索引、后端和CUDA失败传播，不静默回退、不降低精度。
    """
    if marking_backend not in {"source_numba", "source_torch"}:
        raise ValueError("file repair requires source_numba or source_torch")
    if not isinstance(require_zero, bool):
        raise ValueError("require_zero must be bool")
    from .place_surface_final_cleanup import repair_intersections
    from .place_surface_intersection_marking import mark_source_intersections
    input_path, output_path = Path(input_path), Path(output_path)
    started = time.perf_counter()
    input_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()
    vertices, faces = fs.read_geometry(str(input_path))
    read_done = time.perf_counter()
    corrected, diagnostics = repair_intersections(
        vertices=vertices, faces=faces, ripped=np.zeros(len(vertices), dtype=bool),
        marking_backend=marking_backend, device=device,
        candidate_grid_cells_per_axis=candidate_grid_cells_per_axis,
        ordered_neighbor_offsets=ordered_neighbor_offsets, ordered_neighbors=ordered_neighbors)
    compute_done = time.perf_counter()
    # 单独复检当前坐标；不能把旧标记缓存或停止计数代替最终质量检查。
    _, remaining = mark_source_intersections(
        vertices=corrected, faces=faces,
        predicate_backend="torch" if marking_backend == "source_torch" else "numba",
        device=device, candidate_grid_cells_per_axis=candidate_grid_cells_per_axis)
    _, geometric_remaining = mark_intersections(corrected, faces)
    report = {"scope": "complete_intersection_file_API_not_recon_all",
              "input_path": str(input_path), "output_path": str(output_path),
              "input_sha256": input_sha256, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "vertices": len(vertices), "faces": len(faces), "coordinate_units": "surface_RAS_mm",
              "marking_backend": marking_backend, "device": device,
              "candidate_grid_cells_per_axis": candidate_grid_cells_per_axis,
              "require_zero": require_zero, "cleanup": diagnostics,
              "source_intersecting_faces_after_independent_recheck": remaining,
              "geometric_intersecting_faces_after_independent_recheck": geometric_remaining,
              "read_seconds": read_done - started, "compute_seconds": compute_done - read_done,
              "recheck_seconds": time.perf_counter() - compute_done,
              "native_program_called": False, "execution_status": "complete"}
    if ordered_neighbor_offsets is not None:
        report["ordered_neighbor_csr_sha256"] = {
            "offsets": hashlib.sha256(np.asarray(ordered_neighbor_offsets, dtype=np.int32).tobytes()).hexdigest(),
            "neighbors": hashlib.sha256(np.asarray(ordered_neighbors, dtype=np.int32).tobytes()).hexdigest()}
    if require_zero and (remaining or geometric_remaining):
        report.update(execution_status="failed", output_written=False,
                      total_seconds_including_io=time.perf_counter() - started)
        if report_path is not None:
            Path(report_path).parent.mkdir(parents=True, exist_ok=True)
            Path(report_path).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        raise RuntimeError(f"intersection repair left {remaining} source / {geometric_remaining} geometric intersecting faces")
    write_started = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if np.array_equal(np.float32(vertices), corrected):
        if input_path.resolve() != output_path.resolve():
            shutil.copyfile(input_path, output_path)
    else:
        with tempfile.NamedTemporaryFile(prefix=output_path.name + ".fnit-intersection-",
                                         dir=output_path.parent, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            _write_preserving_surface(input_path, temporary, corrected)
            temporary.replace(output_path)
        finally:
            temporary.unlink(missing_ok=True)
    report.update(output_written=True, output_sha256=hashlib.sha256(output_path.read_bytes()).hexdigest(),
                  write_seconds=time.perf_counter() - write_started,
                  total_seconds_including_io=time.perf_counter() - started)
    if report_path is not None:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def remove_intersection_surface(input_path: str | Path, output_path: str | Path,
                                *, binary: str | Path | None = None,
                                assets_dir: str | Path | None = None,
                                marking_backend: str = "source_numba", device: str | None = None,
                                candidate_grid_cells_per_axis: int = 2,
                                ordered_neighbor_offsets: np.ndarray | None = None,
                                ordered_neighbors: np.ndarray | None = None,
                                report_path: str | Path | None = None) -> tuple[int, int]:
    """自有完整修复，返回输入相交面数/标记顶点数；非零残余报错。

    路径/坐标/参数与run_remove_intersection相同；binary与assets_dir保留
    旧调用兼容但不读取或执行，也不查找系统程序。详细轨迹可写report_path。
    两项ordered neighbor CSR显式传递同完整接口，不改变既有None默认。
    """
    report = run_remove_intersection(input_path=input_path, output_path=output_path,
                                   marking_backend=marking_backend, device=device,
                                   candidate_grid_cells_per_axis=candidate_grid_cells_per_axis,
                                   ordered_neighbor_offsets=ordered_neighbor_offsets,
                                   ordered_neighbors=ordered_neighbors,
                                   require_zero=True, report_path=report_path)
    return report["cleanup"]["intersecting_faces_before"], report["cleanup"]["marked_vertices"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--marking-backend", choices=("source_numba", "source_torch"), default="source_numba")
    parser.add_argument("--device")
    parser.add_argument("--candidate-grid-cells-per-axis", type=int, default=2)
    parser.add_argument("--report-path", type=Path)
    args = parser.parse_args()
    faces, vertices = remove_intersection_surface(args.input, args.output,
                                                  binary=args.binary, marking_backend=args.marking_backend,
                                                  device=args.device,
                                                  candidate_grid_cells_per_axis=args.candidate_grid_cells_per_axis,
                                                  report_path=args.report_path)
    print(f"Found {faces} intersecting faces; marked {vertices} vertices")


if __name__ == "__main__":
    main()
