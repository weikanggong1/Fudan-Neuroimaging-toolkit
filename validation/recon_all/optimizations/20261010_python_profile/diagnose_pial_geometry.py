"""只读核验pial清理终态的独立几何相交和参考重复性。

输入(N,3) NumPy坐标、含同序(M,3)面的三角表面、参考表面列表；坐标
均为surface RAS/mm。输出JSON仅有SHA/结构/不同元素数/相交面和顶点数/
读计算秒数，不含坐标或修改表面。复用既有cKDTree/F64三角面谓词；
有相交时保持诊断完成但mesh_quality_passed=False，不能放行生产。
参数非法、非有限坐标、顶点/有序面不对应、输出已存在均报错。
参考为隔离benchmark输入；脚本不运行任何原版阶段程序，无官方CLI。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import nibabel.freesurfer.io as fs
import numpy as np

from fnit.recon_all.mris_remove_intersection_python import mark_intersections


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnose(*, coordinates_path: Path, faces_path: Path,
             reference_surfaces: tuple[Path, ...], output_path: Path) -> dict:
    """具名参数分别指定终态、同序面模板、参考列表、新JSON路径。"""
    if output_path.exists():
        raise FileExistsError(output_path)
    started = time.perf_counter()
    vertices = np.load(coordinates_path, allow_pickle=False)
    _, faces = fs.read_geometry(str(faces_path))
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("coordinates must be finite (N,3)")
    if len(faces) and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("ordered faces refer to missing vertices")
    report = {"scope": "read-only terminal pial geometry diagnosis; not raw T1 run",
              "input_coordinates_sha256": sha(coordinates_path),
              "ordered_faces_file_sha256": sha(faces_path),
              "diagnostic_script_sha256": sha(Path(__file__)),
              "module_sha256": sha(Path(__import__(mark_intersections.__module__, fromlist=["__file__"]).__file__)),
              "vertices": len(vertices), "faces": len(faces),
              "coordinate_dtype": str(vertices.dtype), "coordinate_units": "surface_RAS_mm",
              "cpu_affinity": sorted(__import__("os").sched_getaffinity(0)),
              "references": []}
    for path in reference_surfaces:
        reference_vertices, reference_faces = fs.read_geometry(str(path))
        if reference_vertices.shape != vertices.shape or not np.array_equal(faces, reference_faces):
            raise ValueError("reference vertices/ordered faces do not correspond")
        report["references"].append({"file_sha256": sha(path),
            "different_coordinate_elements": int(np.count_nonzero(vertices != reference_vertices)),
            "different_ordered_face_elements": 0})
    read_done = time.perf_counter()
    marks, count = mark_intersections(vertices, faces)
    report.update(geometric_intersecting_faces=int(count), marked_vertices=int(marks.sum()),
                  read_and_hash_seconds=read_done-started, geometric_recheck_seconds=time.perf_counter()-read_done,
                  diagnosis_seconds_before_metadata_write=time.perf_counter()-started,
                  status="complete_diagnostic", mesh_quality_passed=count==0,
                  original_stage_program_called=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinates-path", required=True, type=Path)
    parser.add_argument("--faces-path", required=True, type=Path)
    parser.add_argument("--reference-surface", action="append", default=[], type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    args = parser.parse_args()
    diagnose(coordinates_path=args.coordinates_path, faces_path=args.faces_path,
             reference_surfaces=tuple(args.reference_surface), output_path=args.output_path)
