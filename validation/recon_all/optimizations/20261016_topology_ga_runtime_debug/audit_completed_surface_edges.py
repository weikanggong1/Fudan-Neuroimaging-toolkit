"""已完成 recon-all 被试的只读表面边长诊断。

输入是完整的纯 Python GA 与混合 C++ FNIT subject，可选已完成官方参考。
逐张读取双侧 orig.premesh/orig/white/pial，坐标为 surface RAS，单位 mm。
输出新的 JSON：唯一边长最大/P99.9/>5 mm 数量、最大三角形面积及固定位置
附近的边长。>5 mm 仅追踪局部异常，并非验收阈值。不跨网格按索引比较。
此诊断没有原软件独立 CLI；相关步骤包括 mris_fix_topology、
mris_remesh 和 mris_place_surface，完整质量门另行验收。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import nibabel.freesurfer.io as fsio
import numpy as np


STAGES = ("orig.premesh", "orig", "white", "pial")
HEMISPHERES = ("lh", "rh")
DEFAULT_FOCUS_RAS_MM = (29.023, 45.086, 2.920)


def _sha256(path: Path) -> str:
    """已有文件→完整 SHA-256；缺失或无法读取时抛 I/O 异常。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_surfaces(subject: Path) -> dict[str, Path]:
    """被试目录→八份必需表面；读几何前一次性验证存在且非空。"""
    paths = {f"{hemi}.{stage}": subject / "surf" / f"{hemi}.{stage}"
             for hemi in HEMISPHERES for stage in STAGES}
    missing = [str(path) for path in paths.values()
               if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("missing or empty required surfaces: " + ", ".join(missing))
    return paths


def _fnit_completion(subject: Path) -> dict[str, str]:
    """完整 FNIT subject→执行、输出与生产网格检查的报告摘要。"""
    path = subject / "fnit-native-free-run.json"
    report = json.loads(path.read_text())
    outputs = report.get("output_validation", {})
    if (report.get("status") != "complete" or
            outputs.get("status") != "passed" or
            outputs.get("profile") != "single-t1-138" or
            outputs.get("expected") != 138 or outputs.get("present") != 138 or
            outputs.get("missing") != [] or
            report.get("mesh_validation", {}).get("status") != "passed"):
        raise ValueError(f"FNIT subject is not complete and quality checked: {subject}")
    return {"status": "verified_complete", "report_sha256": _sha256(path)}


def _edge_summary(lengths: np.ndarray) -> dict[str, float | int | None]:
    """唯一边长数组（mm）→计数、最大、P99.9 和追踪性 >5 mm 数量。"""
    if len(lengths) == 0:
        return {"count": 0, "max_mm": None, "p99_9_mm": None,
                "over_5_mm_count": 0}
    return {"count": int(len(lengths)), "max_mm": float(lengths.max()),
            "p99_9_mm": float(np.quantile(lengths, 0.999)),
            "over_5_mm_count": int(np.count_nonzero(lengths > 5.0))}


def mesh_edge_metrics(*, vertices: np.ndarray, faces: np.ndarray,
                      focus_ras_mm: tuple[float, float, float],
                      focus_radius_mm: float) -> dict[str, object]:
    """N×3 顶点/M×3 面与 RAS 焦点/正半径→独立网格几何统计。

    输入顶点和焦点为 surface RAS/mm，输出边长 mm、面积 mm²。数组无效、
    索引越界、焦点非有限或半径非正时抛 ValueError。不修改输入数组。
    """
    xyz = np.asarray(vertices)
    tri = np.asarray(faces)
    focus = np.asarray(focus_ras_mm, dtype=np.float64)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0 or
            tri.ndim != 2 or tri.shape[1] != 3 or len(tri) == 0 or
            not np.issubdtype(tri.dtype, np.integer) or
            focus.shape != (3,) or not np.isfinite(xyz).all() or
            not np.isfinite(focus).all() or
            not np.isfinite(focus_radius_mm) or focus_radius_mm <= 0 or
            np.any(tri < 0) or np.any(tri >= len(xyz))):
        raise ValueError("invalid mesh, focus RAS, or positive radius")
    xyz64 = xyz.astype(np.float64, copy=False)
    edges = np.concatenate((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]))
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    lengths = np.linalg.norm(xyz64[edges[:, 1]] - xyz64[edges[:, 0]], axis=1)
    corners = xyz64[tri]
    areas = 0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
        axis=1)
    if not np.isfinite(lengths).all() or not np.isfinite(areas).all():
        raise ValueError("non-finite edge length or face area")
    midpoint = 0.5 * (xyz64[edges[:, 0]] + xyz64[edges[:, 1]])
    local = np.linalg.norm(midpoint - focus, axis=1) <= focus_radius_mm
    local_edges = edges[local]
    local_lengths = lengths[local]
    longest = None
    if len(local_lengths):
        index = int(np.argmax(local_lengths))
        first, second = (int(value) for value in local_edges[index])
        longest = {"length_mm": float(local_lengths[index]),
                   "vertex_indices": [first, second],
                   "endpoint_ras_mm": [xyz64[first].tolist(), xyz64[second].tolist()]}
    nearest_vertex = int(np.argmin(np.linalg.norm(xyz64 - focus, axis=1)))
    return {
        "coordinate_space": "surface_RAS", "distance_unit": "mm",
        "area_unit": "mm2", "vertices": int(len(xyz)), "faces": int(len(tri)),
        "unique_edges": _edge_summary(lengths),
        "face_area_max_mm2": float(areas.max()),
        "zero_area_face_count": int(np.count_nonzero(areas == 0)),
        "focus": {"ras_mm": focus.tolist(), "radius_mm": float(focus_radius_mm),
                  "edge_selection": "midpoint_distance_to_focus_le_radius",
                  "nearest_vertex_index": nearest_vertex,
                  "nearest_vertex_distance_mm": float(np.linalg.norm(xyz64[nearest_vertex] - focus)),
                  "local_unique_edges": _edge_summary(local_lengths),
                  "longest_local_edge": longest},
        "over_5_mm_role": "diagnostic_tracking_only_not_acceptance_threshold",
    }


def audit(*, baseline_subject: Path, candidate_subject: Path,
          output_report: Path, reference_subject: Path | None = None,
          focus_ras_mm: tuple[float, float, float] = DEFAULT_FOCUS_RAS_MM,
          focus_radius_mm: float = 20.0) -> dict[str, object]:
    """已完成 subject 目录→新 JSON；输入表面与既有报告保持只读。"""
    start = time.perf_counter()
    roots = {"baseline_python_ga": baseline_subject,
             "candidate_hybrid_cpp": candidate_subject}
    if reference_subject is not None:
        roots["optional_reference"] = reference_subject
    if len({root.resolve() for root in roots.values()}) != len(roots):
        raise ValueError("subject directories must be distinct")
    if any(output_report.resolve().is_relative_to(root.resolve()) for root in roots.values()):
        raise ValueError("diagnostic output must be outside all input subjects")
    if output_report.exists():
        raise FileExistsError(output_report)
    completions = {name: _fnit_completion(root) for name, root in roots.items()
                   if name != "optional_reference"}
    paths = {name: _required_surfaces(root) for name, root in roots.items()}
    subjects = {}
    for name, surfaces in paths.items():
        results = {}
        for label, path in surfaces.items():
            xyz, tri = fsio.read_geometry(str(path))
            results[label] = {"surface_sha256": _sha256(path),
                              **mesh_edge_metrics(vertices=xyz, faces=tri,
                                                  focus_ras_mm=focus_ras_mm,
                                                  focus_radius_mm=focus_radius_mm)}
        subjects[name] = {"completion": completions.get(name, {
            "status": "external_reference_completion_not_verified"}),
            "surfaces": results}
    report = {"schema_version": "1.0", "status": "complete_diagnostic",
              "scope": "independent_mesh_edge_and_area_audit_of_completed_subjects",
              "script_sha256": _sha256(Path(__file__)), "host": platform.node(),
              "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
              "subjects": subjects,
              "indexed_vertex_comparison": "not_performed_distinct_meshes",
              "shape_distance": "not_assessed_here",
              "quality_acceptance": "not_assessed_here",
              "diagnostic_wall_seconds_including_io": time.perf_counter() - start}
    output_report.parent.mkdir(parents=True, exist_ok=True)
    with output_report.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-subject", type=Path, required=True,
                        help="已完成的纯 Python GA FNIT subject 目录")
    parser.add_argument("--candidate-subject", type=Path, required=True,
                        help="已完成的混合 C++ 拓扑 FNIT subject 目录")
    parser.add_argument("--reference-subject", type=Path,
                        help="可选的已完成独立官方参考 subject 目录")
    parser.add_argument("--focus-ras-mm", type=float, nargs=3,
                        default=DEFAULT_FOCUS_RAS_MM, metavar=("R", "A", "S"),
                        help="固定局部焦点的 surface RAS/mm，默认 sub-07 RH 异常坐标")
    parser.add_argument("--focus-radius-mm", type=float, default=20.0,
                        help="按唯一边中点选择局部球体的半径，单位 mm，默认 20")
    parser.add_argument("--output-report", type=Path, required=True,
                        help="新的 JSON 报告路径；不得已有文件")
    args = parser.parse_args()
    report = audit(baseline_subject=args.baseline_subject,
                   candidate_subject=args.candidate_subject,
                   reference_subject=args.reference_subject,
                   focus_ras_mm=tuple(args.focus_ras_mm),
                   focus_radius_mm=args.focus_radius_mm,
                   output_report=args.output_report)
    print(json.dumps({"status": report["status"], "output_report": str(args.output_report),
                      "subjects": list(report["subjects"])}))


if __name__ == "__main__":
    main()
