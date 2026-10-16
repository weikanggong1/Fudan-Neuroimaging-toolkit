"""对两份真实 T1 生成的 orig.premesh 做只读同输入与网格结构检查。

输入是两张 surface RAS/mm 三角表面、自产 conformed 网格的 wm.mgz，
以及可选的两个 FNIT 被试目录。提供被试目录时，逐个核对同侧
orig.nofix、inflated.nofix、qsphere.nofix、brain.mgz、wm.mgz 的 SHA-256。
输出是新的 JSON 报告；此脚本不修改表面，也不运行原版程序。

本诊断没有原软件独立命令。候选的对应原步骤是固定源码独立编译的
``mris_fix_topology -ga -seed 1234``，其产物还要经过标准 remesh 和
自相交清理，才能验收最终 orig。premesh 的 ORIGINAL 几何相交数
在此不判为零；最终 orig 的交数必须另行复检。

真实 ds000114 sub-07 的命令模板（先设置下列实际路径）：

.. code-block:: bash

    FNIT_SOURCE=/path/to/frozen/FNIT/source       # 本次固定源码
    FNIT_PYTHON=/path/to/fnit/conda/bin/python     # 本次 Conda Python
    FNIT_SUBJECT=/path/to/current/sub-07/subject   # 已完成的 FNIT 整例
    NATIVE_SUBJECT=/path/to/new-native/subject     # 隔离 C++ 阶段产物
    REPORT=/path/to/new-native/rh.mesh-check.json # 尚不存在的报告
    PYTHONPATH="$FNIT_SOURCE/src:$FNIT_SOURCE/tools" "$FNIT_PYTHON" \\
      same_input_mesh_check.py \\
      --candidate-surface "$NATIVE_SUBJECT/surf/rh.orig.premesh" \\
      --reference-surface "$FNIT_SUBJECT/surf/rh.orig.premesh" \\
      --wm-volume "$FNIT_SUBJECT/mri/wm.mgz" \\
      --candidate-subject "$NATIVE_SUBJECT" \\
      --reference-subject "$FNIT_SUBJECT" \\
      --hemisphere rh --output-report "$REPORT"

报告的 indexed_distance_mm 只在五项上游 SHA 全同、两份表面具有
相同 WM 头、顶点数和有序面完全一致时计算。其余情况为 null；
双向点到三角面距离属于后续单独的完整形状诊断。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np

from fnit.recon_all.compare_subject import _topology
from fnit.recon_all.topology_genetic_files import _geometry_valid, _same_geometry, _wm_geometry
from fnit.recon_all.topology_volume_python import _nonmanifold_vertex_links


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_hashes(subject: Path, hemisphere: str) -> dict[str, str]:
    """被试目录与半球→五份实际 GA 输入的完整文件 SHA-256。"""
    names = (f"surf/{hemisphere}.orig.nofix",
             f"surf/{hemisphere}.inflated.nofix",
             f"surf/{hemisphere}.qsphere.nofix", "mri/brain.mgz", "mri/wm.mgz")
    return {name: _sha256(subject / name) for name in names}


def _mesh_quality(vertices: np.ndarray, faces: np.ndarray, geometry: dict,
                  wm_geometry: dict) -> dict:
    """表面数组/头→结构、非流形顶点、方向、面积及 WM 空间门。"""
    xyz, tri = np.asarray(vertices), np.asarray(faces)
    finite = bool(xyz.ndim == 2 and xyz.shape[1:] == (3,) and len(xyz) > 0
                  and np.isfinite(xyz).all())
    indices = bool(tri.ndim == 2 and tri.shape[1:] == (3,) and len(tri) > 0
                   and np.issubdtype(tri.dtype, np.integer)
                   and (tri >= 0).all() and (tri < len(xyz)).all())
    geometry_valid = bool(_geometry_valid(geometry))
    geometry_matches_wm = bool(_same_geometry(geometry, wm_geometry))
    if not (finite and indices):
        return {"status": "failed", "finite_coordinates": finite,
                "valid_face_indices": indices, "geometry_valid": geometry_valid,
                "geometry_matches_wm": geometry_matches_wm,
                "topology": {"valid": False, "reason": "invalid mesh arrays"},
                "self_intersections": "not_assessed_for_premesh"}
    faces32 = np.ascontiguousarray(tri, np.int32)
    structural = _topology(xyz, faces32)
    links = int(_nonmanifold_vertex_links(faces32, len(xyz)))
    directed = np.concatenate((faces32[:, [0, 1]], faces32[:, [1, 2]],
                               faces32[:, [2, 0]]))
    _, groups = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True)
    signs = np.where(directed[:, 0] < directed[:, 1], 1, -1)
    conflicts = int(np.count_nonzero(np.bincount(groups, weights=signs)))
    triangles = xyz[faces32].astype(np.float64)
    areas2 = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                     triangles[:, 2] - triangles[:, 0]), axis=1)
    zero_area = int(np.count_nonzero(areas2 == 0))
    passed = bool(structural.get("valid") and links == 0 and conflicts == 0
                  and zero_area == 0 and np.isfinite(areas2).all()
                  and geometry_valid and geometry_matches_wm)
    return {"status": "passed" if passed else "failed",
            "finite_coordinates": finite, "valid_face_indices": indices,
            "geometry_valid": geometry_valid, "geometry_matches_wm": geometry_matches_wm,
            "topology": structural, "nonmanifold_vertices": links,
            "edge_orientation_conflicts": conflicts, "zero_area_faces": zero_area,
            "self_intersections": "not_assessed_for_premesh",
            "coordinate_space": "surface_RAS", "coordinate_unit": "mm"}


def _indexed_summary(candidate: np.ndarray, reference: np.ndarray) -> dict:
    distances = np.linalg.norm(candidate.astype(np.float64) -
                               reference.astype(np.float64), axis=1)
    return {"count": len(distances), "mean_mm": float(distances.mean()),
            "p99_mm": float(np.quantile(distances, 0.99)),
            "max_mm": float(distances.max()),
            "over_0_1_mm": int(np.count_nonzero(distances > 0.1)),
            "coordinate_elements_different": int(np.count_nonzero(candidate != reference))}


def check(*, candidate_surface: Path, reference_surface: Path,
          wm_volume: Path, output_report: Path, hemisphere: str,
          candidate_subject: Path | None = None,
          reference_subject: Path | None = None) -> dict:
    """只读检查并写新 JSON；同索引误差必须经过五输入和有序面证明。"""
    started = time.perf_counter()
    if hemisphere not in ("lh", "rh"):
        raise ValueError("hemisphere must be lh or rh")
    for surface in (candidate_surface, reference_surface):
        if surface.name != f"{hemisphere}.orig.premesh":
            raise ValueError(f"wrong hemisphere or surface stage: {surface}")
    if (candidate_subject is None) != (reference_subject is None):
        raise ValueError("candidate/reference subject must be provided together")
    if output_report.exists():
        raise FileExistsError(output_report)
    if candidate_subject is not None:
        if (candidate_surface.resolve() !=
                (candidate_subject / "surf" / candidate_surface.name).resolve() or
                reference_surface.resolve() !=
                (reference_subject / "surf" / reference_surface.name).resolve()):
            raise ValueError("surface paths must belong to the declared subjects")
        candidate_inputs = _input_hashes(candidate_subject, hemisphere)
        reference_inputs = _input_hashes(reference_subject, hemisphere)
        wm_sha = _sha256(wm_volume)
        same_inputs = (candidate_inputs == reference_inputs and
                       candidate_inputs["mri/wm.mgz"] == wm_sha)
        input_provenance = {"status": "verified_same_inputs" if same_inputs else "different_inputs",
                            "candidate_sha256": candidate_inputs,
                            "reference_sha256": reference_inputs,
                            "declared_wm_sha256": wm_sha}
    else:
        same_inputs = False
        input_provenance = {"status": "not_assessed_missing_subject_roots"}
    wm_geometry = _wm_geometry(nib.load(str(wm_volume)), wm_volume)
    candidate, candidate_faces, candidate_header = fsio.read_geometry(
        str(candidate_surface), read_metadata=True)
    reference, reference_faces, reference_header = fsio.read_geometry(
        str(reference_surface), read_metadata=True)
    candidate_quality = _mesh_quality(candidate, candidate_faces,
                                      candidate_header, wm_geometry)
    reference_quality = _mesh_quality(reference, reference_faces,
                                      reference_header, wm_geometry)
    same_ordered_faces = bool(len(candidate) == len(reference) and
                              np.array_equal(candidate_faces, reference_faces))
    correspondence = bool(same_inputs and same_ordered_faces and
                          candidate_quality["geometry_matches_wm"] and
                          reference_quality["geometry_matches_wm"])
    indexed_distance = _indexed_summary(candidate, reference) if correspondence else None
    report = {
        "status": "complete_diagnostic",
        "scope": "two orig.premesh from real FNIT upstream; final orig is separate",
        "candidate_surface_sha256": _sha256(candidate_surface),
        "reference_surface_sha256": _sha256(reference_surface),
        "wm_sha256": _sha256(wm_volume),
        "script_sha256": _sha256(Path(__file__)),
        "host": platform.node(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "hemisphere": hemisphere, "input_provenance": input_provenance,
        "candidate_quality": candidate_quality, "reference_quality": reference_quality,
        "candidate_vertices": len(candidate), "reference_vertices": len(reference),
        "candidate_faces": len(candidate_faces), "reference_faces": len(reference_faces),
        "ordered_faces_equal": same_ordered_faces,
        "indexed_vertex_correspondence": correspondence,
        "indexed_distance_mm": indexed_distance,
        "bidirectional_vertex_to_triangle_mm": "not_assessed_separate_full_diagnostic",
        "final_orig_quality": "not_assessed",
        "overall_metric_equivalence": "not_assessed",
        "diagnostic_wall_seconds_including_io": time.perf_counter() - started,
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                        allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-surface", type=Path, required=True)
    parser.add_argument("--reference-surface", type=Path, required=True)
    parser.add_argument("--wm-volume", type=Path, required=True)
    parser.add_argument("--candidate-subject", type=Path)
    parser.add_argument("--reference-subject", type=Path)
    parser.add_argument("--hemisphere", choices=("lh", "rh"), required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    report = check(**vars(args))
    print(json.dumps({"status": report["status"],
                      "candidate_quality": report["candidate_quality"]["status"],
                      "reference_quality": report["reference_quality"]["status"],
                      "same_input_status": report["input_provenance"]["status"],
                      "indexed_vertex_correspondence": report["indexed_vertex_correspondence"],
                      "report": str(args.output_report)}))


if __name__ == "__main__":
    main()
