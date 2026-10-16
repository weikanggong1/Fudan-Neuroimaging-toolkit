"""同输入但不同有序网格的两份 orig.premesh：完整双向点到三角面距离。

本脚本只在已保存的 same_input_mesh_check.py 报告证明五份 FNIT 自产
上游 SHA 全同、两个网格结构及 WM 几何头通过、顶点对应不成立后运行。
读取两个 surface RAS/mm 表面，分别对每个顶点查完整目标三角面，
复用仓库既有 reference_surface_distance.py 的 ``_point_to_mesh``。
写出新私有目录中的 report.json 和全部顶点距离 NPZ；不写生产表面。
这里没有同索引误差，也不把顶点到三角面距离称为连续 Hausdorff 距离。
报告没有新的等效阈值。此诊断没有 FreeSurfer 独立命令；对应拓扑
步骤是 ``mris_fix_topology -ga -seed 1234``，最终 orig 另行验收。

真实 ds000114 sub-07 右半球的具名命令模板：

.. code-block:: bash

    FNIT_SOURCE=/path/to/frozen/FNIT/source           # 固定本次源码
    FNIT_PYTHON=/path/to/conda/bin/python             # 本次 Conda Python
    NATIVE_RUN=/path/to/new-native-stage              # 已完成的 C++ 诊断
    FNIT_SUBJECT=/path/to/current/sub-07/subject       # 同输入纯 Python 整例
    OUT=/path/to/new-private-distance-directory       # 必须不存在
    DIST_MODULE="$FNIT_SOURCE/validation/recon_all/accuracy_20261003/task_04/reference_surface_distance.py"
    PYTHONPATH="$FNIT_SOURCE/src:$FNIT_SOURCE/tools" taskset -c 0,1 \\
      "$FNIT_PYTHON" premesh_bidirectional_distance.py \\
      --candidate-surface "$NATIVE_RUN/subject/surf/rh.orig.premesh" \\
      --reference-surface "$FNIT_SUBJECT/surf/rh.orig.premesh" \\
      --mesh-check-report "$NATIVE_RUN/mesh_check.json" \\
      --distance-module "$DIST_MODULE" --output-directory "$OUT"

"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import time

import nibabel.freesurfer.io as fsio
import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(distances: np.ndarray) -> dict:
    return {"count": len(distances), "mean_mm": float(distances.mean()),
            "p50_mm": float(np.quantile(distances, 0.5)),
            "p99_mm": float(np.quantile(distances, 0.99)),
            "max_mm": float(distances.max()),
            "over_0_1_mm": int(np.count_nonzero(distances > 0.1)),
            "over_1_mm": int(np.count_nonzero(distances > 1.0))}


def compare(*, candidate_surface: Path, reference_surface: Path,
            mesh_check_report: Path, distance_module: Path,
            output_directory: Path) -> dict:
    """同输入且不同网格的表面路径→私有双向全顶点距离与完整回执。"""
    started = time.perf_counter()
    if output_directory.exists():
        raise FileExistsError(output_directory)
    check = json.loads(mesh_check_report.read_text())
    if (check.get("status") != "complete_diagnostic" or
            check.get("input_provenance", {}).get("status") != "verified_same_inputs" or
            check.get("candidate_quality", {}).get("status") != "passed" or
            check.get("reference_quality", {}).get("status") != "passed" or
            check.get("candidate_surface_sha256") != _sha256(candidate_surface) or
            check.get("reference_surface_sha256") != _sha256(reference_surface) or
            check.get("indexed_vertex_correspondence") is not False):
        raise ValueError("same-input, quality and different ordered meshes must be proven")
    if distance_module.name != "reference_surface_distance.py":
        raise ValueError("use the declared existing full surface-distance module")
    spec = importlib.util.spec_from_file_location("_fnit_full_premesh_distance", distance_module)
    if spec is None or spec.loader is None:
        raise ImportError(distance_module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "_point_to_mesh", None)):
        raise ImportError("existing _point_to_mesh was not found")
    candidate, candidate_faces = fsio.read_geometry(str(candidate_surface))
    reference, reference_faces = fsio.read_geometry(str(reference_surface))
    if (len(candidate) != check["candidate_vertices"] or
            len(reference) != check["reference_vertices"] or
            len(candidate_faces) != check["candidate_faces"] or
            len(reference_faces) != check["reference_faces"] or
            not np.isfinite(candidate).all() or not np.isfinite(reference).all()):
        raise ValueError("actual decoded mesh differs from completed quality report")
    output_directory.mkdir(parents=True, mode=0o700)
    report_path = output_directory / "report.json"
    report = {"status": "running", "scope": "same-input stage-only orig.premesh comparison",
              "candidate_surface_sha256": _sha256(candidate_surface),
              "reference_surface_sha256": _sha256(reference_surface),
              "mesh_check_report_sha256": _sha256(mesh_check_report),
              "distance_module_sha256": _sha256(distance_module),
              "script_sha256": _sha256(Path(__file__)),
              "host": platform.node(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
              "candidate_vertices": len(candidate), "reference_vertices": len(reference),
              "candidate_faces": len(candidate_faces), "reference_faces": len(reference_faces),
              "indexed_vertex_distance": None,
              "indexed_comparison_reason": "different ordered meshes; vertex correspondence absent",
              "distance_definition": "all source vertices to complete target triangle mesh, both directions",
              "coordinate_space": "surface_RAS", "unit": "mm",
              "continuous_Hausdorff": "not_assessed",
              "final_orig_quality": "not_assessed",
              "overall_metric_equivalence": "not_assessed"}
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    try:
        tick = time.perf_counter()
        forward = module._point_to_mesh(candidate, reference, reference_faces)
        report["candidate_to_reference_seconds"] = time.perf_counter() - tick
        tick = time.perf_counter()
        reverse = module._point_to_mesh(reference, candidate, candidate_faces)
        report["reference_to_candidate_seconds"] = time.perf_counter() - tick
        if (forward.shape != (len(candidate),) or reverse.shape != (len(reference),) or
                not np.isfinite(forward).all() or not np.isfinite(reverse).all() or
                (forward < 0).any() or (reverse < 0).any()):
            raise ValueError("complete finite nonnegative bidirectional distances required")
        checkpoint = output_directory / "all_vertex_distances.private.npz"
        np.savez_compressed(checkpoint, candidate_to_reference=forward,
                            reference_to_candidate=reverse)
        report.update(status="complete_bidirectional_vertex_to_triangle",
                      candidate_to_reference=_summary(forward),
                      reference_to_candidate=_summary(reverse),
                      distance_checkpoint_sha256=_sha256(checkpoint),
                      complete_wall_seconds_after_import=time.perf_counter() - started)
    except Exception as error:
        report.update(status="failed", error=repr(error),
                      incomplete_wall_seconds_after_import=time.perf_counter() - started)
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-surface", type=Path, required=True)
    parser.add_argument("--reference-surface", type=Path, required=True)
    parser.add_argument("--mesh-check-report", type=Path, required=True)
    parser.add_argument("--distance-module", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    result = compare(**vars(args))
    print(json.dumps({"status": result["status"],
                      "report": str(args.output_directory / "report.json"),
                      "seconds": result["complete_wall_seconds_after_import"]}), flush=True)


if __name__ == "__main__":
    main()
