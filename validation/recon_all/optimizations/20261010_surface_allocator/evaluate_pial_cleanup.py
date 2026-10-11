"""只读核对失败 pial 的末态与两次隔离源码参考；不写标准表面。

输入为冻结 FNIT src、失败观察器目录、已完成参考目录和新报告路径。
私有 NPY 只在服务器读取；公开报告仅保存 SHA、差异、相交面编号和统计。
完整有序面对应成立后才逐顶点比较。源标记器的返回面状态只读捕获，
不修改判定、几何或原返回值，诊断耗时与优化器墙钟分开。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def compare(*, source_root: Path, diagnostic_run: Path, reference_run: Path,
            output_json: Path, hemi: str = "lh", threads: int = 4) -> dict:
    """surface RAS/mm，返回同序坐标/面、元数据和源相交状态；失败状态保留。"""
    tick = time.perf_counter()
    if output_json.exists():
        raise FileExistsError(output_json)
    if threads < 1 or hemi not in {"lh", "rh"}:
        raise ValueError("positive threads and lh/rh required")
    sys.path.insert(0, str(source_root))
    import nibabel as nib
    import numpy as np
    from numba import set_num_threads
    from fnit.recon_all.place_surface_intersection_marking import mark_source_intersections
    set_num_threads(threads)
    case = diagnostic_run / "01-cli-enabled"
    child = json.loads((case / "child.private.json").read_text())
    reference = json.loads((reference_run / "summary.private.json").read_text())
    if child["status"] != "failed" or reference["status"] != "complete":
        raise ValueError("requires preserved failed Python and two completed reference runs")
    for name, digest in child["input_sha256"].items():
        if digest != reference["input_sha256"].get(name) or sha(case / "subject" / name) != digest:
            raise ValueError("input SHA differs: " + name)
    white, faces = nib.freesurfer.read_geometry(str(case / "subject" / f"surf/{hemi}.white"))
    last_path, clean_path = (case / name for name in
                            ("last-accepted.private.npy", "cleanup-result.private.npy"))
    last, cleaned = np.load(last_path), np.load(clean_path)
    if last.shape != white.shape or cleaned.shape != white.shape:
        raise ValueError("private snapshots have different ordered vertex count")
    rows, raw = [], []
    for index in (1, 2):
        path = reference_run / f"reference-{index}/subject/surf/{hemi}.pial.T1"
        xyz, other_faces, volume_info = nib.freesurfer.read_geometry(str(path), read_metadata=True)
        ordered = xyz.shape == cleaned.shape and np.array_equal(faces, other_faces)
        distance = np.linalg.norm(cleaned.astype(np.float64) - xyz, axis=1) if ordered else None
        rows.append({"index": index, "reference_output_sha256": sha(path),
                     "same_ordered_faces": bool(ordered), "finite_coordinates": bool(np.isfinite(xyz).all()),
                     "different_coordinate_elements": int(np.count_nonzero(cleaned != xyz)) if ordered else None,
                     "distance_mean_mm": float(distance.mean()) if ordered else None,
                     "distance_p99_mm": float(np.quantile(distance, .99)) if ordered else None,
                     "distance_max_mm": float(distance.max()) if ordered else None,
                     "volume_info": {key: value.tolist() if isinstance(value, np.ndarray) else value
                                     for key, value in volume_info.items()}})
        raw.append(path.read_bytes())
    # 三角格式的两行注释位于3字节magic之后。定位完整字节差异而非归为随机性。
    header = [blob[3:].split(b"\n", 2)[:2] for blob in raw]
    after_header = [blob[3:].split(b"\n", 2)[2] for blob in raw]
    report = {"status": "running_readonly_qc", "scope": "failed producer terminal diagnosis; not successful pial or recon-all",
              "script_sha256": sha(Path(__file__)), "source_marker_sha256": sha(source_root / "fnit/recon_all/place_surface_intersection_marking.py"),
              "input_sha256": child["input_sha256"], "diagnostic_child_sha256": sha(case / "child.private.json"),
              "reference_summary_sha256": sha(reference_run / "summary.private.json"),
              "native_program_sha256": reference["program_sha256"], "original_failed_status_preserved": child["status"],
              "last_accepted_file_sha256": sha(last_path), "cleanup_file_sha256": sha(clean_path),
              "source_cleanup_diagnostic": child["cleanup_observation"]["diagnostic"], "rows": rows,
              "reference_file_metadata": {"header_lines": [[line.decode("utf8", "replace") for line in item] for item in header],
                                          "bytes_after_comment_lines_equal": after_header[0] == after_header[1],
                                          "whole_file_bytes_equal": raw[0] == raw[1]},
              "accepted_steps": len(child["accepted_steps"]), "threads": threads,
              "overall_metric_equivalence": "not_assessed", "whole_pipeline_speedup": None}
    write(output_json, report)
    captured = {}
    def observer(frame, event, arg):
        if event == "return" and frame.f_code is mark_source_intersections.__code__:
            captured["face_flags"] = frame.f_locals["face_marks"].copy()
    qc_tick = time.perf_counter()
    previous = sys.getprofile()
    try:
        sys.setprofile(observer)
        marked, count = mark_source_intersections(cleaned, faces, predicate_backend="numba")
    finally:
        sys.setprofile(previous)
    flags = captured["face_flags"]
    report.update(status="complete_failed_producer_diagnosis", source_intersecting_faces=int(count),
                  source_marked_vertices=int(marked.sum()), residual_face_indices=np.flatnonzero(flags).tolist(),
                  marked_vertex_indices=np.flatnonzero(marked).tolist(), readonly_source_qc_seconds=time.perf_counter()-qc_tick,
                  diagnostic_wall_seconds=time.perf_counter()-tick,
                  original_source_return_and_geometry_unchanged=True)
    write(output_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--diagnostic-run", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--hemi", choices=("lh", "rh"), default="lh")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    compare(source_root=args.source_root, diagnostic_run=args.diagnostic_run,
            reference_run=args.reference_run, output_json=args.output_json, hemi=args.hemi, threads=args.threads)


if __name__ == "__main__":
    main()
