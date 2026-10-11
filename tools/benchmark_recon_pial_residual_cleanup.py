"""冻结失败pial的完整源码清理诊断；不改四轮放置或写标准表面。

服务器原位读取FNIT自产前置、接受末态和清理末态。仅发布哈希、数值
差异及计数；几何快照保留原任务目录。旧参考只在新候选保存后比较。
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def diagnose(*, frozen_source: Path, overlay_source: Path, diagnostic_case: Path,
             reference_run: Path, output_directory: Path,
             expected_input_sha256: dict[str, str], threads: int = 4) -> dict:
    """完整SOAP同输入回放，坐标为surface RAS/mm；独立输出聚合JSON。

    frozen_source为历史FNIT Python源目录，overlay_source为本轮仅含
    清理与marker的冻结FNIT模块目录；不含原程序。diagnostic_case有
    subject七前置与last-accepted/cleanup-result NPY，reference_run有
    两份已完成原参考pial。output_directory须为空；expected_*必须
    包含七前置SHA。threads正整数；错哈希、非有限/不同网格会失败。
    返回实际source/legacy标记、完整SOAP轨迹、rip限制、同序几何差异
    与阶段墙钟。即使仍有残余，诊断返回complete_diagnosis并明确
    producer_mesh_quality=failed；不把诊断完成视作重建通过。
    """
    wall = time.perf_counter()
    if threads < 1:
        raise ValueError("threads must be positive")
    if output_directory.exists():
        raise FileExistsError(output_directory)
    output_directory.mkdir(parents=True)
    if len(expected_input_sha256) != 7:
        raise ValueError("seven declared stage input hashes required")
    for name, expected in expected_input_sha256.items():
        if sha(diagnostic_case / "subject" / name) != expected:
            raise ValueError("stage input hash differs: " + name)
        for repeat in (1, 2):
            path = reference_run / f"reference-{repeat}/subject" / name
            if sha(path) != expected:
                raise ValueError("reference stage input hash differs: " + name)
    sys.path.insert(0, str(frozen_source))
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0, str(overlay_source))
    import nibabel.freesurfer as fs
    import nibabel
    import numpy as np
    import numba
    numba.set_num_threads(threads)
    from fnit.recon_all.place_surface_final_cleanup import pin_medial_wall, repair_intersections
    from fnit.recon_all.place_surface_intersection_marking import mark_source_intersections
    from fnit.recon_all.mris_remove_intersection_python import mark_intersections
    white_path = diagnostic_case / "subject/surf/lh.white"
    white, faces = fs.read_geometry(str(white_path))
    white, faces = white.astype(np.float32), faces.astype(np.int32)
    label_path = diagnostic_case / "subject/label/lh.cortex+hipamyg.label"
    cortex_path = diagnostic_case / "subject/label/lh.cortex.label"
    ripped = np.ones(len(white), dtype=np.bool_)
    ripped[fs.read_label(str(label_path)).astype(np.int64)] = False
    cortex = fs.read_label(str(cortex_path)).astype(np.int64)
    outside_cortex = np.ones(len(white), dtype=np.bool_)
    outside_cortex[cortex] = False
    accepted_path = diagnostic_case / "last-accepted.private.npy"
    terminal_path = diagnostic_case / "cleanup-result.private.npy"
    accepted, terminal = np.load(accepted_path), np.load(terminal_path)
    for value in (accepted, terminal):
        if value.shape != white.shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("finite ordered float32 stage snapshots required")
    pinned = pin_medial_wall(accepted, white, cortex)
    # 写入/计算候选前只核对输入与有序网格，不消费参考输出坐标。
    for repeat in (1, 2):
        _, reference_faces = fs.read_geometry(str(reference_run / f"reference-{repeat}/subject/surf/lh.pial.T1"))
        if not np.array_equal(faces, reference_faces):
            raise ValueError("reference ordered faces differ before replay")
    report = {"schema_version": 1, "status": "running_diagnosis",
              "scope": "frozen failed-pial cleanup replay; not raw-T1 end-to-end",
              "producer_failed_quality_preserved": True,
              "same_input_reference_repeats_prevalidated": True,
              "input_sha256": expected_input_sha256,
              "snapshot_sha256": {"accepted": sha(accepted_path), "terminal": sha(terminal_path)},
              "script_sha256": sha(Path(__file__)), "threads": threads,
              "cpu_affinity": sorted(os.sched_getaffinity(0)),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "gpu_memory_bytes": None, "gpu_computation": False,
              "versions": {"numpy": np.__version__, "numba": numba.__version__, "nibabel": nibabel.__version__},
              "mesh": {"vertices": len(white), "faces": len(faces), "ripped_vertices": int(ripped.sum()),
                       "outside_cortex_vertices": int(outside_cortex.sum())},
              "active_source_sha256": {}, "source_replays": [], "qc": [],
              "overall_metric_equivalence": "not_assessed", "whole_pipeline_speedup": None}
    for name in ("place_surface_final_cleanup", "place_surface_intersection_marking",
                 "place_surface_collision", "place_surface_collision_torch",
                 "place_surface_smoothing", "mris_remove_intersection_python"):
        module = __import__("fnit.recon_all." + name, fromlist=["*"])
        report["active_source_sha256"][name] = sha(Path(module.__file__))
    def difference(a, b):
        d = np.linalg.norm(a.astype(np.float64)-b.astype(np.float64), axis=1)
        return {"different_coordinate_elements": int(np.count_nonzero(a != b)),
                "coordinate_bytes_equal": a.dtype == b.dtype and a.tobytes() == b.tobytes(),
                "mean_distance_mm": float(d.mean()), "p99_distance_mm": float(np.quantile(d, .99)),
                "max_distance_mm": float(d.max())}
    def source_qc(name, value):
        captured = {}
        def observer(frame, event, arg):
            if event == "return" and frame.f_code is mark_source_intersections.__code__:
                captured["faces"] = frame.f_locals["face_marks"].copy()
        tick = time.perf_counter(); previous = sys.getprofile()
        try:
            sys.setprofile(observer)
            marks, count = mark_source_intersections(value, faces, predicate_backend="numba")
        finally:
            sys.setprofile(previous)
        flagged = captured["faces"]
        triplets = ripped[faces[flagged]]
        row = {"state": name, "source_intersecting_faces": int(count),
               "source_marked_vertices": int(marks.sum()),
               "marked_ripped_vertices": int(np.count_nonzero(marks & ripped)),
               "marked_movable_vertices": int(np.count_nonzero(marks & ~ripped)),
               "residual_faces_all_ripped": int(np.count_nonzero(np.all(triplets, axis=1))),
               "residual_faces_any_ripped": int(np.count_nonzero(np.any(triplets, axis=1))),
               "marked_outside_cortex_vertices": int(np.count_nonzero(marks & outside_cortex)),
               "outside_cortex_differs_from_white": int(np.count_nonzero(value[outside_cortex] != white[outside_cortex])),
               "source_face_flags_sha256": hashlib.sha256(flagged.tobytes()).hexdigest(),
               "source_vertex_flags_sha256": hashlib.sha256(marks.tobytes()).hexdigest(),
               "source_qc_seconds": time.perf_counter()-tick}
        tick = time.perf_counter(); legacy_marks, legacy_count = mark_intersections(value, faces)
        row.update(legacy_float64_intersecting_faces=int(legacy_count),
                   legacy_float64_marked_vertices=int(legacy_marks.sum()),
                   source_legacy_vertex_flags_equal=bool(np.array_equal(marks, legacy_marks)),
                   legacy_float64_qc_seconds=time.perf_counter()-tick)
        report["qc"].append(row)
        atomic_write(output_directory / "report.json", report)
    source_qc("original_terminal", terminal)
    # 每次完整源规则重放；不改rip、不增迭代、不越过源停止规则。
    for name, value in (("pinned_before_original_cleanup", pinned), ("additional_terminal_cleanup", terminal)):
        tick = time.perf_counter()
        result, diagnostics = repair_intersections(value, faces, ripped, marking_backend="source_numba")
        seconds = time.perf_counter()-tick
        artifact = output_directory / (name + ".npy")
        np.save(artifact, result)
        report["source_replays"].append({"state": name, "complete_cleanup_seconds": seconds,
                                         "diagnostics": diagnostics,
                                         "artifact_sha256": sha(artifact),
                                         "vs_original_terminal": difference(result, terminal),
                                         "vs_replay_input": difference(result, value)})
        atomic_write(output_directory / "report.json", report)
        source_qc(name + "_output", result)
    # 原参考只在本轮两份候选产物保存后读取。
    references=[]
    for index in (1, 2):
        path=reference_run/f"reference-{index}/subject/surf/lh.pial.T1"
        xyz, other_faces=fs.read_geometry(str(path)); xyz=xyz.astype(np.float32)
        if xyz.shape != terminal.shape or not np.array_equal(faces, other_faces):
            raise ValueError("reference ordered vertex/face correspondence failed")
        references.append(xyz)
        report.setdefault("reference_diagnostics", []).append({"repeat": index, "file_sha256": sha(path),
            "same_ordered_faces": True, "terminal_vs_reference": difference(terminal, xyz)})
    report["reference_repeats_coordinate_bits_equal"]=references[0].tobytes()==references[1].tobytes()
    report["status"]="complete_diagnosis"
    report["producer_mesh_quality"]="failed" if report["qc"][0]["source_intersecting_faces"] else "passed"
    report["diagnostic_wall_seconds"]=time.perf_counter()-wall
    atomic_write(output_directory / "report.json", report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("frozen-source","overlay-source","diagnostic-case","reference-run","output-directory","expected-input-hashes"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--threads",type=int,default=4)
    a=p.parse_args()
    diagnose(frozen_source=a.frozen_source,overlay_source=a.overlay_source,diagnostic_case=a.diagnostic_case,
             reference_run=a.reference_run,output_directory=a.output_directory,
             expected_input_sha256=json.loads(a.expected_input_hashes.read_text()),threads=a.threads)


if __name__ == "__main__":
    main()
