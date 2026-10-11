"""隔离 benchmark：同一自产 white/MRI 上的 Conda 源码参考 pial 重复性。

仅复制输入至独立新目录，复用已有 pial_t1_conda.run_pial_t1；不写生产
目录、不把参考结果送回优化器。输入/程序/源码 SHA、完整 exec/I/O 和
逐元素有序网格、源判定相交数均留报告。没有 GPU 生产后端改动。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def required(hemi: str) -> list[str]:
    return [f"surf/{hemi}.white", f"surf/autodet.gw.stats.{hemi}.dat",
            "mri/brain.finalsurfs.mgz", "mri/wm.mgz", "mri/aseg.presurf.mgz",
            f"label/{hemi}.cortex.label", f"label/{hemi}.cortex+hipamyg.label",
            f"label/{hemi}.aparc.annot"]


def worker(request: dict) -> int:
    """独立参考目录的完整 C++ 调用；生产可执行文件只在此参考路径执行。"""
    sys.path.insert(0, request["source_root"])
    from fnit.recon_all.pial_t1_conda import run_pial_t1
    started = time.perf_counter()
    result = {"status": "running", "program_sha256": sha(Path(request["binary"])),
              "input_sha256": request["input_sha256"]}
    try:
        subject = Path(request["subject"])
        if any(sha(subject/name) != digest for name, digest in request["input_sha256"].items()):
            raise RuntimeError("reference input SHA changed")
        result["stage"] = run_pial_t1(subject_dir=subject, hemi=request["hemi"],
            binary=request["binary"], assets_dir=request["assets_dir"], threads=request["threads"])
        result["status"] = "complete"
    except Exception as error:
        result.update(status="failed", error=repr(error))
    result["worker_wall_seconds"] = time.perf_counter()-started
    write(Path(request["report"]), result)
    return 0 if result["status"] == "complete" else 1


def driver(args) -> int:
    """同一主机/四线程两次 fresh exec；另做源质量检查，不计入优化器墙钟。"""
    tick = time.perf_counter()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    inputs = {name: sha(args.subject/name) for name in required(args.hemi)}
    args.output_dir.mkdir(parents=True)
    sys.path.insert(0, str(args.source_root))
    import numpy as np
    import nibabel as nib
    from numba import set_num_threads
    from fnit.recon_all.place_surface_intersection_marking import mark_source_intersections
    set_num_threads(args.threads)
    reference_wrapper = args.source_root/"fnit/recon_all/pial_t1_conda.py"
    source_marker = args.source_root/"fnit/recon_all/place_surface_intersection_marking.py"
    manifest = {"scope": "isolated Conda fixed-source complete pial reference repeatability, not production or whole recon-all",
                "subject": str(args.subject), "binary": str(args.binary), "assets_dir": str(args.assets_dir),
                "program_sha256": sha(args.binary), "script_sha256": sha(Path(__file__)),
                "reference_wrapper_sha256": sha(reference_wrapper), "source_marker_sha256": sha(source_marker),
                "input_sha256": inputs, "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "threads": args.threads, "source_root": str(args.source_root),
                "ldd": subprocess.run(["ldd", str(args.binary)], capture_output=True, text=True).stdout,
                "isolation_installation_verdict": "not_verified", "overall_metric_equivalence": "not_assessed"}
    write(args.output_dir/"manifest.private.json", manifest)
    rows = []
    for index in (1, 2):
        item_start = time.perf_counter()
        run = args.output_dir/f"reference-{index}"
        subject = run/"subject"
        run.mkdir()
        for name in inputs:
            target = subject/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.subject/name, target)
        request = {"source_root": str(args.source_root), "subject": str(subject), "hemi": args.hemi,
                   "binary": str(args.binary), "assets_dir": str(args.assets_dir), "threads": args.threads,
                   "input_sha256": inputs, "report": str(run/"child.private.json")}
        write(run/"request.private.json", request)
        copy_seconds = time.perf_counter()-item_start
        process_start = time.perf_counter()
        with (run/"reference.log").open("wb") as stream:
            proc = subprocess.run([sys.executable, str(Path(__file__)), "--child-request", str(run/"request.private.json")],
                                  stdout=stream, stderr=subprocess.STDOUT, check=False)
        wall = time.perf_counter()-process_start
        child = json.loads((run/"child.private.json").read_text())
        row = {"index": index, "returncode": proc.returncode, "child": child,
               "copy_seconds": copy_seconds, "fresh_exec_wall_seconds": wall}
        if child["status"] == "complete":
            output = subject/f"surf/{args.hemi}.pial.T1"
            xyz, faces = nib.freesurfer.read_geometry(str(output))
            white, white_faces = nib.freesurfer.read_geometry(str(subject/f"surf/{args.hemi}.white"))
            qc_tick = time.perf_counter()
            marks, count = mark_source_intersections(xyz.astype(np.float32), faces,
                                                     predicate_backend="numba")
            row.update(output_sha256=sha(output), finite_coordinates=bool(np.isfinite(xyz).all()),
                       vertices=len(xyz), faces=len(faces), same_ordered_mesh=np.array_equal(faces, white_faces)
                       and xyz.shape == white.shape, source_intersecting_faces=int(count),
                       source_marked_vertices=int(marks.sum()), source_qc_seconds=time.perf_counter()-qc_tick)
        write(run/"benchmark.private.json", row)
        rows.append(row)
        if proc.returncode:
            break
    comparison = None
    if len(rows) == 2 and all(row["child"]["status"] == "complete" for row in rows):
        a, fa = nib.freesurfer.read_geometry(str(args.output_dir/f"reference-1/subject/surf/{args.hemi}.pial.T1"))
        b, fb = nib.freesurfer.read_geometry(str(args.output_dir/f"reference-2/subject/surf/{args.hemi}.pial.T1"))
        ordered = a.shape == b.shape and np.array_equal(fa, fb)
        distance = np.linalg.norm(a-b, axis=1) if ordered else None
        comparison = {"same_ordered_mesh": ordered, "different_coordinate_elements": int(np.count_nonzero(a!=b)) if ordered else None,
                      "distance_p99_mm": float(np.quantile(distance, .99)) if ordered else None,
                      "distance_max_mm": float(distance.max()) if ordered else None,
                      "whole_file_bytes_equal": rows[0]["output_sha256"] == rows[1]["output_sha256"]}
    result = {**manifest, "status": "complete" if comparison is not None else "failed",
              "rows": rows, "repeatability": comparison, "whole_pipeline_speedup": None,
              "full_driver_wall_seconds": time.perf_counter()-tick,
              "timing_scope": "fresh exec includes Python import and native read/write; source-QC is separate postprocessing, not native optimizer time"}
    write(args.output_dir/"summary.private.json", result)
    return 0 if comparison is not None else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child-request", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--subject", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--assets-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--hemi", choices=("lh", "rh"), default="lh")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.child_request:
        return worker(json.loads(args.child_request.read_text()))
    if not all((args.source_root, args.subject, args.binary, args.assets_dir, args.output_dir)) or args.threads < 1:
        parser.error("all paths and positive threads required")
    return driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
