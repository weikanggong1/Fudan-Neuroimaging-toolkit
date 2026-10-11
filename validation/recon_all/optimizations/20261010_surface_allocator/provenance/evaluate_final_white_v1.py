"""只读评价已完成的缓存阶段；MRI 数值/几何、旧自产输出和程序版本绑定。

不启动 CUDA，不重跑优化器，不写候选/参考影像。baseline-subject 必须是
此前 FNIT 自产阶段，不能把跨 GPU/线程旧墙钟当作本次同条件控制。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import nibabel as nib
import numpy as np


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def compare(*, run_dir: Path, baseline_subject: Path, hemi: str, source_manifest: Path) -> dict:
    manifest = json.loads((run_dir / "manifest.private.json").read_text())
    result = {"scope": "read-only completed stage diagnostic; not official benchmark or whole recon-all",
              "host": platform.node(), "evaluator_sha256": sha(__file__),
              "frozen_source_manifest_sha256": sha(source_manifest),
              "run_manifest_sha256": sha(run_dir / "manifest.private.json"),
              "python_executable_sha256": sha(sys.executable), "inputs": {}, "cases": [],
              "overall_metric_equivalence": "not_assessed", "whole_pipeline_speedup": None}
    cpu = Path("/proc/cpuinfo")
    result["cpu_model"] = next((line.partition(":")[2].strip() for line in cpu.read_text().splitlines()
                                if line.startswith("model name")), None) if cpu.exists() else None
    for name, expected in manifest["input_sha256"].items():
        path = baseline_subject / name
        if sha(path) != expected:
            raise RuntimeError("baseline input differs from frozen experiment: " + name)
        info = {"file_sha256": expected, "bytes": path.stat().st_size}
        if path.suffix == ".mgz":
            image = nib.load(str(path)); array = np.asarray(image.dataobj)
            info.update(shape=list(array.shape), dtype=str(array.dtype),
                        data_sha256=array_sha(array), affine=image.affine.tolist(),
                        voxel_size_mm=list(map(float, image.header.get_zooms()[:3])))
        elif name.endswith("white.preaparc"):
            xyz, faces = nib.freesurfer.read_geometry(str(path))
            info.update(vertices=len(xyz), faces=len(faces), coordinate_dtype=str(xyz.dtype),
                        coordinate_sha256=array_sha(xyz), ordered_faces_sha256=array_sha(faces))
        elif name.endswith(".dat"):
            info["threshold_text"] = path.read_text()
        elif name.endswith(".annot"):
            labels, ctab, names = nib.freesurfer.read_annot(str(path), orig_ids=True)
            info.update(labels_shape=list(labels.shape), label_dtype=str(labels.dtype),
                        labels_sha256=array_sha(labels), ctab_sha256=array_sha(ctab))
        elif name.endswith(".label"):
            labels = nib.freesurfer.read_label(str(path))
            info.update(vertices=len(labels), vertex_indices_sha256=array_sha(labels))
        result["inputs"][name] = info
    original = baseline_subject / "surf" / f"{hemi}.white"
    old_report = baseline_subject / "scripts" / f"{hemi}.white.python.json"
    base, base_faces = nib.freesurfer.read_geometry(str(original))
    result["baseline_output_sha256"] = sha(original)
    result["baseline_coordinate_sha256"] = array_sha(base)
    old = json.loads(old_report.read_text())
    old_volume_path = baseline_subject / "mri/mrisps.white.mgz"
    old_volume = nib.load(str(old_volume_path))
    old_data = np.asarray(old_volume.dataobj)
    result["baseline_stage_report_sha256"] = sha(old_report)
    result["baseline_stage_seconds"] = old["seconds"]
    result["baseline_timing_scope"] = "historical v4: other GPU and two threads per hemisphere; no speed ratio to this old wall"
    first = None
    for path in sorted(run_dir.glob("*/child.private.json")):
        report = json.loads(path.read_text())
        if report["status"] != "complete":
            result["cases"].append({"case": path.parent.name, "status": report["status"]})
            continue
        surface = Path(report["request"]["subject"]) / "surf" / f"{hemi}.white"
        xyz, faces = nib.freesurfer.read_geometry(str(surface))
        volume_path = surface.parent.parent / "mri/mrisps.white.mgz"
        volume = nib.load(str(volume_path))
        volume_data = np.asarray(volume.dataobj)
        same_volume_grid = volume.shape == old_volume.shape and np.array_equal(volume.affine, old_volume.affine)
        ordered = xyz.shape == base.shape and np.array_equal(faces, base_faces)
        distance = np.linalg.norm(xyz - base, axis=1) if ordered else None
        row = {"case": path.parent.name, "status": "complete", "child_report_sha256": sha(path),
               "same_ordered_mesh_to_v4": ordered,
               "coordinate_different_elements_to_v4": int(np.count_nonzero(xyz != base)) if ordered else None,
               "distance_mean_mm_to_v4": float(distance.mean()) if ordered else None,
               "distance_p99_mm_to_v4": float(np.quantile(distance, .99)) if ordered else None,
               "distance_max_mm_to_v4": float(distance.max()) if ordered else None,
               "white_bytes_equal_to_v4": sha(surface) == sha(original),
               "volume_bytes_equal_to_v4": sha(volume_path) == sha(old_volume_path),
               "volume_shape_affine_equal_to_v4": same_volume_grid,
               "volume_dtype_equal_to_v4": str(volume_data.dtype) == str(old_data.dtype),
               "volume_different_voxels_to_v4": int(np.count_nonzero(volume_data != old_data)) if same_volume_grid else None,
               "trial_decisions_equal_to_v4": report["stage"]["per_step"] == old["per_step"],
               "complete_api_seconds": report["complete_api_seconds"], "rusage": report["rusage"],
               "stage_seconds": report["stage"]["stage_seconds"], "torch_memory": report["torch_memory"]}
        if first is not None:
            row["accepted_trajectory_equal_to_first"] = report["accepted_steps"] == first["accepted_steps"]
            row["output_sha256_equal_to_first"] = report["output_sha256"] == first["output_sha256"]
        else:
            first = report
        result["cases"].append(row)
    result["completed_case_count"] = sum(row["status"] == "complete" for row in result["cases"])
    result["running_case_directories"] = [path.name for path in sorted(run_dir.iterdir())
        if path.is_dir() and (path/"request.private.json").exists() and not (path/"child.private.json").exists()]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-subject", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--hemi", choices=("lh", "rh"), default="lh")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(args.output_json)
    value = compare(run_dir=args.run_dir, baseline_subject=args.baseline_subject,
                    hemi=args.hemi, source_manifest=args.source_manifest)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
