"""冻结真实球面/相交表面文件API回归，原生程序只生成隔离参考。

参数全部具名；输出原始私有JSON及表面图。不能把此阶段结果称为原始T1整例。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import nibabel.freesurfer.io as fsio
import numpy as np
import torch
from numba import set_num_threads

from fnit.recon_all.mrisp_paint_python import run_paint_atlas
from fnit.recon_all.mris_remove_intersection_python import run_remove_intersection
from fnit.recon_all.inflate_standard_run import run_standard_inflate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def error(actual, expected):
    delta = np.abs(np.asarray(actual, np.float64) - np.asarray(expected, np.float64))
    return {"different_elements": int(np.count_nonzero(delta)), "maximum": float(delta.max(initial=0)),
            "p99": float(np.percentile(delta, 99)) if delta.size else 0,
            "mean": float(delta.mean()) if delta.size else 0}


def reference(command, output_root, label):
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - started
    (output_root / f"{label}.reference.log").write_text(completed.stdout + completed.stderr)
    if completed.returncode:
        raise RuntimeError(f"reference failed ({completed.returncode}), see isolated benchmark log")
    return {"command": list(map(str, command)), "returncode": completed.returncode,
            "seconds_including_process_start_and_io": elapsed}


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, action="append", default=[])
    parser.add_argument("--atlas-lh", type=Path)
    parser.add_argument("--atlas-rh", type=Path)
    parser.add_argument("--paint-reference", type=Path)
    parser.add_argument("--nofix-reference", type=Path)
    parser.add_argument("--intersection-input", type=Path, action="append", default=[])
    parser.add_argument("--intersection-reference", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--require-zero", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    modules = ["mrisp_paint_python", "mris_register_atlas", "place_surface_smoothing",
               "mris_remove_intersection_python", "place_surface_final_cleanup",
               "place_surface_intersection_marking", "place_surface_collision",
               "place_surface_collision_torch", "place_surface_candidates_torch",
               "inflate_standard_run", "inflate_python", "inflate_torch", "inflate_topology",
               "place_surface_normals", "sphere_standard_python"]
    report = {"scope": "frozen_same_input_file_API_not_raw_T1_whole_case", "code_commit": args.code_commit,
              "candidate_source_sha256": {name: sha(importlib.import_module("fnit.recon_all." + name).__file__)
                                           for name in modules}, "script_sha256": sha(__file__),
              "host": platform.node(), "platform": platform.platform(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
              "threads": args.threads, "device": args.device, "torch_version": torch.__version__,
              "precision": {"matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                            "cudnn_tf32": torch.backends.cudnn.allow_tf32, "autocast": False,
                            "image_coordinate_dtype": "float32", "intersection_predicate_dtype": "float64"},
              "allocator_environment": {name: os.environ.get(name) for name in
                                        ("PYTORCH_NO_CUDA_MEMORY_CACHING", "PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")},
              "paint": [], "intersection": [], "nofix_inflate": [], "overall_equivalence": "not_assessed"}
    for case_number, subject in enumerate(args.subject):
        for hemi in ("lh", "rh"):
            if args.nofix_reference is not None:
                input_surface = subject / "surf" / f"{hemi}.smoothwm.nofix"
                prefix = f"{hemi}.nofix_case{case_number}"
                candidate, official, repeat = (args.output_dir / f"{prefix}.{name}" for name in ("candidate", "reference", "repeat"))
                own = run_standard_inflate(input_surface=input_surface, inflated_output=candidate,
                                           sulc_output=None, save_sulc=False,
                                           backend="torch" if device.type == "cuda" else "numpy", device=args.device)
                control = reference([str(args.nofix_reference), "-no-save-sulc", str(input_surface), str(official)], args.output_dir, prefix)
                repeated = reference([str(args.nofix_reference), "-no-save-sulc", str(input_surface), str(repeat)], args.output_dir, prefix + "_repeat")
                xyz, faces, metadata = fsio.read_geometry(str(candidate), read_metadata=True)
                ref_xyz, ref_faces, ref_metadata = fsio.read_geometry(str(official), read_metadata=True)
                rep_xyz, rep_faces = fsio.read_geometry(str(repeat))
                report["nofix_inflate"].append({"case_number": case_number, "hemisphere": hemi, "candidate": own,
                                               "input_sha256": sha(input_surface), "reference": control,
                                               "reference_repeat": repeated, "reference_program_sha256": sha(args.nofix_reference),
                                               "ordered_faces_equal": np.array_equal(faces, ref_faces),
                                               "candidate_reference_coordinate_error": error(xyz, ref_xyz),
                                               "reference_repeat_coordinate_error": error(rep_xyz, ref_xyz),
                                               "reference_repeat_ordered_faces_equal": np.array_equal(rep_faces, ref_faces),
                                               "geometry_header_equal": set(metadata) == set(ref_metadata) and all(
                                                   np.array_equal(metadata[key], ref_metadata[key]) for key in metadata),
                                               "output_directory_sulc_files": list(map(str, args.output_dir.glob("*.sulc")))})
            if args.paint_reference is None:
                if args.nofix_reference is None:
                    raise ValueError("subjects require paint-reference or nofix-reference")
                continue
            atlas = getattr(args, "atlas_" + hemi)
            if atlas is None or args.paint_reference is None:
                raise ValueError("paint subjects require both atlases and paint-reference")
            sphere = subject / "surf" / f"{hemi}.sphere.reg"
            prefix = f"{hemi}.paint_case{case_number}"
            candidate, official, repeat = (args.output_dir / f"{prefix}.{name}" for name in ("candidate", "reference", "repeat"))
            own = run_paint_atlas(atlas_path=atlas, sphere_path=sphere, output_path=candidate,
                                 frame=6, averages=5, device=args.device)
            control = reference([str(args.paint_reference), "-a", "5", f"{atlas}#6", str(sphere), str(official)],
                                args.output_dir, prefix)
            repeated = reference([str(args.paint_reference), "-a", "5", f"{atlas}#6", str(sphere), str(repeat)],
                                 args.output_dir, prefix + "_repeat")
            values, expected, rerun = map(lambda p: fsio.read_morph_data(str(p)), (candidate, official, repeat))
            report["paint"].append({"case_number": case_number, "hemisphere": hemi, "candidate": own,
                                    "reference": control, "reference_repeat": repeated,
                                    "reference_program_sha256": sha(args.paint_reference),
                                    "reference_repeat_error": error(rerun, expected),
                                    "candidate_reference_error": error(values, expected),
                                    "morph_bytes_equal": candidate.read_bytes() == official.read_bytes()})
    for number, input_path in enumerate(args.intersection_input):
        if args.intersection_reference is None:
            raise ValueError("intersection inputs require intersection-reference")
        prefix = f"intersection_case{number}"
        candidate, official, repeat = (args.output_dir / f"{prefix}.{name}" for name in ("candidate", "reference", "repeat"))
        own = run_remove_intersection(input_path=input_path, output_path=candidate,
                                      marking_backend="source_torch" if device.type == "cuda" else "source_numba",
                                      device=args.device, candidate_grid_cells_per_axis=2,
                                      require_zero=args.require_zero)
        control = reference([str(args.intersection_reference), str(input_path), str(official)], args.output_dir, prefix)
        repeated = reference([str(args.intersection_reference), str(input_path), str(repeat)], args.output_dir, prefix + "_repeat")
        xyz, faces, metadata = fsio.read_geometry(str(candidate), read_metadata=True)
        ref_xyz, ref_faces, ref_metadata = fsio.read_geometry(str(official), read_metadata=True)
        rep_xyz, rep_faces = fsio.read_geometry(str(repeat))
        row = {"candidate": own, "reference": control, "reference_repeat": repeated,
               "reference_program_sha256": sha(args.intersection_reference),
               "ordered_faces_equal": np.array_equal(faces, ref_faces),
               "candidate_reference_coordinate_error": error(xyz, ref_xyz),
               "reference_repeat_coordinate_error": error(rep_xyz, ref_xyz),
               "reference_repeat_ordered_faces_equal": np.array_equal(rep_faces, ref_faces),
               "geometry_header_equal": set(metadata) == set(ref_metadata) and all(
                   np.array_equal(metadata[key], ref_metadata[key]) for key in metadata),
               "raw_file_bytes_equal": candidate.read_bytes() == official.read_bytes()}
        if xyz.shape == ref_xyz.shape and np.array_equal(faces, ref_faces):
            distances = np.linalg.norm(xyz - ref_xyz, axis=1)
            row["same_index_distance_mm"] = {"maximum": float(distances.max(initial=0)),
                                              "p99": float(np.percentile(distances, 99))}
        report["intersection"].append(row)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        report["memory"] = {"target_device": args.device,
                            "torch_allocated_peak_bytes": torch.cuda.max_memory_allocated(device),
                            "torch_reserved_peak_bytes": torch.cuda.max_memory_reserved(device),
                            "parent_child_simultaneous_process_tree_peak_bytes": None,
                            "process_tree_observation": "not_measured_in_this_stage_script"}
    report["benchmark_total_seconds_including_loading_transfers_io_and_references"] = time.perf_counter() - started
    report["execution_status"] = "complete"
    (args.output_dir / "report.private.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"execution_status": "complete", "paint_cases": len(report["paint"]),
                      "intersection_cases": len(report["intersection"])}))


if __name__ == "__main__":
    main()
