#!/usr/bin/env python3
"""Read-only CPU comparison after a fresh official reconstruction/cache check.

Reuse FNIT's existing exact array/payload comparison helper. Full identities and
MRI hashes remain in the private report; the public report contains numbers only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-report", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--case-id", default="sub-CON01")
    parser.add_argument("--reference-helper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("run with CUDA_VISIBLE_DEVICES='' for the CPU-only comparison")
    new = json.loads(args.new_report.read_text())
    if (new.get("status") != "complete" or new.get("second_stage") != "skipped"
            or new.get("subject_bytes_unchanged") is not True):
        raise ValueError("fresh official reconstruction/cache check has not completed")
    binding = json.loads(args.bindings.read_text())["cases"][args.case_id]
    baseline_dir = Path(binding["anatomy"]["directory"])
    candidate_dir = Path(new["first_metadata"]["subject_dir"])
    baseline_meta_path = baseline_dir.parent.parent / "recon_report.json"
    baseline_meta = json.loads(baseline_meta_path.read_text())
    if baseline_meta.get("status") != "completed" or baseline_meta.get("exit_code") != 0:
        raise ValueError("bound existing official reconstruction is not completed")
    raw_verification = [item for item in baseline_meta["input_verification"] if item["kind"] == "raw_t1w"]
    if (len(raw_verification) != 1 or raw_verification[0]["sha256"] != new["raw_t1"]["sha256"]
            or raw_verification[0]["actual_sha256"] != new["raw_t1"]["sha256"]):
        raise ValueError("existing and fresh official runs have no verified identical raw T1")
    spec = importlib.util.spec_from_file_location("fnit_official_comparison", args.reference_helper)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    nib, np = helper.scientific_modules()
    helper_sha = hashlib.sha256(args.reference_helper.read_bytes()).hexdigest()
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    fs_home = Path(new["first_metadata"]["identity"]["options"]["freesurfer_home"])
    actual_version = (fs_home / "build-stamp.txt").read_text().strip()
    source_command = baseline_meta["command"]
    candidate_command = new["first_metadata"]["command"]
    public = {"schema_version": 1, "scope": "same raw T1: existing official output versus fresh official adapter; CPU only",
              "same_raw_t1_verified": True, "source_commit": new["source_commit"],
              "helper_sha256": helper_sha, "comparison_script_sha256": script_sha,
              "versions": {"baseline": baseline_meta["freesurfer_version"]["stdout"].strip(), "candidate": actual_version},
              "threads": {"baseline": baseline_meta.get("cpu_threads", "unknown"), "candidate": new["threads"],
                          "baseline_parallel_flag": "-parallel" in source_command,
                          "candidate_parallel_flag": "-parallel" in candidate_command,
                          "baseline_cpu_affinity": "unknown", "candidate_cpu_affinity": new["cpu_affinity"]},
              "versions_equal": baseline_meta["freesurfer_version"]["stdout"].strip() == actual_version,
              "CPU_only": True, "volumes": {}, "surfaces": {}}
    private = {"baseline_directory": str(baseline_dir), "candidate_directory": str(candidate_dir),
               "baseline_recon_report": str(baseline_meta_path), "raw_t1_verification": raw_verification,
               "fresh_report": str(args.new_report), "scientific_volume_results": {}, "scientific_surface_results": {}}
    touched = {}

    def compare_verified(relative, reader):
        result = helper.compared_files(baseline_dir / relative, candidate_dir / relative, reader, touched)
        if result["baseline_file"]["sha256"] != baseline_meta["anatomy"][relative]["sha256"]:
            raise RuntimeError("existing official scientific output changed since its reconstruction record")
        if result["candidate_file"]["sha256"] != new["first_metadata"]["anatomy"]["files"][relative]["sha256"]:
            raise RuntimeError("fresh official scientific output changed since its verified completion")
        return result

    for relative in ("mri/brain.mgz", "mri/aparc+aseg.mgz", "mri/ribbon.mgz"):
        left, right = baseline_dir / relative, candidate_dir / relative
        result = compare_verified(relative, helper.volume)
        private["scientific_volume_results"][relative] = result
        data = result["voxel_data"]
        grid_equal = (data["shape_equal"] and result["scanner_RAS_affine"]["exact_scientific_array_equal"]
                      and result["surface_RAS_vox2ras_tkr"]["exact_scientific_array_equal"]
                      and result["zooms"]["exact_scientific_array_equal"] and result["orientation"]["equal"])
        summary = {"same_grid": grid_equal, "baseline_shape": data["baseline"]["shape"],
                   "candidate_shape": data["candidate"]["shape"], "dtype_equal": data["dtype_equal"],
                   "affine_max_abs_error": result["scanner_RAS_affine"]["max_abs_error"],
                   "strict_scientific_equal": result["strict_scientific_equal"]}
        if grid_equal:
            summary.update(voxel_elements=data["baseline"]["elements"], changed_voxels=data["numeric_neq"],
                           changed_scalar_bits=data["raw_scalar_bits_neq"], rmse=data["rmse"],
                           max_abs_error=data["max_abs_error"])
            if relative != "mri/brain.mgz":
                a, b = np.asarray(nib.load(left).dataobj), np.asarray(nib.load(right).dataobj)
                summary.update(label_XOR_voxels=int(np.count_nonzero(a != b)),
                               label_XOR_fraction=float(np.count_nonzero(a != b) / a.size),
                               foreground_mask_XOR_voxels=int(np.count_nonzero((a != 0) != (b != 0))))
        else:
            summary["voxel_error_status"] = "not comparable on a shared physical grid; no resampling performed"
        public["volumes"][relative] = summary
    for hemi in ("lh", "rh"):
        for kind in ("white", "pial"):
            relative = f"surf/{hemi}.{kind}"
            left, right = baseline_dir / relative, candidate_dir / relative
            result = compare_verified(relative, helper.surface)
            private["scientific_surface_results"][relative] = result
            faces, coords = result["faces"], result["coordinates"]
            correspondence = faces["exact_scientific_array_equal"] and coords["shape_equal"]
            summary = {"baseline_vertex_count": coords["baseline"]["shape"][0],
                       "candidate_vertex_count": coords["candidate"]["shape"][0],
                       "baseline_face_count": faces["baseline"]["shape"][0],
                       "candidate_face_count": faces["candidate"]["shape"][0],
                       "ordered_faces_equal": faces["exact_scientific_array_equal"],
                       "face_index_elements_changed": faces["numeric_neq"],
                       "vertex_correspondence": "same ordered face arrays and vertex count" if correspondence else "not established",
                       "strict_scientific_equal": result["strict_scientific_equal"]}
            if correspondence:
                a, _ = nib.freesurfer.read_geometry(left)
                b, _ = nib.freesurfer.read_geometry(right)
                distances = np.linalg.norm(a - b, axis=1)
                summary.update(coordinate_elements_changed=coords["numeric_neq"], coordinate_rmse_mm=coords["rmse"],
                               coordinate_max_abs_error_mm=coords["max_abs_error"],
                               vertex_distance_mean_mm=float(distances.mean()), vertex_distance_p95_mm=float(np.percentile(distances, 95)),
                               vertex_distance_max_mm=float(distances.max()),
                               coordinates_payload_equal=result["stored_payload"]["coordinates_payload_equal"])
            else:
                summary["coordinate_error_status"] = "not compared pointwise without vertex correspondence"
            public["surfaces"][relative] = summary
    if any(helper.sha(path) != expected for path, expected in touched.items()):
        raise RuntimeError("scientific input changed during CPU comparison")
    public.update(status="complete", CPU_wall_seconds=time.perf_counter() - started,
                  completed_utc=datetime.now(timezone.utc).isoformat(),
                  all_seven_outputs_exact_equal=all(result["strict_scientific_equal"] for group in (public["volumes"], public["surfaces"]) for result in group.values()))
    private.update(public, scientific_files_read=touched)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "comparison.private.json", private)
    write_json(args.output / "comparison.public.json", public)
    print(json.dumps(public, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
