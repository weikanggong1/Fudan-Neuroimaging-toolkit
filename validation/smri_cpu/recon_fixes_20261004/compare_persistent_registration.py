"""Compare the isolated persistent-team pilot with saved accepted/native outputs."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference-runs", "pilot-runs", "comparator", "report", "cpp-source", "adapter", "library"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    import nibabel as nib
    import numpy as np
    import torch
    torch.set_num_threads(8)
    from fnit.recon_all.mris_register_nonlinear import face_area_normals
    spec = importlib.util.spec_from_file_location("full_registration_comparator", args.comparator)
    comparator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparator)
    reference = json.loads((args.reference_runs / "candidate/record.private.json").read_text())
    pilot = json.loads((args.pilot_runs / "candidate/record.private.json").read_text())
    reference_queue = json.loads((args.reference_runs / "queue/registration_candidate_cpu8_v1/record.json").read_text())
    pilot_queue = json.loads((args.pilot_runs / "queue/persistent_registration_cpu8_v1/record.json").read_text())
    for record in (reference_queue, pilot_queue):
        assert record["status"] == "complete" and record["returncode"] == 0
    comparisons = {}
    saved = args.pilot_runs / "candidate/lh.sphere.reg"
    xyz, faces, volume = nib.freesurfer.read_geometry(saved, read_metadata=True)
    areas, _ = face_area_normals(torch.from_numpy(xyz).float(), torch.from_numpy(faces.astype(np.int64)), signed_sphere=True)
    for name in ("candidate", "official"):
        other = args.reference_runs / name / "lh.sphere.reg"
        other_xyz, other_faces, other_volume = nib.freesurfer.read_geometry(other, read_metadata=True)
        delta = xyz - other_xyz
        comparisons[name] = {
            "ordered_faces_exact": bool(np.array_equal(faces, other_faces)),
            "coordinate_values_exact": bool(np.array_equal(xyz, other_xyz)),
            "coordinate_different_values": int(np.count_nonzero(xyz != other_xyz)),
            "coordinate_max_abs_mm": float(np.abs(delta).max()),
            "coordinate_RMSE_mm": float(np.sqrt(np.mean(delta * delta))),
            "decoded_volume_geometry_exact": comparator.scientific_trajectory({key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in volume.items()}) == comparator.scientific_trajectory({key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in other_volume.items()}),
            "file_sha256": digest(other),
        }
    checks = {"common_source_sha256_exact": reference["common_source_sha256"] == pilot["common_source_sha256"],
              "inputs_exact": reference["registration"]["input_sha256"] == pilot["registration"]["input_sha256"],
              "trajectory_exact": comparator.scientific_trajectory(reference["registration"]) == comparator.scientific_trajectory(pilot["registration"]),
              "negative_saved_faces": int((areas < 0).sum()),
              "finite_coordinates": bool(np.isfinite(xyz).all())}
    gate = all(checks[key] for key in ("common_source_sha256_exact", "inputs_exact", "trajectory_exact", "finite_coordinates")) and checks["negative_saved_faces"] == 0 and all(all(row[key] for key in ("ordered_faces_exact", "coordinate_values_exact", "decoded_volume_geometry_exact")) for row in comparisons.values())
    def arm(record, queued):
        return {"cold_process_wall_seconds": queued["wall_seconds"],
                "sampled_tree_RSS_bytes": queued["maximum_sampled_tree_rss_bytes"],
                "affinity": queued["cpu_affinity"], "threads": queued["max_cpu_threads"],
                "load_before": queued["load_before"], "load_after": queued["load_after"],
                "API_seconds": record["api_seconds_including_io"],
                "averaging_seconds_including_first_JIT": record["averaging_seconds_including_first_JIT"],
                "averaging_calls": record["averaging_calls"],
                "stage_seconds_nested": {part: {key: value for key, value in record["registration"][part].items() if "seconds" in key} for part in ("sulc_pass", "smoothwm_pass")}}
    report = {"scope": "isolated_persistent_CPP_complete_same_input_LH_registration",
              "gate_passed": gate, "checks": checks, "comparisons": comparisons,
              "vertices": len(xyz), "faces": len(faces),
              "file_sha256": digest(saved), "coordinate_sha256": hashlib.sha256(xyz.tobytes()).hexdigest(),
              "input_sha256": pilot["registration"]["input_sha256"],
              "common_source_sha256": pilot["common_source_sha256"],
              "source_sha256": {"cpp": digest(args.cpp_source), "adapter": digest(args.adapter), "library": digest(args.library), "comparator": digest(args.comparator), "worker": digest(__file__)},
              "accepted_reference": arm(reference, reference_queue),
              "isolated_pilot": arm(pilot, pilot_queue),
              "production_adopted": False,
              "timing_limit": "Single shared-node pilot and earlier nonadjacent reference. Not a paired complete speed benchmark. No original T1 reconstruction, other hemisphere, or CUDA changes."}
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"gate_passed": gate, "checks": checks, "time": report["isolated_pilot"]}))
    if not gate:
        raise SystemExit("persistent-team full-stage gate failed")


if __name__ == "__main__":
    main()
