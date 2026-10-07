"""Compare complete saved registration meshes and all accepted trajectories."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scientific_trajectory(value):
    if isinstance(value, dict):
        return {key: scientific_trajectory(item) for key, item in value.items()
                if "seconds" not in key and key not in {
                    "sphere", "smoothwm", "sulc", "atlas", "output", "sulc_seed"}}
    if isinstance(value, list):
        return [scientific_trajectory(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    import nibabel as nib
    import numpy as np
    import torch
    torch.set_num_threads(8)
    from fnit.recon_all.mris_register_nonlinear import face_area_normals

    images, arms = {}, {}
    for name in ("baseline", "candidate", "official"):
        output = args.runs / name / "lh.sphere.reg"
        xyz, faces, volume = nib.freesurfer.read_geometry(output, read_metadata=True)
        images[name] = (xyz, faces)
        queued = json.loads((args.runs / "queue" / ("registration_" + name + "_cpu8_v1") / "record.json").read_text())
        if queued["status"] != "complete" or queued["returncode"] != 0:
            raise ValueError("only complete output-producing processes may be compared")
        areas, _ = face_area_normals(torch.from_numpy(xyz).float(),
                                     torch.from_numpy(faces.astype(np.int64)), signed_sphere=True)
        arms[name] = {
            "vertices": len(xyz), "faces": len(faces),
            "file_sha256": digest(output), "coordinate_sha256": hashlib.sha256(xyz.tobytes()).hexdigest(),
            "ordered_faces_sha256": hashlib.sha256(faces.tobytes()).hexdigest(),
            "finite_coordinates": bool(np.isfinite(xyz).all()),
            "negative_saved_faces": int((areas < 0).sum()),
            "decoded_volume_geometry": {key: val.tolist() if isinstance(val, np.ndarray) else val
                                        for key, val in volume.items()},
            "cold_process_wall_seconds": queued["wall_seconds"],
            "sampled_tree_RSS_bytes": queued["maximum_sampled_tree_rss_bytes"],
            "load_before": queued["load_before"], "load_after": queued["load_after"],
            "threads": queued["max_cpu_threads"], "affinity": queued["cpu_affinity"],
        }
        if name != "official":
            record = json.loads((args.runs / name / "record.private.json").read_text())
            arms[name].update({key: record[key] for key in (
                "versions", "averaging_source_sha256", "common_source_sha256",
                "api_seconds_including_io", "averaging_seconds_including_first_JIT",
                "averaging_calls")})
            registration = record["registration"]
            arms[name]["input_sha256"] = registration["input_sha256"]
            arms[name]["sulc_seed_sha256"] = registration["sulc_seed_sha256"]
            arms[name]["stage_seconds_nested"] = {part: {key: value for key, value in registration[part].items() if "seconds" in key}
                                                   for part in ("sulc_pass", "smoothwm_pass")}
            arms[name]["trajectory"] = {part: scientific_trajectory(registration[part])
                                         for part in ("sulc_pass", "smoothwm_pass")}

    pairs = {}
    for left, right in (("baseline", "candidate"), ("official", "baseline"), ("official", "candidate")):
        a, faces_a = images[left]
        b, faces_b = images[right]
        aligned = a.shape == b.shape and np.array_equal(faces_a, faces_b)
        row = {"ordered_vertex_face_correspondence": aligned,
               "decoded_volume_geometry_equal": arms[left]["decoded_volume_geometry"] == arms[right]["decoded_volume_geometry"]}
        if aligned:
            difference = b - a
            distance = np.linalg.norm(difference, axis=1)
            row.update({"coordinates_exact": bool(np.array_equal(a, b)),
                        "different_coordinate_values": int(np.count_nonzero(a != b)),
                        "coordinate_MAE_mm": float(np.abs(difference).mean()),
                        "coordinate_RMSE_mm": float(np.sqrt(np.mean(difference * difference))),
                        "coordinate_max_abs_mm": float(np.abs(difference).max()),
                        "vertex_displacement_mean_mm": float(distance.mean()),
                        "vertex_displacement_p95_mm": float(np.percentile(distance, 95)),
                        "vertex_displacement_max_mm": float(distance.max())})
        else:
            row["indexed_coordinate_comparison"] = "not_assessed_no_correspondence"
        pairs[left + "_vs_" + right] = row
    old, new = arms["baseline"], arms["candidate"]
    invariant = {key: old[key] == new[key] for key in (
        "common_source_sha256", "input_sha256", "trajectory", "sulc_seed_sha256")}
    invariant.update(pairs["baseline_vs_candidate"])
    passed = all(invariant[key] for key in (
        "common_source_sha256", "input_sha256", "trajectory", "sulc_seed_sha256",
        "ordered_vertex_face_correspondence", "coordinates_exact", "decoded_volume_geometry_equal"))
    report = {"scope": "complete_same_input_LH_registration_not_recon_all_or_cortical_metrics",
              "arms": arms, "pairs": pairs, "old_new_invariants": invariant,
              "old_new_gate_passed": passed,
              "speed_observation": {"old_over_new_cold_process": old["cold_process_wall_seconds"] / new["cold_process_wall_seconds"],
                                    "old_over_new_API": old["api_seconds_including_io"] / new["api_seconds_including_io"],
                                    "paired_repeats": 1, "shared_node": True}}
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"old_new_gate_passed": passed, "pairs": pairs, "speed_observation": report["speed_observation"]}))
    if not passed:
        raise SystemExit("complete old/new registration invariant failed")


if __name__ == "__main__":
    main()
