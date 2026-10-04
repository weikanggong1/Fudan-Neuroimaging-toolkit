"""Score four complete real LH registration runs without publishing private paths."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def public_backend(value):
    result = dict(value)
    for name in ("library", "cache_directory"):
        if name in result and result[name] is not None:
            result[name] = Path(result[name]).name
    if isinstance(result.get("compiler"), dict):
        result["compiler"] = dict(result["compiler"])
        for name in ("command", "resolved_path"):
            if name in result["compiler"]:
                result["compiler"][name] = Path(result["compiler"][name]).name
    return result


def process_time(path):
    names = {
        "User time (seconds)": "user_seconds",
        "System time (seconds)": "system_seconds",
        "File system inputs": "filesystem_inputs",
        "File system outputs": "filesystem_outputs",
        "Maximum resident set size (kbytes)": "maximum_RSS_KiB",
    }
    result = {}
    for line in path.read_text().splitlines():
        key, separator, value = line.strip().partition(": ")
        if separator and key in names:
            result[names[key]] = float(value)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runs", "official-runs", "frozen-source", "comparator", "report"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    source_manifest = json.loads((args.frozen_source / "SOURCE.private.json").read_text())
    source_files_unchanged = all(digest(args.frozen_source / "source" / name) == expected
                                 for name, expected in source_manifest["files"].items())
    import nibabel as nib
    import numpy as np
    import torch
    torch.set_num_threads(8)
    from fnit.recon_all.mris_register_nonlinear import face_area_normals
    spec = importlib.util.spec_from_file_location("saved_full_comparator", args.comparator)
    comparator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparator)

    def geometry(path):
        xyz, faces, volume = nib.freesurfer.read_geometry(path, read_metadata=True)
        volume = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in volume.items()}
        area, _ = face_area_normals(torch.from_numpy(xyz).float(),
                                   torch.from_numpy(faces.astype(np.int64)), signed_sphere=True)
        return xyz, faces, volume, int((area < 0).sum())

    official_path = args.official_runs / "official/lh.sphere.reg"
    official = geometry(official_path)
    reference_queue = json.loads((args.official_runs / "queue/registration_official_cpu8_v1/record.json").read_text())
    assert reference_queue["status"] == "complete" and reference_queue["returncode"] == 0
    old_reference = json.loads((args.official_runs / "candidate/record.private.json").read_text())
    arms, records, meshes = {}, {}, {}
    compile_libraries = {}
    names = ("baseline1", "candidate1", "candidate2", "baseline2")
    for name in names:
        queued_path = args.runs / "queue" / name
        queued = json.loads((queued_path / "record.json").read_text())
        assert queued["status"] == "complete" and queued["returncode"] == 0, name
        record_path = args.runs / name / "record.private.json"
        record = json.loads(record_path.read_text())
        path = args.runs / name / "lh.sphere.reg"
        xyz, faces, volume, negative = geometry(path)
        records[name], meshes[name] = record, (xyz, faces, volume, negative)
        calls, actual_backend = [], []
        for call in record["averaging_calls"]:
            row = dict(call)
            if "CPU_backend" in row:
                private = row["CPU_backend"]
                if private.get("library"):
                    library = Path(private["library"])
                    metadata = json.loads(library.with_suffix(".json").read_text())
                    assert metadata["library_sha256"] == digest(library)
                    compile_libraries[library.name] = public_backend(metadata)
                row["CPU_backend"] = public_backend(private)
            calls.append(row)
            if name.startswith("candidate") and call["iterations"] >= 16:
                backend = call.get("CPU_backend", {})
                actual_backend.append(backend.get("backend") == "cpp" and
                                      backend.get("actual_threads") == 8 and
                                      backend.get("requested_threads") == 8)
        registration = record["registration"]
        arms[name] = {
            "versions": record["versions"], "record_sha256": digest(record_path),
            "averaging_source_sha256": record["averaging_source_sha256"],
            "file_sha256": digest(path),
            "coordinate_sha256": hashlib.sha256(xyz.tobytes()).hexdigest(),
            "ordered_faces_sha256": hashlib.sha256(faces.tobytes()).hexdigest(),
            "finite_coordinates": bool(np.isfinite(xyz).all()),
            "negative_saved_faces": negative,
            "cold_process_wall_seconds": queued["wall_seconds"],
            "API_seconds_including_io": record["api_seconds_including_io"],
            "averaging_seconds_including_first_JIT": record["averaging_seconds_including_first_JIT"],
            "averaging_calls": calls,
            "eligible_calls": len(actual_backend),
            "all_eligible_calls_actual_CPP_threads8": all(actual_backend) if actual_backend else None,
            "stage_seconds_nested": {
                part: {k: v for k, v in registration[part].items() if "seconds" in k}
                for part in ("sulc_pass", "smoothwm_pass")},
            "sampled_tree_RSS_bytes": queued["maximum_sampled_tree_rss_bytes"],
            "maximum_sampled_tree_threads_including_idle_pools": queued["maximum_sampled_tree_threads"],
            "CPU_affinity": queued["cpu_affinity"], "CPU_thread_budget": queued["max_cpu_threads"],
            "load_before": queued["load_before"], "load_after": queued["load_after"],
            "process_time": process_time(queued_path / "time.txt"),
        }

    anchor_record, anchor_geometry = records["baseline1"], meshes["baseline1"]
    comparisons = {}
    for name in names:
        record, mesh = records[name], meshes[name]
        xyz, faces, volume, _ = mesh
        other_xyz, other_faces, other_volume, _ = anchor_geometry
        delta = xyz - official[0]
        comparisons[name] = {
            "common_source_equal_to_baseline1": record["common_source_sha256"] == anchor_record["common_source_sha256"],
            "input_hashes_equal": record["registration"]["input_sha256"] == anchor_record["registration"]["input_sha256"],
            "scientific_trajectory_equal": comparator.scientific_trajectory(record["registration"]) == comparator.scientific_trajectory(anchor_record["registration"]),
            "coordinates_equal_to_baseline1": bool(np.array_equal(xyz, other_xyz)),
            "ordered_faces_equal_to_baseline1": bool(np.array_equal(faces, other_faces)),
            "volume_geometry_equal_to_baseline1": volume == other_volume,
            "coordinates_equal_to_saved_official": bool(np.array_equal(xyz, official[0])),
            "ordered_faces_equal_to_saved_official": bool(np.array_equal(faces, official[1])),
            "volume_geometry_equal_to_saved_official": volume == official[2],
            "official_different_coordinate_values": int(np.count_nonzero(delta)),
            "official_max_abs_coordinate_mm": float(np.abs(delta).max()),
            "official_coordinate_RMSE_mm": float(np.sqrt(np.mean(delta * delta))),
        }
    saved_reference_checks = {
        "same_inputs": anchor_record["registration"]["input_sha256"] == old_reference["registration"]["input_sha256"],
        "same_scientific_trajectory": comparator.scientific_trajectory(anchor_record["registration"]) == comparator.scientific_trajectory(old_reference["registration"]),
    }
    gate = source_files_unchanged and all(saved_reference_checks.values()) and all(
        all(value for key, value in row.items() if not key.startswith("official_"))
        for row in comparisons.values()) and all(
            arm["finite_coordinates"] and arm["negative_saved_faces"] == 0 for arm in arms.values())
    backend_gate = all(arms[n]["all_eligible_calls_actual_CPP_threads8"] for n in ("candidate1", "candidate2"))
    summaries = {}
    for field in ("cold_process_wall_seconds", "API_seconds_including_io", "averaging_seconds_including_first_JIT"):
        before = [arms[n][field] for n in ("baseline1", "baseline2")]
        after = [arms[n][field] for n in ("candidate1", "candidate2")]
        summaries[field] = {
            "baseline_ABBA_median_seconds": statistics.median(before),
            "candidate_ABBA_median_seconds": statistics.median(after),
            "old_over_new_median": statistics.median(before) / statistics.median(after),
            "adjacent_pair_old_over_new": [before[0] / after[0], before[1] / after[1]],
        }
    report = {
        "scope": "complete_same_input_LH_registration_ABBA_not_raw_T1_recon_all",
        "frozen_source_manifest_sha256": digest(args.frozen_source / "SOURCE.private.json"),
        "frozen_source_commit": source_manifest["commit"],
        "frozen_source_file_count": len(source_manifest["files"]),
        "all_frozen_source_files_unchanged": source_files_unchanged,
        "comparator_sha256": digest(__file__), "trajectory_comparator_sha256": digest(args.comparator),
        "arms": arms, "comparisons": comparisons,
        "saved_official_reference": {"file_sha256": digest(official_path),
                                     "cold_process_wall_seconds_historical": reference_queue["wall_seconds"]},
        "saved_reference_checks": saved_reference_checks,
        "common_source_sha256": anchor_record["common_source_sha256"],
        "input_sha256": anchor_record["registration"]["input_sha256"],
        "vertices": len(anchor_geometry[0]), "faces": len(anchor_geometry[1]),
        "CPU_compiled_library_manifests": compile_libraries,
        "numerical_gate_passed": bool(gate), "actual_CPP_backend_gate_passed": bool(backend_gate),
        "speed_observations": summaries,
        "timing_limit": "Two adjacent pairs on a shared node; saved native reference is earlier and not a new paired native benchmark. No claim of all recon-all equivalence or speed.",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("numerical_gate_passed", "actual_CPP_backend_gate_passed", "speed_observations")}))
    if not gate or not backend_gate:
        raise SystemExit("formal full-stage gate failed; retain the report")


if __name__ == "__main__":
    main()
