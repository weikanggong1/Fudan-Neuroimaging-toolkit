"""Score four complete real LH registration runs without publishing private paths."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics


ARMS = ("baseline1", "candidate1", "candidate2", "baseline2")
AFFINITY = (3, 7, 11, 15, 19, 23, 27, 31)
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")
BASELINE_SHA256 = "488b2c2d1784a19e2b461419045482760233636df22c76a079d521f779c7f2c6"
WORKER_SHA256 = "8ab87b7191dfb80e459aa6d85df20f1d993108d26f7fbab058bd04ab829de3dd"
AVERAGE_KEY = "src/fnit/recon_all/mris_register_average_numba.py"
CPP_KEY = "src/fnit/recon_all/_average_cpu_persistent.cpp"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def affinity(value):
    """Reject invalid or repeated cores rather than silently coercing budgets."""
    try:
        if isinstance(value, str):
            value = [int(part) for part in value.split(",")]
        if not isinstance(value, (list, tuple)) or any(type(core) is not int or core < 0 for core in value):
            return None
        return tuple(sorted(value)) if len(set(value)) == len(value) else None
    except (TypeError, ValueError):
        return None


def option(argv, name):
    """Read one explicit argument; duplicate or absent options fail closed."""
    if not isinstance(argv, list) or argv.count(name) != 1:
        return None
    index = argv.index(name) + 1
    return argv[index] if index < len(argv) else None


def wrapped_argv_matches(queued_argv, job_argv):
    """Validate the recorded taskset/time wrapper and its complete job suffix."""
    return (isinstance(queued_argv, list) and len(queued_argv) >= 8
            and queued_argv[:2] == ["taskset", "-c"]
            and affinity(queued_argv[2]) == AFFINITY
            and queued_argv[3:6] == ["/usr/bin/time", "-v", "-o"]
            and isinstance(queued_argv[6], str) and Path(queued_argv[6]).name == "time.txt"
            and queued_argv[7:] == job_argv)


def reference_resource_checks(queue):
    """Historical native timing remains an eight-core CPU reference or fails."""
    environment = queue.get("environment", {})
    return {
        "same_actual_host": queue.get("hostname") == "nodecw7",
        "queue_budget8": type(queue.get("max_cpu_threads")) is int and queue["max_cpu_threads"] == 8,
        "queue_exact_same_eight_cores": affinity(queue.get("cpu_affinity")) == AFFINITY,
        "completed_queue_thread_environment8": all(str(environment.get(key)) == "8" for key in THREAD_VARIABLES),
        "completed_CPU_process_hides_CUDA": environment.get("CUDA_VISIBLE_DEVICES") == "",
    }


def identity_resource_checks(source_files, records, queued, jobs, baseline_sha, worker_sha,
                             compiled_libraries):
    """Bind completed records to source, original jobs and actual resource evidence.

    Baseline NumBa's eight-thread mask is derived from the pinned wrapper and
    worker, the completed job's thread environment and its recorded Torch
    budget. It was not observed separately for every NumBa call.
    ``compiled_libraries`` contains verified *private* build metadata, before
    path sanitization, keyed by each call's library path.
    """
    prefix = "src/fnit/recon_all/"
    expected_common = {Path(name).name: sha for name, sha in source_files.items()
                       if name.startswith(prefix) and "/" not in name[len(prefix):]
                       and name.endswith(".py") and name != AVERAGE_KEY}
    expected_average = source_files.get(AVERAGE_KEY)
    expected_cpp = source_files.get(CPP_KEY)
    job_rows = {job.get("id"): job for job in jobs if isinstance(job, dict)}
    source = {
        "baseline_source_is_pinned_eea929d4": baseline_sha == BASELINE_SHA256,
        "worker_source_is_pinned": worker_sha == WORKER_SHA256,
        "frozen_common_and_candidate_sources_present": bool(expected_common and expected_average and expected_cpp),
    }
    resource = {"four_original_ABBA_jobs_present": len(jobs) == 4 and tuple(job.get("id") for job in jobs) == ARMS}
    versions = records["baseline1"].get("versions")
    for name in ARMS:
        record, queue, job = records[name], queued[name], job_rows.get(name, {})
        expected = baseline_sha if name.startswith("baseline") else expected_average
        source[name] = {
            "averaging_source_matches_expected": bool(expected) and record.get("averaging_source_sha256") == expected,
            "all_common_modules_match_frozen_source": bool(expected_common) and record.get("common_source_sha256") == expected_common,
            "same_runtime_versions": bool(versions) and record.get("versions") == versions,
        }
        job_sha = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
        argv = job.get("argv", [])
        resource[name] = {
            "original_job_sha_matches_completed_queue": bool(job) and queue.get("job_sha256") == job_sha,
            "completed_job_payload_matches_original": bool(job) and queue.get("job") == job,
            "completed_wrapped_argv_matches_original_job": bool(argv) and wrapped_argv_matches(queue.get("argv"), argv),
            "explicit_worker_threads8": option(argv, "--threads") == "8",
            "recorded_Torch_budget8": type(record.get("threads")) is int and record["threads"] == 8,
            "queue_budget8": type(queue.get("max_cpu_threads")) is int and queue["max_cpu_threads"] == 8,
            "recorded_exact_same_eight_cores": affinity(record.get("affinity")) == AFFINITY,
            "queue_exact_same_eight_cores": affinity(queue.get("cpu_affinity")) == AFFINITY,
            "same_actual_host": record.get("hostname") == queue.get("hostname") == "nodecw7",
            "completed_CPU_process_hides_CUDA": queue.get("environment", {}).get("CUDA_VISIBLE_DEVICES") == "",
            "completed_PYTHONPATH_matches_original_job": bool(job.get("env", {}).get("PYTHONPATH"))
                and queue.get("environment", {}).get("PYTHONPATH") == job.get("env", {}).get("PYTHONPATH"),
            "completed_queue_thread_environment8": all(str(queue.get("environment", {}).get(key)) == "8"
                                                        for key in THREAD_VARIABLES),
        }
        call_checks = []
        for call in record.get("averaging_calls", []):
            if call.get("iterations", 0) < 16:
                continue
            backend = call.get("CPU_backend", {})
            if name.startswith("baseline"):
                call_checks.append(not backend or backend.get("backend") == "numba")
                continue
            metadata = compiled_libraries.get(backend.get("library"), {})
            call_checks.append(bool(metadata) and backend.get("backend") == "cpp"
                               and backend.get("requested_threads") == 8 and backend.get("actual_threads") == 8
                               and backend.get("source_sha256") == expected_cpp
                               and metadata.get("source_sha256") == expected_cpp
                               and metadata.get("compiler") == backend.get("compiler")
                               and metadata.get("flags") == backend.get("flags")
                               and metadata.get("abi") == backend.get("abi") == 1
                               and metadata.get("key") == backend.get("cache_key"))
        source[name]["eligible_backend_identity_consistent"] = bool(call_checks) and all(call_checks)

    def passed(value):
        return all(passed(item) if isinstance(item, dict) else item is True for item in value.values())

    return {"source_checks": source, "resource_checks": resource,
            "source_identity_gate_passed": passed(source), "equal_CPU_resources_gate_passed": passed(resource),
            "baseline_NumBa_thread_evidence": {
                "configured_mask": 8,
                "basis": "pinned baseline min(Torch, NumBa config), pinned worker set_num_threads, completed queue environment and record",
                "observed_per_call_team_size": "not_recorded",
                "limit": "same configured budget and affinity; no claim of per-call NumBa team-size observation"}}


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
    parser.add_argument("--cpu-jobs", type=Path,
                        help="Original CPU_jobs.private.json; default inside frozen-source")
    parser.add_argument("--baseline-average-source", type=Path,
                        help="Pinned eea929d4 wrapper; default inside frozen-source")
    args = parser.parse_args()
    jobs_path = args.cpu_jobs or args.frozen_source / "CPU_jobs.private.json"
    baseline_path = args.baseline_average_source or args.frozen_source / "average_baseline_eea929d4.py"
    jobs = json.loads(jobs_path.read_text())["jobs"]
    source_manifest = json.loads((args.frozen_source / "SOURCE.private.json").read_text())
    source_files_unchanged = bool(source_manifest["files"]) and all(digest(args.frozen_source / "source" / name) == expected
                                 for name, expected in source_manifest["files"].items())
    import nibabel as nib
    import numpy as np
    import torch
    torch.set_num_threads(8)
    import fnit.recon_all
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
    arms, records, meshes, queued_records = {}, {}, {}, {}
    compile_libraries = {}
    private_compile_libraries = {}
    names = ARMS
    for name in names:
        queued_path = args.runs / "queue" / name
        queued = json.loads((queued_path / "record.json").read_text())
        assert queued["status"] == "complete" and queued["returncode"] == 0, name
        queued_records[name] = queued
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
                    assert metadata["library_bytes"] == library.stat().st_size
                    private_compile_libraries[str(library)] = metadata
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
        official_correspondence = xyz.shape == official[0].shape and np.array_equal(faces, official[1])
        delta = xyz - official[0] if official_correspondence else None
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
            "official_different_coordinate_values": int(np.count_nonzero(delta)) if delta is not None else None,
            "official_max_abs_coordinate_mm": float(np.abs(delta).max()) if delta is not None else None,
            "official_coordinate_RMSE_mm": float(np.sqrt(np.mean(delta * delta))) if delta is not None else None,
            "official_indexed_coordinate_comparison": "assessed" if official_correspondence else "not_assessed_no_correspondence",
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
    identity = identity_resource_checks(source_manifest["files"], records, queued_records, jobs,
                                         digest(baseline_path),
                                         digest(args.frozen_source / "benchmark_full_registration.py"),
                                         private_compile_libraries)
    identity["reference_resource_checks"] = reference_resource_checks(reference_queue)
    identity["equal_CPU_resources_gate_passed"] = (
        identity["equal_CPU_resources_gate_passed"] and all(identity["reference_resource_checks"].values()))
    source_checks = identity["source_checks"]
    source_checks["scorer_uses_exact_frozen_recon_modules"] = (
        Path(fnit.recon_all.__file__).resolve().parent
        == (args.frozen_source / "source/src/fnit/recon_all").resolve())
    for job in jobs:
        name = job["id"]
        argv = job["argv"]
        expected_wrapper = baseline_path if name.startswith("baseline") else args.frozen_source / "source" / AVERAGE_KEY
        source_checks[name]["original_job_uses_bound_worker_and_wrapper"] = (
            len(argv) > 1 and Path(argv[1]).resolve() == (args.frozen_source / "benchmark_full_registration.py").resolve()
            and option(argv, "--average-source") is not None
            and Path(option(argv, "--average-source")).resolve() == expected_wrapper.resolve())
        source_checks[name]["original_job_imports_frozen_source_first"] = (
            Path(job.get("env", {}).get("PYTHONPATH", "").split(":")[0]).resolve()
            == (args.frozen_source / "source/src").resolve())
    identity["source_identity_gate_passed"] = all(
        all(value.values()) if isinstance(value, dict) else value is True
        for value in source_checks.values())
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
        "CPU_jobs_file_sha256": digest(jobs_path),
        "baseline_averaging_source_sha256": digest(baseline_path),
        "benchmark_worker_sha256": digest(args.frozen_source / "benchmark_full_registration.py"),
        **identity,
        "formal_gate_passed": bool(gate and backend_gate and identity["source_identity_gate_passed"]
                                    and identity["equal_CPU_resources_gate_passed"]),
        "speed_observations": summaries,
        "timing_limit": "Two adjacent pairs on a shared node; saved native reference is earlier and not a new paired native benchmark. No claim of all recon-all equivalence or speed.",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("numerical_gate_passed", "actual_CPP_backend_gate_passed",
                                            "source_identity_gate_passed", "equal_CPU_resources_gate_passed",
                                            "formal_gate_passed", "speed_observations")}))
    if not report["formal_gate_passed"]:
        raise SystemExit("formal full-stage gate failed; retain the report")


if __name__ == "__main__":
    main()
