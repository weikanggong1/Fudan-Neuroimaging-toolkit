"""Read actual completed accuracy runs on CPU; preserve existing metric rules."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

try:
    from . import benchmark_connectome_accuracy_cohort as driver
    from . import benchmark_connectome_raw_cohort_envelope as matrix
    from . import benchmark_connectome_tracking_population as population
except ImportError:
    import benchmark_connectome_accuracy_cohort as driver
    import benchmark_connectome_raw_cohort_envelope as matrix
    import benchmark_connectome_tracking_population as population


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verified_export(wall, name):
    record = wall.get("post_timing_result_export", {})
    require(record.get("status") == "completed", "same-run post-timing export is incomplete")
    entries = [(Path(path), identity) for path, identity in record["files"].items()
               if Path(path).name == name]
    require(len(entries) == 1, f"exactly one actual same-run {name} required")
    path, identity = entries[0]
    require(path.is_file() and path.stat().st_size == identity["size_bytes"] and
            driver.cohort.sha256(path) == identity["sha256"], f"actual exported {name} changed")
    return path


def decisions(ranges):
    values = [item for record in ranges for item in record["comparison_accepted"]]
    return {"passed": sum(item is True for item in values),
            "failed": sum(item is False for item in values),
            "not_assessed": sum(item is None for item in values), "total": len(values)}


def execute(configuration, output_dir, case_ids=None):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU analysis must explicitly hide GPUs")
    started = time.perf_counter()
    configuration = Path(configuration).resolve()
    config = json.loads(configuration.read_bytes())
    manifest, bindings = driver.bound(config["raw_manifest"]), driver.bound(config["input_bindings"])
    cases = driver.validate_plan(config, manifest, bindings)
    selected = list(cases) if case_ids is None else list(case_ids)
    require(selected and len(set(selected)) == len(selected) and set(selected) <= set(cases),
            "distinct actual planned cases required")
    output_dir = driver.cohort.require_fresh(output_dir)
    report = {"status": "running", "start_utc": driver.cohort.utc(),
              "scope": "actual same-raw-case accuracy candidate; CPU read-only matrix and population comparisons",
              "configuration": {"path": str(configuration), "sha256": driver.cohort.sha256(configuration)},
              "raw_manifest": config["raw_manifest"], "input_bindings": config["input_bindings"],
              "analysis_source_sha256": driver.cohort.sha256(__file__), "cases": {},
              "coverage": {"selected_cases": selected, "full_ten_case_cohort": len(selected) == 10}}
    try:
        for case_id in selected:
            job = Path(config["run_root"]) / "candidate" / case_id
            gpu_path, wall_path = job / "gpu_report.json", job / "raw_bids_wall.json"
            gpu, wall = json.loads(gpu_path.read_bytes()), json.loads(wall_path.read_bytes())
            require(gpu.get("status") == "completed" and gpu.get("version") == "candidate" and
                    gpu.get("case_id") == case_id and wall.get("status") == "completed",
                    f"{case_id}: actual new candidate run is incomplete")
            identity = config["declared_source_manifests"]["candidate"]["source_fingerprint"]
            require(gpu["source_before"]["source_fingerprint"] == identity ==
                    gpu["source_after"]["source_fingerprint"], "candidate source identity differs")
            require(Path(gpu["wall_report"]).resolve() == wall_path.resolve(), "actual wall binding differs")
            actual_case = bindings["cases"][case_id]
            official = driver.bound(actual_case["official_reference_manifest"])
            official_root = Path(actual_case["official_reference_manifest"]["path"]).parent
            case_output = output_dir / case_id
            case_output.mkdir()
            matrices = matrix.compare(official_root, [job / "connectome"], [gpu_path],
                                      config["raw_manifest"]["path"], config["raw_manifest"]["sha256"],
                                      case_id, fnit_seeds=[config["seed"]])
            tracks, grid, scalars = (verified_export(wall, name) for name in
                                    ("tracks.tck", "wm_fod_normalized.nii.gz", "track_metrics.npz"))
            official_paths = []
            for seed in official["seeds"]:
                path = official_root / f"seed-{seed}" / "tracks.tck"
                require(driver.cohort.sha256(path) == official["outputs"][str(seed)]["tracks_sha256"],
                        "original official actual TCK changed")
                official_paths.append(path)
            # Verify serialization of the same returned points. No new
            # tracking, scalar sampling, SIFT2 or matrix computation occurs.
            import nibabel as nib
            import numpy as np
            saved_tracks = list(nib.streamlines.load(tracks, lazy_load=False).streamlines)
            with np.load(scalars) as saved:
                endpoints = (np.stack([np.stack((path[0], path[-1])) for path in saved_tracks])
                             if saved_tracks else np.empty((0, 2, 3), dtype=np.float32))
                require(endpoints.dtype == saved["endpoints"].dtype and
                        np.array_equal(endpoints, saved["endpoints"]),
                        "post-timing TCK endpoint bits differ from returned tractogram")
                require(all(len(saved[name]) == len(saved_tracks) for name in
                            ("weights", "lengths", "mean_fa", "endpoints")),
                        "returned scalar and track counts differ")
            track_count = len(saved_tracks)
            del saved_tracks, endpoints
            pops, plot_data = population.compare(
                official_paths, [tracks], grid, dataset=f"{manifest['dataset']} {case_id} independent raw chains",
                n_seeds=config["n_seeds"], official_seeds=official["seeds"],
                fnit_seeds=[config["seed"]], return_data=True)
            population.figure(case_output / "population.png", pops, plot_data)
            del plot_data
            matrices["population_envelope_status"] = pops["population_envelope_status"]
            matrices["population_reason"] = "separate actual same-run TCK analysis; stored-point visit definition preserved"
            driver.cohort.atomic_json(case_output / "matrix_envelope.json", matrices)
            driver.cohort.atomic_json(case_output / "population_envelope.json", pops)
            matrix_ranges = [value for profile in matrices["profiles"].values()
                             for value in profile["ranges"].values()]
            timing = {"candidate_raw_dwi_cli_seconds": gpu["raw_dwi_cli_total_runtime_seconds"],
                      "candidate_export_seconds": wall["post_timing_result_export"]["seconds"],
                      "candidate_memory": gpu["memory_budget"], "baseline_scope": "not_measured_this_case"}
            baseline_path = Path(config["run_root"]) / "baseline" / case_id / "gpu_report.json"
            if baseline_path.is_file():
                baseline = json.loads(baseline_path.read_bytes())
                require(baseline.get("status") == "completed", "paired actual baseline did not complete")
                require(baseline["source_before"]["source_fingerprint"] ==
                        config["declared_source_manifests"]["baseline"]["source_fingerprint"],
                        "paired baseline source differs")
                timing.update(baseline_scope="this_phase_same_raw_case_same_parameters",
                              baseline_raw_dwi_cli_seconds=baseline["raw_dwi_cli_total_runtime_seconds"],
                              baseline_memory=baseline["memory_budget"],
                              candidate_to_baseline_ratio=timing["candidate_raw_dwi_cli_seconds"] /
                              baseline["raw_dwi_cli_total_runtime_seconds"])
            report["cases"][case_id] = {
                "actual_gpu_report": {"path": str(gpu_path), "sha256": driver.cohort.sha256(gpu_path)},
                "actual_wall_report": {"path": str(wall_path), "sha256": driver.cohort.sha256(wall_path)},
                "official_reference_manifest": actual_case["official_reference_manifest"],
                "source_fingerprint": identity, "track_count": track_count,
                "matrix_status": matrices["matrix_envelope_status"],
                "matrix_decisions": decisions(matrix_ranges),
                "population_status": pops["population_envelope_status"],
                "population_decisions": decisions(pops["ranges"].values()),
                "fnit_self_repeat_status": "not_assessed: one actual candidate seed",
                "timing": timing,
                "outputs": {path.name: {"sha256": driver.cohort.sha256(path),
                                        "size_bytes": path.stat().st_size} for path in case_output.iterdir()},
            }
            driver.cohort.atomic_json(output_dir / "report.json", report)
        report["status"] = "analysis_completed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report.update(end_utc=driver.cohort.utc(), cpu_analysis_seconds=time.perf_counter() - started)
        driver.cohort.atomic_json(output_dir / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-id", nargs="+", help="optional completed subset; never labeled full-cohort coverage")
    args = parser.parse_args()
    result = execute(args.configuration, args.output_dir, args.case_id)
    print(json.dumps({"status": result["status"], "coverage": result["coverage"]}))


if __name__ == "__main__":
    main()
