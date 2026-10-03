"""CPU-only analysis of the two actual formal baseline runs; fixed metric helpers."""
from __future__ import annotations

import argparse
import csv
import getpass
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time

EXPECTED_CONFIG = "f8eba1cbf3eab2baa549172b32be2c9d3b1014ad478f6e3702d7987378f52f8c"
EXPECTED_BASELINE = "328c398496c5b90f459381ca462ba6597cbbe5f2e9700f0b1a273f1408962bac"
CASE_IDS = ("sub-CON01", "sub-CON03")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--frozen-tools", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.)
    parser.add_argument("--timeout-hours", type=float, default=24.)
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "explicit CPU GPU isolation required")
    require(socket.gethostname() == "nodecw10", "the actual CPU host must be nodecw10")
    require(1 <= args.poll_seconds <= 60 and 0 < args.timeout_hours <= 48, "invalid wait budget")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.frozen_tools.resolve().parent))
    from tools import analyze_connectome_accuracy_cohort as analysis
    driver, matrix, population = analysis.driver, analysis.matrix, analysis.population
    require(Path(analysis.__file__).resolve().parent == args.frozen_tools.resolve(), "analysis import origin differs")
    config_path = args.configuration.resolve()
    require(driver.cohort.sha256(config_path) == EXPECTED_CONFIG, "formal configuration differs")
    config = json.loads(config_path.read_bytes())
    driver.verify_tools(config)
    manifest, bindings = driver.bound(config["raw_manifest"]), driver.bound(config["input_bindings"])
    cases = driver.validate_plan(config, manifest, bindings)
    require(config["declared_source_manifests"]["baseline"]["source_fingerprint"] == EXPECTED_BASELINE,
            "this is not the declared actual scientific baseline")
    require(config["n_seeds"] == 100000 and config["seed"] == 0, "formal seed budget differs")
    for case_id in CASE_IDS:
        require({"version": "baseline", "case_id": case_id} in config["execution_order"],
                "baseline is not actually planned in this phase")
    helper_paths = {Path(module.__file__).resolve() for name, module in list(sys.modules.items())
                    if name.startswith("tools.") and getattr(module, "__file__", None) and
                    Path(module.__file__).resolve().is_relative_to(args.frozen_tools.resolve())}
    helper_paths.add(Path(__file__).resolve())
    helper_before = {str(path): driver.cohort.sha256(path) for path in sorted(helper_paths)}
    output = driver.cohort.require_fresh(args.output_dir)
    started = time.perf_counter()
    report = {
        "status": "waiting_actual_baselines", "start_utc": driver.cohort.utc(),
        "scope": "two actual completed baseline runs from formal raw12; existing official five-repeat envelope only",
        "version": "baseline", "selected_cases": list(CASE_IDS), "full_ten_baseline_cohort": False,
        "configuration": {"path": str(config_path), "sha256_before": EXPECTED_CONFIG},
        "source_fingerprint": EXPECTED_BASELINE, "raw_manifest": config["raw_manifest"],
        "input_bindings": config["input_bindings"], "helper_sha256_before": helper_before,
        "command": [sys.executable, *sys.argv],
        "environment": {"hostname": socket.gethostname(), "user": getpass.getuser(), "pwd": os.getcwd(),
                        "python": platform.python_version(), "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"]},
        "cases": {case_id: {"status": "not_assessed", "reason": "actual baseline not yet completed",
                             "matrix_decisions": None, "population_decisions": None} for case_id in CASE_IDS},
    }
    driver.cohort.atomic_json(output / "report.json", report)

    def unchanged():
        require(driver.cohort.sha256(config_path) == EXPECTED_CONFIG, "configuration changed during analysis")
        after = {str(path): driver.cohort.sha256(path) for path in sorted(helper_paths)}
        require(after == helper_before, "actual analysis helper changed")
        return after

    def analyze_case(case_id):
        before = time.perf_counter()
        case = cases[case_id]
        bound = bindings["cases"][case_id]
        anatomy_directory = bound["anatomy"]["directory"]
        gpu, wall, gpu_path = analysis.completed_run(config, case, "baseline", anatomy_directory)
        job = Path(config["run_root"]) / "baseline" / case_id
        wall_path = job / "raw_bids_wall.json"
        official = driver.bound(bound["official_reference_manifest"])
        official_root = Path(bound["official_reference_manifest"]["path"]).parent
        tracks, grid, scalars = (analysis.verified_export(wall, name) for name in
                                ("tracks.tck", "wm_fod_normalized.nii.gz", "track_metrics.npz"))
        source_paths = [gpu_path, wall_path, Path(bound["official_reference_manifest"]["path"]), tracks, grid, scalars]
        source_before = {str(path): driver.cohort.sha256(path) for path in source_paths}
        matrices = matrix.compare(official_root, [job / "connectome"], [gpu_path],
                                  config["raw_manifest"]["path"], config["raw_manifest"]["sha256"],
                                  case_id, fnit_seeds=[config["seed"]])
        official_tracks = []
        tck_before = {str(tracks): driver.cohort.sha256(tracks)}
        for seed in official["seeds"]:
            path = official_root / f"seed-{seed}" / "tracks.tck"
            digest = driver.cohort.sha256(path)
            require(digest == official["outputs"][str(seed)]["tracks_sha256"], "original official TCK changed")
            official_tracks.append(path)
            tck_before[str(path)] = digest
        import nibabel as nib
        import numpy as np
        import scipy
        import matplotlib
        saved_tracks = list(nib.streamlines.load(tracks, lazy_load=False).streamlines)
        with np.load(scalars) as saved:
            endpoints = (np.stack([np.stack((path[0], path[-1])) for path in saved_tracks])
                         if saved_tracks else np.empty((0, 2, 3), dtype=np.float32))
            require(endpoints.dtype == saved["endpoints"].dtype and np.array_equal(endpoints, saved["endpoints"]),
                    "same-run exported TCK endpoint bits differ from returned tractogram")
            require(all(len(saved[name]) == len(saved_tracks) for name in
                        ("weights", "lengths", "mean_fa", "endpoints")), "scalar and track counts differ")
        track_count = len(saved_tracks)
        del saved_tracks, endpoints
        pops, plot_data = population.compare(
            official_tracks, [tracks], grid, dataset=f"{manifest['dataset']} {case_id} actual formal baseline",
            n_seeds=config["n_seeds"], official_seeds=official["seeds"], fnit_seeds=[config["seed"]], return_data=True)
        case_output = output / case_id
        case_output.mkdir()
        population.figure(case_output / "population.png", pops, plot_data)
        del plot_data
        matrices["population_envelope_status"] = pops["population_envelope_status"]
        matrices["population_reason"] = "separate same-run baseline TCK analysis; original stored-point visit definition"
        driver.cohort.atomic_json(case_output / "matrix_envelope.json", matrices)
        driver.cohort.atomic_json(case_output / "population_envelope.json", pops)
        matrix_ranges = [record for profile in matrices["profiles"].values() for record in profile["ranges"].values()]
        fields = {field: analysis.decisions([profile["ranges"][field] for profile in matrices["profiles"].values()])
                  for field in matrix.FIELDS}
        atlas_rows = []
        detailed_rows = []
        for atlas, profile in matrices["profiles"].items():
            row = {"atlas": atlas, "nodes": profile["nodes"],
                   **analysis.decisions(profile["ranges"].values())}
            for field, bounds in profile["ranges"].items():
                row[field] = bounds["accepted_count"]
                require(len(bounds["fnit_vs_official"]) == 5, "expected original five cross comparisons")
                detailed_rows.append({"atlas": atlas, "field": field, "criterion": bounds["criterion"],
                                      "threshold": bounds["threshold"], "official_min_max": bounds["official_min_max"],
                                      "five_official_seeds": official["seeds"],
                                      "five_values": bounds["fnit_vs_official"],
                                      "five_decisions": bounds["comparison_accepted"]})
            atlas_rows.append(row)
        with (case_output / "eight_atlas_summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(atlas_rows[0]))
            writer.writeheader()
            writer.writerows(atlas_rows)
        driver.cohort.atomic_json(case_output / "five_comparisons_by_field.json", detailed_rows)
        matrix_count = analysis.decisions(matrix_ranges)
        pop_count = analysis.decisions(pops["ranges"].values())
        require(matrix_count["total"] == 240 and pop_count["total"] == 25, "original decision dimensions differ")
        # Reapply original completed-run/export gates after CPU reads.
        analysis.completed_run(config, case, "baseline", anatomy_directory)
        for name in ("tracks.tck", "wm_fod_normalized.nii.gz", "track_metrics.npz"):
            analysis.verified_export(wall, name)
        tck_after = {str(path): driver.cohort.sha256(path) for path in [tracks, *official_tracks]}
        require(tck_after == tck_before, "actual TCK changed during CPU comparison")
        source_after = {str(path): driver.cohort.sha256(path) for path in source_paths}
        require(source_after == source_before, "actual run/export/official identity changed during comparison")
        helper_after = unchanged()
        result = {
            "status": "analysis_completed", "version": "baseline", "case_id": case_id,
            "source_fingerprint": EXPECTED_BASELINE,
            "actual_gpu_report": {"path": str(gpu_path), "sha256": driver.cohort.sha256(gpu_path)},
            "actual_wall_report": {"path": str(wall_path), "sha256": driver.cohort.sha256(wall_path)},
            "official_reference_manifest": bound["official_reference_manifest"],
            "anatomy_directory": anatomy_directory, "track_count": track_count,
            "endpoint_bits_equal": True, "track_metrics_count_equal": True,
            "matrix_status": matrices["matrix_envelope_status"], "matrix_decisions": matrix_count,
            "matrix_decisions_by_field": fields, "eight_atlas_summary": atlas_rows,
            "population_status": pops["population_envelope_status"], "population_decisions": pop_count,
            "population_five_comparisons_by_field": pops["ranges"],
            "fnit_self_repeat_status": "not_assessed: one actual baseline seed",
            "raw_dwi_cli_seconds": gpu["raw_dwi_cli_total_runtime_seconds"],
            "same_run_export_seconds": wall["post_timing_result_export"]["seconds"],
            "source_sha256_before": source_before, "source_sha256_after": source_after,
            "TCK_sha256_before": tck_before, "TCK_sha256_after": tck_after,
            "helper_sha256_after": helper_after, "configuration_sha256_after": driver.cohort.sha256(config_path),
            "CPU_analysis_seconds": time.perf_counter() - before,
            "package_versions": {"numpy": np.__version__, "nibabel": nib.__version__, "scipy": scipy.__version__,
                                 "matplotlib": matplotlib.__version__},
            "outputs": {p.name: {"sha256": driver.cohort.sha256(p), "size_bytes": p.stat().st_size}
                        for p in case_output.iterdir()},
        }
        driver.cohort.atomic_json(case_output / "case_report.json", result)
        return result

    try:
        for case_id in CASE_IDS:
            gpu_path = Path(config["run_root"]) / "baseline" / case_id / "gpu_report.json"
            while True:
                unchanged()
                if gpu_path.is_file():
                    actual = json.loads(gpu_path.read_bytes())
                    if actual.get("status") == "completed":
                        break
                    require(actual.get("status") not in ("failed", "error"), f"actual {case_id} baseline failed")
                if time.perf_counter() - started > args.timeout_hours * 3600:
                    report["status"] = "timed_out_waiting_actual_baseline"
                    return 2
                report["cases"][case_id] = {"status": "not_assessed", "reason": "actual baseline not yet completed",
                                             "matrix_decisions": None, "population_decisions": None,
                                             "observed_utc": driver.cohort.utc()}
                driver.cohort.atomic_json(output / "report.json", report)
                time.sleep(args.poll_seconds)
            report["status"] = "CPU_analyzing_actual_baseline"
            report["cases"][case_id] = {"status": "analyzing", "matrix_decisions": None, "population_decisions": None}
            driver.cohort.atomic_json(output / "report.json", report)
            result = analyze_case(case_id)
            report["cases"][case_id] = result
            report["status"] = "waiting_actual_baselines"
            driver.cohort.atomic_json(output / "report.json", report)
            print(json.dumps({"case": case_id, "matrix_decisions": result["matrix_decisions"],
                              "population_decisions": result["population_decisions"]}), flush=True)
        report["status"] = "analysis_completed"
        return 0
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report.update(end_utc=driver.cohort.utc(), controller_seconds=time.perf_counter() - started,
                      helper_sha256_after=unchanged(), configuration_sha256_after=driver.cohort.sha256(config_path))
        driver.cohort.atomic_json(output / "report.json", report)


if __name__ == "__main__":
    raise SystemExit(main())
