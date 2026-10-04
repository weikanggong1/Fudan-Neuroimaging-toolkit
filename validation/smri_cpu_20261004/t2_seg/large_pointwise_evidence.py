"""Export the targeted real-T1 CPU repair without private paths or commands.

Output analysis runs on the coordinator, outside inference timing. Reuse the
existing comparator; neither saved arm is resampled or changed.
"""

import argparse
import hashlib
import json
from pathlib import Path

from compare_outputs import compare_csv, compare_segmentation


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    workspace, root = args.workspace, args.run_root
    report = {
        "schema": "fnit_synthseg_large_pointwise_repair/v1",
        "analysis_only": True,
        "input": {"dataset": "OpenNeuro ds000114 1.0.2", "license": "CC0",
                  "doi": "10.18112/openneuro.ds000114.v1.0.2",
                  "shape": [156, 256, 256],
                  "input_sha256": "eb2bc2ff1f30441b0aff54685cfdd7f196bd02dccad4698100a8bad40421de22",
                  "network_shape": [1, 1, 224, 288, 288]},
        "precision": {"dtype": "float32", "new_low_precision": False,
                      "cpu_threads": 8, "cpu_affinity": "32,36,40,44,48,52,56,60",
                      "gpu_memory_budget_bytes": 20_000_000_000},
        "failure": {
            "scope": "large CPU 1x1x1 segmentation likelihood on nodecw10",
            "oneDNN_version": "3.5.3", "torch_version": "2.5.1",
            "input_channels": 24, "output_channels": 33,
            "output_logical_bytes": 2_452_488_192,
            "observed_padded_channels": 48,
            "output_padded_bytes": 3_567_255_552,
            "native_stack": "SIGSEGV in unresolved JIT region after channel reorder",
            "addressing_root_cause": "not established",
            "public_raw_all_v4_status": "failed before saving outputs",
            "public_raw_all_v4_wall_seconds": 44.0615017414093},
        "repair": {"cpu_only": True, "kernel": [1, 1, 1],
                   "minimum_estimated_blocked_bytes": 2**31,
                   "maximum_slab_depth": 32,
                   "target_slab_bytes": 256 * 1024 * 1024,
                   "global_backend_or_cuda_state_writes": False,
                   "unchanged": ["other oneDNN layers", "original guarded CPU backend",
                                 "GPU convolutions", "flipping", "postprocessing"],
                   "v1_to_final_v3": "only outer 5D eligibility check for valid unbatched Conv3d inputs"},
        "source": {}, "records": [], "comparisons": {},
        "official_equivalence": "per-label errors reported; no new bitwise acceptance threshold",
        "gpu_status": "pending_real_regression",
    }
    for label in ("task5_candidate_cpu_v4", "t2_seg/candidate_large_pointwise_v1",
                  "t2_seg/candidate_large_pointwise_v2", "t2_seg/candidate_large_pointwise_v3"):
        path = workspace / label / "SOURCE.private.json"
        manifest = json.loads(path.read_text())
        report["source"][manifest["source_label"]] = {
            "head_commit": manifest["head_commit"],
            "manifest_sha256": sha256(path),
            "file_count": len(manifest["files"]),
            "cpu_conv_sha256": manifest["files"]["src/fnit/synthseg_parc/cpu_conv.py"],
        }
    for version in ("v1", "v2", "v3", "v4"):
        path = workspace / "t2_seg" / f"task2_large_pointwise_{version}.acceptance.private.json"
        report["source"][f"acceptance_{version}_sha256"] = sha256(path)
    report["source"]["exporter_sha256"] = sha256(__file__)
    report["source"]["comparator_sha256"] = sha256(Path(__file__).with_name("compare_outputs.py"))

    def measured(device, version, job_id):
        path = root / f"t2_pointwise_{device}_queue_{version}" / job_id / "record.json"
        if not path.exists():
            return False
        record = json.loads(path.read_text())
        row = {key: record.get(key) for key in (
            "status", "wall_seconds", "returncode", "cpu_affinity", "max_cpu_threads",
            "maximum_sampled_tree_rss_bytes", "maximum_sampled_tree_threads",
            "load_before", "load_after", "started_utc", "finished_utc")}
        row.update(job_id=job_id, device=device, version=version,
                   receipt_sha256=sha256(path))
        guard = root / "t2_seg" / f"pointwise_{device}_{version}" / (job_id + ".guard.json")
        if guard.exists():
            data = json.loads(guard.read_text())
            row["gpu"] = {key: data[key] for key in (
                "cli_api_seconds", "max_allocated_bytes", "max_reserved_bytes",
                "memory_budget_bytes")}
        report["records"].append(row)
        return record["status"] == "complete"

    complete = {}
    for device, version in (("cpu", "v1"), ("cpu", "v2"), ("cpu", "v3"), ("gpu", "v4")):
        plan = workspace / "t2_seg" / f"task2_large_pointwise_{device}_{version}.jobs.private.json"
        report["source"][f"{device}_{version}_plan_sha256"] = sha256(plan)
        for job in json.loads(plan.read_text())["jobs"]:
            complete[job["id"]] = measured(device, version, job["id"])

    def comparison(device, version, first, second, key, strict, detailed, reference_version=None):
        if not complete.get(first) or not complete.get(second):
            report["comparisons"][key] = {"status": "pending"}
            return
        directory = root / "t2_seg" / f"pointwise_{device}_{version}"
        reference_directory = root / "t2_seg" / f"pointwise_{device}_{reference_version or version}"
        seg = compare_segmentation(reference_directory / (first + ".nii.gz"),
                                   directory / (second + ".nii.gz"))
        csv = compare_csv(reference_directory / (first + ".csv"), directory / (second + ".csv"))
        geometry_exact = seg["reference_geometry"] == seg["candidate_geometry"]
        numeric_exact = csv.get("max_absolute_difference_mm3") == 0
        numeric_close = csv.get("comparison_status") == "compared" and all(
            abs(row["signed_difference_mm3"]) <= .01 + 1e-5 * abs(row["reference_mm3"])
            for row in csv.get("columns", []))
        hard_exact = seg.get("different_voxels") == 0
        gate = hard_exact and geometry_exact and csv["column_names_and_order_equal"] and (
            numeric_exact if device == "gpu" else numeric_close)
        if not detailed:
            seg.pop("per_label", None)
            csv.pop("columns", None)
            seg["all_labels_array_equal"] = hard_exact
            csv["all_numeric_columns_equal"] = numeric_exact
        csv.pop("reference_columns", None)
        csv.pop("candidate_columns", None)
        report["comparisons"][key] = {
            "status": "passed" if strict and gate else ("failed" if strict else "reported"),
            "hard_exact": hard_exact, "geometry_exact": geometry_exact,
            "csv_numeric_exact": numeric_exact, "csv_existing_tolerance": numeric_close,
            "segmentation": seg, "soft_volumes": csv}

    prefix = "t2_pointwise_cpu_largeT1_full_"
    comparison("cpu", "v1", prefix + "official_abba1", prefix + "candidate_abba2",
               "largeT1_official_candidate_first", False, True)
    comparison("cpu", "v1", prefix + "official_abba4", prefix + "candidate_abba3",
               "largeT1_official_candidate_second", False, False)
    comparison("cpu", "v1", prefix + "candidate_abba2", prefix + "candidate_abba3",
               "largeT1_candidate_repeat", True, False)
    for case in ("case01", "case02"):
        for mode in ("full", "fast"):
            prefix = f"t2_pointwise_v3_cpu_{case}_{mode}_"
            first = prefix + "baseline_pair1"
            ref_version = None
            if case == "case01" and mode == "full":
                first = "t2_pointwise_v2_cpu_case01_full_baseline_pair1"
                ref_version = "v2"
            comparison("cpu", "v3", first, prefix + "candidate_pair2",
                       f"{case}_{mode}_cpu", True, False, ref_version)
    for mode in ("full", "fast"):
        prefix = f"t2_pointwise_v4_gpu_case01_{mode}_"
        comparison("gpu", "v4", prefix + "baseline_pair1", prefix + "candidate_pair2",
                   f"case01_{mode}_gpu_pair", True, False)
    gpu = [row for key, row in report["comparisons"].items() if "_gpu_" in key]
    if gpu and all(row["status"] != "pending" for row in gpu):
        report["gpu_status"] = "passed" if all(row["status"] == "passed" for row in gpu) else "failed"
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    for private_text in ("/cwStorage/", "/home/", "/mnt/", "sub-02", "FS_LICENSE"):
        if private_text in payload:
            raise ValueError("Private field in public report: " + private_text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload)
    print(json.dumps({"records": len(report["records"]), "gpu_status": report["gpu_status"],
                      "comparisons": {key: row["status"] for key, row in report["comparisons"].items()}}))


if __name__ == "__main__":
    main()
