"""Independent complete-image old/new GPU and CPU/GPU projection checks."""
import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from compare_completed import images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--gpu-run-dir", type=Path, required=True)
    parser.add_argument("--cpu-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.manifest.read_text())["datasets"]["real_run_01"]
    requests = json.loads(args.requests.read_text())
    execution = json.loads((args.gpu_run_dir / "execution.public.json").read_text())
    rows = execution["records"]
    names = [request["name"] for request in requests["requests"]]
    if len(rows) != 10 or [row["name"] for row in rows] != names:
        raise ValueError("All ten recorded complete requests must finish in order")
    if any(row["returncode"] != 0 or row["resource_violation"] for row in rows):
        raise ValueError("A complete request failed or violated its assigned resources")
    if any(row["simultaneous_process_tree_peak_bytes"] is None or
           row["simultaneous_process_tree_peak_bytes"] > 20_000_000_000 for row in rows):
        raise ValueError("Actual process-tree GPU memory evidence is required")
    if len({row["physical_gpu_uuid"] for row in rows}) != 1:
        raise ValueError("All requests must use the same physical GPU")
    mask = np.asarray(nib.load(config["brain_mask"]).dataobj) > 0
    reports = {}
    for request in requests["requests"]:
        name = request["name"]
        report = json.loads((args.gpu_run_dir / name / "report.public.json").read_text())
        if not report["executed"] or report["device"] != "cuda:0" or report["shape"][-1] != 490:
            raise ValueError("A recorded full 490-frame CUDA call is missing")
        if report["threads"] != request["threads"] or report["cpu_affinity"] != sorted(request["cpus"]):
            raise ValueError("GPU-host CPU budget differs from the prepared request")
        for relative, actual in report["source_sha256"].items():
            expected = requests["frozen_sha256"][str(Path(request["source_path"]) / "fnit" / relative)]
            if actual != expected:
                raise ValueError("Executed GPU module differs from its frozen source")
        if not report["tf32_matmul"] or not report["tf32_cudnn"]:
            raise ValueError("Default TF32 flags were not active")
        reports[name] = report
    results, timing = {}, {}
    for case in ("all", "all_bandpass"):
        timing[case] = []
        for position, backend in (("A1", "baseline"), ("B1", "candidate"),
                                  ("B2", "candidate"), ("A2", "baseline")):
            name = f"{case}_{position}_{backend}"
            row = next(row for row in rows if row["name"] == name)
            timing[case].append({"position": position, "backend": backend,
                                "api_wall_seconds": reports[name]["api_wall_seconds"],
                                "process_seconds": row["process_seconds"]})
        for candidate_position, baseline_position in (("B1", "A1"), ("B2", "A2")):
            candidate_name = f"{case}_{candidate_position}_candidate"
            baseline_name = f"{case}_{baseline_position}_baseline"
            if reports[candidate_name]["input_sha256"] != reports[baseline_name]["input_sha256"]:
                raise ValueError("Old/new actual input bytes differ")
            candidate = args.gpu_run_dir / candidate_name / "cleaned.nii.gz"
            baseline = args.gpu_run_dir / baseline_name / "cleaned.nii.gz"
            value = images(candidate, baseline, mask)
            value["headers_binary_equal"] = nib.load(candidate).header.binaryblock == nib.load(baseline).header.binaryblock
            value["all_values_equal"] = value["full_grid"]["max_absolute_difference"] == 0
            if not value["all_values_equal"] or not value["headers_binary_equal"]:
                raise ValueError("The unchanged default GPU path differs from its baseline")
            results[f"default_{case}_{candidate_position}_{baseline_position}"] = value
        candidate_name = f"{case}_gpu_afni_compat"
        cpu_directory = args.cpu_run_dir / f"confounds_{case}_t8_afni_compat"
        cpu_report = json.loads((cpu_directory / "report.public.json").read_text())
        if not cpu_report["executed"] or cpu_report["input_sha256"] != reports[candidate_name]["input_sha256"]:
            raise ValueError("Complete CPU/GPU compatibility inputs differ")
        results[f"afni_compat_{case}_cpu_gpu"] = images(
            args.gpu_run_dir / candidate_name / "cleaned.nii.gz",
            cpu_directory / "cleaned.nii.gz", mask)
    report = {
        "schema_version": 1, "dataset_alias": "real_run_01", "frames": 490,
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "full_comparisons": len(results), "unchanged_default_gpu_accepted": True,
        "same_physical_gpu_verified": True, "maximum_gpu_bytes": 20_000_000_000,
        "simultaneous_process_tree_peak_bytes": max(row["simultaneous_process_tree_peak_bytes"] for row in rows),
        "tf32_matmul": True, "tf32_cudnn": True, "results": results,
        "abba_api_and_process_timing": timing,
        "timing_scope": "Each API includes normal read/compute/write; process includes imports and provenance hashes",
        "timing_status": "Shared-GPU paired observations; no stable speedup inference",
        "shared_gpu_utilization_percent_ranges": [row["utilization_percent_range"] for row in rows],
        "maximum_sampling_gap_seconds": max(row["maximum_sampling_gap_seconds"] for row in rows),
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"complete_comparisons": len(results), "default_gpu_accepted": True,
                      "private_images_exported": False}))


if __name__ == "__main__":
    main()
