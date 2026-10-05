"""Collect saved raw-T1 pairs and official reference differences; no inference."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--comparator", type=Path, required=True)
    p.add_argument("--official-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--require-complete", action="store_true")
    args = p.parse_args()
    import nibabel as nib
    import numpy as np
    spec = importlib.util.spec_from_file_location("saved_compare", args.comparator)
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)
    queue = json.loads((args.run / "queue.private.json").read_text())
    if args.require_complete:
        assert queue["status"] == "complete"
    official_manifest = json.loads(args.official_manifest.read_text())
    ref_jobs = {("seg33" if "synthseg_case02" in row["id"] else
                 "parc-fast" if "parc_fast_case02" in row["id"] else "parc"): row
                for row in official_manifest["jobs"] if "official" in row["id"]}
    def geo_equal(first, second):
        return (first.shape == second.shape and
                first.get_data_dtype() == second.get_data_dtype() and
                np.array_equal(first.affine, second.affine) and
                np.array_equal(first.header["pixdim"], second.header["pixdim"]) and
                first.header.binaryblock == second.header.binaryblock and
                [(e.get_code(), e._raw) for e in first.header.extensions] ==
                [(e.get_code(), e._raw) for e in second.header.extensions])
    def numeric_exact(csv):
        return csv["comparison_status"] == "compared" and csv["max_absolute_difference_mm3"] == 0
    report = {"schema": "fnit_seg_decoder_full_pairs/v1", "queue_status": queue["status"],
              "collector_sha256": sha(__file__), "comparator_sha256": sha(args.comparator),
              "official_manifest_sha256": sha(args.official_manifest),
              "device": queue["device"], "hostname": queue["hostname"],
              "controller_sha256": queue["controller_sha256"], "worker_sha256": queue["worker_sha256"],
              "affinity": queue["affinity"], "configured_environment": queue["configured_environment"],
              "cudnn_tf32": queue["cudnn_tf32"], "pairs": {}, "jobs": []}
    for job in queue["jobs"]:
        summary = {key: job[key] for key in ("name", "arm", "mode", "returncode", "wall_seconds", "load_after")}
        if job["returncode"] == 0:
            worker = json.loads((args.run / job["name"] / "full.private.json").read_text())
            worker = {key: value for key, value in worker.items() if key not in ("source_import", "python_version")}
            summary["worker"] = worker
        if "gpu_samples" in job:
            # Keep only anonymous physical-GPU state and sampler timing. MRI
            # paths, usernames and other jobs' command lines never enter this.
            samples = job["gpu_samples"]
            summary["gpu_monitor"] = {"before": job["gpu_before"], "after": job["gpu_after"],
                "sample_count": len(samples), "failed_samples": sum(s["returncode"] != 0 for s in samples),
                "max_sample_gap_seconds": max((b["monotonic"] - a["monotonic"]
                    for a, b in zip(samples, samples[1:])), default=None),
                "samples": [{key: value for key, value in sample.items() if key != "monotonic"} for sample in samples]}
        report["jobs"].append(summary)
    for mode in ("seg33", "parc", "parc-fast"):
        first, second = args.run / (mode + "_baseline"), args.run / (mode + "_candidate")
        if not (first / "full.private.json").exists() or not (second / "full.private.json").exists():
            continue
        expected_outputs = ("segmentation",) if mode == "seg33" else (
            "segmentation", "cortical_parcellation", "combined")
        pair = {"saved_maps": {}, "old_new_csv": compare.compare_csv(first / "volumes.csv", second / "volumes.csv")}
        for name in expected_outputs:
            old, new = first / (name + ".nii.gz"), second / (name + ".nii.gz")
            result = compare.compare_segmentation(old, new)
            result["header_and_world_geometry_exact"] = geo_equal(nib.load(old), nib.load(new))
            pair["saved_maps"][name] = result
        csv = pair["old_new_csv"]
        csv["numeric_exact"] = numeric_exact(csv)
        csv["predeclared_tolerance_passed"] = (csv["comparison_status"] == "compared" and
            all(abs(row["signed_difference_mm3"]) <= 0.01 + 1e-5 * abs(row["reference_mm3"]) for row in csv["columns"]))
        main_name = "segmentation" if mode == "seg33" else "combined"
        ref_map, ref_csv = map(Path, ref_jobs[mode]["expected_outputs"])
        pair["official_output_sha256"] = {"map": compare.sha256(ref_map), "csv": compare.sha256(ref_csv)}
        old, new = first / (main_name + ".nii.gz"), second / (main_name + ".nii.gz")
        pair["official_vs_baseline"] = compare.compare_segmentation(ref_map, old)
        pair["official_vs_candidate"] = compare.compare_segmentation(ref_map, new)
        pair["official_vs_baseline_csv"] = compare.compare_csv(ref_csv, first / "volumes.csv")
        pair["official_vs_candidate_csv"] = compare.compare_csv(ref_csv, second / "volumes.csv")
        ref, a, b = nib.load(ref_map), nib.load(old), nib.load(new)
        if ref.shape == a.shape == b.shape and np.array_equal(ref.affine, a.affine) and np.array_equal(a.affine, b.affine):
            x, y, z = (np.asanyarray(image.dataobj) for image in (ref, a, b))
            old_error, new_error = x != y, x != z
            pair["official_error_changes"] = {"introduced": int(np.count_nonzero(~old_error & new_error)),
                "removed": int(np.count_nonzero(old_error & ~new_error)),
                "retained": int(np.count_nonzero(old_error & new_error)),
                "retained_with_changed_label": int(np.count_nonzero(old_error & new_error & (y != z)))}
        else:
            pair["official_error_changes"] = {"status": "geometry_not_identical_no_resampling"}
        pair["old_new_passed"] = (all(row["comparison_status"] == "compared" and row["different_voxels"] == 0
                                      and row["header_and_world_geometry_exact"] for row in pair["saved_maps"].values())
                                  and (csv["predeclared_tolerance_passed"] if queue["device"] == "cpu" else csv["numeric_exact"]))
        report["pairs"][mode] = pair
    report["all_completed_pairs_passed"] = bool(report["pairs"]) and all(row["old_new_passed"] for row in report["pairs"].values())
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"queue_status": queue["status"], "modes": list(report["pairs"]),
                      "all_completed_pairs_passed": report["all_completed_pairs_passed"]}))


if __name__ == "__main__":
    main()
