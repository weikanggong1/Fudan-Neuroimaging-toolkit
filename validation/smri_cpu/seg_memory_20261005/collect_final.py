"""Collect same-node native timing and the targeted fast BA saved-output gates."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import nibabel as nib
    import numpy as np
    root = args.root
    w = root / "workspaces/smri_cpu_20261004"
    run = root / "runs/smri_cpu_20261004/remaining_20261004"
    comparator = w / "t2_seg/compare_outputs.py"
    assert sha(comparator) == "61eaa50d008e1958f48c5f718d6ad6805fccf7142a91de83c266a91733b7bbf8"
    spec = importlib.util.spec_from_file_location("saved_compare", comparator)
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)
    native_work = w / "remaining_20261004/seg_memory_official_node7_v1"
    native_run = run / "seg_memory_official_node7_v1"
    source = json.loads((native_work / "source_binding.private.json").read_text())
    # Recheck the immutable original software/resources after all fits.
    software = Path("/public/software/apps/Freesurfer/8.2.0-1")
    for relative, entry in source["sources"].items():
        assert sha(software / relative) == entry["sha256"]
    for name, entry in source["weights"].items():
        assert (software / "models" / name).stat().st_size == entry["bytes"]
        assert sha(software / "models" / name) == entry["sha256"]
    cpu_run = run / "seg_memory_full_cpu_v1"
    first = json.loads((cpu_run / "queue.private.json").read_text())
    repeat_run = run / "seg_memory_full_cpu_fast_ba_v1"
    repeat = json.loads((repeat_run / "queue.private.json").read_text())
    assert first["status"] == repeat["status"] == "complete"
    assert first["affinity"] == repeat["affinity"] == source["cpu_affinity"]
    assert first["hostname"] == repeat["hostname"] == source["hostname"] == "nodecw7"
    old_manifest = json.loads((w / "t2_seg/corrected_case02_v1_jobs.private.json").read_text())
    old_refs = {}
    for row in old_manifest["jobs"]:
        if "official" in row["id"]:
            mode = "parc-fast" if "--fast" in row["argv"] else "parc" if "--parc" in row["argv"] else "seg33"
            old_refs[mode] = row["expected_outputs"]
    report = {"schema": "fnit_seg_decoder_official_node7_and_fast_ba/v1",
              "collector_sha256": sha(__file__), "source_binding": source,
              "source_binding_sha256": sha(native_work / "source_binding.private.json"),
              "official_job_manifest_sha256": sha(native_work / "jobs.private.json"),
              "comparator_sha256": sha(comparator), "official_modes": {}, "fast_abba": {},
              "scope": "same-node original command cold time; numerical reference-only comparisons; no models rerun by collector"}
    jobs = {(row["mode"], row["arm"]): row for row in first["jobs"]}
    for mode in ("seg33", "parc", "parc-fast"):
        record_path = native_run / "records" / ("node7_" + mode) / "record.json"
        record = json.loads(record_path.read_text())
        assert record["status"] == "complete" and record["returncode"] == 0
        assert record["hostname"] == source["hostname"] and record["cpu_affinity"] == source["cpu_affinity"]
        assert record["max_cpu_threads"] == 8
        assert record["job"]["env"]["FNIT_BENCHMARK_SOURCE_BINDING_SHA256"] == report["source_binding_sha256"]
        map_path, csv_path = native_run / (mode + ".nii.gz"), native_run / (mode + ".csv")
        current = cpu_run / (mode + "_candidate")
        name = "segmentation" if mode == "seg33" else "combined"
        row = {"status": record["status"], "returncode": record["returncode"],
               "hostname": record["hostname"], "cpu_affinity": record["cpu_affinity"],
               "max_cpu_threads": record["max_cpu_threads"], "environment": record["environment"],
               "wall_seconds": record["wall_seconds"], "load_before": record["load_before"],
               "load_after": record["load_after"],
               "maximum_sampled_tree_rss_bytes": record["maximum_sampled_tree_rss_bytes"],
               "maximum_sampled_tree_threads": record["maximum_sampled_tree_threads"],
               "resource_samples": record["resource_samples"],
               "sampled_peak_affinities": sorted({r["affinity"] for r in record["peak_tree_sample"]["processes"]}),
               "record_sha256": sha(record_path), "source_binding_sha256": report["source_binding_sha256"],
               "native_output_sha256": {"map": compare.sha256(map_path), "csv": compare.sha256(csv_path)},
               "old_native_vs_node7_native": compare.compare_segmentation(old_refs[mode][0], map_path),
               "old_native_vs_node7_native_csv": compare.compare_csv(old_refs[mode][1], csv_path),
               "node7_native_vs_candidate": compare.compare_segmentation(map_path, current / (name + ".nii.gz")),
               "node7_native_vs_candidate_csv": compare.compare_csv(csv_path, current / "volumes.csv")}
        row["fnit_cold_wall_seconds"] = {arm: jobs[(mode, arm)]["wall_seconds"] for arm in ("baseline", "candidate")}
        row["fnit_api_seconds"] = {arm: json.loads((cpu_run / (mode + "_" + arm) / "full.private.json").read_text())["api_seconds"]
                                   for arm in ("baseline", "candidate")}
        row["timing_scope"] = "official module/CLI import+whole inference+main map+CSV; FNIT cold worker includes hash preflight and all API maps+CSV"
        report["official_modes"][mode] = row
    baseline, candidate = repeat_run / "parc-fast_baseline", repeat_run / "parc-fast_candidate"
    ba = {"controller_sha256": repeat["controller_sha256"], "worker_sha256": repeat["worker_sha256"],
          "arm_order": repeat["arm_order"], "saved_maps": {}, "jobs": []}
    for name in ("segmentation", "cortical_parcellation", "combined"):
        paths = [directory / (name + ".nii.gz") for directory in
                 (cpu_run / "parc-fast_baseline", cpu_run / "parc-fast_candidate", baseline, candidate)]
        images = [nib.load(path) for path in paths]
        values = [np.asanyarray(image.dataobj) for image in images]
        ba["saved_maps"][name] = {"all_four_saved_file_shas_equal": len({compare.sha256(path) for path in paths}) == 1,
            "saved_file_sha256": [compare.sha256(path) for path in paths],
            "all_four_arrays_exact": all(np.array_equal(values[0], value) for value in values[1:]),
            "all_four_headers_exact": all(image.header.binaryblock == images[0].header.binaryblock and
                                           np.array_equal(image.affine, images[0].affine) for image in images[1:])}
    ba["csv"] = compare.compare_csv(baseline / "volumes.csv", candidate / "volumes.csv")
    ba["csv"]["all_four_saved_file_shas_equal"] = len({compare.sha256(directory / "volumes.csv") for directory in
        (cpu_run / "parc-fast_baseline", cpu_run / "parc-fast_candidate", baseline, candidate)}) == 1
    times = {arm: [] for arm in ("baseline", "candidate")}
    cold = {arm: [] for arm in ("baseline", "candidate")}
    for controller, directory, group in ((first, cpu_run, "AB"), (repeat, repeat_run, "BA")):
        for job in controller["jobs"]:
            if job["mode"] != "parc-fast":
                continue
            worker = json.loads((directory / job["name"] / "full.private.json").read_text())
            times[job["arm"]].append(worker["api_seconds"])
            cold[job["arm"]].append(job["wall_seconds"])
            ba["jobs"].append({"group": group, "name": job["name"], "returncode": job["returncode"],
                               "cold_wall_seconds": job["wall_seconds"], "api_seconds": worker["api_seconds"],
                               "maximum_rss_kib": worker["maximum_rss_kib"],
                               "source_files": worker["source_files"], "cpu_join_calls": worker["cpu_join_calls"]})
    ba["api_medians_seconds"] = {arm: float(np.median(value)) for arm, value in times.items()}
    ba["cold_medians_seconds"] = {arm: float(np.median(value)) for arm, value in cold.items()}
    ba["candidate_api_relative_change"] = ba["api_medians_seconds"]["candidate"] / ba["api_medians_seconds"]["baseline"] - 1
    ba["passed"] = (all(r["all_four_arrays_exact"] and r["all_four_headers_exact"] for r in ba["saved_maps"].values())
                    and ba["csv"]["max_absolute_difference_mm3"] == 0 and ba["csv"]["all_four_saved_file_shas_equal"])
    report["fast_abba"] = ba
    report["status"] = "complete"
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "fast_ba_passed": ba["passed"], "official_cold_seconds":
                      {mode: value["wall_seconds"] for mode, value in report["official_modes"].items()}}))


if __name__ == "__main__":
    main()
