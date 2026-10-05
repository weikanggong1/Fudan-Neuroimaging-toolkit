"""Append one declared native seg33 repeat to the immutable same-node report."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    work = root / "workspaces/smri_cpu_20261004"
    run = root / "runs/smri_cpu_20261004/remaining_20261004"
    official_work = work / "remaining_20261004/seg_memory_official_node7_v1"
    official_run = run / "seg_memory_official_node7_v1"
    report = json.loads(args.input.read_text())
    assert report["status"] == "complete"
    assert sha(args.input) == "03e220aec0cc93cf26599987076b74410554871441188e7e379dfa82da55dc3d"
    source = report["source_binding"]
    assert sha(official_work / "source_binding.private.json") == report["source_binding_sha256"]
    software = Path("/public/software/apps/Freesurfer/8.2.0-1")
    for relative, entry in source["sources"].items():
        assert sha(software / relative) == entry["sha256"]
    for name, entry in source["weights"].items():
        resource = software / "models" / name
        assert resource.stat().st_size == entry["bytes"] and sha(resource) == entry["sha256"]
    comparator = work / "t2_seg/compare_outputs.py"
    assert sha(comparator) == report["comparator_sha256"]
    spec = importlib.util.spec_from_file_location("saved_compare", comparator)
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)
    record_path = official_run / "records_repeat1/node7_seg33_repeat1/record.json"
    record = json.loads(record_path.read_text())
    assert record["status"] == "complete" and record["returncode"] == 0
    assert record["hostname"] == source["hostname"] == "nodecw7"
    assert record["cpu_affinity"] == source["cpu_affinity"]
    assert record["max_cpu_threads"] == 8
    assert record["job"]["env"]["FNIT_BENCHMARK_SOURCE_BINDING_SHA256"] == report["source_binding_sha256"]
    map_path = official_run / "seg33-repeat1.nii.gz"
    csv_path = official_run / "seg33-repeat1.csv"
    candidate = run / "seg_memory_full_cpu_v1/seg33_candidate"
    row = {key: record[key] for key in (
        "status", "returncode", "hostname", "cpu_affinity", "max_cpu_threads", "environment",
        "wall_seconds", "load_before", "load_after", "maximum_sampled_tree_rss_bytes",
        "maximum_sampled_tree_threads", "resource_samples")}
    row.update({
        "record_sha256": sha(record_path),
        "job_manifest_sha256": sha(official_work / "jobs33repeat.private.json"),
        "source_binding_sha256": report["source_binding_sha256"],
        "sampled_peak_affinities": sorted({r["affinity"] for r in record["peak_tree_sample"]["processes"]}),
        "native_output_sha256": {"map": sha(map_path), "csv": sha(csv_path)},
        "first_native_vs_repeat": compare.compare_segmentation(official_run / "seg33.nii.gz", map_path),
        "first_native_vs_repeat_csv": compare.compare_csv(official_run / "seg33.csv", csv_path),
        "node7_repeat_vs_candidate": compare.compare_segmentation(map_path, candidate / "segmentation.nii.gz"),
        "node7_repeat_vs_candidate_csv": compare.compare_csv(csv_path, candidate / "volumes.csv"),
        "timing_scope": report["official_modes"]["seg33"]["timing_scope"],
        "interpretation": "Declared repeat after a 373-second first cold observation; retain both, do not average them into a typical native time or use the first as a speedup denominator.",
    })
    report["schema"] = "fnit_seg_decoder_official_node7_and_fast_ba/v2"
    report["parent_report_sha256"] = sha(args.input)
    report["repeat_collector_sha256"] = sha(__file__)
    report["seg33_declared_native_repeat"] = row
    report["completed_full_fnit_arms"] = 16
    report["completed_official_native_arms"] = 4
    report["cpu_official_speed_goal_met"] = False
    report["timing_limitations"] = [
        "The first native seg33 cold observation is anomalous; its detailed cause is not established.",
        "Each native mode has one regular cold observation, not a repeated stable benchmark.",
        "Candidate ordinary parc and parc-fast remain slower than their same-node native cold observations.",
        "FNIT worker also verifies resource hashes and saves every API output; API and whole-process clocks are reported separately.",
    ]
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "complete", "repeat_wall_seconds": row["wall_seconds"],
        "repeat_native_different_voxels": row["first_native_vs_repeat"]["different_voxels"],
        "repeat_candidate_different_voxels": row["node7_repeat_vs_candidate"]["different_voxels"]}))


if __name__ == "__main__":
    main()
