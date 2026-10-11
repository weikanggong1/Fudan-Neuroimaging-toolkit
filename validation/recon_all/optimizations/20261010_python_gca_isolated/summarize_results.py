"""只读验证公开SHA和8份完整GCA收据，重建CSV与摘要；不运行影像算法。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def summarize_results(*, reports_directory: Path, output_directory: Path) -> dict:
    """输入12份公开JSON及清单；返回两例阶段摘要并写wall/内存/分段CSV。

    两个具名参数均必填目录Path，输出目录须不存在。时间单位秒，显存
    字节，LTA为type-0 source→atlas voxel；平移单位是目标体素，非毫米。
    校验公开文件SHA、原始receiptSHA、输入/源码与矩阵后写出。动态
    producer checkpoint SHA差异保留为来源限制，不冒充一致。缺文件、
    不完整阶段或关键合同失败抛异常；不会改变输入或执行MRI/GPU。
    """
    if output_directory.exists():
        raise FileExistsError(output_directory)
    manifest = json.loads((reports_directory / "PUBLIC_EXPORT_MANIFEST.json").read_text())
    for relative, binding in manifest["files"].items():
        content = (reports_directory / relative).read_bytes()
        if hashlib.sha256(content).hexdigest() != binding["published_sha256"]:
            raise ValueError("published SHA mismatch: " + relative)
    verified = json.loads((reports_directory / "summary.provenance_v2.json").read_text())
    if verified["status"] != "complete_stages" or verified["complete_stage_count"] != 8:
        raise ValueError("eight completed calls required")
    required = ("producer_fixed_configuration_matches_snapshot",
                "producer_source_manifest_current_sha_matches",
                "producer_source_manifest_semantics_match", "executed_module_sha_verified",
                "input_hashes_verified", "parent_state_unchanged", "parent_live_tensor_unchanged")
    rows, phases, subjects, source_manifests = [], [], {}, set()
    for subject in ("sub06", "sub07"):
        data = verified["subjects"][subject]
        hashes = set()
        configuration_times = {"cpu64": [], "torch1024": []}
        for configuration in configuration_times:
            for repeat in (1, 2):
                key = f"{configuration}-r{repeat}"
                relative = f"{subject}-{key}.json"
                receipt = json.loads((reports_directory / relative).read_text())
                binding = data["stages"][key]
                source_manifests.add(receipt["producer_source_manifest_sha256"])
                if receipt["status"] != "complete" or not all(binding[item] for item in required):
                    raise ValueError("stage/source/input/parent-state contract: " + relative)
                if binding["receipt_sha256"] != manifest["files"][relative]["raw_sha256"]:
                    raise ValueError("original receipt SHA mismatch: " + relative)
                hashes.add(receipt["output_lta_sha256"])
                comparison = data["comparisons"][key + "-vs-cpu64-r1"]
                if (comparison["right_sha256"] != receipt["output_lta_sha256"] or
                    comparison["different_elements"] != 0 or comparison["max_abs_error"] != 0 or
                    comparison["p99_abs_error"] != 0 or not comparison["complete_file_bytes_equal"]):
                    raise ValueError("LTA contract: " + relative)
                wall = receipt["full_call_seconds"]
                configuration_times[configuration].append(wall)
                worker, memory = receipt["worker"], receipt["memory"]
                rows.append({"subject": subject, "configuration": configuration, "repeat": repeat,
                    "threads": receipt["threads"], "candidate_chunk": receipt["candidate_chunk"],
                    "inverse_backend": receipt["inverse_backend"], "wall_seconds": wall,
                    "different_matrix_elements": comparison["different_elements"],
                    "max_abs_error": comparison["max_abs_error"], "p99_abs_error": comparison["p99_abs_error"],
                    "complete_lta_bytes_equal": comparison["complete_file_bytes_equal"],
                    "child_allocated_peak_bytes": worker.get("allocated_peak_bytes"),
                    "child_reserved_peak_bytes": worker.get("reserved_peak_bytes"),
                    "peak_tree_total_bytes": memory.get("peak_tree_total_bytes"),
                    "peak_target_device_used_bytes": memory.get("peak_target_device_used_bytes"),
                    "sampling_interval_seconds": memory.get("sampling_interval_seconds"),
                    "max_observed_interval_seconds": memory.get("max_observed_interval_seconds"),
                    "failed_samples": memory.get("failed_samples"),
                    "memory_status": memory["status"],
                    "historical_producer_checkpoint_sha_matches_poststage": binding["producer_benchmark_snapshot_sha_matches"],
                    "input_sha_verified": binding["input_hashes_verified"],
                    "executed_source_sha_verified": binding["executed_module_sha_verified"],
                    "output_lta_sha256": receipt["output_lta_sha256"]})
                for name, seconds in worker["timing"].items():
                    phases.append({"subject": subject, "configuration": configuration,
                                   "repeat": repeat, "phase": name, "seconds": seconds})
        if len(hashes) != 1:
            raise ValueError("per-subject output SHA differs")
        subjects[subject] = {"matrix_and_complete_file": "exact_same_input",
            "times": {key: {"wall_seconds": times, "median_seconds": statistics.median(times)}
                      for key, times in configuration_times.items()},
            "isolated_vs_v3_inprocess": data["comparisons"]["isolated-vs-v3-inprocess"],
            "output_lta_sha256": hashes.pop()}
    if len(source_manifests) != 1:
        raise ValueError("one immutable producer source manifest required")
    output_directory.mkdir(parents=True)
    for name, values in (("stage_calls.csv", rows), ("worker_phases.csv", phases)):
        with (output_directory / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(values)
    result = {"status": "complete_stages", "complete_stage_count": len(rows),
        "scope": "same-input isolated self-owned GCA stage observations; A/B/A/B shared-load runs, not ABBA or whole-pipeline acceleration",
        "source_manifest_sha256": source_manifests.pop(),
        "overall_metric_equivalence": "not_assessed", "subjects": subjects,
        "memory": "parent+child process namespace ownership unresolved: null, not zero; card upper bounds include other jobs",
        "provenance": "eight stage-start producer checkpoint SHAs preserved; historical raw checkpoint bytes not captured. Poststage checkpoint SHAs differ because producer was live. Fixed configuration, source manifest, executed source and copied/input bytes verified separately.",
        "summarizer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output_directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize_results(**vars(args)), indent=2))
