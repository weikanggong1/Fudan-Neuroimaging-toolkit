"""Collect scalar evidence from completed private workers; no tensor arithmetic."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(part)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run, plan = args.run, json.loads(args.plan.read_text())
    queue_path = run / "queue.private.json"
    queue = json.loads(queue_path.read_text())
    assert queue["status"] == "bounded_trial_complete"
    assert queue["plan_sha256"] == sha(args.plan)
    assert queue["source_files"] == plan["source_files"]
    order = ["contracts", "resume", "A1_baseline", "B1_candidate", "B2_candidate", "A2_baseline"]
    assert [job["name"] for job in queue["jobs"]] == order
    assert all(job["status"] == "completed" and job["exit_code"] == 0 for job in queue["jobs"])
    contracts_path = run / "CONTRACTS.private.json"
    contracts = json.loads(contracts_path.read_text())
    assert contracts["status"] == "passed" and contracts["torch_version"] == "2.5.1"
    assert contracts["helper_sha256"] == plan["prototype_files"]["channel_batch.py"]
    assert contracts["contract_sha256"] == plan["prototype_files"]["check_contracts.py"]
    assert all(row["bit_different_values"] == 0 for row in contracts["cases"])
    private = {name: json.loads((run / name / "report.private.json").read_text()) for name in order[1:]}
    reference_path = run / "A1_baseline" / "baseline_blur.private.npy"
    reference = {"file_sha256": sha(reference_path), "bytes": reference_path.stat().st_size,
                 "value_sha256": private["A1_baseline"]["output"]["value_sha256"],
                 "source_arm": "A1_baseline", "sha_measured_at_collection": True}
    excluded = {"source_before", "source_after", "hostname", "torch_configuration", "torch_parallel_info", "weights"}
    arms = {}
    for name, report in private.items():
        assert report["source_before"] == report["source_after"] == plan["source_files"]
        assert report["worker_sha256"] == plan["prototype_files"]["real_trial.py"]
        assert report["plan_sha256"] == sha(args.plan)
        assert report["helper_sha256"] == plan["prototype_files"]["channel_batch.py"]
        assert report["source_unchanged"] and report["flags_unchanged"] and report["RSS_gate_passed"]
        assert report["maximum_RSS_bytes"] <= plan["gates"]["maximum_RSS_bytes"]
        assert report["torch_version"] == "2.5.1" and report["threads"] == report["interop_threads"] == 8
        assert report["cpu_affinity"] == plan["cpu_affinity"]
        assert report["initial_flags"] == report["final_flags"] and not report["final_flags"]["cuda_initialized"]
        assert report["input_sha256"] == plan["input_sha256"]
        arms[name] = {key: value for key, value in report.items() if key not in excluded}
        arms[name]["private_report_sha256"] = sha(run / name / "report.private.json")
        if name == "resume":
            assert report["status"] == "terminal_preblur_saved" and not report["full_CNN_executed"]
            assert report["checkpoint_files"] == plan["checkpoint_files"]
            continue
        gate = report["bit_gate"]
        assert report["preblur_file"] == private["resume"]["preblur_file"]
        assert report["output"]["value_sha256"] == reference["value_sha256"]
        assert gate["all_finite"] and gate["input_unchanged"] and gate["shape_stride_dtype_same"] and gate["output_independent"]
        if name == "A1_baseline":
            assert not gate["comparison_executed"]
            arms[name]["comparison_role"] = "Reference generation only; natural zero is not an old/new gate."
        else:
            assert gate["comparison_executed"] and gate["bit_different_values"] == 0 and gate["max_absolute"] == 0
            arms[name]["comparison_reference"] = reference
    baseline = [private[name]["operation_seconds_with_trial_guard"] for name in ("A1_baseline", "A2_baseline")]
    candidate = [private[name]["operation_seconds_with_trial_guard"] for name in ("B1_candidate", "B2_candidate")]
    old, new = statistics.median(baseline), statistics.median(candidate)
    result = {
        "schema": "fnit_seg_cpu_blur_trial_public/v1", "status": "bounded_trial_complete_exact_no_speed_gain",
        "scope": "One real saved decoder-tail resume and one posterior old/new ABBA; not a complete T1 or native benchmark.",
        "pending_candidate_commit": "08d6f8826e56af0844786643e655fb30a0a34496",
        "plan_sha256": sha(args.plan), "collector_sha256": sha(__file__),
        "queue_private_sha256": sha(queue_path), "contracts_private_sha256": sha(contracts_path),
        "source_files": plan["source_files"], "prototype_files": plan["prototype_files"],
        "input_sha256": plan["input_sha256"], "checkpoint_files": plan["checkpoint_files"],
        "prepared_values_sha256": plan["prepared_values_sha256"], "weights": plan["weights"],
        "cpu_affinity": plan["cpu_affinity"], "threads": 8, "actual_job_order": queue["jobs"],
        "contracts": contracts, "reference": reference, "arms": arms,
        "summary": {
            "actual_bit_comparison_arms": ["B1_candidate", "B2_candidate", "A2_baseline"],
            "compared_FP32_values_per_arm": 33 * 192 * 224 * 256,
            "all_actual_comparisons_bit_exact": True, "all_sources_flags_unchanged": True,
            "all_RSS_below_32GB": True, "baseline_operation_seconds": baseline,
            "candidate_operation_seconds": candidate, "baseline_two_arm_median_seconds": old,
            "candidate_two_arm_median_seconds": new, "candidate_median_percent_change": 100 * (new / old - 1),
            "baseline_max_RSS_bytes": max(private[name]["maximum_RSS_bytes"] for name in ("A1_baseline", "A2_baseline")),
            "candidate_max_RSS_bytes": max(private[name]["maximum_RSS_bytes"] for name in ("B1_candidate", "B2_candidate")),
            "stable_speed_improvement_demonstrated": False, "production_changed": False,
            "full_T1_maps_CSV_assessed": False, "new_native_or_full_CNN_run": False,
            "timing_limit": "Shared CPU, two operations per path, initial baseline cold boundary; no stable speedup or whole-API inference."
        },
        "watchdog": json.loads((run / "OUTER_TIMEOUT.private.json").read_text())
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "result_sha256": sha(args.output), "summary": result["summary"]}))


if __name__ == "__main__":
    main()
