"""Rebuild scalar failure accounting from unchanged original receipts; no Torch."""
import hashlib
import json
from pathlib import Path


def main():
    leaf = Path(__file__).resolve().parent
    queue = json.loads((leaf / "QUEUE.json").read_text())
    arm = json.loads((leaf / "A1_baseline/report.json").read_text())
    closed = json.loads((leaf / "INDEX_CLOSED.json").read_text())
    assert queue["status"] == "bounded_real_layer_queue_failed_stopped"
    assert len(queue["jobs"]) == 1 and queue["jobs"][0]["name"] == "A1_baseline"
    assert queue["jobs"][0]["exit_code"] == 1
    counts = {name: arm[name] for name in ("direct_layer_calls", "candidate_copy_calls", "candidate_SGEMM_calls")}
    assert all(value == 0 for value in counts.values())
    assert arm["error"] == "contiguous singleton-batch value hash required"
    assert not arm["comparison_executed"] and not arm["completed"] and not arm["valid_real_layer_arm"]
    assert not arm["postcondition_failures"]
    for name in ("bindings_unchanged", "runtime_bindings_unchanged", "PLAN_unchanged", "flags_unchanged"):
        assert arm[name]
    assert queue["bindings_unchanged"] and queue["PLAN_unchanged"]
    assert closed["controllers_absent"] and closed["baseline_reference_file_absent"] and closed["source4_source14_exact"]
    files = {}
    for name in ("QUEUE.json", "A1_baseline/report.json", "INDEX_CLOSED.json"):
        data = (leaf / name).read_bytes()
        files[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    result = {
        "schema": "fnit_real_layer_harness_metadata_failure_summary/v1",
        "status": "harness_parameter_hash_rank_error_before_any_convolution_stopped",
        "original_receipts": files, "frozen_PLAN": arm["PLAN"],
        "attempted_arms": 1, "attempted_arm": "A1_baseline", "exit_code": 1,
        "actual_numeric_counts": counts, "actual_bit_comparisons": 0,
        "baseline_reference_saved": False, "remaining_arms_dispatched": 0,
        "candidate_numerical_failure_assessed": False, "speed_result_assessed": False,
        "failure": "value_sha incorrectly required ndim5 for singleton-batch weight ndim6 [1,24,72,3,3,3]",
        "joined_check_source_order_only_not_saved_scalar_result": True,
        "source_runtime_flags_postconditions_passed": True,
        "CUDA_initialized_after": arm["flags_after"]["CUDA_initialized"],
        "maximum_RSS_bytes_only_preconvolution": arm["maximum_RSS_bytes"],
        "observational_seconds_not_convolution_timing": {
            "worker_with_IO_hash": arm["worker_observation_seconds_with_IO_hash_comparison"],
            "fresh_worker_outer": queue["jobs"][0]["worker_wall_seconds"],
            "controller": queue["controller_observation_seconds"],
        },
        "loadavg_before_after": [arm["loadavg_before"], arm["loadavg_after"]],
        "INDEX_closed_status": closed["status"], "numerical_retries": 0,
        "v1_frozen_sources_and_PLAN_preserved": True,
        "future_v2_requires_new_review": True,
    }
    (leaf / "SUMMARY.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "actual_numeric_counts": counts}))


if __name__ == "__main__":
    main()
