"""Recompute accepted scalar hash/real-layer ABBA accounting; no Torch import."""
import hashlib
import json
from pathlib import Path
import statistics


def identity(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main():
    leaf = Path(__file__).resolve().parent
    plan = json.loads((leaf / "PLAN.json").read_text())
    queue = json.loads((leaf / "QUEUE.json").read_text())
    hashes = json.loads((leaf / "HASH_CONTRACTS.json").read_text())
    closed = json.loads((leaf / "INDEX_CLOSED.json").read_text())
    assert queue["status"] == "four_real_layer_arms_complete_three_actual_bit_gates"
    assert queue["bindings_unchanged"] and queue["PLAN_unchanged"]
    assert hashes["valid_hash_contracts"] and hashes["completed"]
    assert len(hashes["rows"]) == 4 and len(hashes["negative_guards"]) == 3
    assert hashes["convolution_calls"] == hashes["candidate_copy_calls"] == hashes["candidate_SGEMM_calls"] == 0
    assert all(row["exact"] and row["input_bytes_unchanged"] and row["raw_C_bytes_sha256"] == row["channel_sha256"] for row in hashes["rows"])
    assert hashes["rows"][1]["legacy5_sha256"] == hashes["rows"][1]["channel_sha256"]
    assert all(row["expected_rejection"] for row in hashes["negative_guards"])
    assert all(hashes[key] for key in ("bindings_unchanged", "runtime_unchanged", "PLAN_unchanged", "flags_unchanged"))
    assert closed["controllers_absent"] and closed["six_shared_locks"] and closed["source5_source14_exact"]
    assert len(queue["jobs"]) == 4 and [row["name"] for row in queue["jobs"]] == plan["ABBA"]
    rows = []
    for job in queue["jobs"]:
        assert job["exit_code"] == 0 and job["status"] == "completed"
        filename = job["name"] + "/report.json"
        assert identity(leaf / filename) == job["report"]
        arm = json.loads((leaf / filename).read_text())
        assert arm["valid_real_layer_arm"] and arm["completed"] and not arm["postcondition_failures"]
        assert arm["PLAN"] == queue["PLAN"] == identity(leaf / "PLAN.json")
        assert arm["bindings_before"] == arm["bindings_after"] == queue["bindings_before"] == queue["bindings_after"]
        assert arm["runtime_bindings_before"] == arm["runtime_bindings_after"]
        assert all(arm[key] for key in ("bindings_unchanged", "runtime_bindings_unchanged", "PLAN_unchanged", "flags_unchanged", "joined_input_unchanged", "parameter_values_unchanged"))
        assert arm["comparison_executed"] == (job["name"] != "A1_baseline")
        assert arm["bit_gate"]["different_bits"] == 0 and arm["bit_gate"]["max_abs"] == 0
        assert arm["bit_gate"]["finite"] and arm["bit_gate"]["shape_stride_dtype_equal"] and arm["bit_gate"]["output_independent"]
        assert arm["direct_layer_calls"] == 1 and not arm["flags_after"]["CUDA_initialized"]
        assert arm["joined_input_value_sha256"] == plan["joined_input_value_sha256"]
        assert arm["candidate_copy_calls"] == arm["candidate_SGEMM_calls"] == (14 if job["mode"] == "candidate" else 0)
        assert arm["output"]["shape"] == plan["output_shape"] and arm["output"]["dtype"] == "torch.float32"
        assert arm["maximum_RSS_bytes"] <= plan["limits"]["maximum_RSS_bytes"]
        for key in ("whole_model_calls", "ELU_BN_head_softmax_calls", "native_calls", "GPU_calls", "new_compilation_calls"):
            assert arm[key] == 0
        if job["name"] != "A1_baseline":
            assert arm["reference_file_unchanged"]
        rows.append({"name": job["name"], "mode": job["mode"], "receipt": identity(leaf / filename),
                     "exit_code": job["exit_code"], "comparison_executed": arm["comparison_executed"],
                     "different_bits": arm["bit_gate"]["different_bits"], "max_abs": arm["bit_gate"]["max_abs"],
                     "output": arm["output"], "operation_seconds": arm["operation_seconds_including_candidate_guards_counters"],
                     "operation_usage": arm["operation_usage"], "maximum_RSS_bytes": arm["maximum_RSS_bytes"],
                     "worker_seconds_with_IO_hash_comparison": arm["worker_observation_seconds_with_IO_hash_comparison"],
                     "fresh_process_outer_seconds": job["worker_wall_seconds"],
                     "loadavg_before_after": [arm["loadavg_before"], arm["loadavg_after"]],
                     "copy_SGEMM_calls": [arm["candidate_copy_calls"], arm["candidate_SGEMM_calls"]]})
    assert len({row["output"]["value_sha256"] for row in rows}) == 1
    assert len({tuple(row["output"]["stride"]) for row in rows}) == 1
    original = [row for row in rows if row["mode"] == "baseline"]
    candidate = [row for row in rows if row["mode"] == "candidate"]
    original_median = statistics.median(row["operation_seconds"] for row in original)
    candidate_median = statistics.median(row["operation_seconds"] for row in candidate)
    summary = {
        "schema": "fnit_columns_real_layer_v2_complete_summary/v1",
        "status": "one_saved_real_layer_ABBA_three_actual_bit_gates_passed_pending_production_review",
        "PLAN": identity(leaf / "PLAN.json"), "source5": plan["scientific_sources"], "production_source14": plan["source14"],
        "hash_contract": {"receipt": identity(leaf / "HASH_CONTRACTS.json"), "positive": 4, "negative": 3,
                          "raw_bytes_exact": True, "legacy5D_exact": True, "convolution_copy_SGEMM": [0, 0, 0]},
        "arms": rows, "actual_real_layer_calls": 4, "actual_old_new_bit_comparisons": 3,
        "FP32_values_per_comparison": 264241152, "output_full_value_sha256": rows[0]["output"]["value_sha256"],
        "shape_stride_dtype_finite_noalias_input_parameter_flags_source_RSS_gates_passed": True,
        "paired_layer_clock_observation_only": {
            "baseline_two_arm_median_seconds": original_median,
            "candidate_two_arm_median_seconds": candidate_median,
            "descriptive_baseline_over_candidate_ratio": original_median / candidate_median,
            "scope": "One up3.conv0 only; including candidate guards/provider/counters, excluding load/join/hash/IO; not whole/native speed ratio",
            "baseline_mean_user_seconds": statistics.mean(row["operation_usage"]["user_seconds"] for row in original),
            "candidate_mean_user_seconds": statistics.mean(row["operation_usage"]["user_seconds"] for row in candidate),
            "baseline_mean_system_seconds": statistics.mean(row["operation_usage"]["system_seconds"] for row in original),
            "candidate_mean_system_seconds": statistics.mean(row["operation_usage"]["system_seconds"] for row in candidate),
            "baseline_mean_minor_faults": statistics.mean(row["operation_usage"]["minor_faults"] for row in original),
            "candidate_mean_minor_faults": statistics.mean(row["operation_usage"]["minor_faults"] for row in candidate),
        },
        "largest_worker_RSS_bytes": max(row["maximum_RSS_bytes"] for row in rows),
        "controller_observation_seconds": queue["controller_observation_seconds"],
        "reference_file_current_identity": closed["reference_file_after"],
        "controllers_absent_common_lock_released": closed["controllers_absent"],
        "INDEX_close": identity(leaf / "INDEX_CLOSED.json"),
        "remaining_scope": ["production integration/fallback", "clean Conda compiler/install", "whole original-T1 API output/header/CSV", "CUDA untouched-route/output/performance gates"],
        "whole_CNN_native_GPU_speed_or_precision_assessed": False, "production_adopted": False,
        "new_compile_calls": 0, "scientific_retries": 0, "v1_rank_failure_preserved": True,
    }
    (leaf / "SUMMARY.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": summary["status"], "actual_bit_gates": 3, "ratio_observation_only": original_median / candidate_median}))


if __name__ == "__main__":
    main()
