"""Recompute scalar contract gates from original receipts; no Torch or math."""
from pathlib import Path
import hashlib
import json


def identity(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main():
    leaf = Path(__file__).resolve().parent
    read = lambda name: json.loads((leaf / name).read_text())
    plan, queue, metadata, report = [read(name) for name in
                                  ("PLAN.json", "QUEUE.json", "INTERFACE_LOAD.json", "CONTRACTS.json")]
    receipts = read("RECEIPTS.json")
    for name, record in receipts["receipts"].items():
        assert identity(leaf / name) == {key: record[key] for key in ("bytes", "sha256")}
    frozen = identity(leaf / "PLAN.json")
    assert frozen["sha256"] == "46e7f26f8f7f17cbcf40424418af787f4893720bf4dc8ba40e208d9e0881e673"
    for name, digest in plan["prototype_sources"].items():
        assert identity(leaf / name)["sha256"] == digest
    rows, copies, guards = [report[name] for name in ("numeric_rows", "copy_rows", "guard_rows")]
    numeric_fields = ("finite", "shape_stride_dtype_equal", "input_unchanged", "weight_bias_unchanged",
                      "output_independent", "workspace_released_on_return", "poisoned_columns_each_slab")
    gates = {
        "metadata_exit0_contract_exit0": len(queue["arms"]) == 2 and all(arm["PID_returncode"] == 0 for arm in queue["arms"]),
        "original_receipts_match_frozen_plan": all(obj["PLAN"] == frozen for obj in (queue, metadata, report)),
        "original_binary_reused_exact": report["binary"] == metadata["binary"] == {key: plan["previous_frozen_files"]["columns_reuse.so"][key] for key in ("bytes", "sha256")},
        "metadata_ABI_provider_gate": metadata["ABI_metadata"] == 10404 and metadata["Torch_handle_global_same_address"] and metadata["provider_sha256"] == report["provider_sha256"] == plan["provider_sha256"],
        "six_complete_FP32_contracts_exact": len(rows) == 6 and [row["case"] for row in rows] == [row["name"] for row in plan["numeric_cases"]] and all(row["different_bits"] == 0 and row["max_abs"] == 0 and row["old_candidate_slab_depth"] == 14 and row["explicit_workspace_allocations"] == 3 and all(row[field] for field in numeric_fields) for row in rows),
        "thirteen_copy_prefix_and_poison_gates": len(copies) == 13 and all(row["different_bits"] == 0 and row["poisoned_before_copy"] and row.get("unconsumed_tail_poison_unchanged", True) and row["compact_prefix_stride"] == [row["M"], 1] for row in copies),
        "twenty_three_fallback_guards": len(guards) == report["fallback_calls"] == 23 and all(row["fallback_identity"] and row["numeric_calls"] == 0 for row in guards),
        "declared_actual_counts": report["copy_oracle_calls"] == 13 and report["candidate_copy_calls"] == report["candidate_SGEMM_calls"] == 12 and report["injected_copy_error_calls"] == 1,
        "normal_exception_flags_workspace": all(report[field] for field in ("normal_scope_flags_restored", "default_compute_rejected", "old_fallback_error_propagated", "provider_error_propagated_before_allocation", "copy_error_propagated", "exception_workspace_released", "exception_flags_restored")),
        "source_flags_weight_parameters_hooks_unchanged": report["sources_before"] == report["sources_after"] == metadata["sources_before"] == metadata["sources_after"] == queue["bindings_before"] and report["flags_before"] == report["flags_after"] == metadata["flags_before"] == metadata["flags_after"] and all(report[field] for field in ("sources_unchanged", "flags_unchanged", "weight_file_unchanged", "parameter_bits_unchanged", "global_hook_tables_unchanged")) and not report["postcondition_failures"],
        "same_declared_CPU8_context": report["Torch"] == "2.5.1" and report["affinity"] == plan["affinity"] and report["threads"] == report["interop_threads"] == 8,
        "bounded_resource_no_CUDA": report["RSS_maximum_bytes"] <= 32_000_000_000 and not report["flags_after"]["CUDA_initialized"],
        "no_MRI_native_wholeCNN_compile_production": all(report[field] == 0 for field in ("MRI_calls", "native_calls", "model_forward_calls", "new_compilation_calls")) and not report["production_changed"] and not report["accepted_for_MRI"] and all(metadata[field] == 0 for field in ("compilation_calls", "copy_calls", "SGEMM_calls", "MRI_calls")),
        "completed_bounded_contracts": report["completed"] and report["valid_bounded_contracts"] and report["status"] == "bounded_contracts_passed_no_MRI" and queue["status"] == "metadata_and_bounded_contract_workers_exit0_no_MRI" and receipts["controllers_absent"],
    }
    summary = {
        "schema": "fnit_columns_v2_mechanical_summary/v1",
        "status": "bounded_contracts_passed_pending_real_layer_review",
        "frozen_PLAN": frozen,
        "scalar_receipt_identities": {name: identity(leaf / name) for name in ("QUEUE.json", "INTERFACE_LOAD.json", "CONTRACTS.json")},
        "gates": gates, "all_declared_gates_passed": all(gates.values()),
        "numeric_cases": [{key: row[key] for key in ("case", "shape", "different_bits", "max_abs")} for row in rows],
        "actual_reported_counts": {name: report[name] for name in ("copy_oracle_calls", "candidate_copy_calls", "candidate_SGEMM_calls", "injected_copy_error_calls", "fallback_calls")},
        "process_maxRSS_bytes": report["RSS_maximum_bytes"],
        "worker_observation_seconds": report["worker_observation_seconds"],
        "controller_worker_wall_seconds": {arm["script"]: arm["worker_wall_seconds"] for arm in queue["arms"]},
        "queue_wait_seconds": queue["wait_seconds"],
        "MRI_authorized": False, "production_integration_accepted": False,
        "official_accuracy_or_speed_claim": False, "full_model_accuracy_or_speed_claim": False,
        "scope_notes": [
            "Six synthetic inputs use the actual C72->24 weights; no MRI or official CNN ran.",
            "Reported counts are scoped Python wrappers, not an all-time native trace.",
            "Injected copy-error call is a Python stub, not a C++ copy invocation.",
            "Provider error before allocation is source-order evidence; no allocation counter was installed for that negative case.",
            "The before/after oneDNN=True state is restored; numerical candidate scopes used oneDNN=False and no-grad.",
            "Meta-device and static guards do not prove complete CUDA output or performance.",
            "Diagnostic timings include imports, source hashing, weight loading, poison and comparisons; no speed ratio.",
            "Small M contracts do not prove bit equality at real M=802816/573440; separate layer gate required.",
        ],
    }
    assert summary["all_declared_gates_passed"], gates
    (leaf / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"all_declared_gates_passed": True, "gate_count": len(gates), "summary": identity(leaf / "SUMMARY.json")}))


if __name__ == "__main__":
    main()
