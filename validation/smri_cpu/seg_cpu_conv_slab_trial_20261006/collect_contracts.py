"""Rebuild scalar-only rejection summary; no Torch, MRI or numerical model call."""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer", type=Path)
    parser.add_argument("--leaf", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    leaf = args.leaf
    public_names = {
        "CONTRACTS.private.json": "CONTRACTS.json",
        "contracts_queue.private.json": "QUEUE.json",
        "contracts_outer_timeout.private.json": "WATCHDOG.json",
        "TASK_BOUND.private.json": "TASK_BOUND.json",
        "contracts.log": "contracts.log",
    }
    if args.transfer is not None:
        envelope = json.loads(args.transfer.read_text())
        assert envelope["schema"] == "fnit_scalar_diagnostic_transfer/v1"
        assert set(envelope["files"]) == set(public_names)
        for private_name, public_name in public_names.items():
            bound = envelope["files"][private_name]
            data = base64.b64decode(bound["data_base64"], validate=True)
            assert len(data) == bound["bytes"] and digest(data) == bound["sha256"]
            target = leaf / public_name
            if target.exists():
                assert target.read_bytes() == data, "existing receipt must not be overwritten"
            else:
                target.write_bytes(data)
    plan = json.loads((leaf / "PLAN.json").read_text())
    report = json.loads((leaf / "CONTRACTS.json").read_text())
    queue = json.loads((leaf / "QUEUE.json").read_text())
    watchdog = json.loads((leaf / "WATCHDOG.json").read_text())
    bound = json.loads((leaf / "TASK_BOUND.json").read_text())
    assert report["plan_sha256"] == digest((leaf / "PLAN.json").read_bytes()) == queue["plan_sha256"]
    assert report["source_before"] == report["source_after"] == plan["source_files"] == queue["source_files"]
    assert report["weight_sha256"] == plan["weight"]["sha256"]
    assert report["helper_sha256"] == plan["prototype_files"]["candidate.py"]
    assert report["worker_sha256"] == plan["prototype_files"]["check_contracts.py"]
    for name, sha256 in plan["prototype_files"].items():
        assert digest((leaf / name).read_bytes()) == sha256
    assert queue["controller_sha256"] == plan["prototype_files"]["run_trial.py"]
    assert queue["status"] == "stopped_at_first_failed_gate"
    assert len(queue["jobs"]) == 1 and queue["jobs"][0]["exit_code"] == 2
    assert report["status"] == "bit_gate_failed_stop_before_MRI"
    assert report["initial_flags"] == report["final_flags"]
    assert not report["final_flags"]["cuda_initialized"] and not report["real_MRI_executed"]
    assert report["fallback_calls"] == 0 and report["remaining_contracts_skipped_on_first_nonexact"]
    assert report["cpu_affinity"] == plan["cpu_affinity"] and report["threads"] == report["interop_threads"] == 8
    assert report["torch_version"] == "2.5.1"
    assert watchdog["status"] == "controller_finished_or_identity_changed_before_deadline"
    assert bound["deadline_epoch"] == queue["outer_deadline_epoch"] == watchdog["deadline_epoch"]
    rows = []
    for row in report["rows"]:
        total = math.prod(row["shape"]) // 72 * 24
        assert all(row[name] for name in ("finite", "shape_stride_dtype_equal", "input_unchanged", "weight_bias_unchanged", "output_independent"))
        rows.append({**row, "output_value_count": total, "bit_difference_fraction": row["bit_different_values"] / total,
                     "bitexact": row["bit_different_values"] == 0})
    assert [row["name"] for row in rows] == [case["name"] for case in plan["contracts"][:len(rows)]]
    assert all(row["bitexact"] for row in rows[:-1]) and not rows[-1]["bitexact"]
    result = {
        "schema": "fnit_seg_CPU_conv64Mi_rejection/v1",
        "status": "rejected_at_first_nonexact_contract",
        "production_integration": False,
        "source_commit": plan["source_commit"],
        "source14_unchanged": True,
        "scientific_prototype5_unchanged": True,
        "actual_weight_sha256": report["weight_sha256"],
        "plan_sha256": report["plan_sha256"],
        "scope": "Target Torch 2.5.1, actual model weights and synthetic FP32 inputs; old CPU versus private candidate only",
        "official_comparison_executed": False,
        "official_accuracy_direction": "unknown",
        "real_MRI_layer_arms": 0,
        "ABBA_arms": 0,
        "full_CNN_native_GPU_arms": 0,
        "whole_T1_map_CSV_status": "not_assessed",
        "speed_conclusion": "none; no real-layer timer or benchmark was run",
        "executed_contracts": rows,
        "skipped_numeric_contracts": [case["name"] for case in plan["contracts"][len(rows):]],
        "skipped_fallback_contracts": ["grad_enabled", "oneDNN_enabled", "training", "float64_guard", "non_CPU_meta", "CPU_autocast", "forward_hook"],
        "skipped_exception_propagation_contract": True,
        "guard_status": "not_executed_after_first_nonexact_arithmetic_contract",
        "torch_version": report["torch_version"],
        "threads": report["threads"],
        "interop_threads": report["interop_threads"],
        "cpu_affinity": report["cpu_affinity"],
        "initial_flags": report["initial_flags"],
        "final_flags": report["final_flags"],
        "flags_unchanged": True,
        "synthetic_worker_maximum_RSS_bytes": report["maximum_RSS_bytes"],
        "controller_PID": queue["controller_PID"],
        "worker_receipt": queue["jobs"][0],
        "timing_boundary": "worker wall includes import, model weight loading, contracts and reporting; not isolated convolution timing",
        "initial_enqueue_epoch": bound["initial_enqueue_epoch"],
        "outer_deadline_epoch": bound["deadline_epoch"],
        "deadline_not_reset": True,
        "receipts": {name: {"bytes": (leaf / name).stat().st_size, "sha256": digest((leaf / name).read_bytes())}
                     for name in public_names.values()},
    }
    json_write(leaf / "RESULT.json", result)
    print(json.dumps({"status": result["status"], "actual_bit_contracts": len(rows), "failed_case": rows[-1]["name"],
                      "result_sha256": digest((leaf / "RESULT.json").read_bytes())}))


if __name__ == "__main__":
    main()
