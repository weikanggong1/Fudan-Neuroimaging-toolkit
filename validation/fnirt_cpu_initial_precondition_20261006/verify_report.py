"""Read-only metadata/SHA/scalar checks; no NumPy, Torch or Numba imports."""
import ast
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[1]


def identity(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def read_json(path):
    def invalid(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(path.read_text(), parse_constant=invalid)


def check():
    collection = read_json(BASE / "COLLECTION_BINDINGS.public.json")
    assert not collection["private_array_content_transferred"]
    assert not collection["scientific_worker_repeated"]
    assert len(collection["files"]) == 27
    assert sum(row["bytes"] for row in collection["files"].values()) == 107162
    for name, row in collection["files"].items():
        assert identity(BASE / name) == {key: row[key] for key in ("bytes", "sha256")}, name
        assert row["mode"] == "0o600", name
    assert collection["transport"] == {"raw_chunk_cap_bytes": 45000, "batches": 3,
                                       "whole_file_hashes_reverified": True}
    freeze = read_json(BASE / "source/freeze.public.json")
    assert identity(BASE / "source/freeze.public.json")["sha256"] == "7f5cdc7572aac6b27752d4ca47bffc1be3bd2087f4c751ce1b1e02fcbcc130e5"
    assert len(freeze["files"]) == 11
    for name, row in freeze["files"].items():
        assert identity(BASE / "source" / name) == row, name
    expected = read_json(BASE / "source/expected.public.json")
    assert len(expected["production_source"]) == 17 and len(expected["source_extra"]) == 2
    for section in ("production_source", "source_extra"):
        for name, row in expected[section].items():
            assert identity(REPO / name) == row, name
    for name, row in expected["candidate_source_provenance"].items():
        assert identity(BASE / "source" / name) == row, name
    assert len(expected["files"]) == 3
    assert expected["lm_tau"]["value"] == .001
    assert identity(BASE / "run/initial/summary.public.json") == {
        "bytes": 17028, "sha256": "38715327435477707a512a367848c165028bf10278e1723162fd4e6f5086db46"}
    s = read_json(BASE / "run/initial/summary.public.json")
    assert s["status"] == "bounded_initial_precondition_scalar_control_completed"
    assert s["valid_bounded_diagnostic"] and s["probe_completed"]
    assert not s["runtime_integration_accepted"]
    assert s["checkpoint_arrays_read"] == ["gradient_half", "diagonal_half"]
    assert s["prefix_calls"] == {"current_torch": 1, "owned_cpu": 1}
    assert s["dot_calls"] == {"current_torch": 2, "owned_cpu": 2}
    assert s["norm_calls"] == {"current_torch": 1, "owned_cpu": 1}
    for name in ("callback_calls", "PCG_calls", "native_calls", "evaluate_calls", "linearize_calls",
                 "images_read", "unexpected_zero_rhs_return"):
        assert s[name] == 0, name
    assert not s["new_H"]
    assert s["input_system_bitexact_to_stage3"]
    assert s["prefix_ast"] == expected["prefix_ast"]
    for kind, row in s["initial_direction_gates"].items():
        assert row["bitexact"]
        assert row["actual_sha256"] == row["expected_sha256"] == expected["expected_initial_direction_sha256"][kind]
    inner = s["inner_floor"]
    assert inner["value_changed_count"] == inner["bits_changed_count"] == 0
    assert inner["supplied_weights_sha256"] == inner["effective_weights_sha256"]
    controls = s["precondition_controls"]
    for name in ("old_reciprocal_multiply_vs_candidate_division",
                 "old_reciprocal_multiply_vs_division_same_effective_weights"):
        row = controls[name]
        assert row["different_bits"] == 332 and row["max_ordered_ULP"] == 1
        assert row["max_abs"] == 1.7763568394002505e-15
        assert row["left_sha256"] == expected["expected_initial_direction_sha256"]["current_torch"]
        assert row["right_sha256"] == expected["expected_initial_direction_sha256"]["owned_cpu"]
    assert controls["candidate_division_vs_division_same_effective_weights"]["different_bits"] == 0
    assert controls["candidate_division_vs_division_same_effective_weights"]["max_abs"] == 0
    assert len(s["scalar_ledger"]) == 6
    for row in s["scalar_ledger"]:
        assert struct.pack("<d", row["value"]).hex() == row["FP64_little_endian_hex"]
    ledger = s["scalar_ledger"]
    assert [row["value"] for row in ledger] == [5.227043232336545, 1.5627520580425893,
        5.227043232336546, 1.5627520580425889, 1.5627520580425889, 1.562752058042589]
    assert ledger[1]["input_value_sha256"] == ledger[4]["input_value_sha256"]
    assert ledger[3]["input_value_sha256"] == ledger[5]["input_value_sha256"]
    assert ledger[0]["input_value_sha256"] == ledger[2]["input_value_sha256"]
    assert s["immutable_values_before"] == s["immutable_values_after"] and s["operand_values_unchanged"]
    assert s["bindings_before"] == s["bindings_after"] and len(s["bindings_before"]) == 22
    assert s["all_source_and_inputs_unchanged"] and not s["binding_failures_after"]
    assert s["flags_before"] == s["flags_after"] and s["flags_unchanged"]
    assert not s["flags_before"]["cuda_initialized"]
    assert s["fsl_dso_snapshot_before"] == s["fsl_dso_snapshot_after"] == []
    assert not s["postcondition_bookkeeping_failures"]
    assert s["affinity"] == [32, 36, 40, 44, 48, 52, 56, 60]
    assert s["torch_threads"] == 8 and s["interop_threads"] == 1
    assert s["address_space_cap_bytes"] == 20_000_000_000
    for name in ("initial.exitcode", "initial.science.exitcode", "preflight_after.exitcode"):
        assert (BASE / "run" / name).read_text().strip() == "0"
    for phase in ("before", "after"):
        pre = read_json(BASE / "run" / ("preflight_" + phase + ".public.json"))
        assert pre["all_bindings_match"] and not pre["failures"]
    launch = read_json(BASE / "run/launch.public.json")
    assert launch["enqueued_once"] and launch["controller_pid"] == 189857
    assert launch["controller_start_ticks"] == 534431758
    assert launch["freeze_sha256"] == identity(BASE / "source/freeze.public.json")["sha256"]
    assert launch["enqueue_source"] == identity(BASE / "source/operations/enqueue_once.py")
    completed = read_json(BASE / "run/index_completed.public.json")
    assert completed["status"] == "initial_completed" and completed["permissions_preserved"]
    assert len(completed["six_locks"]) == 6
    assert completed["entry_value"]["summary"] == identity(BASE / "run/initial/summary.public.json")
    acc = read_json(BASE / "ACCEPTANCE.public.json")
    for name in ("prefix_calls", "dot_calls", "norm_calls", "initial_direction_gates", "inner_floor",
                 "precondition_controls", "rho_cross_control", "rhs_norm_control"):
        assert acc[name] == s[name], name
    assert acc["completed"] and not acc["runtime_integration_accepted"]
    interpretation = read_json(BASE / "INTERPRETATION.public.json")
    assert interpretation["precondition_expression_difference_established_for_this_saved_state"]
    assert not interpretation["final_6_9665_percent_step_difference_uniquely_explained"]
    assert not interpretation["current_vs_official_RHS_7_3657e_7_explained"]
    assert not interpretation["runtime_integration_accepted"]
    upload = read_json(BASE / "UPLOAD_BINDINGS.public.json")
    assert upload["all_transfers_completed"] and len(upload["files"]) == 14
    assert not upload["numerical_arrays_transferred"] and not upload["scientific_worker_enqueued_by_uploader"]
    assert all(row["transfer_exit_code"] == 0 for row in upload["files"].values())
    transport = read_json(BASE / "METADATA_TRANSPORT_RECEIPTS.public.json")
    assert transport["readonly_metadata_only"] and len(transport["requests"]) == 4
    assert all(row["exit_code"] == 0 and not row["scientific_worker_repeated"] for row in transport["requests"])
    for path in BASE.rglob("*"):
        if not path.is_file():
            continue
        assert path.suffix not in (".npz", ".npy", ".f64", ".so", ".nii", ".gz")
        if path.suffix == ".json":
            read_json(path)
        elif path.suffix == ".py":
            compile(ast.parse(path.read_text()), str(path), "exec")
        elif path.suffix == ".sh":
            subprocess.run(["/bin/bash", "-n", str(path)], check=True)
        if path.suffix == ".md":
            for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
                if not target.startswith(("http://", "https://", "#")):
                    assert (path.parent / target.split("#", 1)[0]).exists(), (path, target)
        text = path.read_text()
        assert not re.search(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])", text), path
        assert not re.search(r"(?i)(?:BEGIN (?:RSA |OPENSSH )?PRIVATE KEY|gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,})", text), path
    manifest = BASE / "MANIFEST.public.json"
    if manifest.exists():
        for name, row in read_json(manifest)["files"].items():
            assert identity(BASE / name) == row, name
    return {"status": "readonly_report_checks_passed", "raw_files_verified": 27,
            "frozen_payload_files": 11, "current_source_count": 19, "binding_count": 22,
            "initial_direction_SHA_bridge_exact": True, "actual_inner_floor_changed_count": 0,
            "initial_precondition_bits_different": 332, "maximum_ordered_FP64_ULP": 1,
            "actual_dot_norm_counts": [4, 2], "scientific_calls_repeated": False,
            "NumPy_Torch_NumBa_imported": False, "private_arrays_present": False,
            "runtime_integration_accepted": False}


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
