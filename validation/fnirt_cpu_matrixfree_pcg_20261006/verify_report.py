"""Read-only stdlib report verification; no numerical imports or callbacks."""
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[1]


def identity(path):
    value = path.read_bytes()
    return {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}


def read_json(path):
    def invalid(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(path.read_text(), parse_constant=invalid)


def check():
    collections = read_json(BASE / "COLLECTION_BINDINGS.public.json")
    assert not collections["private_array_content_transferred"] and not collections["scientific_worker_repeated"]
    assert len(collections["files"]) == 26
    for name, record in collections["files"].items():
        assert identity(BASE / name) == {key: record[key] for key in ("bytes", "sha256")}, name
        assert record["mode"] == "0o600", name
        assert Path(name).suffix not in (".npz", ".npy", ".f64", ".so", ".nii", ".gz")
    assert sum(record["bytes"] for record in collections["files"].values()) == 258085
    assert collections["transport"]["raw_chunk_cap_bytes"] == 45000
    assert collections["transport"]["whole_file_hashes_reverified"]
    freeze = read_json(BASE / "source/freeze.public.json")
    assert identity(BASE / "source/freeze.public.json")["sha256"] == "7ee1eb753514693ff7f00c64cdea43c12fcfb2d83293b717c8db3deeef9166fd"
    assert len(freeze["files"]) == 10
    for name, record in freeze["files"].items():
        assert identity(BASE / "source" / name) == record
    expected = read_json(BASE / "source/expected.public.json")
    source_count = 0
    for section in ("production_source", "source_extra"):
        for name, record in expected[section].items():
            assert identity(REPO / name) == record, name
            source_count += 1
    assert source_count == 21 and len(expected["files"]) == 6
    assert expected["lm_tau"]["value"] == .001
    for name, record in expected["candidate_source_provenance"].items():
        assert identity(BASE / "source" / name) == record
    result = read_json(BASE / "run/stage3/summary.public.json")
    assert identity(BASE / "run/stage3/summary.public.json") == {
        "bytes": 149216, "sha256": "8730d37a2faace0d100b24c94230ca676f19a3c1aa019c6759e50f5a4ebf0cd3"}
    assert result["status"] == "bounded_actual_matrixfree_two_PCG_completed"
    assert result["valid_bounded_diagnostic"] and result["probe_completed"]
    assert not result["runtime_integration_accepted"] and not result["numerical_improvement_established"]
    assert result["solver_calls"] == {"current_torch": 1, "owned_cpu": 1}
    assert result["callback_calls"] == {"current_torch": 51, "owned_cpu": 55}
    assert not result["full_H_materialized"] and result["existing_H_loaded_only_for_unit_restore_gate"]
    for name in ("evaluate_calls", "linearize_calls", "gradient_calls", "bending_constructor_calls",
                 "unexpected_solver_calls", "native_process_calls", "strict_csc_calls"):
        assert result[name] == 0, name
    assert result["unit_restore_gates"] == {
        kind: {"column": 0, "different_bits": 0, "bitexact": True} for kind in result["arms"]}
    assert result["two_arm_storage_disjoint"] and result["two_arm_system_values_bitexact"] and result["restored_full_g_diag_bitexact"]
    assert result["checkpoint_values_before"] == result["checkpoint_values_after"]
    assert result["system_values_before"] == result["system_values_after"]
    assert result["checkpoint_operand_values_unchanged"] and result["system_operand_values_unchanged"]
    assert result["bindings_before"] == result["bindings_after"] and len(result["bindings_before"]) == 27
    assert result["all_source_and_inputs_unchanged"] and not result["binding_failures_after"]
    assert result["flags_before"] == result["flags_after"] and result["flags_unchanged"]
    assert not result["flags_before"]["cuda_initialized"]
    assert result["fsl_dso_snapshot_before"] == result["fsl_dso_snapshot_after"] == []
    assert not result["postcondition_bookkeeping_failures"]
    assert result["affinity"] == [32, 36, 40, 44, 48, 52, 56, 60]
    assert result["torch_threads"] == 8 and result["interop_threads"] == 1
    assert result["address_space_cap_bytes"] == 20_000_000_000
    private_hashes = {"current_torch": "4d659ee6ec4fa1d4fc3dccbf2edbadfee1e19aced62f648db4cf88ca5c613575",
                      "owned_cpu": "418e3d2a7a1aede4acd9a7a37deb99cdf7757e547017e66efe68f5b898d0e93c"}
    for kind, iterations in (("current_torch", 49), ("owned_cpu", 53)):
        layouts = result["restored_layouts"][kind]
        assert len(layouts) == 68
        for name, record in layouts.items():
            assert record["restored_values_bitexact"]
            assert record["restored_byte_strides"] == record["recorded_byte_strides"]
            assert record["logical_value_sha256"] == result["checkpoint_values_before"][kind][name]
        arm = result["arms"][kind]
        assert arm["PCG"]["iterations"] == iterations and arm["PCG"]["converged"]
        assert arm["PCG"]["solver_callback_count"] == iterations
        assert not arm["PCG"]["possible_denominator_stop_from_bound_source"]
        assert arm["true_residual"]["relative_le_declared_tolerance"]
        assert 0 <= arm["PCG"]["relative_residual"] <= .001
        assert 0 <= arm["true_residual"]["relative_l2"] <= .001
        assert arm["true_residual"]["common_norm"] == "torch.linalg.vector_norm"
        assert arm["true_residual"]["rhs_l2"] == 5.227043232336545
        ledger = arm["callback_ledger"]
        assert [row["call"] for row in ledger] == list(range(1, iterations + 3))
        assert ledger[0]["phase"] == "unit_restore_gate" and not ledger[0]["damped"]
        assert ledger[-1]["phase"] == "true_residual" and ledger[-1]["damped"]
        assert all(row["phase"] == "pcg_iteration" and row["damped"] for row in ledger[1:-1])
        assert ledger[-1]["direction_sha256"] == private_hashes[kind]
        assert arm["private_solution"] == {"bytes": 9416, "sha256": private_hashes[kind], "mode": "0o600",
                                           "dtype": "little-endian FP64", "shape": [1177]}
    old, new = result["arms"]["current_torch"], result["arms"]["owned_cpu"]
    assert old["input_hashes"] == new["input_hashes"]
    assert old["callback_ledger"][0]["direction_sha256"] == new["callback_ledger"][0]["direction_sha256"]
    assert old["callback_ledger"][0]["action_sha256"] == new["callback_ledger"][0]["action_sha256"]
    assert old["callback_ledger"][1]["direction_sha256"] != new["callback_ledger"][1]["direction_sha256"]
    assert old["callback_ledger"][1]["direction_sha256"] == "e63284f6f4198c09827ee9815aae3d9b1720585a88d658c6eee129c382749cf4"
    assert new["callback_ledger"][1]["direction_sha256"] == "b3fea96e550d2a7c35369a62e942775b6345524a028a7529af57d9075aaff7ea"
    assert result["two_step_comparison"]["different_bits"] == 1177
    assert result["two_step_comparison"]["relative_l2_to_current_torch"] == 0.06966489368470974
    assert result["two_step_comparison"]["max_abs"] == 0.7198678639661988
    for name in ("stage3.exitcode", "stage3.science.exitcode", "preflight_after.exitcode"):
        assert (BASE / "run" / name).read_text().strip() == "0"
    for phase in ("before", "after"):
        preflight = read_json(BASE / "run" / ("preflight_" + phase + ".public.json"))
        assert preflight["all_bindings_match"] and not preflight["failures"]
        assert preflight["production_source_count"] == 17
    launch = read_json(BASE / "run/launch.public.json")
    assert launch["enqueued_once"] and launch["controller_pid"] == 173178 and launch["controller_start_ticks"] == 534104746
    assert launch["freeze_sha256"] == identity(BASE / "source/freeze.public.json")["sha256"]
    assert not launch["native_same_system_oracle"]
    assert launch["enqueue_source"] == identity(BASE / "source/operations/enqueue_once.py")
    completed = read_json(BASE / "run/index_completed.public.json")
    assert completed["status"] == "stage3_completed" and completed["permissions_preserved"]
    assert len(completed["six_locks"]) == 6
    acc = read_json(BASE / "ACCEPTANCE.public.json")
    assert acc["arms"] == {kind: {key: arm[key] for key in ("PCG", "true_residual", "private_solution", "coordinate_units")}
                           for kind, arm in result["arms"].items()}
    assert acc["two_step_comparison"] == result["two_step_comparison"]
    interpretation = read_json(BASE / "INTERPRETATION.public.json")
    assert interpretation["prior_materialized_comparison"]["not_the_same_floating_point_system_as_this_probe"]
    assert not interpretation["runtime_integration_accepted"] and not interpretation["additional_math_performed_for_interpretation"]
    upload = read_json(BASE / "UPLOAD_BINDINGS.public.json")
    assert upload["all_transfers_completed"] and not upload["numerical_arrays_transferred"]
    assert not upload["scientific_worker_enqueued_by_uploader"]
    assert all(row["transfer_exit_code"] == 0 for row in upload["files"].values())
    transport = read_json(BASE / "METADATA_TRANSPORT_RECEIPTS.public.json")
    assert transport["readonly_metadata_only"] and len(transport["requests"]) == 8
    assert all(row["exit_code"] == 0 and not row["scientific_worker_repeated"] for row in transport["requests"])
    for path in BASE.rglob("*"):
        if not path.is_file():
            continue
        assert path.suffix not in (".npz", ".npy", ".f64", ".so", ".nii", ".gz"), path
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
    manifest_path = BASE / "MANIFEST.public.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        for name, record in manifest["files"].items():
            assert identity(BASE / name) == record, name
    return {"status": "readonly_report_checks_passed", "raw_files_verified": 26, "active_frozen_payload_files": 10,
            "current_sources_verified": 21, "actual_callback_counts": result["callback_calls"],
            "unit_restore_bits_exact": True, "first_PCG_direction_fork_confirmed_from_saved_hashes": True,
            "scientific_calls_repeated": False, "NumPy_Torch_NumBa_imported": False,
            "private_arrays_present": False, "runtime_integration_accepted": False}


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
