"""Read-only stdlib verifier of saved source/scalar metadata; no replay."""
import ast
import hashlib
import json
from pathlib import Path
import re


def identity(path):
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def check():
    leaf = Path(__file__).resolve().parent
    load = lambda name: json.loads((leaf / name).read_text())
    frozen = load("source/freeze.public.json")
    assert identity(leaf / "source/freeze.public.json")["sha256"] == "354aa28b19de672b4704fc3a3b3ef15e668794ecc8c9027a19edc5e80191fd47"
    assert len(frozen["files"]) == 15
    for name, record in frozen["files"].items():
        assert identity(leaf / "source" / name) == record, name
    collection = load("COLLECTION_BINDINGS.public.json")
    assert len(collection["files"]) == 29
    for name, record in collection["files"].items():
        assert identity(leaf / name) == {key: record[key] for key in ("bytes", "sha256")}, name
        assert record["mode"] == "0o600", name
    assert sum(record["bytes"] for record in collection["files"].values()) == 190195
    assert collection["private_array_content_transferred"] is False
    assert collection["transport"] == {"raw_chunk_cap_bytes": 45000, "batches": 5, "whole_file_hashes_reverified": True}
    summary = load("run/moving/summary.public.json")
    expected = load("source/expected.public.json")
    assert identity(leaf / "run/moving/summary.public.json") == {
        "bytes": 64571, "sha256": "c0d8cd7cb545a7d69bf15369ca0903aa88368aa2f1ee533a9f8bae3d2c8ed9d8"}
    assert summary["control_completed"] and summary["status"] == "moving_input_control_completed_no_solver"
    assert not summary["production_integration_accepted"] and not summary["postcheck_errors"]
    assert len(summary["gates"]) == 12
    assert all(row.get("bitexact", row.get("exact")) is True for row in summary["gates"].values())
    assert summary["calls"] == {name: expected["limits"][name] for name in summary["calls"]}
    assert summary["calls"]["sampler"] == 1 and summary["calls"]["actual_bending_normal"] == 1
    assert summary["calls"]["LM_gradient_prefix"] == summary["calls"]["FSL_order_prefix"] == 2
    assert len(summary["checkpoint_arrays_read"]) == 32 and summary["checkpoint_arrays_read"] == expected["checkpoint_arrays_read"]
    assert not any("diagonal" in name or "weight" in name for name in summary["checkpoint_arrays_read"])
    assert len(expected["production_source"]) == 17 and len(expected["source_extra"]) == 4
    assert len(expected["files"]) == 12 and len(summary["bindings_before"]) == 33
    assert summary["bindings_before"] == summary["bindings_after"]
    assert summary["harness_bindings"] == summary["harness_bindings_after"]
    assert summary["all_source_inputs_harness_unchanged"] and not summary["binding_failures_after"]
    assert summary["checkpoint_values_before"] == summary["checkpoint_values_after"]
    assert summary["derived_operand_values_before"] == summary["derived_operand_values_after"]
    assert summary["flags_before"] == summary["flags_after"] and not summary["flags_after"]["cuda_initialized"]
    assert summary["checkpoint_operands_unchanged"] and summary["derived_operands_unchanged"] and summary["flags_unchanged"]
    assert summary["fsl_dso_snapshot_before"] == summary["fsl_dso_snapshot_after"] == []
    assert not any(summary[key] for key in ("new_H", "diag_read_or_created", "normal_cache_created", "raw_MRI_read"))
    assert summary["affinity"] == [32,36,40,44,48,52,56,60]
    assert summary["torch_threads"] == 8 and summary["interop_threads"] == 1
    assert summary["address_space_cap_bytes"] == 20_000_000_000
    for name in ("moving.exitcode", "moving.science.exitcode", "preflight_after.exitcode"):
        assert int((leaf / "run" / name).read_text()) == 0
    assert summary["lambda_policy"]["native_SSD_explicitly_saved"] is False
    assert summary["lambda_policy"]["native_SSD_bit_identity_established"] is False
    assert summary["lambda_policy"]["new_SSD_does_not_update_effective_lambda"] is True
    assert summary["native_state"]["regularization_lambda"] == expected["fixed_effective_lambda"] == 9049.463427795125
    assert summary["arms"]["current_saved_moving"]["count"] == summary["arms"]["official_saved_moving_only"]["count"] == 14341
    assert summary["arms"]["current_saved_moving"]["full_g_LM_sha256"] == expected["files"]["current_g"]["sha256"]
    assert summary["arms"]["current_saved_moving"]["full_g_FSL_order_sha256"] == expected["files"]["current_fsl_g"]["sha256"]
    for name, item in summary["sampler_output_contract"].items():
        assert item["device"] == "cpu" and item["dtype"] == ("torch.bool" if name == "valid_mask" else "torch.float32")
    interpretation = load("INTERPRETATION.public.json")
    assert len(interpretation["rows"]) == 10
    for row in interpretation["rows"]:
        current_key = "current_LM_vs_saved_native" if row["convention"] == "LM" else "current_FSL_order_vs_saved_native"
        new_key = "moving_only_LM_vs_saved_native" if row["convention"] == "LM" else "moving_only_FSL_order_vs_saved_native"
        comparison = summary["full_g_samepoint_comparison"]
        baseline = comparison[current_key]["full"] if row["block"] == "full" else comparison[current_key]["blocks"][row["block"]]
        changed = comparison[new_key]["full"] if row["block"] == "full" else comparison[new_key]["blocks"][row["block"]]
        assert row["baseline"] == baseline and row["moving_only"] == changed
        assert row["relative_L2_error_reduction_fraction"] == 1 - changed["relative_l2"] / baseline["relative_l2"]
        assert not changed["bitexact"]
        if row["block"] == "full":
            assert changed["different_bits"] == 1177
    assert interpretation["full_g_mixed_scale_dominated"]
    assert not interpretation["unique_remaining_error_cause_established"]
    assert not interpretation["warp_registration_or_segmentation_improvement_established"]
    assert not interpretation["production_runtime_integration_supported"]
    for name in ("preflight_before.public.json", "preflight_after.public.json"):
        report = load("run/" + name)
        assert report["all_bindings_match"] and not report["failures"]
        assert report["main_head_at_read"] == "db61cebc9d3683720e6c0a288db3cdc7d6f30a7c"
    launch = load("run/launch.public.json")
    assert launch["enqueued_once"] and launch["controller_pid"] == 10886
    assert launch["controller_deadline_unix"] - launch["enqueue_unix"] == 19600
    locks = [".INDEX.codex.lock", ".index.lock", "INDEX.json.lock", "INDEX.md.lock", "admin/index-update.lock", "admin/index.update.lock"]
    for phase in ("prepared", "queued", "completed"):
        index = load("run/index_" + phase + ".public.json")
        assert index["entry"] == "fnirt_rhs_moving_control_20261006" and index["six_locks"] == sorted(locks)
        assert index["permissions_preserved"]
    assert load("run/index_completed.public.json")["entry_value"]["control_completed"]
    assert load("ACCEPTANCE.public.json")["gates"] == summary["gates"]
    assert load("ACCEPTANCE.public.json")["actual_calls"] == summary["calls"]
    links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", (leaf / "README.md").read_text())
    for target in links:
        if not target.startswith(("https://", "http://")):
            assert (leaf / target.split("#", 1)[0]).exists(), target
    forbidden_suffixes = {".npz", ".npy", ".f64", ".so", ".nii", ".gz", ".pkl", ".pickle"}
    files = [path for path in leaf.rglob("*") if path.is_file()]
    for path in files:
        assert path.suffix not in forbidden_suffixes, path.name
        raw = path.read_text()
        assert not re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", raw), path.name
        assert not re.search(r"(?i)(?:password|access_token|authorization)\s*[:=]\s*[\"']?\S+", raw), path.name
        if path.suffix == ".json":
            json.loads(raw)
        if path.suffix == ".py":
            compile(ast.parse(raw), str(path), "exec")
    manifest_path = leaf / "MANIFEST.public.json"
    if manifest_path.exists():
        manifest = load(manifest_path.name)
        for name, record in manifest["files"].items():
            assert identity(leaf / name) == record, name
    return {"scope": "stdlib saved-report verification only; no numeric replay", "passed": True,
            "original_metadata_files": 29, "all12_gates_passed": True, "source_binding_count": 21,
            "input_binding_count": 12, "zero_array_payloads": True, "scientific_replays": 0,
            "one_science_dispatch_confirmed": True, "local_links_checked": len([x for x in links if not x.startswith("http")])}


if __name__ == "__main__":
    result = check()
    print(json.dumps(result, indent=2))
