"""Read-only stdlib report integrity checks; no numerical imports/workers."""
import ast
import hashlib
import json
import math
from pathlib import Path


def identity(path):
    value = path.read_bytes()
    return {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}


def check():
    root = Path(__file__).resolve().parent
    load = lambda name: json.loads((root / name).read_text())
    collection = load("COLLECTION_BINDINGS.public.json")
    assert not collection["private_array_content_transferred"]
    assert not collection["scientific_worker_repeated"]
    for name, record in collection["files"].items():
        assert identity(root / name) == {key: record[key] for key in ("bytes", "sha256")}, name
    freeze = load("source/freeze.public.json")
    assert identity(root / "source/freeze.public.json")["sha256"] == "782927b4984cf53b62a487fd7dde79b8da72ad238878b2ab0a08e6b1ef789150"
    for name, record in freeze["files"].items():
        assert identity(root / "source" / name) == record, name
    s = load("run/stage2/summary.public.json")
    for name in ("stage2.exitcode", "stage2.science.exitcode", "preflight_after.exitcode"):
        assert (root / "run" / name).read_text().strip() == "0"
    assert s["valid_bounded_diagnostic"] and s["probe_completed"]
    assert s["callback_calls"] == {"optimized": 7, "reference": 7} and s["strict_csc_calls"] == 7
    assert all(s[name] == 0 for name in ("evaluate_calls", "linearize_calls", "gradient_calls", "bending_constructor_calls", "solver_calls", "native_process_calls"))
    assert not s["full_H_materialized"] and not s["runtime_integration_accepted"]
    assert len(s["directions"]) == 7
    assert all(row["original_unit_bridge_bitexact"] for row in s["directions"][:3])
    assert all(row["optimized_vs_reference"]["different_bits"] == 0 for row in s["directions"])
    assert [row["optimized_vs_existing_CSC"]["different_bits"] for row in s["directions"]] == [0, 0, 0, 877, 1128, 1143, 1107]
    assert s["bindings_before"] == s["bindings_after"] and len(s["bindings_before"]) == 27
    assert s["flags_before"] == s["flags_after"] and not s["flags_after"]["cuda_initialized"]
    assert s["fsl_dso_snapshot_before"] == s["fsl_dso_snapshot_after"] == []
    assert s["restored_full_g_diag_bitexact"] and s["two_arm_storage_disjoint"]
    assert s["checkpoint_values_before"] == s["checkpoint_values_after"]
    assert s["checkpoint_operand_values_unchanged"] and not s["checkpoint_after_hash_failures"]
    assert not s["postcondition_bookkeeping_failures"]
    for kind, layout in s["restored_layouts"].items():
        assert len(layout) == 68
        for name, row in layout.items():
            assert row["recorded_byte_strides"] == row["restored_byte_strides"]
            assert row["restored_values_bitexact"]
            assert row["logical_value_sha256"] == s["checkpoint_values_before"][kind][name]
    for p in root.rglob("*.py"):
        ast.parse(p.read_text(), filename=str(p))
    def finite(value):
        if isinstance(value, dict): return all(finite(v) for v in value.values())
        if isinstance(value, list): return all(finite(v) for v in value)
        return not isinstance(value, float) or math.isfinite(value)
    for p in root.rglob("*.json"):
        assert finite(json.loads(p.read_text())), p
    manifest = root / "MANIFEST.public.json"
    if manifest.exists():
        entries = load("MANIFEST.public.json")["files"]
        for name, record in entries.items(): assert identity(root / name) == record, name
        assert set(entries) == {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p != manifest}
    return {"status": "PASS", "as_run_files": len(collection["files"]), "optimized_callbacks": 7,
            "reference_callbacks": 7, "CSC_actions": 7, "checker_numerical_execution": False, "private_arrays_read": False}


if __name__ == "__main__":
    print(json.dumps(check(), sort_keys=True))
