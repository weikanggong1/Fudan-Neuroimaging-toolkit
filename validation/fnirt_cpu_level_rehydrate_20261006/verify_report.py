"""Read-only stdlib integrity checks; never imports scientific dependencies."""
from pathlib import Path
import ast
import hashlib
import json
import math
import struct


def identity(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def check():
    root = Path(__file__).resolve().parent
    load = lambda name: json.loads((root / name).read_text())
    collection = load("COLLECTION_BINDINGS.public.json")
    for name, expected in collection["files"].items():
        assert identity(root / name) == {k: expected[k] for k in ("bytes", "sha256")}, name
    freeze = load("source/freeze.public.json")
    for name, expected in freeze["files"].items():
        assert identity(root / "source" / name) == expected, name
    summary = load("run/stage1/summary.public.json")
    expected = load("source/expected.public.json")
    assert (root / "run/stage1.exitcode").read_text().strip() == "0"
    assert summary["stage1_passed"] and summary["accepted_checkpoint"]
    assert summary["evaluate_calls"] == summary["linearize_calls"] == 1
    assert all(summary[name] == 0 for name in ("callback_calls", "solver_calls", "native_process_calls"))
    assert not summary["full_H_materialized"]
    for name, values in summary["scalar_gates"].items():
        assert values["bitexact"]
        assert struct.pack("<d", values["actual"]) == struct.pack("<d", values["reference"])
        assert values["reference"] == expected["expected_state_before_lambda_override"][name]
    for name, values in summary["vector_gates"].items():
        assert values["bitexact"] and values["different_bits"] == 0
        assert values["actual_value_sha256"] == expected["files"][name]["sha256"]
        assert values["reference_file_sha256"] == values["actual_value_sha256"]
    assert summary["bindings_before"] == summary["bindings_after"]
    assert len(summary["bindings_before"]) == 29
    assert summary["flags_before"] == summary["flags_after"]
    assert not summary["flags_after"]["cuda_initialized"]
    assert summary["fsl_dso_snapshot_before"] == summary["fsl_dso_snapshot_after"] == []
    checkpoint = collection["private_checkpoint_identity_only"]
    assert checkpoint["array_content_transferred"] is False
    assert checkpoint["mode"] == "0o600" and checkpoint["directory_mode"] == "0o700"
    for name in ("bytes", "sha256", "array_count"):
        assert checkpoint[name] == summary["checkpoint"][name]
    assert len(summary["checkpoint"]["arrays"]) == checkpoint["array_count"] == 68
    assert summary["checkpoint"]["normal_cache"] == {"packed_layout": None, "scratch": None, "layout_copy_bytes": 0}
    assert not summary["checkpoint"]["closure_identity_preserved"]
    for path in root.rglob("*.py"):
        ast.parse(path.read_text(), filename=str(path))
    def finite(value):
        if isinstance(value, dict):
            return all(finite(v) for v in value.values())
        if isinstance(value, list):
            return all(finite(v) for v in value)
        return not isinstance(value, float) or math.isfinite(value)
    for path in root.rglob("*.json"):
        assert finite(json.loads(path.read_text())), path
    manifest_path = root / "MANIFEST.public.json"
    if manifest_path.exists():
        manifest = load("MANIFEST.public.json")
        for name, entry in manifest["files"].items():
            assert identity(root / name) == entry, name
        assert set(manifest["files"]) == {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p != manifest_path}
    return {"status": "PASS", "bound_as_run_files": len(collection["files"]), "evaluate_calls": 1,
            "linearize_calls": 1, "science_executed_by_checker": False, "private_arrays_downloaded": False}


if __name__ == "__main__":
    print(json.dumps(check(), sort_keys=True))
