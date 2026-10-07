"""Check prepared scalar-control metadata using only the standard library.

No image arrays, weight files, scientific libraries or remote calls are read.
"""
from pathlib import Path
import hashlib
import json
import re
import ast


def identity(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main():
    leaf = Path(__file__).resolve().parent
    root = leaf.parents[1]
    manifest = json.loads((leaf / "MANIFEST.public.json").read_text())
    assert set(manifest["files"]) == {p.name for p in leaf.iterdir() if p.is_file() and p.name != "MANIFEST.public.json"}
    for name, wanted in manifest["files"].items():
        assert identity(leaf / name) == wanted, name
    freeze = json.loads((leaf / "freeze.public.json").read_text())
    assert set(freeze["files"]) == set(manifest["files"]) - {"freeze.public.json"}
    for name, wanted in freeze["files"].items():
        assert identity(leaf / name) == wanted, name
    context = json.loads((leaf / "CONTEXT.public.json").read_text())
    plan = json.loads((leaf / "PLAN.public.json").read_text())
    expected = json.loads((leaf / "EXPECTED.public.json").read_text())
    audit = json.loads((leaf / "SOURCE_AUDIT.public.json").read_text())
    assert plan["status"] == "prepared_only_zero_new_science_upload_enqueue"
    assert plan["worker_definition_status"] == "implemented_frozen_not_executed_not_authorized"
    assert expected["schema_arrays"] == 68 and set(expected["arrays_to_restore"]) == {"fixed", "state_residual", "state_mask", "scale"}
    assert expected["baseline_count"] == 14341
    assert audit["live_canonical_repository"]["commit"] == plan["main_head_at_preparation"]
    for section in [context["prior_evidence"], audit["FNIT_sources"]]:
        for name, wanted in section.items():
            relative = wanted.get("repository_relative_path", name)
            assert not Path(relative).is_absolute() and ".." not in Path(relative).parts
            assert identity(root / relative) == {k: wanted[k] for k in ["bytes", "sha256"]}, relative
    assert audit["reference_scan_header"]["header348_bytes"] == 348
    assert audit["reference_scan_header"]["srow_x"] == [-8.0, 0.0, 0.0, 90.0]
    assert plan["resource_limits"]["science_seconds"] == 60
    assert plan["resource_limits"]["six_INDEX_locks_common_timeout_seconds"] == 25
    for name in ["contracts.py", "scalar_control.py", "controller.py"]:
        ast.parse((leaf / name).read_text())
    assert context["new_operations"]["checkpoint_array_read"] == 0
    assert context["new_operations"]["scientific_math_worker"] == 0
    assert context["new_operations"]["upload"] == context["new_operations"]["enqueue"] == 0
    assert context["observed_saved_values"]["projection_control_scale_bit0"] is True
    assert context["remaining_component_caches"]["matching_filenames"] == []
    assert len(audit["installed_reference_sources"]) == 6
    readme = (leaf / "README.md").read_text()
    for number in range(1, 8):
        assert "## " + str(number) + "." in readme
    for filename in manifest["files"]:
        data = (leaf / filename).read_text()
        assert not re.search(r"(?<![0-9])(?:10|192[.]168)[.][0-9]+[.][0-9]+[.][0-9]+", data), filename
        assert not re.search(r"gh[pousr]_[A-Za-z0-9]{20,}|BEGIN (?:OPENSSH|RSA) PRIVATE KEY", data), filename
    print(json.dumps({"status": "passed", "payload_files": len(manifest["files"]), "scientific_runs": 0, "uploads": 0, "enqueue": 0, "production_changes": 0}))


if __name__ == "__main__":
    main()
