"""Verify stored scalar/hash provenance with stdlib only; no scientific replay."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import re


def identity(path):
    b = path.read_bytes()
    return {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}


def check():
    leaf = Path(__file__).resolve().parent
    load = lambda n: json.loads((leaf / n).read_text())
    total_files, total_bytes = 0, 0
    statuses = {"v1": ("25bfacc620dd27770a994c708e190fd4d50e55ce49b06440ba8d390d6b351d9d", 14, 77738, [1, 1, 0],
                        "fd5fd0b04ec9434aa49b7457f366812147fe58e682ac323db185fdd68e02c2b1", 37674),
                "v2": ("780e109610e4f7f1f84bb639185b97950347e18de6fcd68e7f511e1bdead6a93", 15, 89880, [0, 0, 0],
                        "2aca393df523196d0f22f2c1bd8e0ad9a2af0f95f0108beba9f0a64e5faf501c", 62103)}
    summaries = {}
    for version, (freeze_sha, count, pid, rc, summary_sha, summary_bytes) in statuses.items():
        source, run = leaf / version / "source", leaf / version / "run"
        freeze = json.loads((source / "freeze.public.json").read_text())
        assert identity(source / "freeze.public.json")["sha256"] == freeze_sha
        assert len(freeze["files"]) == count and freeze["science_calls_at_freeze"] == 0
        for n, record in freeze["files"].items():
            assert identity(source / n) == record, (version, n)
        collection = load(version.upper() + "_COLLECTION_BINDINGS.public.json")
        for n, record in collection["files"].items():
            assert identity(leaf / version / n) == {k: record[k] for k in ("bytes", "sha256")}, n
            if n == "source/UPLOAD_SERVER_VERIFY.public.json":
                assert record["mode"] == "0o644", n  # non-frozen hash-only metadata inside private700 W
            else:
                assert record["mode"] == "0o600", n
        assert not collection["private_array_content_transferred"] and not collection["scientific_worker_repeated"]
        total_files += len(collection["files"])
        total_bytes += sum(x["bytes"] for x in collection["files"].values())
        s = json.loads((run / "projection/summary.public.json").read_text()); summaries[version] = s
        assert identity(run / "projection/summary.public.json") == {"bytes": summary_bytes, "sha256": summary_sha}
        assert [int((run / n).read_text()) for n in ("projection.exitcode", "projection.science.exitcode", "preflight_after.exitcode")] == rc
        expected = json.loads((source / "expected.public.json").read_text())
        assert len(expected["production_source"]) == 17 and len(expected["source_extra"]) == 6
        assert len(expected["files"]) == 9 and len(expected["checkpoint_arrays_read"]) == 29
        assert s["source_binding_count"] == 23 and s["input_file_binding_count"] == 9
        assert s["bindings_before"] == s["bindings_after"] and len(s["bindings_before"]) == 32
        assert s["harness_bindings"] == s["harness_bindings_after"] == freeze["files"]
        assert s["checkpoint_values_before"] == s["checkpoint_values_after"] and len(s["checkpoint_values_before"]) == 29
        assert s["derived_operand_values_before"] == s["derived_operand_values_after"]
        assert all(s[k] for k in ("flags_unchanged", "checkpoint_operands_unchanged", "derived_operands_unchanged", "all_source_inputs_harness_unchanged"))
        assert s["flags_before"] == s["flags_after"] and not s["flags_after"]["cuda_initialized"]
        assert s["fsl_dso_snapshot_before"] == s["fsl_dso_snapshot_after"] == []
        assert s["postcheck_errors"] == [] and not s["production_integration_accepted"]
        assert s["affinity"] == [32, 36, 40, 44, 48, 52, 56, 60]
        assert s["torch_threads"] == 8 and s["interop_threads"] == 1 and s["address_space_cap_bytes"] == 20000000000
        assert s["runtime"]["actual_prefix_matches_declared_alias"] is True
        for phase in ("before", "after"):
            p = json.loads((run / ("preflight_" + phase + ".public.json")).read_text())
            assert p["all_bindings_match"] and p["failures"] == []
            assert p["main_head_at_read"] == "7ff215ee86c49414b2fa6156fdf6769aa54d00f8"
            assert p["bindings"] == s["bindings_before"] and p["harness_bindings"] == freeze["files"]
        launch = json.loads((run / "launch.public.json").read_text())
        assert launch["freeze_sha256"] == freeze_sha and launch["enqueued_once"]
        assert launch["controller_pid"] == int((run / "controller.pid").read_text()) == pid
        assert launch["controller_deadline_unix"] - launch["enqueue_unix"] == 360
        assert launch["scientific_child_timeout_seconds"] == 180
        for phase in ("prepared", "queued", "completed"):
            p = json.loads((run / ("index_" + phase + ".public.json")).read_text())
            assert len(p["six_locks"]) == 6 and p["permissions_preserved"]
            assert p["six_lock_timeout_seconds"] == 25 and p["six_lock_mode"] == "LOCK_EX|LOCK_NB"
        p = json.loads((run / "index_completed.public.json").read_text())["entry_value"]
        assert p["summary"] == identity(run / "projection/summary.public.json") and p["actual_calls"] == s["calls"]
    assert (total_files, total_bytes) == (57, 386818)
    v1, v2 = summaries["v1"], summaries["v2"]
    assert v1["status"] == "failed_first_mismatch_no_retry" and not v1["control_completed"]
    assert v1["error"] == {"type": "RuntimeError", "message": "saved moving header bytes"}
    assert all(x == 0 for x in v1["calls"].values())
    assert v1["images_read"] == ["current_fixed_level_npy", "official_fixed"]
    assert len(v1["gates"]) == 2 and all(x["bitexact"] for x in v1["gates"].values())
    assert v2["status"] == "projection_control_completed_no_solver" and v2["control_completed"]
    assert v2["calls"] == {"sampler": 1, "coordinates": 1, "current_projection": 1, "candidate_projection": 1,
        "state_scalar_prefix": 1, "LM_gradient_prefix": 0, "FSL_order_prefix": 2, "actual_bending_normal": 1,
        "cached_gradient_bending_reads": 2, "evaluate": 0, "linearize": 0, "gradient_method": 0,
        "dense_field_expansion": 0, "design_diagonal": 0, "PCG": 0, "SCG": 0, "H_callback": 0, "native": 0}
    assert len(v2["gates"]) == 17
    for name, row in v2["gates"].items():
        if "bitexact" in row: assert row["bitexact"], name
        elif "exact" in row: assert row["exact"], name
        elif "passed" in row: assert row["passed"], name
        else: assert row["reference_sha256"] == row["actual_sha256"], name
    assert v2["divisor_contract"]["operator"] == "aten.div.Tensor" and v2["divisor_contract"]["numel"] == 3
    assert v2["divisor_contract"]["division_operand_shape"] == [3, 1, 1, 1] and not v2["divisor_contract"]["scalar_or_0D_operand"]
    assert v2["scale_g_unchanged"] and v2["projection_value_change"]["different_bits"] == 12267
    m = v2["full_g_samepoint_comparison"]
    assert m["current_projection_vs_saved_native"]["full"]["relative_l2"] == 2.483601764762327e-7
    assert m["division_projection_vs_saved_native"]["full"]["relative_l2"] == 2.483864966670823e-7
    assert m["division_vs_current_projection"]["blocks"]["global_scale"]["bitexact"]
    assert m["division_vs_current_projection"]["full"]["different_bits"] == 1051
    assert m["division_projection_vs_saved_native"]["full"]["relative_l2"] > m["current_projection_vs_saved_native"]["full"]["relative_l2"]
    accept = load("ACCEPTANCE.public.json")
    assert accept["v2"]["full_g_samepoint_comparison"] == m
    assert all(accept[k] is False for k in ("production_integration_accepted", "native_gradient_equivalence_established", "complete_registration_accepted", "production_GPU_changed"))
    audit = load("HEADER_REPRESENTATION_AUDIT.public.json")
    assert audit["raw348_matches_original"] and audit["spatial_fields_bitexact"]
    assert audit["changed_byte_offsets"] == [110, 111, 114, 115, 118, 119]
    assert audit["changed_header_fields"] == ["vox_offset", "scl_slope", "scl_inter"]
    assert audit["nifti_header_pixel_read_calls"] == 0 and not audit["controller_alive"]
    assert v2["header_representation_bridge"]["stored348_identity_exact"] and v2["header_representation_bridge"]["spatial_fields_bitexact"]
    for row in v2["header_representation_bridge"]["spatial_fields"]: assert row["bitexact"]
    a, b = (ast.parse((leaf / ver / "source/projection_control.py").read_text()) for ver in ("v1", "v2"))
    for tree in (a, b):
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "read_official":
                for block in node.body:
                    if isinstance(block, ast.If) and isinstance(block.test, ast.Compare) and ast.unparse(block.test) == "label == 'official_moving'":
                        block.body = block.body[1:]
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == "header_contract": node.names = [ast.alias(name="classify_header")]
    assert ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False)
    for n in ("projection_adapter.py", "expected.public.json", "gradient_body.py", "projection_io.py", "preflight_projection.py"):
        assert identity(leaf / "v1/source" / n) == identity(leaf / "v2/source" / n)
    csv_rows = list(csv.DictReader((leaf / "BLOCK_METRICS.csv").open()))
    assert len(csv_rows) == 15
    for row in csv_rows:
        item = m[row["comparison"]]
        item = item["full"] if row["block"] == "full_mixed_units" else item["blocks"][row["block"]]
        for key in ("relative_l2", "max_abs", "rmse"): assert float(row[key]) == item[key]
    links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", (leaf / "README.md").read_text())
    for target in links:
        if not target.startswith(("https://", "http://")): assert (leaf / target).exists(), target
    forbidden = {".npz", ".npy", ".f64", ".so", ".nii", ".gz", ".pkl", ".pickle", ".nbc", ".nbi", ".cxx", ".h"}
    files = [p for p in leaf.rglob("*") if p.is_file()]
    for p in files:
        assert p.suffix not in forbidden
        raw = p.read_text()
        assert not re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", raw), p.name
        assert not re.search(r"(?i)(?:password|access_token|authorization)\s*[:=]\s*[\"']?\S+", raw), p.name
        if p.suffix == ".json": json.loads(raw)
        if p.suffix == ".py": compile(ast.parse(raw), str(p), "exec")
    manifest_path = leaf / "MANIFEST.public.json"
    if manifest_path.exists():
        manifest = load(manifest_path.name)
        names = {p.relative_to(leaf).as_posix() for p in files if p != manifest_path}
        assert names == set(manifest["files"])
        for n, record in manifest["files"].items(): assert identity(leaf / n) == record, n
    return {"passed": True, "scope": "stored metadata stdlib check; no numerical imports or scientific replay",
            "original_metadata_files": total_files, "original_metadata_bytes": total_bytes, "source_bindings": 23,
            "input_bindings": 9, "v1_RC": [1, 1, 0], "v1_sampler_projection_RHS_calls": 0,
            "v2_RC": [0, 0, 0], "v2_gates_passed": 17, "v2_total_RHS_improved": False,
            "candidate_integration_accepted": False, "private_array_downloads": 0, "new_scientific_runs": 0}


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
