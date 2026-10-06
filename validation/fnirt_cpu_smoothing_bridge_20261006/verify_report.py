"""Read saved report metadata with stdlib only; never execute the worker."""
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
    freeze_sha = "6efcc5d57568e26431dbb21c6054a3ccd6abde15be6e86ff28cbddb790eb9bbd"
    head = "db61cebc9d3683720e6c0a288db3cdc7d6f30a7c"
    frozen = load("source/freeze.public.json")
    assert identity(leaf / "source/freeze.public.json")["sha256"] == freeze_sha
    assert len(frozen["files"]) == 12
    for name, record in frozen["files"].items():
        assert identity(leaf / "source" / name) == record, name
    # Prepared records keep their original pre-authorization state.
    assert frozen["science_worker_started"] is False
    assert frozen["upload_started"] is False
    collection = load("COLLECTION_BINDINGS.public.json")
    assert len(collection["files"]) == 26
    for name, record in collection["files"].items():
        assert identity(leaf / name) == {k: record[k] for k in ("bytes", "sha256")}, name
        assert record["mode"] == "0o600", name
    assert sum(row["bytes"] for row in collection["files"].values()) == 135334
    assert collection["private_array_content_transferred"] is False
    assert collection["scientific_worker_repeated"] is False
    assert collection["transport"] == {
        "raw_chunk_cap_bytes": 45000, "batches": 4, "whole_file_hashes_reverified": True}
    expected = load("source/expected.public.json")
    summary_identity = {
        "bytes": 26637,
        "sha256": "45fb40c70d5fb12b326449ebacc46f1e9ae714996599c128d24cc04719108c8b"}
    assert identity(leaf / "run/smoothing/summary.public.json") == summary_identity
    summary = load("run/smoothing/summary.public.json")
    assert summary["status"] == "saved_current_smoothing_bridge_passed_no_registration"
    assert summary["accepted"] is True and summary["production_changed"] is False
    assert summary["gates"] == {
        "declared_header_orientation_and_zoom": True,
        "plain_current_matches_legacy_saved_bits": True,
        "adapter_matches_saved_native_bits": True}
    assert summary["calls"] == expected["limits"] == {
        "mature_blur": 2, "compiled_cpu_helper": 2, "adapter": 1,
        "normalization": 0, "RHS": 0, "H": 0, "PCG": 0,
        "evaluate": 0, "linearize": 0, "native": 0, "GPU": 0}
    assert len(summary["arms"]) == 2
    for record in summary["arms"].values():
        assert record["shape"] == [224, 288, 288]
        assert record["dtype"] == "float32" and record["values"] == 18579456
        assert record["different_bits"] == 0 and record["all_value_bits_exact"] is True
        assert record["max_abs"] == record["rmse"] == record["relative_l2"] == 0.0
    assert len(expected["production_source"]) == 17 and len(expected["source_extra"]) == 5
    assert len(expected["files"]) == 9 and len(expected["header_bindings"]) == 5
    expected_bindings = {}
    for category in ("production_source", "source_extra"):
        expected_bindings.update({category + "/" + name: row
                                 for name, row in expected[category].items()})
    expected_bindings.update({"input/" + name: {k: row[k] for k in ("bytes", "sha256")}
                             for name, row in expected["files"].items()})
    expected_headers = {name: row["metadata"] for name, row in expected["header_bindings"].items()}
    assert summary["bindings_before"] == summary["bindings_after"] == expected_bindings
    assert len(summary["bindings_before"]) == 31
    assert summary["harness_before"] == summary["harness_after"] == frozen["files"]
    assert summary["headers_before"] == summary["headers_after"] == expected_headers
    assert summary["source_inputs_and_harness_unchanged"] is True
    assert summary["operands_before"] == summary["operands_after"]
    assert summary["all_available_operands_unchanged"] is True
    assert summary["flags_before"] == summary["flags_after"]
    assert summary["flags_unchanged"] and not summary["flags_after"]["cuda_initialized"]
    assert summary["fsl_dso_snapshot_before"] == summary["fsl_dso_snapshot_after"] == []
    assert summary["no_fsl_dso_in_recorded_snapshots"] is True
    assert summary["main_head_at_read"] == head
    assert summary["affinity"] == expected["cpu_affinity"] == [32, 36, 40, 44, 48, 52, 56, 60]
    runtime = summary["runtime"]
    assert runtime["versions"] == expected["runtime_versions"]
    assert runtime["prefix_path_sha256"] == expected["prefix_path_sha256"]
    assert runtime["interpreter"] == expected["python_interpreter"]
    assert runtime["canonical_default_resolves_to_actual_prefix"] is True
    assert runtime["torch_threads"] == runtime["numba_threads"] == runtime["numba_config_threads"] == 8
    assert runtime["torch_interop_threads"] == 1
    assert runtime["environment"]["CUDA_VISIBLE_DEVICES"] == ""
    assert runtime["numba_cache_is_new_run_local"] is True
    assert summary["numba_cache_empty_before_numerical_import"] is True
    context = summary["numba_dispatcher_context"]
    assert context["selected_by_current_helper"] == "parallel" and context["threading_layer"] == "omp"
    assert context["serial_signatures"] == [] and len(context["parallel_signatures"]) == 1
    assert "Array(float32, 5, 'F'" in context["parallel_signatures"][0]
    assert "Array(float64, 1, 'C'" in context["parallel_signatures"][0]
    assert context["assembly_assessed"] is False
    assert context["per_axis_native_runtime_arrays_available"] is False
    assert summary["native_reference_decode"] == {
        "actual_dtype": "float32", "proxy_slope": 1.0,
        "proxy_intercept": 0.0, "reference_cast_performed": False}
    for name in ("smoothing.exitcode", "smoothing.science.exitcode", "preflight_after.exitcode"):
        assert int((leaf / "run" / name).read_text()) == 0
    for phase in ("before", "after"):
        record = load("run/preflight_" + phase + ".public.json")
        assert record["main_head_at_read"] == head and record["all_bindings_match"] is True
        assert record["failures"] == [] and record["image_values_decoded"] == 0
        assert record["production_source_count"] == 17 and record["import_dependency_source_count"] == 5
        assert record["bindings"] == expected_bindings and record["headers"] == expected_headers
        assert record["harness_bindings"] == frozen["files"]
    upload = load("source/UPLOAD_SERVER_VERIFY.public.json")
    assert upload["canonical_head"] == head and upload["all_files_and_inputs_match"] is True
    assert upload["worker_started"] is False and upload["arrays_uploaded"] is False
    assert upload["scientific_calls"] == 0 and upload["workspace_mode"] == 0o700
    assert all(mode == 0o600 for mode in upload["permissions"].values())
    launch = load("run/launch.public.json")
    assert launch["freeze_sha256"] == freeze_sha and launch["enqueued_once"] is True
    assert launch["controller_pid"] == int((leaf / "run/controller.pid").read_text()) == 51502
    assert launch["controller_start_ticks"] == 535257672
    assert launch["controller_deadline_unix"] - launch["enqueue_unix"] == 19600
    assert launch["maximum_calls"] == summary["calls"]
    locks = sorted([".INDEX.codex.lock", ".index.lock", "INDEX.json.lock", "INDEX.md.lock",
                    "admin/index-update.lock", "admin/index.update.lock"])
    for phase in ("prepared", "queued", "completed"):
        record = load("run/index_" + phase + ".public.json")
        assert record["entry"] == "fnirt_cpu_smoothing_bridge_20261006"
        assert record["six_locks"] == locks and record["permissions_preserved"] is True
    completed = load("run/index_completed.public.json")["entry_value"]
    assert completed["accepted"] is True and completed["status"] == "smoothing_completed"
    assert completed["all_exitcodes"] == {name: 0 for name in (
        "smoothing.exitcode", "smoothing.science.exitcode", "preflight_after.exitcode")}
    assert completed["summary"] == summary_identity and completed["actual_calls"] == summary["calls"]
    final = load("FINAL_CONTEXT.public.json")
    assert final["canonical_head"] == head and final["summary"] == summary_identity
    assert final["controller_proc_exists"] is False and final["no_new_scientific_calls"] is True
    assert final["compiled_cache_content_exported"] is False and final["machine_assembly_assessed"] is False
    assert final["worker_output_files"] == ["summary.public.json"]
    acceptance = load("ACCEPTANCE.public.json")
    assert acceptance["value_bits_gates"] == summary["arms"]
    assert acceptance["actual_calls"] == summary["calls"]
    assert all(acceptance[key] is False for key in (
        "complete_registration_accepted", "runtime_integration_accepted", "production_change", "gpu_change"))
    interpretation = load("INTERPRETATION.public.json")
    assert interpretation["runtime"] == runtime and interpretation["numba_dispatcher_context"] == context
    assert interpretation["native_decode"] == summary["native_reference_decode"]
    assert interpretation["old_full_stage_rejected"]["coeff_rmse_baseline_candidate"] == [0.084011, 0.127315]
    assert interpretation["next_runtime_patch_authorized"] is False
    for key, value in interpretation["clocks"].items():
        assert value == summary[key]
    history = load("STATUS_FAILURES.public.json")
    assert history["science_enqueued_once"] is True and history["science_retry"] is False
    assert history["science_failures"] == [] and history["rejected_full_stage_history_retained"] is True
    links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", (leaf / "README.md").read_text())
    local_links = [target for target in links if not target.startswith(("https://", "http://"))]
    for target in local_links:
        assert (leaf / target.split("#", 1)[0]).exists(), target
    forbidden_suffixes = {".npz", ".npy", ".f64", ".so", ".nii", ".gz", ".pkl", ".pickle", ".nbc", ".nbi", ".cxx", ".h"}
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
        names = {path.relative_to(leaf).as_posix() for path in files if path != manifest_path}
        assert names == set(manifest["files"]), names.symmetric_difference(manifest["files"])
        for name, record in manifest["files"].items():
            assert identity(leaf / name) == record, name
    return {"scope": "stdlib saved-metadata checks; zero numerical imports or scientific replay",
            "passed": True, "original_metadata_files": 26, "original_metadata_bytes": 135334,
            "harness_payloads": 12, "source_bindings": 22, "input_bindings": 9, "header_bindings": 5,
            "image_value_bits_gates": 2, "values_per_gate": 18579456,
            "all_three_gates_passed": True, "all_three_exit_codes_zero": True,
            "local_links_checked": len(local_links), "scientific_replays": 0,
            "forbidden_file_payloads": 0, "basic_IP_credential_pattern_hits": 0}


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
