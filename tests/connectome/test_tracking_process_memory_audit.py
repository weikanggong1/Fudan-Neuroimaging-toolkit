"""CPU provenance and memory-gate controls; no CUDA calls or real benchmark."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest


@pytest.fixture(scope="module")
def audit():
    source = Path(__file__).resolve().parents[2] / "tools/audit_connectome_tracking_process_memory.py"
    spec = importlib.util.spec_from_file_location("_tracking_memory_audit_cpu_test", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _control(mode="paired"):
    hashes = {name: digit * 64 for name, digit in (("benchmark_module", "a"),
              ("candidate_tracking", "b"), ("candidate_fod_module", "c"))}
    inputs = {name: {"sha256": "1" * 64, "size_bytes": 10}
              for name in ("fod", "five_tissue", "gmwmi")}
    output = {"accepted_streamlines": 1, "total_path_points": 2, "seeds_attempted": 100000,
              "output_sha256": "e" * 64, "digest_format": "raw byte unit control",
              "field_sha256": {key: "2" * 64 for key in
                               ("point_counts", "packed_points", "accepted_seeds", "lengths_mm", "endpoints")},
              "streamlines": [{"point_count": 2, "sha256": "3" * 64}]}
    options = {"n_seeds": 100000, "lmax": 8, "seed": 0, "batch_size": 8192,
               "arc_proposals": 16, "max_length_mm": 250., "min_length_mm": None, "step_mm": None,
               "max_angle_degrees": 45., "cutoff": .1, "power": .5, "compile_arc": False,
               "five_tissue_spacing_mm": [1., 1., 1.]}
    reference = {"status": "passed", "all_strict_equal": True, "fod_mode": mode,
                 "fixed_fod_function_identity_equal": mode == "shared",
                 "fod_function_identity": {variant: {name: True for name in ("real_sh", "tracking_sh_precomputed")}
                                           for variant in ("baseline", "candidate")},
                 "source_sha256": {"benchmark": hashes["benchmark_module"],
                                   "baseline_tracking": "f" * 64,
                                   "candidate_tracking": hashes["candidate_tracking"],
                                   "baseline_fod": "d" * 64 if mode == "paired" else hashes["candidate_fod_module"],
                                   "candidate_fod": hashes["candidate_fod_module"]},
                 "input_files": deepcopy(inputs), "options": options,
                 "runs": [{"variant": "baseline", "phase": "warmup", "n_seeds": 100000, "output": deepcopy(output)},
                          {"variant": "baseline", "phase": "hot", "n_seeds": 100000, "output": deepcopy(output)}]}
    return reference, hashes, inputs


@pytest.mark.parametrize("mode", ["shared", "paired"])
def test_oracle_comes_from_current_baseline_with_its_own_fod(audit, mode):
    reference, sources, inputs = _control(mode)
    options, oracle, provenance = audit.validate_reference(reference, sources, inputs)
    assert options == reference["options"]
    assert oracle["accepted_streamlines"] == 1  # Not historical H100 27411 or 27537.
    assert oracle["output_sha256"] == "e" * 64
    assert provenance["fod_mode"] == mode
    assert provenance["full_baseline_output_count"] == 2
    assert (provenance["baseline_fod_sha256"] == provenance["candidate_fod_sha256"]) == (mode == "shared")
    assert audit.compare_output(oracle, deepcopy(oracle))["all_equal"]


@pytest.mark.parametrize("change", ["source", "fod", "input_sha", "input_size", "baseline_output", "missing_paths", "missing_full", "binding", "options", "failed"])
def test_reference_rejects_wrong_provenance_or_mixed_outputs(audit, change):
    reference, sources, inputs = _control()
    if change == "source":
        reference["source_sha256"]["benchmark"] = "0" * 64
    elif change == "fod":
        reference["source_sha256"]["candidate_fod"] = "0" * 64
    elif change == "input_sha":
        inputs["fod"]["sha256"] = "0" * 64
    elif change == "input_size":
        inputs["fod"]["size_bytes"] += 1
    elif change == "baseline_output":
        reference["runs"][1]["output"]["streamlines"][0]["sha256"] = "0" * 64
    elif change == "missing_paths":
        del reference["runs"][0]["output"]["streamlines"]
    elif change == "missing_full":
        for run in reference["runs"]:
            run["n_seeds"] = 128
    elif change == "binding":
        reference["fod_function_identity"]["baseline"]["real_sh"] = False
    elif change == "options":
        del reference["options"]["power"]
    else:
        reference["all_strict_equal"] = False
    with pytest.raises(ValueError):
        audit.validate_reference(reference, sources, inputs)


def test_shared_reference_cannot_hide_different_fod_sources(audit):
    reference, sources, inputs = _control("shared")
    reference["source_sha256"]["baseline_fod"] = "0" * 64
    with pytest.raises(ValueError, match="shared FOD binding"):
        audit.validate_reference(reference, sources, inputs)


def test_old_shared_report_without_new_fod_metadata_remains_usable(audit):
    reference, sources, inputs = _control("shared")
    del reference["fod_mode"]
    del reference["fod_function_identity"]
    del reference["source_sha256"]["baseline_fod"]
    reference["source_sha256"]["fixed_fod"] = reference["source_sha256"].pop("candidate_fod")
    assert audit.validate_reference(reference, sources, inputs)[2]["fod_mode"] == "shared"


@pytest.mark.parametrize("field", ["output_sha256", "field_sha256", "streamlines", "accepted_streamlines"])
def test_changed_raw_byte_digest_cannot_pass_output_gate(audit, field):
    reference, sources, inputs = _control()
    oracle = audit.validate_reference(reference, sources, inputs)[1]
    actual = deepcopy(oracle)
    if field == "field_sha256":
        actual[field]["packed_points"] = "0" * 64
    elif field == "streamlines":
        actual[field][0]["sha256"] = "0" * 64
    elif field == "accepted_streamlines":
        actual[field] = 0
        actual["streamlines"] = []
    else:
        actual[field] = "0" * 64
    assert not audit.compare_output(oracle, actual)["all_equal"]


@pytest.mark.parametrize("mutation", [None, "other_gpu", "other_platform", "other_torch"])
def test_runtime_uses_current_physical_gpu_metadata(audit, mutation):
    runtime = {"torch_version": "2.5.1", "cpu_threads": 8}
    device = {"name": "NVIDIA A100", "total_memory_bytes": 80_000_000_000, "uuid": "unit-device"}
    reference = dict(runtime, device=deepcopy(device))
    if mutation == "other_gpu":
        reference["device"]["uuid"] = "other-device"
    elif mutation == "other_platform":
        reference["device"]["name"] = "NVIDIA H100"
    elif mutation == "other_torch":
        reference["torch_version"] = "2.4.0"
    if mutation is None:
        audit.validate_runtime(reference, runtime, device)
    else:
        with pytest.raises(ValueError):
            audit.validate_runtime(reference, runtime, device)


@pytest.mark.parametrize("mutation", [None, "limit", "unknown", "zero", "coverage", "allocator", "output", "snapshot"])
def test_memory_gate_requires_strict_process_budget_and_complete_sampling(audit, mutation):
    peaks = {"tracking": {"peak_allocated_bytes": 1_000_000_000, "peak_reserved_bytes": 2_000_000_000}}
    process = {"whole_audit_selected_process_sampled_peak_bytes": 3_000_000_000,
               "coverage_complete_under_declared_gap_limit": True}
    output_ok = snapshot_ok = True
    if mutation == "limit":
        process["whole_audit_selected_process_sampled_peak_bytes"] = 20_000_000_000
    elif mutation == "unknown":
        process["whole_audit_selected_process_sampled_peak_bytes"] = None
    elif mutation == "zero":
        process["whole_audit_selected_process_sampled_peak_bytes"] = 0
    elif mutation == "coverage":
        process["coverage_complete_under_declared_gap_limit"] = False
    elif mutation == "allocator":
        peaks["tracking"]["peak_reserved_bytes"] = 18_000_000_001
    elif mutation == "output":
        output_ok = False
    elif mutation == "snapshot":
        snapshot_ok = False
    gate = audit.memory_gate(peaks, process, allocator_cap=18_000_000_000, process_budget=20_000_000_000,
                             execution_ok=True, output_ok=output_ok, snapshot_ok=snapshot_ok)
    assert gate["passed_sampled_full_process_memory_gate"] == (mutation is None)
    assert gate["continuous_memory_upper_bound_proven"] is False


@pytest.mark.parametrize("mutation", [None, "sha", "helper"])
def test_source_loader_checks_file_sha_before_accepting_a_helper(audit, tmp_path, mutation):
    source = tmp_path / "control.py"
    source.write_text("def helper():\n    return 'CPU control'\n")
    expected = audit.file_sha256(source)
    if mutation == "sha":
        expected = "0" * 64
    functions = ("missing_helper",) if mutation == "helper" else ("helper",)
    if mutation is None:
        module = audit.load_module("_audit_helper_unit_control", source, expected, functions)
        assert module.helper() == "CPU control"
    else:
        with pytest.raises(ValueError):
            audit.load_module("_audit_helper_unit_control", source, expected, functions)


@pytest.mark.parametrize("mutation", ["budget", "gap", "overwrite"])
def test_cli_rejects_unsafe_budget_gap_or_overwrite(audit, tmp_path, mutation):
    arguments = []
    for name in ("benchmark-module", "candidate-tracking", "candidate-fod-module", "reference-report", "monitor-module"):
        arguments += ["--" + name, str(tmp_path / name), "--" + name + "-sha256", "a" * 64]
    for name in ("fod", "five-tissue", "gmwmi"):
        arguments += ["--" + name, str(tmp_path / name)]
    arguments += ["--output", str(tmp_path / ("reference-report" if mutation == "overwrite" else "output.json"))]
    if mutation == "budget":
        arguments += ["--memory-budget-gb", "21"]
    elif mutation == "gap":
        arguments += ["--sample-interval-seconds", "1", "--max-sample-gap-seconds", ".5"]
    with pytest.raises(SystemExit) as failure:
        audit.parse_args(arguments)
    assert failure.value.code == 2
