"""Validation arithmetic/contracts only; fixtures are not performance evidence."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[2] / "validation/dmri_pipeline/public10_20261002/compare_public10.py"
spec = importlib.util.spec_from_file_location("public10_comparison", SOURCE)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def save_image(path, values, affine=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(values, dtype=np.float32),
                           np.eye(4) if affine is None else affine), str(path))
    return comparison.load_image(path)


def write_report(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path.name


def matched_provenance(backend="tbss"):
    digest = "a" * 64
    raw_names = ("AP.nii.gz", "AP.bval", "AP.bvec", "AP.json", "PA.nii.gz", "PA.bval", "PA.json")
    candidate = {"input_and_resource_provenance": {
        **{"raw_" + name: {"sha256": digest} for name in raw_names},
        "fa_template": {"sha256": digest}, "synthstrip_weights": {"sha256": digest}}}
    reference = {"input_files": {name: {"sha256": digest} for name in raw_names},
                 "templates": {"FA_reference": {"sha256": digest}},
                 "SynthStrip": {"weights": {"sha256": digest}}}
    if backend == "tbss":
        candidate["input_and_resource_provenance"]["fa_skeleton"] = {"sha256": digest}
        reference["templates"]["FA_skeleton"] = {"sha256": digest}
    else:
        for role in ("t1", "t1_template", "tensor_template"):
            candidate["input_and_resource_provenance"][role] = {"sha256": digest}
        reference["input_files"]["T1w.nii.gz"] = {"sha256": digest}
        reference["templates"].update({"T1_reference": {"sha256": digest}, "tensor_reference": {"sha256": digest}})
    return candidate, reference


def test_chunk_statistics_use_all_values_and_stable_centered_moments():
    x = np.array([1e6, 1e6 + 1, 1e6 + 3, 1e6 + 5, 1e6 + 7])
    y = np.array([1e6 + .5, 1e6 + 2, 1e6 + 4, 1e6 + 6, 1e6 + 7.5])
    accumulator = comparison.Statistics(5)
    accumulator.update(x[:2], y[:2])
    accumulator.update(x[2:], y[2:])
    result = accumulator.result()
    assert result["pearson_r"] == pytest.approx(np.corrcoef(x, y)[0, 1], abs=1e-10)
    assert result["rmse"] == pytest.approx(np.sqrt(np.mean((x - y)**2)))
    assert result["p95_absdiff"] == np.percentile(np.abs(x - y), 95)
    assert result["different_elements"] == 5
    assert result["p95_sampling"]["method"] == "all_elements"


def test_empty_and_constant_correlations_are_unavailable():
    empty = comparison.Statistics(0).result()
    assert empty["pearson_r"] is None and empty["rmse"] is None
    constant = comparison.Statistics(3)
    constant.update(np.ones(3), np.ones(3))
    result = constant.result()
    assert result["pearson_r"] is None and result["values_exact"]


def test_fixed_roi_keeps_lost_support_and_geometry_is_not_resampled(tmp_path):
    x = save_image(tmp_path / "candidate.nii.gz", [[[0., 2., 0.]]])
    y = save_image(tmp_path / "reference.nii.gz", [[[1., 2., 0.]]])
    roi = comparison.derived_mask(y, np.ones((1, 1, 3), bool))
    result = comparison.image_pair(x, y, roi)
    assert result["gate"]["passed"]
    assert result["statistics"]["fixed_roi"]["rmse"] == pytest.approx(np.sqrt(1 / 3))
    assert result["statistics"]["common_nonzero_support_in_fixed_roi"]["rmse"] == 0
    assert result["support"]["fixed_roi"]["dice"] == pytest.approx(2 / 3)
    affine = np.eye(4)
    affine[0, 3] = .1
    wrong_grid = save_image(tmp_path / "wrong.nii.gz", [[[1., 2., 0.]]], affine)
    result = comparison.image_pair(x, wrong_grid, roi)
    assert not result["gate"]["passed"] and result["statistics"] is None


def test_singleton_normalization_and_nonfinite_mask_rejection(tmp_path):
    a = save_image(tmp_path / "a.nii.gz", np.ones((2, 2, 2)))
    b = save_image(tmp_path / "b.nii.gz", np.ones((2, 2, 2, 1)))
    assert comparison.geometry_gate(a, b)["passed"]
    vector = save_image(tmp_path / "vector.nii.gz", np.ones((2, 2, 2, 3)))
    nifti_vector = save_image(tmp_path / "vector5d.nii.gz", np.ones((2, 2, 2, 1, 3)))
    assert comparison.geometry_gate(vector, nifti_vector)["passed"]
    corrupt = save_image(tmp_path / "bad.nii.gz", np.full((2, 2, 2), np.nan))
    mask = comparison.derived_mask(corrupt, corrupt["values"] != 0)
    assert mask["values"] is None
    assert not comparison.image_pair(a, b, mask)["gate"]["passed"]


def test_missing_file_does_not_expose_private_path(tmp_path):
    secret = tmp_path / "private_person" / "image.nii.gz"
    unavailable = comparison.load_image(secret)
    text = json.dumps(unavailable)
    assert str(tmp_path) not in text and "private_person" not in text
    assert unavailable["metadata"]["error_type"] == "FileNotFoundError"


def test_original_output_paths_match_the_two_reference_contracts():
    root = Path("root")
    assert comparison.map_path(root, "tbss", "skeleton", "FA", reference=True) == Path("root/tbss/stats/all_FA_skeletonised.nii.gz")
    assert comparison.map_path(root, "mmorf", "standard", "FA", reference=True) == Path("root/mmorf/standard/FA.nii.gz")
    assert comparison.map_path(root, "mmorf", "native", "OD", reference=True) == Path("root/native/NODDI_OD.nii.gz")


def test_full_mmorf_map_contract_and_missing_map_are_distinct(tmp_path):
    candidate, reference = tmp_path / "candidate", tmp_path / "original"
    template = tmp_path / "template.nii.gz"
    save_image(template, np.ones((2, 2, 2)))
    for root in (candidate, reference):
        save_image(root / "eddy/nodif_brain_mask.nii.gz", np.ones((2, 2, 2)))
        for space in ("native", "standard"):
            for name in comparison.MAP_NAMES:
                save_image(comparison.map_path(root, "mmorf", space, name, reference=root == reference),
                           np.arange(8).reshape(2, 2, 2))
    args = SimpleNamespace(case_id="case01", backend="mmorf", candidate_dir=candidate,
                           reference_dir=reference, reference_roi=template,
                           bvals=None, fa_skeleton=None, skeleton_threshold=2000)
    result = comparison.compare_case(args)
    assert result["status"] == "complete" and result["map_geometry_and_finite_pairs_passed"] == 18
    assert result["maps"]["standard"]["FA"]["statistics"]["fixed_roi"]["values_exact"]
    assert result["upstream"]["eddy_complete_dwi"]["statistics"] is None
    assert str(tmp_path) not in json.dumps(result)
    comparison.map_path(reference, "mmorf", "standard", "ISOVF", reference=True).unlink()
    result = comparison.compare_case(args)
    assert result["status"] == "required_map_gate_failed"
    assert result["map_geometry_and_finite_pairs_passed"] == 17
    assert not result["numerical_equivalence_claimed"]


def test_aggregate_retains_every_planned_case_and_failure(tmp_path):
    row = {"case_id": "case01", "backend": "tbss"}
    row["candidate_report"] = write_report(tmp_path / "candidate.json", {
        "case_id": "case01", "registration_backend": "tbss", "status": "failed",
        "failure": {"exception_type": "OutOfMemoryError"}, "timing": {"api_wall_seconds": 10}})
    row["reference_report"] = write_report(tmp_path / "reference.json", {
        "case_id": "case01", "registration_backend": "tbss", "status": "complete", "total_processing_seconds": 100})
    report = comparison.aggregate_plan({"planned_cases": [row]}, manifest_root=tmp_path)
    assert report["expected_branch_pairs"] == len(report["cases"]) == 20
    assert report["status"] == "incomplete"
    failed = next(item for item in report["cases"] if item["case_id"] == "case01" and item["backend"] == "tbss")
    assert failed["failure_types"] == {"candidate": "OutOfMemoryError"}
    assert failed["paired_ratios"]["processing_reference_over_candidate"] is None
    assert report["branches"]["tbss"]["paired_runs_complete"] == 0
    assert len(report["branches"]["tbss"]["missing_or_failed_case_ids"]) == 10
    assert str(tmp_path) not in json.dumps(report)


def test_aggregate_uses_paired_ratios_includes_accuracy_failures(tmp_path):
    rows = []
    for index, (candidate_time, reference_time) in enumerate(((1, 10), (100, 200)), 1):
        row = {"case_id": f"case{index:02d}", "backend": "tbss"}
        candidate_provenance, reference_provenance = matched_provenance()
        row["candidate_report"] = write_report(tmp_path / f"candidate{index}.json", {
            **candidate_provenance,
            "case_id": row["case_id"], "registration_backend": "tbss", "status": "complete",
            "timing": {"api_wall_seconds": candidate_time}, "source_python_sha256": {"foo.py": "hash"}})
        row["reference_report"] = write_report(tmp_path / f"reference{index}.json", {
            **reference_provenance,
            "case_id": row["case_id"], "registration_backend": "tbss", "status": "complete",
            "total_processing_seconds": reference_time})
        row["comparison_report"] = write_report(tmp_path / f"comparison{index}.json", {
            "case_id": row["case_id"], "registration_backend": "tbss", "status": "required_map_gate_failed"})
        rows.append(row)
    result = comparison.aggregate_plan({"planned_cases": rows}, expected_case_count=2, manifest_root=tmp_path)
    branch = result["branches"]["tbss"]
    assert branch["paired_runs_complete"] == 2 and branch["paired_comparisons_complete"] == 0
    assert branch["paired_ratios"]["processing_reference_over_candidate"]["median"] == 6
    assert branch["paired_ratios"]["processing_reference_over_candidate"]["median"] != 105 / 50.5


def test_aggregate_rejects_duplicate_and_external_failed_process(tmp_path):
    row = {"case_id": "case01", "backend": "mmorf"}
    with pytest.raises(ValueError, match="unexpected_or_duplicate"):
        comparison.aggregate_plan({"planned_cases": [row, row]})
    row["candidate_report"] = write_report(tmp_path / "candidate.json", {"status": "complete"})
    row["reference_report"] = write_report(tmp_path / "reference.json", {"status": "complete"})
    row["candidate_process_metrics"] = write_report(tmp_path / "time.json", {"exit_status": 1, "wall_seconds": 10})
    report = comparison.aggregate_plan({"planned_cases": [row]}, expected_case_count=1, manifest_root=tmp_path)
    assert not report["cases"][0]["paired_runs_complete"]


def test_gnu_time_parser_omits_command_and_recovers_hours(tmp_path):
    path = tmp_path / "time.txt"
    path.write_text('Command being timed: "/private/sub/secret command"\n'
                    'Elapsed (wall clock) time (h:mm:ss or m:ss): 1:02:03.45\n'
                    'Maximum resident set size (kbytes): 123456\nExit status: 0\n')
    metrics = comparison.process_metrics(path)
    assert metrics["wall_seconds"] == pytest.approx(3723.45)
    assert metrics["max_rss_kib"] == 123456
    assert "secret" not in json.dumps(metrics)


def test_observer_json_prefers_sibling_gnu_clock_and_keeps_gpu_scope(tmp_path):
    (tmp_path / "time.txt").write_text("Elapsed (wall clock) time (h:mm:ss or m:ss): 0:12.50\n"
                                      "Maximum resident set size (kbytes): 4321\nExit status: 0\n")
    path = tmp_path / "process_metrics.json"
    path.write_text(json.dumps({"wall_seconds": 15.7, "exit_code": 0,
                               "sampled_own_gpu_memory_peak_mib": 100,
                               "sampled_other_gpu_memory_peak_mib": 300,
                               "gpu_samples": [{"pid": 777, "path": "/private/ignored"}]}))
    metrics = comparison.process_metrics(path)
    assert metrics["wall_seconds"] == 12.5 and metrics["observer_wall_seconds"] == 15.7
    assert metrics["wall_time_source"] == "gnu_time_v"
    assert metrics["max_rss_kib"] == 4321 and metrics["exit_status"] == 0
    assert metrics["sampled_own_gpu_memory_peak_mib"] == 100
    assert metrics["sampled_other_gpu_memory_peak_mib"] == 300
    assert "ignored" not in json.dumps(metrics)


def test_paired_binding_omits_absent_optional_pa_bvec_and_requires_all_resources():
    candidate, reference = matched_provenance("mmorf")
    result = comparison.paired_input_binding(candidate, reference, "mmorf")
    assert result["status"] == "matched" and "PA.bvec" not in result["roles"]
    reference["templates"]["tensor_reference"]["sha256"] = "b" * 64
    assert comparison.paired_input_binding(candidate, reference, "mmorf")["status"] == "mismatched"
    del reference["templates"]["tensor_reference"]
    assert comparison.paired_input_binding(candidate, reference, "mmorf")["status"] == "unavailable"


def test_gradient_antipodal_match_and_zero_direction_failure(tmp_path):
    candidate, reference = tmp_path / "candidate", tmp_path / "reference"
    (candidate / "eddy").mkdir(parents=True)
    (reference / "eddy").mkdir(parents=True)
    bvals = tmp_path / "bvals"
    np.savetxt(bvals, [0, 1000, 2000])
    x = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    np.savetxt(candidate / "eddy/data.eddy_rotated_bvecs", x)
    np.savetxt(reference / "eddy/data.eddy_rotated_bvecs", -x)
    result = comparison.gradients(candidate, reference, bvals)
    assert result["compared_dwi_volumes"] == 1 and result["excluded_zero_direction_volumes"] == 1
    assert result["directed_angle_degrees"]["max"] == 180
    assert result["antipodal_angle_degrees"]["max"] == 0


def test_reference_rerun_retains_failed_attempt_without_inflating_speed(tmp_path):
    candidate_provenance, reference_provenance = matched_provenance("mmorf")
    candidate_dir, reference_dir, initial_dir = (tmp_path / name for name in ("candidate", "recovered", "initial_failed"))
    for directory in (candidate_dir, reference_dir, initial_dir):
        directory.mkdir()
    row = {"case_id": "case01", "backend": "mmorf"}
    write_report(candidate_dir / "report.json", {**candidate_provenance, "status": "complete",
        "case_id": "case01", "registration_backend": "mmorf", "timing": {"api_wall_seconds": 20}})
    write_report(reference_dir / "report.json", {**reference_provenance, "status": "complete",
        "case_id": "case01", "registration_backend": "mmorf", "total_processing_seconds": 200})
    write_report(initial_dir / "report.json", {"status": "failed", "failure_type": "RuntimeError",
        "total_processing_seconds": 340, "failure_message": "/private/person/secret_cuda_command",
        "subprocess_steps": [{"arguments": ["secret_cuda_command", "/private/input.nii.gz"]}]})
    write_report(candidate_dir / "process_metrics.json", {"exit_code": 0, "wall_seconds": 53,
        "sampled_own_gpu_memory_peak_mib": 1000, "sampled_other_gpu_memory_peak_mib": 0})
    write_report(reference_dir / "process_metrics.json", {"exit_code": 0, "wall_seconds": 503,
        "sampled_own_gpu_memory_peak_mib": 1100, "sampled_other_gpu_memory_peak_mib": 0})
    write_report(initial_dir / "process_metrics.json", {"exit_code": 1, "wall_seconds": 349,
        "sampled_own_gpu_memory_peak_mib": 1234, "sampled_other_gpu_memory_peak_mib": 4321,
        "gpu_samples": [{"private_path": "/private/ignored"}]})
    for directory, wall, code, rss in ((candidate_dir, "0:50.00", 0, 100),
                                       (reference_dir, "8:20.00", 0, 200),
                                       (initial_dir, "5:45.67", 1, 321)):
        (directory / "time.txt").write_text(f"Elapsed (wall clock) time (h:mm:ss or m:ss): {wall}\n"
            f"Maximum resident set size (kbytes): {rss}\nExit status: {code}\n")
    row.update(candidate_report="candidate/report.json", reference_report="recovered/report.json",
               candidate_process_metrics="candidate/process_metrics.json",
               reference_process_metrics="recovered/process_metrics.json",
               initial_failed_reference_report="initial_failed/report.json",
               initial_failed_reference_process_metrics="initial_failed/process_metrics.json",
               reference_rerun_reason="native MMORF failed; fresh complete raw rerun; logs in /private/failed/run")
    row["comparison_report"] = write_report(tmp_path / "comparison.json", {
        "case_id": "case01", "registration_backend": "mmorf", "status": "complete"})
    report = comparison.aggregate_plan({"planned_cases": [row]}, expected_case_count=1, manifest_root=tmp_path)
    selected = next(item for item in report["cases"] if item["backend"] == "mmorf")
    retained = selected["retained_initial_reference_attempt"]
    assert selected["paired_runs_complete"]
    assert selected["paired_ratios"]["processing_reference_over_candidate"] == 10
    assert selected["paired_ratios"]["full_process_reference_over_candidate"] == 10
    assert retained["status"] == "failed" and retained["failure_type"] == "RuntimeError"
    assert retained["report"]["sha256"] == comparison.file_hash(initial_dir / "report.json")
    assert retained["timing_seconds"] == {"processing": 340, "full_process": 345.67}
    assert retained["memory"] == {"max_rss_kib": 321, "own_gpu_sampled_peak_mib": 1234,
                                  "other_gpu_sampled_peak_mib": 4321}
    assert retained["external_process_metrics"]["exit_status"] == 1
    assert not retained["included_in_paired_speed_ratios"]
    attempts = report["reference_attempts"]
    assert attempts["initial"]["observed_attempts"] == attempts["initial"]["failed_attempts"] == 1
    assert attempts["rerun"]["observed_attempts"] == attempts["rerun"]["successful_attempts"] == 1
    assert attempts["selected_primary"]["successful_attempts"] == 1
    assert attempts["all"]["observed_attempts"] == 2 and attempts["all"]["success_fraction"] == .5
    assert report["branches"]["mmorf"]["reference_attempts"] == {
        **attempts, "planned_primary_cases": 1}
    serialized = json.dumps(report)
    assert str(tmp_path) not in serialized and "secret_cuda_command" not in serialized
    assert "/private" not in serialized and "[private filesystem path]" in serialized


def test_mmorf_startup_observation_fallback_preserves_unknown_retry_guard(tmp_path):
    reference = {
        "FSL_binaries": {"mmorf": {"sha256": "a" * 64, "bytes": 1234, "path": "/private/program"}},
        "MMORF": {"native_gpu_observations": [
            {"name": "mmorf", "seconds": 3.5, "exit_code": -6, "arguments": ["secret_command"]},
            {"name": "mmorf_startup_retry_2", "seconds": 30.0, "exit_code": 0,
             "sampled_process_gpu_memory_peak_mib": 1200},
            {"name": "applywarp_FA", "seconds": 2.0, "exit_code": 0},
        ]}}
    result = comparison.reference_mmorf_startups({}, reference, "mmorf", tmp_path)
    assert result["available"] and result["source"] == "outer_reference_public_observations"
    assert not result["helper_report"]["available"]
    assert [item["attempt"] for item in result["attempts"]] == [1, 2]
    assert result["attempts"][0]["retry_allowed"] is None
    assert result["attempts"][1]["startup_retry"] is None
    assert result["native_program"]["sha256"] == "a" * 64
    assert result["native_program"]["version"] is None
    assert "secret_command" not in json.dumps(result) and "/private" not in json.dumps(result)
    assert comparison.reference_mmorf_startups({}, reference, "tbss", tmp_path) is None
    unavailable = comparison.reference_mmorf_startups({}, {}, "mmorf", tmp_path)
    assert not unavailable["available"] and unavailable["attempts"] == []


def test_mmorf_private_helper_metadata_is_whitelisted_and_preferred(tmp_path):
    helper_file = tmp_path / "reference_output/mmorf/official_mmorf_report.json"
    helper_file.parent.mkdir(parents=True)
    helper = {
        "binaries": {"mmorf": {"sha256": "b" * 64, "bytes": 5678, "path": "/private/program"}},
        "reference": {"MMORF": "0.3.2", "MMORF_source_commit": "c" * 40},
        "startup_retry_policy": {"maximum_attempts": 3, "delay_seconds": 2,
            "scope": "constructor before registration; log in /private/log",
            "all_attempts_included_in_processing_time": True, "numerical_parameters_changed": False,
            "ignored_private_option": "secret_policy"},
        "subprocess_steps": [
            {"name": "mmorf", "seconds": 1.25, "exit_code": -6,
             "sampled_process_gpu_memory_peak_mib": 100,
             "arguments": ["secret_command", "/private/input"],
             "startup_retry": {"attempt": 1, "retry_allowed": True, "log_sha256": "d" * 64,
                 "reason": "diagnosed_constructor_allocation_before_registration",
                 "exception": "secret_exception"}},
            {"name": "mmorf_startup_retry_2", "seconds": 20.0, "exit_code": 0},
        ], "exception": "secret_exception"}
    helper_file.write_text(json.dumps(helper), encoding="utf-8")
    row = {"reference_dir": "reference_output"}
    outer = {"MMORF": {"native_gpu_observations": [{"name": "mmorf", "seconds": 999, "exit_code": 0}]}}
    result = comparison.reference_mmorf_startups(row, outer, "mmorf", tmp_path)
    assert result["source"] == "private_helper_report" and result["available"]
    assert result["helper_report"]["sha256"] == comparison.file_hash(helper_file)
    assert result["native_program"] == {"sha256": "b" * 64, "size_bytes": 5678,
                                        "version": "0.3.2", "source_commit": "c" * 40}
    assert result["startup_retry_policy"]["all_attempts_included_in_processing_time"]
    assert result["attempts"][0]["seconds"] == 1.25
    assert result["attempts"][0]["retry_allowed"] is True
    assert result["attempts"][0]["startup_retry"]["log_sha256"] == "d" * 64
    assert result["attempts"][1]["retry_allowed"] is None
    serialized = json.dumps(result)
    assert str(tmp_path) not in serialized and "/private" not in serialized
    assert "secret_" not in serialized


def test_candidate_failure_retention_uses_fresh_primary_source_and_renderer(tmp_path):
    import csv

    candidate_provenance, reference_provenance = matched_provenance("mmorf")
    old_dir, fresh_dir, reference_dir = (tmp_path / name for name in ("old_failed", "fresh_primary", "original"))
    for directory in (old_dir, fresh_dir, reference_dir):
        directory.mkdir()
    old_commit, fresh_commit = "1" * 40, "2" * 40
    write_report(old_dir / "report.json", {"status": "failed", "source_commit_supplied": old_commit,
        "source_python_sha256": {"module.py": "a" * 64}, "timing": {"api_wall_seconds": 300},
        "failure": {"exception_type": "OutOfMemoryError", "message": "/private/person/secret_failure"},
        "memory": {"peak_allocated_bytes": 19_000_000_000, "peak_reserved_bytes": 20_000_000_000}})
    write_report(fresh_dir / "report.json", {**candidate_provenance, "case_id": "case02",
        "registration_backend": "mmorf", "status": "complete", "source_commit_supplied": fresh_commit,
        "source_python_sha256": {"module.py": "b" * 64}, "timing": {"api_wall_seconds": 20}})
    write_report(reference_dir / "report.json", {**reference_provenance, "case_id": "case02",
        "registration_backend": "mmorf", "status": "complete", "total_processing_seconds": 200})
    for directory, wall, code in ((old_dir, "5:05.00", 1), (fresh_dir, "0:50.00", 0), (reference_dir, "8:20.00", 0)):
        write_report(directory / "process_metrics.json", {"exit_code": code, "wall_seconds": 999,
            "sampled_own_gpu_memory_peak_mib": 1234, "sampled_other_gpu_memory_peak_mib": 0})
        (directory / "time.txt").write_text(f"Elapsed (wall clock) time (h:mm:ss or m:ss): {wall}\n"
            f"Maximum resident set size (kbytes): 12345\nExit status: {code}\n")
    comparison_path = tmp_path / "comparison.json"
    write_report(comparison_path, {"case_id": "case02", "registration_backend": "mmorf", "status": "complete"})
    row = {"case_id": "case02", "backend": "mmorf",
           "candidate_report": "fresh_primary/report.json", "reference_report": "original/report.json",
           "comparison_report": "comparison.json", "candidate_process_metrics": "fresh_primary/process_metrics.json",
           "reference_process_metrics": "original/process_metrics.json", "initial_failed_candidate_report": "old_failed/report.json",
           "initial_failed_candidate_process_metrics": "old_failed/process_metrics.json",
           "candidate_rerun_reason": "AMICO workspace fixed; fresh fixed-source raw run"}
    aggregate = comparison.aggregate_plan({"planned_cases": [row]}, manifest_root=tmp_path)
    selected = next(item for item in aggregate["cases"] if item["case_id"] == "case02" and item["backend"] == "mmorf")
    retained = selected["retained_initial_candidate_attempt"]
    assert selected["source_commit"] == fresh_commit and retained["source_commit"] == old_commit
    assert selected["source_python_manifest_sha256"] != retained["source_python_manifest_sha256"]
    assert aggregate["source_commits"] == [fresh_commit]
    assert retained["failure_type"] == "OutOfMemoryError" and retained["timing_seconds"] == {"processing": 300, "full_process": 305}
    assert retained["memory"]["peak_reserved_bytes"] == 20_000_000_000
    assert not retained["included_in_paired_speed_ratios"]
    assert selected["paired_ratios"]["processing_reference_over_candidate"] == 10
    assert selected["paired_ratios"]["full_process_reference_over_candidate"] == 10
    counts = aggregate["candidate_attempts"]
    assert counts["initial"]["failed_attempts"] == 1 and counts["rerun"]["successful_attempts"] == 1
    assert counts["all"]["observed_attempts"] == 2 and counts["all"]["successful_attempts"] == 1
    assert aggregate["branches"]["mmorf"]["candidate_attempts"]["selected_primary"]["successful_attempts"] == 1

    renderer_file = SOURCE.with_name("render_report.py")
    renderer_spec = importlib.util.spec_from_file_location("public10_renderer_candidate_test", renderer_file)
    renderer = importlib.util.module_from_spec(renderer_spec)
    renderer_spec.loader.exec_module(renderer)
    aggregate_file = tmp_path / "aggregate.json"
    aggregate_file.write_text(json.dumps(aggregate), encoding="utf-8")
    output_dir = tmp_path / "rendered"
    binding = renderer.render(aggregate_file, output_dir)
    with (output_dir / "cases.csv").open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    record = next(item for item in records if item["case_id"] == "case02" and item["backend"] == "mmorf")
    assert record["source_commit"] == fresh_commit
    assert record["retained_initial_candidate_source_commit"] == old_commit
    assert record["retained_initial_candidate_failure_type"] == "OutOfMemoryError"
    assert float(record["processing_reference_over_candidate"]) == 10
    assert float(record["full_process_reference_over_candidate"]) == 10
    assert record["retained_initial_candidate_failure_used_for_ratio"] == "False"
    assert binding["candidate_full_run_attempts"]["all"] == {"observed": 2, "successful": 1, "failed": 1, "pending": 0}
    markdown = (output_dir / "RESULTS.md").read_text(encoding="utf-8")
    assert "最初保留的 FNIT 失败运行" in markdown and "OutOfMemoryError" in markdown
    assert "case01 的两个旧版成功运行属于回归诊断" in markdown
    assert "secret_failure" not in json.dumps(aggregate) and "/private" not in markdown
