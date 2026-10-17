"""报告metadata白名单与失败状态契约；不模拟影像或执行科学benchmark。"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "validation/recon_all/optimizations/20261017_sphere_normals/summarize_speed_reports.py"
SPEC = importlib.util.spec_from_file_location("fnit_public_speed_summary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
SHA = "a" * 64
COMMIT = "b" * 40


class PublicSpeedSummaryTest(unittest.TestCase):
    def test_register_keeps_false_strict_and_hides_all_private_strings(self):
        child = {"status": "complete", "host": "private-host", "subject": "/secret-user/subject",
            "source_version": COMMIT, "output_sha256": SHA, "gpu_uuid_actual": "GPU-private-uuid",
            "source_sha256": {"fnit/recon_all/mris_register_nonlinear.py": SHA,
                              "/secret-user/other.py": SHA, "fnit/vendor/unrelated.py": SHA},
            "api_total_seconds_including_io": 12., "wall_seconds_including_io_jit_sync_init": 13.,
            "cuda_allocation_cache_disabled": True, "gpu_peak_allocated_bytes": 0,
            "gpu_peak_reserved_bytes": 0, "traceback": "private-host 10.111.222.33 secret-user"}
        pair = {"repeat": 0, "strict_reproduction": False, "comparison": {"coordinates_equal": False,
                "different_coordinates": 3, "reference": "/secret-user/reference"},
                "baseline": {"stage_report": child, "returncode": 0, "process_wall_seconds_including_imports": 14.},
                "candidate": {"stage_report": child, "returncode": 0, "process_wall_seconds_including_imports": 14.}}
        report = {"status": "complete", "baseline_version": COMMIT, "candidate_version": COMMIT,
                  "strict_reproduction": False, "runs": [pair, {**pair, "repeat": 1}]}
        output = MODULE.register_summary(report)
        self.assertIs(output["strict_reproduction"], False)
        self.assertIs(output["runs"][0]["comparison"]["coordinates_equal"], False)
        self.assertEqual(output["runs"][0]["source"]["actual_commit"], COMMIT)
        self.assertEqual(list(output["runs"][0]["source"]["scientific_file_sha256"]),
                         ["src/fnit/recon_all/mris_register_nonlinear.py"])
        self.assertIsNone(output["runs"][0]["memory"]["pytorch"]["allocated_peak_bytes"])
        self.assertEqual(output["same_arm_repetition"]["baseline"]["API_wrapper_variation"]["samples"], 2)
        serialized = json.dumps(output)
        for value in ("secret-user", "private-host", "GPU-private-uuid", "10.111.222.33", "unrelated.py"):
            self.assertNotIn(value, serialized)

    def test_curvature_comparison_failure_and_three_output_groups_remain_separate(self):
        comparisons = {name: {"same_bytes": True, "different_elements": 0, "max_abs": 0.}
                       for name in (*MODULE.FINAL_MAPS, *MODULE.EARLY_MAPS)}
        comparisons["inflated.H"] = {"same_bytes": False, "different_elements": 5, "max_abs": 1e-7}
        comparisons["curv.stats"] = {"same_bytes": False, "header_lines_equal": False,
            "numeric_fields_exact": True, "ordered_structure_equal": True,
            "numerical_and_semantic_equal": True, "reference_header_lines": ["# user private"]}
        trials = [{"iteration": i, "mode": mode, "complete_api_seconds": 10. + i,
                   "comparison_to_first_legacy": comparisons, "output_sha256": {},
                   "torch_memory": {"reserved_entry_bytes": 100, "reserved_peak_bytes": 200},
                   "process_tree_memory": {"status": "ownership_unresolved", "peak_tree_total_bytes": 999,
                                             "peak_target_device_used_bytes": 3000}}
                  for i, mode in enumerate(("legacy", "deferred", "deferred", "legacy"))]
        output = MODULE.curvature_summary({"status": "comparison_failed", "code_version": COMMIT,
            "order": ["legacy", "deferred", "deferred", "legacy"],
            "cases": [{"case": "private-user-case", "hemisphere": "lh", "trials": trials,
                       "all_output_values_and_semantics_equal": False}]})
        self.assertEqual(output["execution_status"], "comparison_failed")
        self.assertFalse(output["cases"][0]["all_output_values_and_semantics_equal"])
        self.assertEqual(len(output["cases"][0]["trials"][0]["final_eight_maps"]), 8)
        self.assertEqual(len(output["cases"][0]["trials"][0]["early_four_maps"]), 4)
        self.assertIs(output["cases"][0]["trials"][0]["final_stats"]["numeric_fields_exact"], True)
        self.assertIsNone(output["cases"][0]["trials"][0]["memory"]["owned_simultaneous_peak_bytes"])
        self.assertEqual(output["cases"][0]["trials"][0]["memory"]["sampled_target_device_peak_bytes"], 3000)
        self.assertNotIn("private-user-case", json.dumps(output))
        self.assertFalse(output["run_coverage_complete"])

    def test_standard_failed_quality_worker_retains_API_but_is_not_performance_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "worker.json"
            path.write_text(json.dumps({"status": "running", "phase": "saved_surface_quality_running",
                "stage_seconds_including_io_and_synchronization": 149., "timing_qualification": "normal",
                "host": "secret-host", "target_gpu_uuid": "GPU-secret"}))
            output = MODULE.sphere_summary({"status": "failed", "baseline_commit": COMMIT, "candidate_commit": COMMIT,
                "runs": [{"case": "public-input", "hemisphere": "lh", "mode": "timing", "arm": "A",
                          "exit_code": -11, "status": "running", "worker_report": str(path),
                          "worker_report_sha256": MODULE.sha(path), "parent_full_wall_seconds": 155.}]})
            self.assertEqual(output["execution_status"], "failed")
            self.assertEqual(output["runs"][0]["stage_API_seconds"], 149.)
            self.assertFalse(output["runs"][0]["performance_sample_eligible"])
            self.assertEqual(output["same_arm_repetition"][0]["stage_API_variation"]["status"], "not_assessed")
            self.assertNotIn(directory, json.dumps(output))

    def test_standard_rejects_changed_child_report_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "worker.json"
            path.write_text('{"status":"complete"}')
            output = MODULE.sphere_summary({"status": "complete", "baseline_commit": COMMIT,
                "candidate_commit": COMMIT, "runs": [{"worker_report": str(path), "worker_report_sha256": SHA}]})
            self.assertEqual(output["runs"][0]["child_metadata_binding"], "hash_mismatch")
            self.assertFalse(output["runs"][0]["performance_sample_eligible"])

    def test_pair_missing_or_failed_runs_do_not_get_speed_or_equivalence_pass(self):
        output = MODULE.speed_pair_summary({"schema_version": 1,
            "scope": "two-raw-T1-four-empty-directory-paired-whole-runs", "status": "running", "runs": []})
        self.assertFalse(output["run_coverage_complete"])
        self.assertEqual(output["performance_acceptance"], "not_assessed")
        self.assertTrue(all(row["observed_CLI_reduction_percent"] is None for row in output["within_case_observations"]))

    def test_unknown_and_omitted_schemas_preserve_not_assessed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unknown.json"
            path.write_text(json.dumps({"status": "pass-from-some-new-schema", "host": "private-host", "secret": "private-value"}))
            output = MODULE.summarize({"evaluation": [path]})
            self.assertEqual(output["reports"]["evaluation"][0]["original_status"], "unknown")
            self.assertEqual(output["reports"]["evaluation"][0]["summary"]["assessment"], "not_assessed")
            self.assertEqual(output["reports"]["register_v2"]["status"], "not_assessed")
            self.assertEqual(output["overall_metric_equivalence"], "not_assessed")
            self.assertNotIn("private-value", json.dumps(output))

    def _monitor_fixture(self):
        totals = {"sampled_frames": 5, "resolved_frames": 0, "unknown_or_failed_frames": 5,
            "maximum_observed_interval_seconds": .5002, "owned_simultaneous_sampled_peak_bytes": None,
            "partial_attributed_same_snapshot_peak_bytes": 0, "target_device_sampled_peak_bytes": 42000000000,
            "target_compute_process_sum_sampled_peak_bytes": 41000000000}
        return {"scope": MODULE.MONITOR_SCOPE, "status": "complete_monitoring", "child_started": True,
            "command_exit_code": 0, "cuda_initialized_at_entry": False, "cuda_initialized_after": False,
            "source_commit": COMMIT, "source_manifest_sha256": SHA, "source_manifest_declared_files": 639,
            "files_sha256": {name: SHA for name in MODULE.MONITOR_FILE_ROLES},
            "files_unchanged_after_monitoring": {name: True for name in MODULE.MONITOR_FILE_ROLES},
            "sampling_interval_seconds": .5, "query_backend": "nvml", "wrapper_totals": totals,
            "per_run_observed_samples": {}, "run_metadata_observed": {}, "samples_jsonl_sha256": SHA,
            "host": "private-host", "uid": 9999, "target_gpu_uuid": "GPU-private-uuid",
            "command": ["/secret-user/science.sh"], "samples_path": "/secret-user/unreadable.jsonl"}

    def test_outer_probe_is_not_whole_monitoring_and_unknown_own_stays_null(self):
        raw = self._monitor_fixture()
        raw["wrapper_totals"]["owned_simultaneous_sampled_peak_bytes"] = 123
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.private.json"
            path.write_text(json.dumps(raw))
            record = MODULE.summarize({"outer_monitor": [path]})["reports"]["outer_monitor"][0]
            output = record["summary"]
            self.assertEqual(record["original_status"], "complete_monitoring")
            self.assertEqual(output["wrapper_observed_frames"]["sampled_frames"], 5)
            self.assertIsNone(output["wrapper_observed_frames"]["owned_simultaneous_sampled_peak_bytes"])
            self.assertEqual(output["metadata_bound_run_blocks"], 0)
            self.assertFalse(output["samples_jsonl_read_or_verified_by_summary"])
            self.assertEqual(output["scientific_E2E_completion"], "not_assessed_by_monitor")
            self.assertEqual(output["20GB_owned_memory_acceptance"], "not_assessed")
            for private in (directory, "secret-user", "private-host", "GPU-private-uuid", '"uid"'):
                self.assertNotIn(private, json.dumps(record))

    def test_outer_failure_and_partial_or_unknown_metadata_never_get_memory_acceptance(self):
        raw = self._monitor_fixture()
        raw.update(status="monitor_failed", command_exit_code=1, traceback="private-host secret-user 10.111.222.33")
        raw["wrapper_totals"].update(resolved_frames=5, unknown_or_failed_frames=0,
            owned_simultaneous_sampled_peak_bytes=100, maximum_observed_interval_seconds=float("nan"))
        raw["20GB_owned_memory_acceptance"] = "passed"
        output = MODULE.outer_monitor_summary(raw)
        self.assertEqual(output["monitoring_status"], "monitor_failed")
        self.assertEqual(output["child_exit_code"], 1)
        self.assertTrue(output["monitor_error_present"])
        self.assertIsNone(output["wrapper_observed_frames"]["owned_simultaneous_sampled_peak_bytes"])
        self.assertIsNone(output["wrapper_observed_frames"]["maximum_observed_interval_seconds"])
        self.assertEqual(output["20GB_owned_memory_acceptance"], "not_assessed")
        self.assertNotIn("private-host", json.dumps(output, allow_nan=False))
        raw["status"] = "some_future_complete"
        self.assertEqual(MODULE.outer_monitor_summary(raw)["monitoring_status"], "unknown")

    def test_outer_per_run_requires_explicit_metadata_and_never_parses_private_run_names(self):
        raw = self._monitor_fixture()
        private = "/private-case-user/path-case"
        raw["per_run_observed_samples"] = {private: dict(raw["wrapper_totals"]), "missing-metadata": {}}
        raw["run_metadata_observed"] = {private: {"run_id": private, "case": "private-case-name",
            "arm": "candidate", "code_version": COMMIT, "input_sha256": SHA,
            "benchmark_report_sha256_observed": SHA, "pair_report_sha256_observed": SHA,
            "status": "complete", "exit_code": 0, "benchmark_execution": "complete"}}
        output = MODULE.outer_monitor_summary(raw)
        self.assertEqual(output["reported_per_run_blocks"], 2)
        self.assertEqual(output["metadata_bound_run_blocks"], 1)
        self.assertEqual(output["unbound_per_run_blocks"], 1)
        self.assertEqual(output["runs"][0]["run_index"], 0)
        self.assertEqual(output["runs"][0]["arm"], "candidate")
        self.assertEqual(output["runs"][0]["observed_frames"]["partial_attributed_same_snapshot_peak_bytes"], 0)
        self.assertNotIn("private-case", json.dumps(output))
        self.assertNotIn("missing-metadata", json.dumps(output))

    def test_outer_inconsistent_counts_or_manifest_cannot_be_resolved_peak(self):
        raw = self._monitor_fixture()
        raw["wrapper_totals"].update(resolved_frames=5, unknown_or_failed_frames=0, owned_simultaneous_sampled_peak_bytes=100)
        output = MODULE.outer_monitor_summary(raw)
        self.assertEqual(output["wrapper_observed_frames"]["owned_simultaneous_sampled_peak_bytes"], 100)
        self.assertEqual(output["wrapper_observed_frames"]["20GB_owned_memory_acceptance"], "not_assessed")
        raw["wrapper_totals"]["resolved_frames"] = 4
        output = MODULE.outer_monitor_summary(raw)
        self.assertFalse(output["wrapper_observed_frames"]["reported_counts_consistent"])
        self.assertIsNone(output["wrapper_observed_frames"]["owned_simultaneous_sampled_peak_bytes"])
        raw["wrapper_totals"]["resolved_frames"] = 5
        raw["source_manifest_sha256"] = "c" * 64
        output = MODULE.outer_monitor_summary(raw)
        self.assertFalse(output["source"]["declared_sources_bound_and_unchanged"])
        self.assertIsNone(output["wrapper_observed_frames"]["owned_simultaneous_sampled_peak_bytes"])

    def _evaluation_fixture(self, root, *, main_status="complete"):
        main = root / "evaluation.json"
        main.write_text(json.dumps({"status": main_status, "candidate_profile": "mixed-native-surfaces",
            "tested_code_version": COMMIT, "execution_complete": True,
            "strict_reproduction": {"checked": 138, "passed": 137, "failed_files": ["surf/lh.white"]},
            "host": "private-host"}))
        metric = {"matched_regions": 68, "mae": .02, "maximum_absolute_error": .12,
            "p90_absolute_relative_error_percent": 2., "missing_in_candidate": [], "extra_in_candidate": [],
            "zero_reference_regions": [], "worst_regions_by_relative_error": ["lh/private-label"],
            "per_region": {"lh/private-label": {"reference": 2., "candidate": 2.12,
                "absolute_error": .12, "signed_relative_difference_percent": 6.}}}
        distance = {"mean_mm": .1, "p99_mm": .2, "max_mm": .3}
        reports = {
            "strict": {"checked": 138, "passed": 137, "files": {"surf/lh.white": {"pass": False}}},
            "dice": {"files": {"aseg.mgz": {"different_voxels": 3, "minimum_dice": 0., "p05_dice": .9,
                "median_dice": .99, "worst_labels": ["77"], "per_label": {"77": {"name": "private-label-name",
                    "reference_voxels": 0, "candidate_voxels": 1, "intersection_voxels": 0, "dice": 0.}}}}},
            "surface": {"stages": {"lh": {"white": {"ordered_faces_equal": False,
                "candidate_to_reference_triangle": distance, "reference_to_candidate_triangle": distance}}}},
            "geometry": {"common_conform_and_tkras": True, "surfaces": {"lh.white": {
                "surface_ras_mm_comparable": True, "vertex_correspondence": False}}},
            "region": {"aparc_68": {"ThickAvg": metric}, "aseg": metric, "wmparc": metric},
            "all_regions": {"status": "measured", "atlases": {"aparc": {name: metric for name in MODULE.REGION_FIELDS}}},
            "no_th3": {"atlases": {"aparc": metric}, "reference_stats_modes": {"command": "/private-user/tool"}},
            "local": {"maps": {"surf/lh.thickness": {"status": "not_assessed_vertex_correspondence"}}, "annotations": {}},
            "quality_candidate": {"status": "measured", "hemispheres": {"lh": {"topology": {"euler": 2, "valid": True},
                "white_pial_crossings": {"status": "measured", "proper_transverse_pairs": 3,
                                          "white_faces_examined": 100, "white_faces_total": 100}}}},
            "quality_reference": {"status": "measured", "hemispheres": {}}}
        records = {}
        for role, value in reports.items():
            filename = ("quality_candidate/report.json" if role == "quality_candidate" else
                        "quality_official/report.json" if role == "quality_reference" else role + "_candidate_vs_official.json")
            path = root / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
            records[role] = {"path": str(path), "sha256": MODULE.sha(path)}
        binding = root / "binding.json"
        binding.write_text(json.dumps({"schema": "fnit-evaluation-summary-bindings-v1",
            "evaluation_report": {"path": str(main), "sha256": MODULE.sha(main)},
            "comparison_label": "candidate_vs_official", "reports": records}))
        return main, binding, records

    def test_completed_bound_evaluation_extracts_existing_numbers_without_promoting_equivalence(self):
        with tempfile.TemporaryDirectory() as directory:
            main, binding, _ = self._evaluation_fixture(Path(directory))
            record = MODULE.summarize({"evaluation": [main], "evaluation_bindings": [binding]})["reports"]["evaluation"][0]
            output = record["summary"]
            self.assertEqual(output["subreport_binding"]["status"], "complete")
            self.assertEqual(output["strict_reproduction"]["status"], "difference_detected")
            metrics = output["numeric_error_metrics"]
            self.assertEqual(metrics["segmentation_dice"]["maps"]["aseg.mgz"]["reported_worst_labels"][0]["dice"], 0.)
            self.assertEqual(metrics["surface_distances"]["final_white_pial_directions_with_complete_statistics"], 2)
            self.assertEqual(metrics["surface_distances"]["expected_surfaces"], 20)
            self.assertEqual(metrics["surface_distances"]["reported_surfaces"], 1)
            self.assertEqual(metrics["all_cortical_statistics"]["reported_atlases"], 1)
            self.assertEqual(metrics["common_no_th3_statistics"]["aparc"]["mae"], .02)
            self.assertEqual(metrics["quality_candidate"]["hemispheres"]["lh"]["white_pial_crossings"]["proper_transverse_pairs"], 3)
            self.assertEqual(metrics["local_comparison_coverage"]["maps"]["status_counts"]["not_assessed_vertex_correspondence"], 1)
            self.assertEqual(output["overall_metric_equivalence"], "not_assessed")
            for private in (directory, "private-host", "private-label", "private-label-name", "private-user"):
                self.assertNotIn(private, json.dumps(record))

    def test_failed_main_does_not_read_even_valid_bound_subreports(self):
        with tempfile.TemporaryDirectory() as directory:
            main, binding, _ = self._evaluation_fixture(Path(directory), main_status="failed")
            output = MODULE.summarize({"evaluation": [main], "evaluation_bindings": [binding]})["reports"]["evaluation"][0]["summary"]
            self.assertEqual(output["subreport_binding"]["status"], "main_not_complete")
            self.assertEqual(output["numeric_error_metrics"], "not_assessed")
            self.assertEqual(output["execution_status"], "failed")

    def test_changed_child_is_excluded_and_missing_binding_never_reuses_old_values(self):
        with tempfile.TemporaryDirectory() as directory:
            main, binding, records = self._evaluation_fixture(Path(directory))
            Path(records["surface"]["path"]).write_text('{"stages":{}}')
            output = MODULE.summarize({"evaluation": [main], "evaluation_bindings": [binding]})["reports"]["evaluation"][0]["summary"]
            self.assertEqual(output["subreport_binding"]["status"], "partial_or_failed_binding")
            self.assertEqual(output["subreport_binding"]["subreport_binding_status"]["surface"], "hash_mismatch")
            self.assertNotIn("surface_distances", output["numeric_error_metrics"])
            unbound = MODULE.summarize({"evaluation": [main]})["reports"]["evaluation"][0]["summary"]
            self.assertEqual(unbound["numeric_error_metrics"], "not_assessed")


if __name__ == "__main__":
    unittest.main()
