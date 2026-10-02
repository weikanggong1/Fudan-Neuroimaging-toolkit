"""Stdlib two-origin binding tests; byte fixtures never model MRI performance."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import benchmark_connectome_anatomy_prep as prep
from tools import benchmark_connectome_raw_cohort as cohort
from tools import benchmark_connectome_staged_gpu as staged


_spec = importlib.util.spec_from_file_location(
    "multi_origin_staged_fixture", Path(__file__).with_name("test_staged_gpu_binding_tool.py"))
_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixtures)


class MultiOriginFixture(_fixtures.StagedFixture):
    def setUp(self):
        super().setUp()
        self.second_root = self.root / "second_fresh_official_anatomy"
        self.second_root.mkdir()
        self.second_config = copy.deepcopy(self.prep_config)
        self.second_config.update(run_root=str(self.second_root), cpu_jobs=8,
                                  selected_cases=[case["case_id"] for case in self.cases[2:]],
                                  full_manifest_case_count=10)
        self.second_config_path = self.second_root / "anatomy_prep_config.json"
        self.second_manifest_path = self.second_root / "input_manifest.json"
        cohort.atomic_json(self.second_config_path, self.second_config)
        self.second_manifest_path.write_bytes(self.prep_manifest_path.read_bytes())
        self.second_case = self.cases[2]
        with patch.object(cohort, "worker", side_effect=self.fake_official_worker):
            self.second_prepared = prep.worker({"action": "recon", "config": self.second_config,
                                               "case": self.second_case, "version": "candidate"})
        self.assertEqual(self.second_prepared["status"], "completed")
        self.second_job = self.second_root / "candidate" / self.second_case["case_id"]
        self.second_subject = cohort.recon_command(self.second_config, self.second_case, self.second_job)[2]
        self.second_report = self.second_job / "anatomy_prep_report.json"
        self.second_driver = self.root / "second_preparation_driver"
        self.second_driver.mkdir()
        self.second_state = copy.deepcopy(self.prep_state)
        record = copy.deepcopy(next(iter(self.prep_state["cases"].values())))
        record.update(case_id=self.second_case["case_id"], subject=self.second_case["subject"],
                      preparation_report=self.second_prepared,
                      preparation_worker_wall_seconds=self.second_prepared["preparation_worker_wall_seconds"])
        self.second_state.update(config=self.second_config, requested_cases=8,
            fresh_namespace={"status": "claimed_fresh_anatomy_namespace", "path": str(self.second_root)},
            cases={"candidate/" + self.second_case["case_id"]: record})
        cohort.atomic_json(self.second_driver / "status.json", self.second_state)
        self.declarations = [
            {"prep_config": str(self.prep_config_path), "prep_driver_report_dir": str(self.prep_driver),
             "case_ids": [case["case_id"] for case in self.cases[:2]]},
            {"prep_config": str(self.second_config_path), "prep_driver_report_dir": str(self.second_driver),
             "case_ids": [case["case_id"] for case in self.cases[2:]]}]
        self.declaration_path = self.root / "two_original_preparation_bindings.json"
        cohort.atomic_json(self.declaration_path, {"bindings": self.declarations})
        self.multi_reports = self.root / "new_two_origin_gpu_driver"
        self.multi_options = SimpleNamespace(prep_bindings=self.declaration_path, prep_config=None,
            prep_driver_report_dir=None, gpu_config=self.gpu_config_path,
            run_root=self.root / "new_two_origin_gpu_outputs", report_dir=self.multi_reports,
            preflight_only=True, timeout_hours=1., poll_seconds=.001)
        for path in (self.second_config_path, self.second_manifest_path, self.second_report,
                     self.second_job / "recon_report.json", self.prep_driver / "status.json",
                     self.second_driver / "status.json", self.declaration_path):
            self.frozen_original_bytes[path] = path.read_bytes()

    def read_origins(self, declarations=None):
        return staged.read_preparation_origins(
            self.declarations if declarations is None else declarations, self.multi_reports)

    def bound_configuration(self):
        config, manifest, cases, raw_records = staged.derive_config(self.multi_options)
        self.multi_reports.mkdir()
        for record in raw_records:
            Path(record["path"]).write_bytes(record["bytes"])
        return config, manifest, cases

    def second_checked(self, config):
        selected = staged.for_case(config, self.second_case)
        return {"status": "actual_fresh_preparation_revalidated",
            "prep_config_sha256": selected["preparation_config"]["sha256"],
            "preparation_report": copy.deepcopy(self.second_prepared),
            "preparation_report_binding": {"path": str(self.second_report), "sha256": cohort.sha256(self.second_report)},
            "anatomy_subject_dir": str(self.second_subject),
            "anatomy_geometry": {"status": "actual_images_surfaces_annotations_read"}}


class OriginManifestAndMappingTests(MultiOriginFixture):
    def test_two_origins_cover_ten_cases_with_exact_original_bytes_and_no_config_merge(self):
        origins, mapping, manifest, cases, records = self.read_origins()
        self.assertEqual(len(origins), 2)
        self.assertEqual(manifest, self.manifest)
        self.assertEqual(cases, self.cases)
        self.assertEqual(mapping, {case["case_id"]: int(index >= 2) for index, case in enumerate(self.cases)})
        self.assertEqual(records[0]["bytes"], self.prep_config_path.read_bytes())
        self.assertEqual(records[1]["bytes"], self.second_config_path.read_bytes())
        self.assertNotEqual(records[0]["bytes"], records[1]["bytes"])
        self.assertFalse(self.multi_reports.exists())
        self.assert_original_bytes_unchanged()

    def test_case_map_must_be_complete_unique_and_known(self):
        changes = []
        missing = copy.deepcopy(self.declarations)
        missing[1]["case_ids"].pop()
        changes.append(missing)
        duplicate = copy.deepcopy(self.declarations)
        duplicate[1]["case_ids"].append(self.cases[0]["case_id"])
        changes.append(duplicate)
        unknown = copy.deepcopy(self.declarations)
        unknown[1]["case_ids"][-1] = "unknown-case"
        changes.append(unknown)
        repeated = copy.deepcopy(self.declarations)
        repeated[0]["case_ids"].append(repeated[0]["case_ids"][0])
        changes.append(repeated)
        empty = copy.deepcopy(self.declarations)
        empty[1]["case_ids"] = []
        changes.append(empty)
        for declared in changes:
            with self.subTest(declared=declared), self.assertRaises(ValueError):
                self.read_origins(declared)

    def test_origin_cannot_declare_case_outside_its_actual_selected_preparation_plan(self):
        changed = copy.deepcopy(self.declarations)
        changed[0]["case_ids"][0], changed[1]["case_ids"][0] = changed[1]["case_ids"][0], changed[0]["case_ids"][0]
        with self.assertRaises(ValueError):
            self.read_origins(changed)

    def test_duplicate_original_path_and_forged_binding_fields_are_rejected(self):
        for change in ({"prep_config": str(self.prep_config_path)}, {"candidate_source": "pretend"}):
            declared = copy.deepcopy(self.declarations)
            declared[1].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.read_origins(declared)

    def test_full_manifests_require_same_bytes_even_if_json_is_semantically_identical(self):
        self.second_manifest_path.write_bytes(self.second_manifest_path.read_bytes() + b"\n")
        with self.assertRaises(ValueError):
            self.read_origins()

    def test_changed_raw_identity_or_incomplete_manifest_is_rejected(self):
        for change in ("hash", "subject", "count"):
            manifest = copy.deepcopy(self.manifest)
            if change == "hash":
                manifest["cases"][2]["input_files"][0]["sha256"] = "0" * 64
            elif change == "subject":
                manifest["cases"][2]["subject"] = manifest["cases"][0]["subject"]
            else:
                manifest["cases"].pop()
            cohort.atomic_json(self.second_manifest_path, manifest)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.read_origins()

    def test_official_identity_atlas_and_future_scientific_parameters_must_match(self):
        for change in ("official", "atlas", "future"):
            config = copy.deepcopy(self.second_config)
            if change == "official":
                config["official_origin"]["identity"]["version"] += " different-version"
            elif change == "atlas":
                config["atlases"] = ["fs-aparc-a2009s"]
            else:
                config["future_gpu_parameters"]["n_seeds"] += 1
            cohort.atomic_json(self.second_config_path, config)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.read_origins()

    def test_every_origin_must_remain_unknown_source_and_gpu_not_started(self):
        for change in ({"sources": {}}, {"frozen_sources": {}}, {"gpu_started": True},
                       {"candidate_source": "candidate"}, {"cpu_threads": 16}):
            config = copy.deepcopy(self.second_config)
            config.update(change)
            cohort.atomic_json(self.second_config_path, config)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.read_origins()


class OriginSelectionAndValidationTests(MultiOriginFixture):
    def test_case_b_selects_its_config_and_driver_without_mutating_global_or_originals(self):
        config, _, _ = self.bound_configuration()
        before = copy.deepcopy(config)
        selected = staged.for_case(config, self.second_case)
        self.assertEqual(selected["preparation_config"]["original_path"], str(self.second_config_path))
        self.assertEqual(selected["preparation_driver_status_path"], str(self.second_driver / "status.json"))
        self.assertEqual(staged.original_prep_config(selected), self.second_config)
        self.assertEqual(staged.original_prep_config(config, self.second_case), self.second_config)
        self.assertEqual(config, before)
        self.assert_original_bytes_unchanged()

    def test_invalid_case_origin_indices_and_membership_are_rejected(self):
        config, _, _ = self.bound_configuration()
        for value in (None, True, -1, 2, 0):
            changed = copy.deepcopy(config)
            changed["case_origin_index"][self.second_case["case_id"]] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                staged.for_case(changed, self.second_case)

    def test_case_b_clean_child_receives_its_actual_original_binding_not_origin_a(self):
        config, _, _ = self.bound_configuration()
        checked = self.second_checked(config)
        with patch.object(staged.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, json.dumps(checked), "")) as child:
            self.assertEqual(staged.verify_preparation(config, self.second_case), checked)
        call = child.call_args
        payload = json.loads(call.kwargs["input"])
        self.assertEqual(payload["binding"]["original_path"], str(self.second_config_path))
        self.assertEqual(payload["binding"]["sha256"], cohort.sha256(self.second_config_path))
        self.assertEqual(payload["case"], self.second_case)
        self.assertNotIn("PYTHONPATH", call.kwargs["env"])
        self.assertEqual(call.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
        self.assert_original_bytes_unchanged()

    def test_original_b_runtime_is_really_checked_in_clean_child(self):
        config, _, _ = self.bound_configuration()
        checked = staged.verify_preparation(staged.for_case(config, self.second_case))
        self.assertEqual(checked["status"], "original_preparation_runtime_verified")
        self.assertEqual(checked["prep_config_sha256"], cohort.sha256(self.second_config_path))
        self.assert_original_bytes_unchanged()

    def test_worker_preflight_really_checks_both_original_runtimes_without_gpu_dispatch(self):
        config, _, _ = self.bound_configuration()
        with patch.object(staged.rerun, "build_resources", return_value={"files": []}) as resources, \
                patch.object(cohort, "worker") as gpu:
            result = staged.worker({"config": config, "action": "preflight"})
        resources.assert_called_once_with(config, source_version="candidate")
        gpu.assert_not_called()
        self.assertEqual([item["prep_config_sha256"] for item in result["original_preparation"]],
                         [cohort.sha256(self.prep_config_path), cohort.sha256(self.second_config_path)])
        self.assertTrue(all(item["status"] == "original_preparation_runtime_verified"
                            for item in result["original_preparation"]))
        self.assert_original_bytes_unchanged()

    def test_case_b_driver_record_is_read_from_b_and_incomplete_b_cannot_use_completed_a(self):
        config, _, _ = self.bound_configuration()
        record, observation = staged.preparation_driver_case(config, self.second_case)
        self.assertEqual(record["preparation_report"], self.second_prepared)
        self.assertEqual(observation["path"], str(self.second_driver / "status.json"))
        changed = copy.deepcopy(self.second_state)
        changed["cases"]["candidate/" + self.second_case["case_id"]]["status"] = "cpu_running"
        cohort.atomic_json(self.second_driver / "status.json", changed)
        with self.assertRaises(ValueError):
            staged.preparation_driver_case(config, self.second_case)

    def test_changed_original_b_config_or_its_snapshot_is_rejected(self):
        config, _, _ = self.bound_configuration()
        selected = staged.for_case(config, self.second_case)
        for path in (self.second_config_path, Path(selected["preparation_config"]["snapshot_path"])):
            contents = path.read_bytes()
            path.write_bytes(contents + b"\n")
            with self.subTest(path=path), self.assertRaises(ValueError):
                staged.original_prep_config(config, self.second_case)
            path.write_bytes(contents)

    def test_mapping_declaration_bytes_and_both_derived_fields_are_frozen(self):
        config, _, _ = self.bound_configuration()
        staged.verify_source(config)
        changed = copy.deepcopy(config)
        first, second = self.cases[0]["case_id"], self.second_case["case_id"]
        changed["case_origin_index"][first], changed["case_origin_index"][second] = 1, 0
        changed["preparation_origins"][0]["case_ids"][0] = second
        changed["preparation_origins"][1]["case_ids"][0] = first
        with self.assertRaises(ValueError):
            staged.verify_source(changed)
        self.declaration_path.write_bytes(self.declaration_path.read_bytes() + b"\n")
        with self.assertRaises(ValueError):
            staged.verify_source(config)

    def test_binding_and_loading_b_keep_b_subject_config_and_report_hash(self):
        config, _, _ = self.bound_configuration()
        checked = self.second_checked(config)
        job = Path(config["run_root"]) / "candidate" / self.second_case["case_id"]
        with patch.object(staged, "verify_preparation", return_value=checked):
            binding = staged.bind_anatomy(config, self.second_case, "candidate", job)
            loaded = staged.load_anatomy(config, self.second_case, "candidate", job)
        self.assertEqual(binding["prep_config_sha256"], cohort.sha256(self.second_config_path))
        self.assertEqual(binding["original_preparation"]["anatomy_subject_dir"], str(self.second_subject))
        self.assertEqual(loaded["staged_anatomy"]["prepared_anatomy_subject_dir"], str(self.second_subject))
        self.assertFalse((job / "freesurfer").exists())
        self.assertFalse((job / "connectome").exists())
        self.assert_original_bytes_unchanged()

    def test_new_gpu_outputs_and_driver_must_be_fresh_and_separate_from_b(self):
        for key, path in (("run_root", self.second_root / "nested"),
                          ("report_dir", self.second_root / "nested_driver"),
                          ("run_root", self.second_driver / "nested"),
                          ("report_dir", self.second_driver / "nested_driver")):
            original = getattr(self.multi_options, key)
            setattr(self.multi_options, key, path)
            with self.subTest(key=key, path=path), self.assertRaises(ValueError):
                staged.derive_config(self.multi_options)
            setattr(self.multi_options, key, original)

    def test_preflight_only_writes_each_original_byte_snapshot_and_starts_no_gpu(self):
        calls = []
        def remote(config, action, case, log_path):
            calls.append(action)
            return {"resources": {"files": []}, "original_preparation": []}
        with patch.object(staged, "remote", side_effect=remote):
            self.assertEqual(staged.run(self.multi_options), 0)
        self.assertEqual(calls, ["preflight"])
        self.assertFalse(self.multi_options.run_root.exists())
        state = json.loads((self.multi_reports / "status.json").read_text())
        self.assertEqual(state["status"], "preflight_only_GPU_not_started")
        self.assertFalse(state["continuous_cold_pipeline"])
        self.assertEqual((self.multi_reports / "original_prep_config.0.bytes.json").read_bytes(), self.prep_config_path.read_bytes())
        self.assertEqual((self.multi_reports / "original_prep_config.1.bytes.json").read_bytes(), self.second_config_path.read_bytes())
        self.assertEqual((self.multi_reports / "original_prep_bindings.bytes.json").read_bytes(), self.declaration_path.read_bytes())
        self.assertEqual((self.multi_reports / "prep_driver_observation_at_bind.0.bytes.json").read_bytes(),
                         (self.prep_driver / "status.json").read_bytes())
        self.assertEqual((self.multi_reports / "prep_driver_observation_at_bind.1.bytes.json").read_bytes(),
                         (self.second_driver / "status.json").read_bytes())
        self.assert_original_bytes_unchanged()


if __name__ == "__main__":
    unittest.main()
