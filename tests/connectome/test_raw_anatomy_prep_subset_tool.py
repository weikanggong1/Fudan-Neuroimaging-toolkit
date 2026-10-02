"""Stdlib subset dispatch protocol tests; no MRI/GPU performance simulation."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import threading
import unittest
from unittest.mock import patch

from tools import benchmark_connectome_anatomy_prep as prep
from tools import benchmark_connectome_raw_cohort as cohort


_spec = importlib.util.spec_from_file_location(
    "subset_preparation_fixture", Path(__file__).with_name("test_raw_anatomy_prep_tool.py"))
_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixtures)


class SubsetFixture(_fixtures.PreparationFixture):
    def setUp(self):
        super().setUp()
        self.prior_root = Path(self.config["run_root"])
        self.prior_root.mkdir()
        self.prior_config_path = self.prior_root / "anatomy_prep_config.json"
        cohort.atomic_json(self.prior_config_path, self.config)
        cohort.atomic_json(self.prior_root / "input_manifest.json", self.manifest)
        self.prior_driver = self.root / "prior_stopped_preparation_driver"
        self.prior_driver.mkdir()
        self.stop = self.prior_driver / "STOP_DISPATCH"
        self.stop.write_bytes(b"Explicitly stop new preparation dispatch.\n")
        # Directories alone establish started jobs; they are never treated as complete.
        self.started = self.cases[:2]
        for case in self.started:
            (self.prior_root / "candidate" / case["case_id"]).mkdir(parents=True)
        self.selected = [case["case_id"] for case in self.cases[2:]]
        self.subset_options = copy.copy(self.options)
        self.subset_options.run_root = self.root / "new_subset_anatomy"
        self.subset_options.report_dir = self.root / "new_subset_driver"
        self.subset_options.cpu_jobs = 8
        self.subset_options.selected_cases = self.selected
        self.subset_options.avoid_preparations = [(self.prior_config_path, self.prior_driver)]
        self.original_bytes = {path: path.read_bytes() for path in
                               (self.prior_config_path, self.prior_root / "input_manifest.json", self.stop)}

    def derive(self):
        return prep.derive_config(self.subset_options)

    def assert_prior_unchanged(self):
        for path, raw in self.original_bytes.items():
            self.assertEqual(path.read_bytes(), raw, str(path))

    def remote_fixture(self, config, action, case, log_path):
        if action == "preflight":
            return {"status": "verified_official_cpu_environment", "gpu_started": False}
        return prep.worker({"action": action, "config": config, "case": case, "version": "candidate"})


class SubsetConfigurationTests(SubsetFixture):
    def test_avoid_prior_preparation_without_explicit_selection_is_rejected(self):
        self.subset_options.selected_cases = None
        self.subset_options.cpu_jobs = 2
        with self.assertRaises(ValueError):
            self.derive()

    def test_selected_eight_keep_canonical_ten_case_manifest_and_unknown_candidate(self):
        config, manifest, cases = self.derive()
        self.assertEqual(manifest, self.manifest)
        self.assertEqual(len(manifest["cases"]), 10)
        self.assertEqual([case["case_id"] for case in cases], self.selected)
        self.assertEqual(config["selected_cases"], self.selected)
        self.assertEqual(config["full_manifest_case_count"], 10)
        self.assertEqual(config["candidate_source"], "unknown")
        self.assertFalse(config["gpu_started"])
        self.assertNotIn("sources", config)
        self.assertNotIn("frozen_sources", config)
        self.assertEqual(config["official_origin"]["identity"], self.config["official_origin"]["identity"])
        self.assert_prior_unchanged()

    def test_selection_is_explicit_nonempty_unique_and_known(self):
        for selected in ([], [self.selected[0], self.selected[0]], ["unknown-case"]):
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                prep.select_cases(self.cases, selected)
        self.assertEqual(prep.select_cases(self.cases, None), self.cases)
        self.assertEqual(prep.select_cases(self.cases, self.selected[::-1]), self.cases[2:][::-1])

    def test_selection_config_rejects_non_list_duplicate_and_non_string_entries(self):
        config, _, _ = self.derive()
        for selected in ([], "C02", ["C02", "C02"], [None]):
            changed = copy.deepcopy(config)
            changed["selected_cases"] = selected
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                prep.validate_config(changed)

    def test_eight_jobs_require_explicit_selection_and_cpu_threads_remain_eight(self):
        config, _, _ = self.derive()
        for jobs in range(1, 9):
            changed = copy.deepcopy(config)
            changed["cpu_jobs"] = jobs
            prep.validate_config(changed)
        for key, value in (("cpu_jobs", 0), ("cpu_jobs", 9), ("cpu_jobs", True), ("cpu_threads", 16)):
            changed = copy.deepcopy(config)
            changed[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                prep.validate_config(changed)
        changed = copy.deepcopy(config)
        changed.pop("selected_cases")
        with self.assertRaises(ValueError):
            prep.validate_config(changed)

    def test_existing_started_case_in_prior_root_is_never_reconstructed_again(self):
        self.subset_options.selected_cases = [self.started[0]["case_id"]]
        with self.assertRaises(FileExistsError):
            self.derive()
        self.assertFalse(self.subset_options.run_root.exists())
        self.assert_prior_unchanged()

    def test_dangling_prior_case_symlink_also_means_started(self):
        job = self.prior_root / "candidate" / self.selected[0]
        job.symlink_to(self.root / "missing_foreign_subject")
        with self.assertRaises(FileExistsError):
            self.derive()

    def test_prior_stop_must_exist_and_keep_original_bytes(self):
        config, _, _ = self.derive()
        self.stop.write_bytes(b"Changed stop declaration.\n")
        with self.assertRaises(ValueError):
            prep.verify_prior_preparations(config)
        self.stop.unlink()
        with self.assertRaises(ValueError):
            prep.verify_prior_preparations(config)

    def test_prior_config_and_full_manifest_are_hash_bound(self):
        config, _, _ = self.derive()
        self.prior_config_path.write_bytes(self.prior_config_path.read_bytes() + b"\n")
        with self.assertRaises(ValueError):
            prep.verify_prior_preparations(config)
        self.prior_config_path.write_bytes(self.original_bytes[self.prior_config_path])
        manifest_path = self.prior_root / "input_manifest.json"
        manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")
        with self.assertRaises(ValueError):
            prep.verify_prior_preparations(config)

    def test_prior_config_must_describe_source_independent_official_preparation(self):
        for change in ({"gpu_started": True}, {"candidate_source": "candidate"}, {"cpu_threads": 16}):
            changed = copy.deepcopy(self.config)
            changed.update(change)
            cohort.atomic_json(self.prior_config_path, changed)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.derive()

    def test_new_output_and_driver_cannot_overlap_prior_output_or_driver(self):
        for key, path in (("run_root", self.prior_root / "nested"),
                          ("report_dir", self.prior_root / "nested_driver"),
                          ("run_root", self.prior_driver / "nested"),
                          ("report_dir", self.prior_driver / "nested_driver")):
            original = getattr(self.subset_options, key)
            setattr(self.subset_options, key, path)
            with self.subTest(key=key, path=path), self.assertRaises(ValueError):
                self.derive()
            setattr(self.subset_options, key, original)


class SubsetWorkerAndDriverTests(SubsetFixture):
    def test_worker_rejects_non_selected_case_before_creating_directory_or_running_official_tool(self):
        config, _, _ = self.derive()
        with patch.object(cohort, "worker") as official, self.assertRaises(ValueError):
            prep.worker({"action": "recon", "config": config, "case": self.started[0], "version": "candidate"})
        official.assert_not_called()
        self.assertFalse(Path(config["run_root"]).exists())
        with self.assertRaises(ValueError):
            prep.validate_preparation_result(config, self.started[0], {})

    def test_prior_started_state_is_rechecked_before_every_worker(self):
        config, _, cases = self.derive()
        (self.prior_root / "candidate" / cases[0]["case_id"]).mkdir()
        with patch.object(cohort, "worker") as official, self.assertRaises(FileExistsError):
            prep.worker({"action": "recon", "config": config, "case": cases[0], "version": "candidate"})
        official.assert_not_called()

    def test_only_selected_eight_are_dispatched_and_reported_without_reusing_prior_subjects(self):
        calls = []
        lock = threading.Lock()
        def remote(config, action, case, log_path):
            if action == "recon":
                with lock:
                    calls.append(case["case_id"])
            return self.remote_fixture(config, action, case, log_path)
        with patch.object(prep, "remote", side_effect=remote), \
                patch.object(cohort, "worker", side_effect=self.fake_official_worker), \
                patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(prep.run(self.subset_options), 0)
        state = json.loads((self.subset_options.report_dir / "status.json").read_text())
        self.assertEqual(sorted(calls), sorted(self.selected))
        self.assertEqual(state["requested_cases"], 8)
        self.assertEqual(state["completed_cases"], 8)
        self.assertEqual(state["status"], "completed_anatomy_preparation")
        self.assertFalse(state["gpu_started"])
        self.assertFalse(state["full_pipeline_benchmark"])
        self.assertFalse(state["comparison_ready"])
        manifest = json.loads((self.subset_options.run_root / "input_manifest.json").read_text())
        self.assertEqual(manifest, self.manifest)
        self.assertEqual(len(state["cases"]), 8)
        for case in self.started:
            self.assertFalse((self.subset_options.run_root / "candidate" / case["case_id"]).exists())
        self.assert_prior_unchanged()


if __name__ == "__main__":
    unittest.main()
