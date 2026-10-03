"""Stdlib staged-binding protocol tests; byte fixtures are not MRI benchmarks."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from tools import benchmark_connectome_raw_cohort as cohort
from tools import benchmark_connectome_anatomy_prep as prep
from tools import benchmark_connectome_raw_rerun as rerun
from tools import benchmark_connectome_staged_gpu as staged


fixture_spec = importlib.util.spec_from_file_location(
    "staged_preparation_fixture", Path(__file__).with_name("test_raw_anatomy_prep_tool.py")
)
fixture_module = importlib.util.module_from_spec(fixture_spec)
fixture_spec.loader.exec_module(fixture_module)


class StagedFixture(fixture_module.PreparationFixture):
    """Freeze actual original helper bytes apart from later candidate source."""

    def setUp(self):
        super().setUp()
        self.prep_root = Path(self.config["run_root"])
        self.prep_root.mkdir()
        self.original_helpers = self.root / "frozen_preparation_helpers"
        self.original_helpers.mkdir()
        original_cohort = self.original_helpers / Path(cohort.__file__).name
        original_prep = self.original_helpers / Path(prep.__file__).name
        original_cohort.write_bytes(Path(cohort.__file__).read_bytes())
        original_prep.write_bytes(Path(prep.__file__).read_bytes())
        self.config.update(worker_script=str(original_cohort), worker_script_sha256=cohort.sha256(original_cohort),
                           anatomy_prep_script=str(original_prep), anatomy_prep_script_sha256=cohort.sha256(original_prep))
        self.config["future_gpu_parameters"].update(
            gpu_python=sys.executable, gpu_cpu_threads=8, seed=0, eddy_gp_seed=12345,
            device="cuda:0", gpu_uuid="GPU-protocol-fixture", cuda_visible_devices="1", gpu_path_prefix=[],
            gpu_lock="/tmp/fnit-recon-five-20261002-gongwk.gpu.lock")
        self.config["frozen_runtime_files"] = prep.freeze_runtime_files(self.config)
        self.prep_config = copy.deepcopy(self.config)
        self.prep_config_path = self.prep_root / "anatomy_prep_config.json"
        self.prep_manifest_path = self.prep_root / "input_manifest.json"
        cohort.atomic_json(self.prep_config_path, self.prep_config)
        cohort.atomic_json(self.prep_manifest_path, self.manifest)
        self.case = self.cases[0]
        self.prepared_result = self.prepare()
        self.assertEqual(self.prepared_result["status"], "completed")
        self.prep_job = self.prep_root / "candidate" / self.case["case_id"]
        self.prep_report_path = self.prep_job / "anatomy_prep_report.json"
        self.recon_report_path = self.prep_job / "recon_report.json"
        self.subject = cohort.recon_command(self.prep_config, self.case, self.prep_job)[2]
        self.prep_driver = self.root / "staged_preparation_driver"
        self.prep_driver.mkdir()
        self.prep_state = {
            "scope": prep.SCOPE, "status": "running", "candidate_source": "unknown", "gpu_started": False,
            "gpu_status": "GPU_not_started", "config": self.prep_config, "requested_cases": 10,
            "fresh_namespace": {"status": "claimed_fresh_anatomy_namespace", "path": str(self.prep_root)},
            "cases": {"candidate/" + self.case["case_id"]: {
                "status": "completed", "version": "candidate", "case_id": self.case["case_id"], "subject": self.case["subject"],
                "start_utc": "2026-10-03T00:00:00+00:00", "end_utc": "2026-10-03T01:00:00+00:00",
                "timing": {"head_case_wall_seconds": 3600., "recon_command_seconds": .000001,
                           "cpu_driver_queue_seconds": 3.}, "preparation_report": self.prepared_result,
                "preparation_worker_wall_seconds": self.prepared_result["preparation_worker_wall_seconds"],
            }},
        }
        cohort.atomic_json(self.prep_driver / "status.json", self.prep_state)
        self.candidate_source = self.root / "later_frozen_candidate_source"
        (self.candidate_source / "src/fnit").mkdir(parents=True)
        (self.candidate_source / "src/fnit/cli.py").write_text(
            "import argparse\np = argparse.ArgumentParser()\n"
            "p.add_argument('--compile-arc')\np.add_argument('--tracking-batch-size')\n"
        )
        for relative in ("src/fnit/flirt/core.py", "src/fnit/connectome/pipeline.py", "src/fnit/weights.py",
                         "src/fnit/connectome/atlas_manifest.json", "pyproject.toml", "environment.yml"):
            path = self.candidate_source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# Candidate source identity fixture; never imported.\n")
        self.wall_script = self.root / "common_wall_evaluator.py"
        self.wall_script.write_bytes(b"Common wall byte fixture; never executed.")
        self.candidate_manifest = cohort.source_manifest(self.candidate_source)
        self.resources_path = self.root / "staged_resources.json"
        cohort.atomic_json(self.resources_path, {"files": []})
        self.frozen_original_bytes = {path: path.read_bytes() for path in
                                      (self.prep_config_path, self.prep_manifest_path,
                                       self.prep_report_path, self.recon_report_path, original_cohort, original_prep)}
        self.stage_reports = self.root / "new_staged_binding_driver"
        self.stage_reports.mkdir()
        self.prep_snapshot = self.stage_reports / "original_prep_config.bytes.json"
        self.prep_snapshot.write_bytes(self.prep_config_path.read_bytes())
        self.gpu_config_path = self.root / "ready_candidate_gpu_config.json"
        self.gpu_supplied = {
            **copy.deepcopy(self.prep_config["future_gpu_parameters"]), "atlases": self.prep_config["atlases"],
            "sources": {"candidate": str(self.candidate_source)},
            "frozen_sources": {"candidate": self.candidate_manifest}, "candidate_ready": True,
            "cuda_alloc_conf": staged.ALLOCATOR, "fnit_weights": str(self.root / "declared_gpu_weights"),
            "wall_script": str(self.wall_script), "wall_script_sha256": cohort.sha256(self.wall_script),
            "candidate_cli_arguments": ["--tracking-batch-size", "8192", "--compile-arc"],
        }
        cohort.atomic_json(self.gpu_config_path, self.gpu_supplied)
        self.stage_config = {
            **copy.deepcopy(self.gpu_supplied), "staged_mode": staged.MODE,
            "run_root": str(self.root / "new_staged_gpu_outputs"), "pilot": False, "cpu_threads": 8,
            "cpu_python": self.prep_config["cpu_python"],
            "anatomy_validation_python": self.prep_config["anatomy_validation_python"],
            "preparation_config": {"original_path": str(self.prep_config_path), "snapshot_path": str(self.prep_snapshot),
                                   "sha256": cohort.sha256(self.prep_config_path)},
            "preparation_driver_status_path": str(self.prep_driver / "status.json"),
            "worker_script": cohort.__file__, "worker_script_sha256": cohort.sha256(cohort.__file__),
            "staged_worker_script": staged.__file__, "staged_worker_sha256": cohort.sha256(staged.__file__),
            "resource_helper_script": rerun.__file__, "resource_helper_sha256": cohort.sha256(rerun.__file__),
            "gpu_configuration": {"path": str(self.gpu_config_path), "sha256": cohort.sha256(self.gpu_config_path)},
            "input_manifest": {"path": str(self.prep_manifest_path), "sha256": cohort.sha256(self.prep_manifest_path)},
            "resources_manifest": {"path": str(self.resources_path), "sha256": cohort.sha256(self.resources_path)},
            "stop_dispatch_path": str(self.stage_reports / "STOP_DISPATCH"),
        }
        self.gpu_job = Path(self.stage_config["run_root"]) / "candidate" / self.case["case_id"]

    def assert_original_bytes_unchanged(self):
        for path, contents in self.frozen_original_bytes.items():
            self.assertEqual(path.read_bytes(), contents, str(path))

    def checked_case(self):
        """Mock success protocol only; never claim these bytes are MRI arrays."""
        return {
            "status": "actual_fresh_preparation_revalidated",
            "prep_config_sha256": self.stage_config["preparation_config"]["sha256"],
            "preparation_report": copy.deepcopy(self.prepared_result),
            "preparation_report_binding": {"path": str(self.prep_report_path), "sha256": cohort.sha256(self.prep_report_path)},
            "anatomy_subject_dir": str(self.subject), "anatomy_geometry": {"status": "actual_images_surfaces_annotations_read"},
        }

    def bind(self):
        with patch.object(staged, "verify_preparation", return_value=self.checked_case()):
            return staged.bind_anatomy(self.stage_config, self.case, "candidate", self.gpu_job)

    def load(self):
        with patch.object(staged, "verify_preparation", return_value=self.checked_case()):
            return staged.load_anatomy(self.stage_config, self.case, "candidate", self.gpu_job)


class OriginalPreparationBindingTests(StagedFixture):
    def test_original_config_remains_unknown_source_and_bytes_unchanged(self):
        original = staged.original_prep_config(self.stage_config)
        self.assertEqual(original, self.prep_config)
        self.assertEqual(original["candidate_source"], "unknown")
        self.assertNotIn("sources", original)
        self.assertNotIn("frozen_sources", original)
        self.assert_original_bytes_unchanged()

    def test_modified_original_or_snapshot_even_only_whitespace_is_rejected(self):
        for path in (self.prep_config_path, self.prep_snapshot):
            contents = path.read_bytes()
            path.write_bytes(contents + b"\n")
            with self.subTest(path=path), self.assertRaises(ValueError):
                staged.original_prep_config(self.stage_config)
            path.write_bytes(contents)

    def test_original_config_cannot_pretend_that_candidate_or_gpu_was_known(self):
        for change in ({"candidate_source": "later_candidate"}, {"sources": {"candidate": "/pretend"}},
                       {"frozen_sources": {}}, {"gpu_started": True}):
            changed = copy.deepcopy(self.prep_config)
            changed.update(change)
            cohort.atomic_json(self.prep_config_path, changed)
            self.prep_snapshot.write_bytes(self.prep_config_path.read_bytes())
            self.stage_config["preparation_config"]["sha256"] = cohort.sha256(self.prep_config_path)
            with self.subTest(change=change), self.assertRaises(ValueError):
                staged.original_prep_config(self.stage_config)

    def test_real_clean_child_verifies_only_original_frozen_runtime(self):
        with patch.dict(staged.os.environ, {"PYTHONPATH": str(self.candidate_source / "src"), "CUDA_VISIBLE_DEVICES": "7"}):
            checked = staged.verify_preparation(self.stage_config)
        self.assertEqual(checked["status"], "original_preparation_runtime_verified")
        self.assertEqual(checked["prep_config_sha256"], cohort.sha256(self.prep_config_path))
        self.assert_original_bytes_unchanged()

    def test_clean_child_detects_changed_original_helper(self):
        helper = Path(self.prep_config["worker_script"])
        helper.write_bytes(helper.read_bytes() + b"\n# changed original helper\n")
        with self.assertRaisesRegex(RuntimeError, "original preparation validation failed"):
            staged.verify_preparation(self.stage_config)

    def test_case_success_protocol_uses_original_python_config_and_clean_environment(self):
        checked = self.checked_case()
        with patch.dict(staged.os.environ, {"PYTHONPATH": "/unapproved/candidate", "CUDA_VISIBLE_DEVICES": "7"}), \
                patch.object(staged.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, json.dumps(checked), "")) as run:
            returned = staged.verify_preparation(self.stage_config, self.case)
        self.assertEqual(returned, checked)
        command = run.call_args.args[0]
        self.assertEqual(command[0], self.prep_config["anatomy_validation_python"])
        self.assertEqual(command[1], "-c")
        self.assertIn("prep.verify_runtime_files(c)", command[2])
        self.assertIn("prep.validate_preparation_result(c, case, prepared)", command[2])
        payload = json.loads(run.call_args.kwargs["input"])
        self.assertEqual(payload["binding"], self.stage_config["preparation_config"])
        self.assertEqual(payload["case"], self.case)
        self.assertNotIn("PYTHONPATH", run.call_args.kwargs["env"])
        self.assertEqual(run.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")

    def test_child_nonzero_or_invalid_identity_status_never_authorizes_case(self):
        checks = [subprocess.CompletedProcess([], 1, "", "real anatomy read failed")]
        for changed in ({"status": "names_only"}, {"prep_config_sha256": "f" * 64},
                        {"anatomy_geometry": {"status": "names_only"}}):
            checked = self.checked_case()
            checked.update(changed)
            checks.append(subprocess.CompletedProcess([], 0, json.dumps(checked), ""))
        for response in checks:
            with self.subTest(response=response), patch.object(staged.subprocess, "run", return_value=response), \
                    self.assertRaises((RuntimeError, ValueError)):
                staged.verify_preparation(self.stage_config, self.case)

    def test_raw_input_change_fails_original_validator_before_mri_array_read(self):
        raw_dwi = next(item for item in self.case["input_files"] if item["kind"] == "raw_dwi")
        Path(raw_dwi["path"]).write_bytes(b"Raw DWI changed after preparation.")
        with self.assertRaisesRegex(RuntimeError, "raw input hash mismatch"):
            staged.verify_preparation(self.stage_config, self.case)


class CandidateFreezeTests(StagedFixture):
    def test_full_candidate_ledger_and_scheduling_flags_are_accepted(self):
        self.assertEqual(staged.verify_source(self.stage_config), self.candidate_manifest)
        staged.validate_gpu_configuration(self.prep_config, self.gpu_supplied)
        self.assert_original_bytes_unchanged()

    def test_candidate_cannot_be_labeled_baseline_or_have_an_additional_source(self):
        for names in (("baseline",), ("candidate", "baseline")):
            config = copy.deepcopy(self.stage_config)
            config["sources"] = {name: str(self.candidate_source) for name in names}
            config["frozen_sources"] = {name: self.candidate_manifest for name in names}
            with self.subTest(names=names), self.assertRaises(ValueError):
                staged.verify_source(config)

    def test_not_ready_mode_allocator_and_lock_changes_are_rejected(self):
        for field, value in (("candidate_ready", False), ("staged_mode", "other"), ("cuda_alloc_conf", ""),
                             ("gpu_lock", "/tmp/different.lock")):
            config = copy.deepcopy(self.stage_config)
            config[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                staged.verify_source(config)

    def test_changed_scientific_source_or_full_ledger_metadata_is_rejected(self):
        for key, value in (("directory", "/other/source"), ("git_commit", "different_commit"), ("git_status", "dirty")):
            config = copy.deepcopy(self.stage_config)
            config["frozen_sources"]["candidate"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                staged.verify_source(config)
        (self.candidate_source / "src/fnit/connectome/pipeline.py").write_text("# altered scientific source\n")
        with self.assertRaises(ValueError):
            staged.verify_source(self.stage_config)

    def test_missing_required_source_file_is_rejected_even_with_refrozen_fingerprint(self):
        (self.candidate_source / "environment.yml").unlink()
        self.stage_config["frozen_sources"]["candidate"] = cohort.source_manifest(self.candidate_source)
        with self.assertRaises(ValueError):
            staged.verify_source(self.stage_config)

    def test_gpu_configuration_and_manifest_bytes_must_stay_frozen(self):
        for path in (self.gpu_config_path, self.prep_manifest_path):
            contents = path.read_bytes()
            path.write_bytes(contents + b"\n")
            with self.subTest(path=path), self.assertRaises(ValueError):
                staged.verify_source(self.stage_config)
            path.write_bytes(contents)

    def test_only_exact_scheduling_flags_are_allowed(self):
        self.assertEqual(staged.extra_arguments(["--compile-arc", "--tracking-batch-size", "1024"]),
                         ["--compile-arc", "--tracking-batch-size", "1024"])
        for arguments in (["--overwrite"], ["--skip-topup"], ["--corrected-dwi", "/old/dwi"], ["--float16"],
                          ["--tracking-batch-size", "0"], ["--tracking-batch-size", "1.5"],
                          ["--compile-arc", "--compile-arc"], ["--tracking-batch-size"]):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                staged.extra_arguments(arguments)

    def test_candidate_scheduling_option_must_exist_in_actual_frozen_cli(self):
        (self.candidate_source / "src/fnit/cli.py").write_text("import argparse\np=argparse.ArgumentParser()\n")
        self.stage_config["frozen_sources"]["candidate"] = cohort.source_manifest(self.candidate_source)
        with self.assertRaises(ValueError):
            staged.verify_source(self.stage_config)

    def test_shared_scientific_settings_and_resume_fields_cannot_be_changed(self):
        for field, value in (("n_seeds", 1), ("seed", 1), ("atlases", ["fs-aparc-a2009s"]),
                             ("skip_eddy", True), ("resume", True), ("run_root", "/old/output")):
            supplied = copy.deepcopy(self.gpu_supplied)
            supplied[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                staged.validate_gpu_configuration(self.prep_config, supplied)


class NewGPUAnatomyBindingTests(StagedFixture):
    def test_binding_writes_separate_new_file_without_reconstruction_or_input_copy(self):
        binding = self.bind()
        self.assertEqual(binding["status"], "completed")
        self.assertFalse(binding["raw_dwi_preprocessing_resumed"])
        self.assertFalse(binding["raw_dwi_outputs_copied"])
        self.assertEqual(binding["candidate_source_fingerprint"], self.candidate_manifest["source_fingerprint"])
        self.assertEqual(sorted(path.name for path in self.gpu_job.iterdir()), ["staged_anatomy_binding.json"])
        self.assert_original_bytes_unchanged()

    def test_loading_uses_exact_original_fresh_subject_and_later_candidate_binding(self):
        self.bind()
        loaded = self.load()
        self.assertEqual(loaded["status"], "completed")
        self.assertEqual(Path(loaded["staged_anatomy"]["prepared_anatomy_subject_dir"]), self.subject)
        self.assertEqual(loaded["staged_anatomy"]["candidate_source_fingerprint"], self.candidate_manifest["source_fingerprint"])
        self.assertFalse(loaded["staged_anatomy"]["recon_all_reused_from_other_round"])
        self.assert_original_bytes_unchanged()

    def test_existing_even_empty_gpu_directory_is_not_resumed(self):
        self.gpu_job.mkdir(parents=True)
        with self.assertRaises((FileExistsError, ValueError)):
            self.bind()
        self.assertFalse((self.gpu_job / "staged_anatomy_binding.json").exists())

    def test_existing_gpu_partial_or_foreign_reconstruction_is_not_resumed(self):
        for name in ("gpu_report.json", "raw_bids_wall.json", "connectome", "freesurfer"):
            self.gpu_job.mkdir(parents=True)
            path = self.gpu_job / name
            path.mkdir() if name in ("connectome", "freesurfer") else path.touch()
            with self.subTest(name=name), self.assertRaises((FileExistsError, ValueError)):
                self.bind()
            path.rmdir() if path.is_dir() else path.unlink()
            self.gpu_job.rmdir()

    def test_wrong_version_or_case_namespace_is_rejected(self):
        for version, job in (("baseline", self.gpu_job), ("candidate", self.prep_job),
                             ("candidate", self.gpu_job.parent / "OTHER")):
            with self.subTest(version=version, job=job), self.assertRaises(ValueError):
                staged.assert_new_job(self.stage_config, self.case, version, job)

    def test_driver_incomplete_case_or_timing_error_never_authorizes_binding(self):
        key = "candidate/" + self.case["case_id"]
        for change in ({"status": "running"}, {"status": "failed"}, {"timing_error": {"message": "invalid"}},
                       {"validation_error": {"message": "invalid"}}, {"timing": None}):
            state = copy.deepcopy(self.prep_state)
            state["cases"][key].update(change)
            cohort.atomic_json(self.prep_driver / "status.json", state)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.bind()
            self.assertFalse(self.gpu_job.exists())

    def test_changed_driver_case_after_binding_is_rejected(self):
        self.bind()
        state = copy.deepcopy(self.prep_state)
        state["cases"]["candidate/" + self.case["case_id"]]["end_utc"] = "2026-10-03T01:01:00+00:00"
        cohort.atomic_json(self.prep_driver / "status.json", state)
        with self.assertRaises(ValueError):
            self.load()

    def test_direct_load_rejects_candidate_source_change_after_binding(self):
        self.bind()
        (self.candidate_source / "src/fnit/connectome/pipeline.py").write_text("# source changed after binding\n")
        with self.assertRaises(ValueError):
            self.load()

    def test_binding_cannot_claim_another_source_or_resumed_dwi(self):
        original = self.bind()
        path = self.gpu_job / "staged_anatomy_binding.json"
        for change in ({"candidate_source_fingerprint": "f" * 64}, {"raw_dwi_preprocessing_resumed": True},
                       {"prep_config_sha256": "f" * 64}, {"case_id": "OTHER"}):
            altered = copy.deepcopy(original)
            altered.update(change)
            cohort.atomic_json(path, altered)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.load()


class WorkerCallbackAndStopTests(StagedFixture):
    def test_default_cli_subject_is_unchanged_and_explicit_subject_only_changes_that_argument(self):
        config = {**self.prep_config, **self.prep_config["future_gpu_parameters"]}
        default = cohort.cli_command(config, self.case, self.prep_job)
        subject_index = default.index("--freesurfer-subject-dir") + 1
        self.assertEqual(default[subject_index], str(self.subject))
        custom_subject = self.root / "explicit_verified_prepared_subject"
        explicit = cohort.cli_command(config, self.case, self.prep_job, anatomy_subject=custom_subject)
        expected = default.copy()
        expected[subject_index] = str(custom_subject)
        self.assertEqual(explicit, expected)

    def test_normal_worker_keeps_default_loader_and_explicit_loader_is_only_an_override(self):
        config = copy.deepcopy(self.stage_config)
        payload = {"action": "gpu", "config": config, "case": self.case, "version": "candidate"}
        with patch.object(cohort, "load_recon_for_gpu", side_effect=RuntimeError("ordinary loader gate")) as normal, \
                patch.object(cohort.subprocess, "run") as run:
            result = cohort.worker(payload)
        normal.assert_called_once()
        run.assert_not_called()
        self.assertEqual(result["error"]["message"], "ordinary loader gate")
        config["run_root"] = str(self.root / "another_explicit_callback_output")
        with patch.object(cohort, "load_recon_for_gpu") as normal, \
                patch.object(cohort.subprocess, "run") as run:
            def explicit_loader(*_):
                raise RuntimeError("explicit staged loader gate")
            result = cohort.worker(payload, anatomy_loader=explicit_loader)
        normal.assert_not_called()
        run.assert_not_called()
        self.assertEqual(result["error"]["message"], "explicit staged loader gate")

    def test_stopped_gpu_dispatch_never_calls_the_gpu_worker(self):
        Path(self.stage_config["stop_dispatch_path"]).touch()
        with patch.object(rerun, "verify_resources", return_value={}), patch.object(cohort, "worker") as run:
            result = staged.worker({"action": "gpu", "config": self.stage_config, "case": self.case, "version": "candidate"})
        self.assertEqual(result, {"status": "dispatch_paused", "gpu_started": False})
        run.assert_not_called()

    def test_stop_gate_is_rechecked_in_subject_callback_after_lock_wait(self):
        Path(self.stage_config["stop_dispatch_path"]).touch()
        with patch.object(rerun, "verify_resources") as resources, self.assertRaisesRegex(RuntimeError, "STOP_DISPATCH"):
            staged.anatomy_subject(self.stage_config, self.case, self.gpu_job)
        resources.assert_not_called()

    def test_staged_worker_refuses_previous_gpu_attempt_before_calling_cohort(self):
        self.bind()
        for name in ("gpu_report.json", "staged_eligibility.json", "raw_bids_wall.json", "raw_bids_wall.log", "connectome"):
            path = self.gpu_job / name
            path.mkdir() if name == "connectome" else path.write_bytes(b"Preserve old GPU attempt.")
            with self.subTest(name=name), patch.object(rerun, "verify_resources", return_value={}), \
                    patch.object(cohort, "worker") as run, self.assertRaises(FileExistsError):
                staged.worker({"action": "gpu", "config": self.stage_config, "case": self.case, "version": "candidate"})
            run.assert_not_called()
            if path.is_dir():
                path.rmdir()
            else:
                self.assertEqual(path.read_bytes(), b"Preserve old GPU attempt.")
                path.unlink()

    def test_staged_worker_supplies_only_explicit_anatomy_and_scheduling_callbacks(self):
        self.bind()
        failed = {"status": "failed", "error": {"type": "RuntimeError", "message": "protocol fixture, no GPU run"}}
        with patch.object(rerun, "verify_resources", return_value={}), \
                patch.object(cohort, "worker", return_value=failed) as run:
            result = staged.worker({"action": "gpu", "config": self.stage_config, "case": self.case, "version": "candidate"})
        self.assertEqual(result["gpu_result"], failed)
        self.assertEqual(run.call_args.kwargs["anatomy_loader"], staged.load_anatomy)
        self.assertEqual(run.call_args.kwargs["anatomy_subject"], staged.anatomy_subject)
        self.assertEqual(run.call_args.kwargs["extra_cli_arguments"], self.stage_config["candidate_cli_arguments"])
        self.assert_original_bytes_unchanged()


class StrictMemoryEligibilityTests(unittest.TestCase):
    @staticmethod
    def budget():
        return {"status": "observed_below_budget", "measurements": {
            "process_tree": 12_000_000_000, "allocated_bytes": 10_000_000_000, "reserved_bytes": 15_000_000_000}}

    def test_three_distinct_measurement_groups_and_supported_peak_aliases_are_eligible(self):
        self.assertTrue(staged.memory_eligible(self.budget()))
        budget = self.budget()
        budget["measurements"] = {"process_tree": 12_000_000_000,
                                  "peak_allocated_bytes": 10_000_000_000, "max_memory_reserved_bytes": 15_000_000_000}
        self.assertTrue(staged.memory_eligible(budget))

    def test_duplicate_allocation_aliases_do_not_substitute_for_a_reserved_measurement(self):
        budget = self.budget()
        del budget["measurements"]["reserved_bytes"]
        budget["measurements"]["peak_allocated_bytes"] = 10_000_000_000
        self.assertFalse(staged.memory_eligible(budget))

    def test_every_group_requires_a_finite_non_boolean_value_strictly_below_twenty_gb(self):
        for group in ("process_tree", "allocated_bytes", "reserved_bytes"):
            for value in (None, True, -1, float("nan"), float("inf"), 20_000_000_000, 20_000_000_001):
                budget = self.budget()
                budget["measurements"][group] = value
                with self.subTest(group=group, value=value):
                    self.assertFalse(staged.memory_eligible(budget))
        budget = self.budget()
        budget["status"] = "not_fully_measured"
        self.assertFalse(staged.memory_eligible(budget))

    def test_monitor_issues_prevent_eligibility_even_with_three_complete_measurements(self):
        budget = self.budget()
        budget["monitor_issues"] = ["process tree sampler failed"]
        self.assertFalse(staged.memory_eligible(budget))

    def test_another_peak_alias_cannot_hide_an_over_budget_or_invalid_measurement(self):
        for alias in ("peak_allocated_bytes", "peak_reserved_bytes"):
            for value in (True, -1, float("nan"), float("inf"), 20_000_000_000, 20_000_000_001):
                budget = self.budget()
                budget["measurements"][alias] = value
                with self.subTest(alias=alias, value=value):
                    self.assertFalse(staged.memory_eligible(budget))


class StagedTimingTests(unittest.TestCase):
    @staticmethod
    def prepared():
        return {"start_utc": "2026-10-03T00:00:00+00:00", "end_utc": "2026-10-03T01:00:00+00:00",
                "preparation_report": {"recon_command_seconds": 3500.}, "preparation_worker_wall_seconds": 3550.,
                "timing": {"head_case_wall_seconds": 3600., "cpu_driver_queue_seconds": 3.}}

    @staticmethod
    def gpu():
        return {"status": "completed", "gpu_lock_queue_seconds": 20., "raw_dwi_cli_total_runtime_seconds": 95.}

    def test_stage_walls_and_observed_gap_are_separate_from_cold_pipeline(self):
        prepared = self.prepared()
        original = copy.deepcopy(prepared)
        timing = staged.staged_timing(prepared, self.gpu(), "2026-10-03T03:00:00+00:00", "2026-10-03T03:02:00+00:00", 120.)
        self.assertEqual(timing["preparation_head_wall_seconds"], 3600.)
        self.assertEqual(timing["raw_dwi_head_wall_seconds"], 120.)
        self.assertEqual(timing["raw_dwi_head_wall_excluding_gpu_queue_seconds"], 100.)
        self.assertEqual(timing["prep_to_gpu_observed_utc_gap_seconds"], 7200.)
        self.assertEqual(timing["staged_observed_utc_elapsed_seconds"], 10920.)
        self.assertFalse(timing["continuous_cold_pipeline"])
        self.assertEqual(prepared, original)

    def test_invalid_node_timers_and_head_queue_are_rejected(self):
        for change in ({"preparation_worker_wall_seconds": 3400.}, {"preparation_worker_wall_seconds": 3601.}):
            prepared = self.prepared()
            prepared.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                staged.staged_timing(prepared, self.gpu(), "2026-10-03T03:00:00+00:00", "2026-10-03T03:02:00+00:00", 120.)
        for wall in (-1., float("nan"), 19.):
            with self.subTest(wall=wall), self.assertRaises(ValueError):
                staged.staged_timing(self.prepared(), self.gpu(), "2026-10-03T03:00:00+00:00", "2026-10-03T03:02:00+00:00", wall)

    def test_timing_error_cannot_overwrite_actual_gpu_result(self):
        gpu = {"status": "failed", "error": {"type": "RuntimeError", "message": "real raw-DWI failure"}}
        response = {"gpu_result": gpu, "eligibility": {"status": "not_eligible"}}
        record = {"anatomy_binding": {"preparation_driver_record": self.prepared()}}
        with patch.object(staged, "staged_timing", side_effect=ValueError("independent timer error")):
            staged.collect_gpu_result(record, response, "2026-10-03T03:00:00+00:00", "2026-10-03T03:02:00+00:00", 120.)
        self.assertEqual(record["gpu_result"], gpu)
        self.assertEqual(record["gpu_result"]["error"], gpu["error"])
        self.assertEqual(record["error"], gpu["error"])
        self.assertEqual(record["status"], "failed_gpu_execution")
        self.assertEqual(record["timing_error"]["message"], "independent timer error")


if __name__ == "__main__":
    unittest.main()
