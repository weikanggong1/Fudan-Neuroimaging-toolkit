"""Stdlib Task05 reader contracts; byte fixtures are not MRI/GPU benchmarks."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

_spec = importlib.util.spec_from_file_location(
    "selected_actual_origin_fixture", Path(__file__).with_name("test_actual_gpu_origin_bindings.py"))
_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixtures)
compare, origins = _fixtures.compare, _fixtures.origins


class SelectedOriginFixture(unittest.TestCase):
    def setUp(self):
        fixture = _fixtures.ActualOriginTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        for name in ("root", "options", "cases", "manifest", "source", "original_config",
                     "original_GPU", "original_wall", "subject_dir", "new_root", "preflight", "path"):
            setattr(self, name, getattr(fixture, name))
        self.write, self.identity = fixture.write, fixture.identity
        self.case = self.cases[0]
        self.job = self.new_root / "candidate" / self.case["case_id"]
        self.job.mkdir(parents=True)
        # Task05 never writes a manifest at the replacement run root.
        (self.new_root / "input_manifest.json").unlink()
        (self.new_root / "staged_gpu_config.json").unlink()
        self.write(self.options.candidate_root / "input_manifest.json", self.manifest)
        science_worker = self.root / "frozen_original_science_worker.py"
        science_worker.write_bytes(b"Frozen original helper identity fixture; never executed.\n")
        recovery_worker = self.root / "selected_recovery_worker.py"
        recovery_worker.write_bytes(b"Selected recovery helper identity fixture; never executed.\n")
        resource = self.root / "original_scientific_resource.bytes"
        resource.write_bytes(b"Scientific resource identity fixture; never loaded.\n")
        scientific = {"role": "scientific_resource", "path": str(resource),
                      "sha256": compare.anatomy.sha(resource), "size_bytes": resource.stat().st_size}
        self.old_resources = {"files": [scientific, {"role": "gpu_python", "path": sys.executable,
            "sha256": compare.anatomy.sha(sys.executable), "size_bytes": Path(sys.executable).stat().st_size}]}
        old_resource_path = self.write(self.options.candidate_root / "resources.json", self.old_resources)
        self.original_config.update(worker_script=str(science_worker), worker_script_sha256=compare.anatomy.sha(science_worker),
            resources_manifest=self.identity(old_resource_path), stop_dispatch_path=str(self.root / "old-driver/STOP_DISPATCH"))
        self.original_config_path = Path(fixture.declaration["original"]["configuration"]["path"])
        self.write(self.original_config_path, self.original_config)
        original_driver = {"config": self.original_config, "status": "failed_or_incomplete_staged_raw_cohort",
            "end_utc": "2026-10-03T00:00:00+00:00", "dispatch_paused": True,
            "cases": {"candidate/" + self.case["case_id"]: {"status": "failed_gpu_eligibility", "gpu_report": self.original_GPU}}}
        self.original_driver_path = self.write(self.options.candidate_driver, original_driver)
        self.original_snapshot_path = Path(fixture.declaration["original"]["driver_snapshot"]["path"])
        self.original_snapshot_path.write_bytes(self.original_driver_path.read_bytes())
        self.isolated_python = self.root / "isolated_monitor_python"
        self.isolated_python.symlink_to(Path(sys.executable).resolve())
        self.preflight["gpu_python"] = str(self.isolated_python)
        self.preflight["new_runtime"]["gpu_python"] = str(self.isolated_python)
        self.preflight_path = Path(fixture.declaration["replacement"]["runtime_preflight"]["path"])
        self.write(self.preflight_path, self.preflight)
        self.resources = {"files": [copy.deepcopy(scientific), {"role": "gpu_python", "path": str(self.isolated_python),
            "sha256": compare.anatomy.sha(self.isolated_python), "size_bytes": self.isolated_python.stat().st_size}]}
        resource_path = self.write(self.job / "recovery_resources.json", self.resources)
        self.new_config = copy.deepcopy(self.original_config)
        self.new_config.update(run_root=str(self.new_root), gpu_python=str(self.isolated_python),
            stop_dispatch_path=str(self.root / "selected-driver/STOP_DISPATCH"), resources_manifest=self.identity(resource_path))
        self.config_path = self.write(self.job / "recovery_config.json", self.new_config)
        self.proof = {"mode": origins.SELECTED_MODE, "arm": "candidate", "case_id": self.case["case_id"], "case": self.case,
            "original_config": self.identity(self.original_config_path), "original_driver": self.identity(self.original_driver_path),
            "old_case_reports": {name: {"binding": fixture.declaration["original"][key], "value": copy.deepcopy(value)}
                for name, key, value in (("gpu_report.json", "GPU_report", self.original_GPU),
                                         ("raw_bids_wall.json", "wall_report", self.original_wall))},
            "FS_recomputed": False, "old_DWI_outputs_used": False, "raw_DWI_execution_policy": "complete_from_raw",
            "anatomy_subject_dir": str(self.subject_dir), "same_round_anatomy": {"fixture_only": True},
            "resources": self.resources, "runtime": {"scientific_runtime_equal": True, "CUDA_initialized": False,
                "original": copy.deepcopy(self.preflight["original_runtime"]), "isolated": copy.deepcopy(self.preflight["new_runtime"])}}
        self.proof_path = self.write(self.job / "recovery_binding.json", self.proof)
        self.GPU = copy.deepcopy(self.original_GPU)
        self.GPU["identity"]["python"] = str(self.isolated_python)
        self.GPU["gpu_memory"]["process"].update(backend="pynvml", status="measured", errors=[], failed_samples=0,
            samples=10, sample_interval_seconds=.5, max_observed_interval_seconds=.5)
        self.GPU["memory_budget"].update(status="observed_below_budget", monitor_issues=[])
        self.GPU.update(worker_wall_seconds=50., gpu_command_wall_seconds=40., gpu_lock_queue_seconds=2.,
                        wall_report=str(self.job / "raw_bids_wall.json"))
        self.wall = copy.deepcopy(self.original_wall)
        self.wall["total_runtime_seconds"] = 30.
        self.wall["cli_arguments"][-1] = str(self.job / "connectome")
        self.wall["inputs"]["prepared/dwi"]["path"] = "new-actual-output"
        self.wall["selected_inputs"].update(dwi=str(self.job / "connectome/preproc/eddy/data.nii.gz"),
            bvecs=str(self.job / "connectome/preproc/eddy/data.eddy_rotated_bvecs"))
        self.gpu_path = self.write(self.job / "gpu_report.json", self.GPU)
        self.wall_path = self.write(self.job / "raw_bids_wall.json", self.wall)
        self.eligibility = {"status": "execution_complete_memory_observed_below_budget", "execution_status": "completed",
            "validation_error": None, "memory_budget": self.GPU["memory_budget"], "full_ten_complete": False}
        self.eligibility_path = self.write(self.job / "recovery_eligibility.json", self.eligibility)
        self.response = {"binding": self.proof, "new_config": self.new_config, "gpu_result": self.GPU,
            "configuration": self.identity(self.config_path), "eligibility": self.eligibility, "job_root": str(self.job),
            "science_worker": self.identity(science_worker), "recovery_worker": self.identity(recovery_worker),
            "resources_manifest": self.identity(resource_path), "GPU_report": self.identity(self.gpu_path),
            "wall_report": self.identity(self.wall_path)}
        self.record = {"case_id": self.case["case_id"], "version": "candidate", "status": "completed",
                       "response": self.response, "recovery_head_wall_seconds": 1.}
        self.state = {"mode": origins.SELECTED_MODE, "status": "completed_selected_subset",
            "selected_attempted": 1, "selected_completed": 1, "full_ten_complete": False,
            "cases": {"candidate/" + self.case["case_id"]: self.record}}
        self.driver_path = self.write(self.root / "selected-driver/status.json", self.state)
        self.declaration = copy.deepcopy(fixture.declaration)
        self.declaration["original"].update(configuration=self.identity(self.original_config_path),
                                             driver_snapshot=self.identity(self.original_snapshot_path))
        self.declaration["replacement"].update(root=str(self.job), driver_status=self.identity(self.driver_path),
            configuration=self.identity(self.config_path), runtime_preflight=self.identity(self.preflight_path))
        for key in ("science_worker", "recovery_worker", "resources_manifest"):
            self.declaration[key] = self.response[key]
        self.persist_declaration()

    def persist_declaration(self):
        self.write(self.path, {"schema_version": 1, "scope": "explicit_actual_GPU_monitor_recovery", "bindings": [self.declaration]})

    def load(self):
        return origins.load_bindings(self.path, self.cases, self.options)[0]["candidate", self.case["case_id"]]

    def persist_driver(self):
        self.write(self.driver_path, self.state)
        self.declaration["replacement"]["driver_status"] = self.identity(self.driver_path)
        self.persist_declaration()

    def persist_gpu(self):
        self.write(self.gpu_path, self.GPU)
        self.write(self.wall_path, self.wall)
        self.response["GPU_report"] = self.identity(self.gpu_path)
        self.response["wall_report"] = self.identity(self.wall_path)
        self.persist_driver()


class SelectedDeclarationTests(SelectedOriginFixture):
    def test_exact_task05_extras_normalize_case_job_and_driver_binding_without_mutation(self):
        before = copy.deepcopy(self.declaration)
        normalized, selected = origins.normalize_selected_declaration(self.declaration)
        self.assertEqual(self.declaration, before)
        self.assertEqual(normalized["replacement"]["root"], str(self.new_root))
        self.assertEqual(normalized["replacement"]["driver_status"], str(self.driver_path))
        self.assertEqual(set(normalized), {"arm", "case_id", "reason", "original", "replacement"})
        self.assertEqual(selected["driver_identity"], self.identity(self.driver_path))

    def test_partial_or_unknown_extra_fields_are_rejected(self):
        for field in ("science_worker", "recovery_worker", "resources_manifest"):
            declaration = copy.deepcopy(self.declaration)
            declaration.pop(field)
            with self.subTest(field=field), self.assertRaises(ValueError):
                origins.normalize_selected_declaration(declaration)
        declaration = {**self.declaration, "undocumented": True}
        with self.assertRaises(ValueError):
            origins.normalize_selected_declaration(declaration)

    def test_bound_driver_and_workers_reject_byte_changes_or_wrong_sha(self):
        for field in ("science_worker", "recovery_worker", "resources_manifest"):
            declaration = copy.deepcopy(self.declaration)
            declaration[field]["sha256"] = "0" * 64
            with self.subTest(field=field), self.assertRaises(ValueError):
                origins.normalize_selected_declaration(declaration)
        self.driver_path.write_bytes(self.driver_path.read_bytes() + b"\n")
        with self.assertRaises(ValueError):
            origins.normalize_selected_declaration(self.declaration)

    def test_running_incomplete_or_failed_selected_driver_does_not_qualify(self):
        for status in ("preflighting", "running", "dispatch_paused", "selected_recovery_failed_or_ineligible"):
            self.state["status"] = status
            self.persist_driver()
            with self.subTest(status=status), self.assertRaises(ValueError):
                origins.normalize_selected_declaration(self.declaration)
        self.state.update(status="completed_selected_subset", selected_completed=0)
        self.persist_driver()
        with self.assertRaises(ValueError):
            origins.normalize_selected_declaration(self.declaration)

    def test_case_job_and_configuration_path_must_match_declared_run_root(self):
        for root in (str(self.new_root), str(self.new_root / "candidate/wrong-case")):
            declaration = copy.deepcopy(self.declaration)
            declaration["replacement"]["root"] = root
            with self.subTest(root=root), self.assertRaises(ValueError):
                origins.normalize_selected_declaration(declaration)
        config_copy = self.write(self.root / "foreign/recovery_config.json", self.new_config)
        declaration = copy.deepcopy(self.declaration)
        declaration["replacement"]["configuration"] = self.identity(config_copy)
        with self.assertRaises(ValueError):
            origins.normalize_selected_declaration(declaration)


class SelectedProofAndRecordTests(SelectedOriginFixture):
    def test_four_metadata_changes_load_from_original_manifest_without_new_manifest(self):
        self.assertFalse((self.new_root / "input_manifest.json").exists())
        binding = self.load()
        self.assertEqual(binding["replacement_configuration"], self.new_config)
        self.assertEqual(origins.selected_origin(self.options, "candidate", self.case["case_id"],
            {("candidate", self.case["case_id"]): binding})[:2], (self.new_root, self.driver_path))
        self.assertFalse((self.new_root / "input_manifest.json").exists())

    def test_any_additional_scientific_change_or_changed_original_raw_manifest_is_rejected(self):
        for key, value in (("n_seeds", 50000), ("gpu_cpu_threads", 16), ("undocumented_science", True)):
            config = {**self.new_config, key: value}
            self.write(self.config_path, config)
            self.declaration["replacement"]["configuration"] = self.identity(self.config_path)
            self.persist_declaration()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.load()
        self.write(self.config_path, self.new_config)
        self.declaration["replacement"]["configuration"] = self.identity(self.config_path)
        self.persist_declaration()
        manifest = copy.deepcopy(self.manifest)
        manifest["cases"][0]["input_files"][0]["sha256"] = "1" * 64
        self.write(self.options.candidate_root / "input_manifest.json", manifest)
        with self.assertRaises(ValueError):
            self.load()

    def test_original_driver_copy_may_have_different_path_but_requires_identical_bytes(self):
        binding = self.load()
        proof, _, _ = origins.selected_proof(binding, self.case)
        self.assertNotEqual(proof["original_driver"]["path"], binding["declaration"]["original"]["driver_snapshot"]["path"])
        self.assertEqual(proof["original_driver"]["sha256"], binding["declaration"]["original"]["driver_snapshot"]["sha256"])
        self.original_driver_path.write_bytes(self.original_driver_path.read_bytes() + b"\n")
        with self.assertRaises(ValueError):
            origins.selected_proof(binding, self.case)

    def test_driver_nested_response_must_match_actual_proof_configuration_and_eligibility(self):
        binding = self.load()
        config, timing, GPU, response = origins.selected_driver_record(binding, self.record, self.state, self.case)
        self.assertEqual(config, self.new_config)
        self.assertEqual(GPU, self.GPU)
        self.assertEqual(response, self.response)
        self.assertEqual(timing["recovery_head_wall_seconds"], 1.)
        self.assertNotIn("parent_full_wall_seconds", timing)
        for field in ("binding", "new_config", "eligibility"):
            record = copy.deepcopy(self.record)
            record["response"][field] = {"forged": True}
            with self.subTest(field=field), self.assertRaises(ValueError):
                origins.selected_driver_record(binding, record, self.state, self.case)
        self.write(self.eligibility_path, {**self.eligibility, "validation_error": {"type": "HashError"}})
        with self.assertRaises(ValueError):
            origins.selected_driver_record(binding, self.record, self.state, self.case)

    def test_actual_proof_policy_case_and_non_monitor_resource_changes_are_rejected(self):
        binding = self.load()
        for field, value in (("old_DWI_outputs_used", True), ("FS_recomputed", True), ("case", {})):
            self.write(self.proof_path, {**self.proof, field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                origins.selected_proof(binding, self.case)
        proof = copy.deepcopy(self.proof)
        proof["resources"]["files"][0]["sha256"] = "0" * 64
        self.write(self.proof_path, proof)
        with self.assertRaises(ValueError):
            origins.selected_proof(binding, self.case)

    def test_completed_case_requires_positive_finite_non_boolean_independent_head_time(self):
        binding = self.load()
        for value in (None, True, 0., -1., float("nan"), float("inf")):
            record = copy.deepcopy(self.record)
            record["recovery_head_wall_seconds"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                origins.selected_driver_record(binding, record, self.state, self.case)
        record = copy.deepcopy(self.record)
        record["status"] = "failed_or_ineligible"
        with self.assertRaises(ValueError):
            origins.selected_driver_record(binding, record, self.state, self.case)

    def test_actual_eligibility_cannot_claim_failed_execution_or_full_ten_completion(self):
        for change in ({"status": "not_eligible"}, {"validation_error": {"type": "HashError"}},
                       {"full_ten_complete": True}):
            eligibility = {**self.eligibility, **change}
            self.write(self.eligibility_path, eligibility)
            self.response["eligibility"] = eligibility
            self.persist_driver()
            binding = self.load()
            with self.subTest(change=change), self.assertRaises(ValueError):
                origins.selected_driver_record(binding, self.record, self.state, self.case)

    def test_same_gpu_timers_are_checked_without_comparing_independent_head_timer(self):
        binding = self.load()
        result = origins.verify_replacement(binding, self.GPU, self.wall, self.case, self.subject_dir)
        self.assertEqual(result["status"], "actual_eligible_replacement_verified")
        # The independent head duration is intentionally smaller in this schema fixture.
        self.assertLess(self.record["recovery_head_wall_seconds"], self.GPU["worker_wall_seconds"])
        for key, value in (("gpu_command_wall_seconds", 20.), ("worker_wall_seconds", 35.),
                           ("gpu_lock_queue_seconds", 51.), ("worker_wall_seconds", float("nan"))):
            original = self.GPU[key]
            self.GPU[key] = value
            self.persist_gpu()
            binding = self.load()
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                origins.verify_replacement(binding, self.GPU, self.wall, self.case, self.subject_dir)
            self.GPU[key] = original

    def test_readers_leave_all_original_and_recovery_artifacts_byte_identical(self):
        before = {str(path): compare.anatomy.sha(path) for path in self.root.rglob("*") if path.is_file()}
        binding = self.load()
        origins.normalize_selected_declaration(self.declaration)
        origins.selected_proof(binding, self.case)
        origins.selected_driver_record(binding, self.record, self.state, self.case)
        origins.verify_replacement(binding, self.GPU, self.wall, self.case, self.subject_dir)
        after = {str(path): compare.anatomy.sha(path) for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(after, before)
        self.assertNotIn("config", self.state)
        self.assertNotIn("timing", self.record)
        self.assertFalse((self.new_root / "input_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
