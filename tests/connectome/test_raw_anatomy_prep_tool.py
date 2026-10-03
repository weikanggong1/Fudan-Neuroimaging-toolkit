"""Stdlib preparation protocol tests; byte fixtures are not MRI benchmarks."""
import copy
import io
import json
from pathlib import Path
import shlex
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import benchmark_connectome_raw_cohort as cohort
from tools import benchmark_connectome_anatomy_prep as prep


class PreparationFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw = self.root / "new_raw_bids"
        self.raw.mkdir()
        dataset = self.raw / "dataset_description.json"
        dataset.write_bytes(b"Dataset identity fixture.")
        self.cases = []
        for index in range(10):
            subject = f"S{index:02d}"
            t1 = self.raw / f"sub-{subject}/anat/sub-{subject}_T1w.nii.gz"
            dwi = self.raw / f"sub-{subject}/dwi/sub-{subject}_dwi.nii.gz"
            files = [("raw_t1w", t1), ("raw_dwi", dwi), ("bval", dwi.with_suffix(".bval")),
                     ("bvec", dwi.with_suffix(".bvec")), ("dwi_json", dwi.with_suffix(".json")),
                     ("dataset_description", dataset)]
            for kind, path in files:
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    path.write_bytes(("Raw input identity fixture: " + kind).encode())
            self.cases.append({"case_id": f"C{index:02d}", "subject": subject, "bids_root": str(self.raw),
                               "t1w": str(t1), "input_files": [{"kind": kind, "path": str(path),
                                                                 "sha256": cohort.sha256(path)} for kind, path in files]})
        self.manifest = {"dataset": "protocol-fixture", "snapshot": "fixed", "license": "CC0",
                         "source_url": "https://example.org/fixture", "download_completed_utc": "2026-10-03T00:00:00+00:00",
                         "downloaded_new": True, "cases": self.cases}
        self.old_root = self.root / "original_round"
        self.old_root.mkdir()
        cohort.atomic_json(self.old_root / "input_manifest.json", self.manifest)
        official_home = self.root / "official FreeSurfer's installation"
        executable = official_home / "bin/recon-all"
        executable.parent.mkdir(parents=True)
        executable.write_bytes(b"Official executable identity fixture; not executed.")
        executable.chmod(0o755)
        setup = official_home / "SetUpFreeSurfer.sh"
        setup.write_bytes(b"Official setup identity fixture; not executed.")
        self.original_config = {"run_root": str(self.old_root), "cpu_threads": 8, "cpu_jobs": 2,
                                "cpu_host": "nodecw10", "cpu_python": sys.executable,
                                "anatomy_validation_python": sys.executable, "cpu_control_path": "/tmp/verified socket",
                                "recon_all": str(executable), "freesurfer_home": str(official_home), "atlases": ["fs-aparc"],
                                "pilot": False, "sources": {"baseline": "/no_candidate_source_access"},
                                "frozen_sources": {"baseline": {"source_fingerprint": "not-used"}},
                                "gpu_host": "gpucw1", "gpu_lock": "/tmp/do-not-acquire", "n_seeds": 100000,
                                "atlas_options": ["--mni-template", "/declared/future/resource.nii.gz"]}
        job = self.old_root / "baseline" / self.cases[0]["case_id"]
        command, launch, _ = cohort.recon_command(self.original_config, self.cases[0], job)
        self.origin_report = {"action": "recon", "case_id": self.cases[0]["case_id"], "version": "baseline",
                              "exit_code": 0, "command": command, "launch_arguments": launch,
                              "freesurfer_version": {"returncode": 0, "stdout": "freesurfer official fixture", "stderr": ""},
                              "executable_sha256": cohort.sha256(executable), "setup_script_sha256": cohort.sha256(setup)}
        cohort.atomic_json(job / "recon_report.json", self.origin_report)
        self.driver = self.root / "original_driver"
        self.driver.mkdir()
        self.origin = {"config": self.original_config,
                       "fresh_namespace": {"status": "claimed_fresh_namespace", "path": str(self.old_root)}}
        cohort.atomic_json(self.driver / "status.json", self.origin)
        self.options = SimpleNamespace(origin_driver_report_dir=self.driver, run_root=self.root / "fresh_candidate_anatomy",
                                       report_dir=self.root / "fresh_preparation_driver", worker_script=Path(cohort.__file__),
                                       cpu_jobs=2, cpu_threads=8, poll_seconds=1.)
        self.config, _, _ = prep.derive_config(self.options)

    def fake_official_worker(self, payload):
        """Mock execution only; write byte-bound protocol outputs for IO gates."""
        config, case = payload["config"], payload["case"]
        self.assertEqual(payload["action"], "recon")
        self.assertEqual(payload["version"], "candidate")
        job = Path(config["run_root"]) / "candidate" / case["case_id"]
        cohort.require_fresh(job)
        command, launch, subject = cohort.recon_command(config, case, job)
        for relative in (*cohort.ANATOMY, "scripts/recon-all.done"):
            path = subject / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("Anatomy byte fixture: " + relative).encode())
        identity = config["official_origin"]["identity"]
        report = {"status": "completed", "action": "recon", "case_id": case["case_id"], "subject": case["subject"],
                  "version": "candidate", "cpu_threads": 8, "exit_code": 0, "command": command, "launch_arguments": launch,
                  "raw_input_provenance": copy.deepcopy(case["input_files"]),
                  "executable_sha256": identity["executable_sha256"], "setup_script_sha256": identity["setup_script_sha256"],
                  "freesurfer_version": {"returncode": 0, "stdout": identity["version"], "stderr": ""},
                  "anatomy_geometry": {"status": "actual_images_surfaces_annotations_read"},
                  "anatomy": cohort.check_anatomy(subject, config["atlases"]), "recon_command_seconds": .000001}
        cohort.atomic_json(job / "recon_report.json", report)
        return report

    def prepare(self):
        with patch.object(cohort, "worker", side_effect=self.fake_official_worker):
            return prep.worker({"action": "recon", "config": self.config, "case": self.cases[0], "version": "candidate"})


class ConfigurationTests(PreparationFixture):
    def test_candidate_source_is_unknown_and_not_present_in_config(self):
        prep.reject_source_fields(self.config)
        self.assertNotIn("sources", self.config)
        self.assertNotIn("frozen_sources", self.config)
        self.assertEqual(self.config["candidate_source"], "unknown")
        self.assertEqual(self.config["gpu_status"], "GPU_not_started")
        self.assertEqual(self.config["future_gpu_parameters"]["n_seeds"], 100000)

    def test_source_fields_are_rejected_even_if_nested(self):
        for field in ("sources", "frozen_sources"):
            config = copy.deepcopy(self.config)
            config["future_gpu_parameters"][field] = {}
            with self.subTest(field=field), self.assertRaises(ValueError):
                prep.validate_config(config)

    def test_cpu_threads_are_exactly_eight_and_jobs_are_bounded(self):
        for field, value in (("cpu_threads", 16), ("cpu_threads", True), ("cpu_jobs", 3), ("cpu_jobs", True)):
            config = copy.deepcopy(self.config)
            config[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                prep.validate_config(config)

    def test_future_gpu_parameters_never_authorize_gpu_execution(self):
        config = copy.deepcopy(self.config)
        config["gpu_started"] = True
        with self.assertRaises(ValueError):
            prep.validate_config(config)
        with self.assertRaises(ValueError):
            prep.worker({"action": "gpu", "config": self.config, "case": self.cases[0], "version": "candidate"})

    def test_formal_manifest_requires_exactly_ten_distinct_raw_subjects(self):
        for cases in (self.cases[:9], self.cases + [self.cases[0]]):
            changed = copy.deepcopy(self.manifest)
            changed["cases"] = cases
            cohort.atomic_json(self.old_root / "input_manifest.json", changed)
            with self.subTest(count=len(cases)), self.assertRaises(ValueError):
                prep.derive_config(self.options)

    def test_run_and_reports_cannot_overlap_original_anatomy_or_each_other(self):
        for field, path in (("run_root", self.old_root / "nested"), ("report_dir", self.old_root),
                            ("report_dir", self.options.run_root / "nested_reports")):
            options = copy.copy(self.options)
            setattr(options, field, path)
            with self.subTest(field=field, path=path), self.assertRaises(ValueError):
                prep.derive_config(options)

    def test_original_official_executable_and_manifest_changes_are_detected(self):
        executable = Path(self.config["recon_all"])
        executable.write_bytes(executable.read_bytes() + b"changed")
        with self.assertRaises(ValueError):
            prep.verify_runtime_files(self.config)

    def test_original_official_version_requires_an_actual_exit_zero_probe(self):
        path = Path(self.config["official_origin"]["report"]["path"])
        report = copy.deepcopy(self.origin_report)
        report["exit_code"] = 1
        cohort.atomic_json(path, report)
        with self.assertRaises(ValueError):
            prep.derive_config(self.options)

    def test_recon_command_uses_new_subject_and_safe_literal_raw_path(self):
        case = copy.deepcopy(self.cases[0])
        case["t1w"] = "/shared/raw input $(do-not-run)'_T1w.nii.gz"
        job = Path(self.config["run_root"]) / "candidate" / case["case_id"]
        command, launch, subject = cohort.recon_command(self.config, case, job)
        self.assertEqual(command[command.index("-i") + 1], case["t1w"])
        self.assertEqual(command[-3:], ["-all", "-openmp", "8"])
        self.assertTrue(subject.is_relative_to(job))
        self.assertNotIn(case["t1w"], launch[2])
        ssh = cohort.ssh_command("nodecw10", None, None, launch)
        self.assertEqual(shlex.split(ssh[-1]), launch)


class PreparationIOTests(PreparationFixture):
    def test_fresh_result_has_both_input_hash_sets_and_real_reconstruction_binding(self):
        result = self.prepare()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["input_verification_before"], cohort.verify_inputs(self.cases[0]))
        self.assertEqual(result["input_verification_after"], result["input_verification_before"])
        prep.validate_preparation_result(self.config, self.cases[0], result)
        self.assertFalse(result["gpu_started"])
        self.assertFalse(result["recon_all_reused"])

    def test_existing_empty_case_is_not_resumed(self):
        job = Path(self.config["run_root"]) / "candidate" / self.cases[0]["case_id"]
        job.mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertFalse((job / "anatomy_prep_report.json").exists())

    def test_original_baseline_reports_remain_byte_identical(self):
        old_report = Path(self.config["official_origin"]["report"]["path"])
        contents = old_report.read_bytes()
        self.prepare()
        self.assertEqual(old_report.read_bytes(), contents)

    def test_changed_dwi_after_official_run_is_a_failed_preparation(self):
        def mutate(payload):
            report = self.fake_official_worker(payload)
            record = next(item for item in self.cases[0]["input_files"] if item["kind"] == "raw_dwi")
            Path(record["path"]).write_bytes(b"Raw DWI changed while CPU recon ran.")
            return report
        with patch.object(cohort, "worker", side_effect=mutate):
            result = prep.worker({"action": "recon", "config": self.config, "case": self.cases[0], "version": "candidate"})
        self.assertEqual(result["status"], "failed")
        self.assertIn("raw input hash mismatch", result["error"]["message"])

    def test_names_only_anatomy_validation_cannot_complete(self):
        def invalid(payload):
            report = self.fake_official_worker(payload)
            report["anatomy_geometry"] = {"status": "filenames_only"}
            return report
        with patch.object(cohort, "worker", side_effect=invalid):
            result = prep.worker({"action": "recon", "config": self.config, "case": self.cases[0], "version": "candidate"})
        self.assertEqual(result["status"], "failed")
        self.assertIn("actual MRI", result["error"]["message"])

    def test_report_and_anatomy_mutations_cannot_pass_completion_gate(self):
        result = self.prepare()
        subject = cohort.recon_command(self.config, self.cases[0],
                                       Path(self.config["run_root"]) / "candidate" / self.cases[0]["case_id"])[2]
        (subject / "mri/brain.mgz").write_bytes(b"Anatomy changed after preparation.")
        with self.assertRaises(ValueError):
            prep.validate_preparation_result(self.config, self.cases[0], result)


class DriverAndTimingTests(PreparationFixture):
    def remote_fixture(self, config, action, case, log_path):
        self.assertNotIn("sources", config)
        if action == "preflight":
            return {"status": "verified_official_cpu_environment", "gpu_started": False}
        return prep.worker({"action": action, "config": config, "case": case, "version": "candidate"})

    def test_completed_driver_is_only_preparation_and_requires_ten_actual_bound_reports(self):
        with patch.object(prep, "remote", side_effect=self.remote_fixture), \
                patch.object(cohort, "worker", side_effect=self.fake_official_worker), \
                patch("sys.stdout", new_callable=io.StringIO):
            code = prep.run(self.options)
        self.assertEqual(code, 0)
        state = json.loads((self.options.report_dir / "status.json").read_text())
        self.assertEqual(state["status"], "completed_anatomy_preparation")
        self.assertEqual(state["completed_cases"], 10)
        self.assertEqual(state["candidate_source"], "unknown")
        self.assertEqual(state["gpu_status"], "GPU_not_started")
        self.assertFalse(state["full_pipeline_benchmark"])
        self.assertFalse(state["comparison_ready"])
        self.assertTrue((self.options.report_dir / "cases.csv").is_file())
        self.assertTrue((self.options.run_root / "anatomy_prep_config.json").is_file())

    def test_existing_driver_directory_is_never_overwritten(self):
        self.options.report_dir.mkdir()
        (self.options.report_dir / "keep.txt").write_bytes(b"Keep original report.")
        with self.assertRaises(FileExistsError):
            prep.run(self.options)
        self.assertEqual((self.options.report_dir / "keep.txt").read_bytes(), b"Keep original report.")

    def test_existing_empty_run_directory_is_not_resumed(self):
        self.options.run_root.mkdir()
        with patch.object(prep, "remote", side_effect=self.remote_fixture):
            self.assertEqual(prep.run(self.options), 1)
        self.assertFalse((self.options.run_root / "anatomy_prep_config.json").exists())

    def test_stop_sentinel_prevents_new_cpu_dispatch_without_stopping_active_workers(self):
        calls = []
        lock = threading.Lock()
        def remote(config, action, case, log_path):
            if action == "recon":
                with lock:
                    calls.append(case["case_id"])
                    (self.options.report_dir / "STOP_DISPATCH").touch()
            return self.remote_fixture(config, action, case, log_path)
        with patch.object(prep, "remote", side_effect=remote), \
                patch.object(cohort, "worker", side_effect=self.fake_official_worker), \
                patch("sys.stdout", new_callable=io.StringIO):
            code = prep.run(self.options)
        self.assertEqual(code, 1)
        self.assertGreaterEqual(len(calls), 1)
        self.assertLessEqual(len(calls), 2)
        state = json.loads((self.options.report_dir / "status.json").read_text())
        self.assertTrue(state["dispatch_paused"])
        self.assertLess(state["completed_cases"], 10)

    def test_head_timer_has_cpu_queue_and_no_cross_host_timestamp_math(self):
        timing = prep.case_timing(20., 32., 15., 10.)
        self.assertEqual(timing["head_case_wall_seconds"], 12.)
        self.assertEqual(timing["cpu_driver_queue_seconds"], 5.)
        self.assertEqual(timing["recon_command_seconds"], 10.)
        for arguments in ((20., 19., 15., 1.), (20., 32., 21., 1.), (20., 32., 15., 13.),
                          (20., 32., 15., float("nan"))):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                prep.case_timing(*arguments)

    def test_timing_error_preserves_successful_real_preparation_results(self):
        with patch.object(prep, "remote", side_effect=self.remote_fixture), \
                patch.object(cohort, "worker", side_effect=self.fake_official_worker), \
                patch.object(prep, "case_timing", side_effect=ValueError("independent timing error")), \
                patch("sys.stdout", new_callable=io.StringIO):
            code = prep.run(self.options)
        self.assertEqual(code, 1)
        state = json.loads((self.options.report_dir / "status.json").read_text())
        self.assertEqual(state["completed_cases"], 10)
        self.assertEqual(state["status"], "completed_anatomy_preparation_with_timing_errors")
        self.assertFalse(state["timing_complete"])
        self.assertFalse(state["comparison_ready"])
        for record in state["cases"].values():
            self.assertEqual(record["status"], "completed")
            self.assertEqual(record["preparation_report"]["status"], "completed")
            self.assertEqual(record["timing_error"]["message"], "independent timing error")

    def test_head_start_is_saved_before_cpu_ssh_begins(self):
        observed = []
        def remote(config, action, case, log_path):
            if action == "recon":
                state = json.loads((self.options.report_dir / "status.json").read_text())
                record = state["cases"]["candidate/" + case["case_id"]]
                self.assertEqual(record["status"], "cpu_running")
                self.assertTrue(record["start_utc"])
                self.assertGreaterEqual(record["cpu_driver_queue_seconds"], 0.)
                observed.append(case["case_id"])
            return self.remote_fixture(config, action, case, log_path)
        with patch.object(prep, "remote", side_effect=remote), \
                patch.object(cohort, "worker", side_effect=self.fake_official_worker), \
                patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(prep.run(self.options), 0)
        self.assertEqual(len(observed), 10)

    def test_collection_keeps_actual_official_error_separate_from_timing_error(self):
        record = {}
        actual = {"status": "failed", "error": {"type": "RuntimeError", "message": "official recon-all exited 1"}}
        context = {"result": actual, "start_utc": "2026-10-03T00:00:00+00:00",
                   "end_utc": "2026-10-03T01:00:00+00:00",
                   "timing_error": {"type": "ValueError", "message": "independent timer problem"}}
        prep.collect_preparation_result(record, context)
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["error"], actual["error"])
        self.assertEqual(record["preparation_report"], actual)
        self.assertEqual(record["timing_error"], context["timing_error"])


if __name__ == "__main__":
    unittest.main()
