"""Stdlib protocol tests; byte fixtures are not MRI accuracy benchmarks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import benchmark_connectome_raw_cohort as cohort
from tools import benchmark_connectome_raw_recovery as recovery


SERIALIZATION_ERROR = (
    "actual nibabel anatomy validation failed: Traceback (most recent call last):\n"
    '  File "cohort.py", line 1, in <module>\n'
    "TypeError: Object of type int32 is not JSON serializable"
)


class RecoveryFixture(unittest.TestCase):
    """Use real files for provenance, and mock only MRI array validation."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run_root = self.root / "new_run"
        self.job = self.run_root / "baseline" / "sub-CON03"
        self.job.mkdir(parents=True)
        self.t1 = self.root / "raw" / "sub-CON03" / "ses-preop" / "anat" / "sub-CON03_ses-preop_T1w.nii.gz"
        self.t1.parent.mkdir(parents=True)
        self.t1.write_bytes(b"Raw T1 provenance unit-test bytes; not a medical image.")
        self.case = {
            "case_id": "sub-CON03", "subject": "CON03", "session": "preop",
            "bids_root": str(self.root / "raw"), "t1w": str(self.t1),
            "input_files": [{"kind": "raw_t1w", "path": str(self.t1),
                             "sha256": cohort.sha256(self.t1)}],
        }
        self.config = {
            "run_root": str(self.run_root), "cpu_threads": 8,
            "recon_all": str(self.root / "official_fs" / "bin" / "recon-all"),
            "freesurfer_home": str(self.root / "official_fs"),
            "atlases": ["fs-aparc"], "anatomy_validation_python": "/usr/bin/python3",
            "worker_script_sha256": cohort.sha256(cohort.__file__),
        }
        executable = Path(self.config["recon_all"])
        executable.parent.mkdir(parents=True)
        executable.write_bytes(b"Official executable identity fixture; never executed.")
        setup = Path(self.config["freesurfer_home"]) / "SetUpFreeSurfer.sh"
        setup.write_bytes(b"Official setup identity fixture; never executed.")
        command, launch, self.subject = cohort.recon_command(self.config, self.case, self.job)
        for relative in (*cohort.ANATOMY, "scripts/recon-all.done"):
            path = self.subject / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("Anatomy IO fixture: " + relative).encode())
        self.anatomy = cohort.check_anatomy(self.subject, self.config["atlases"])
        self.geometry = {"status": "actual_images_surfaces_annotations_read", "images": {},
                         "surfaces": {}, "annotations": {}}
        self.original = {
            "schema_version": cohort.SCHEMA_VERSION, "action": "recon",
            "case_id": self.case["case_id"], "subject": self.case["subject"],
            "version": "baseline", "status": "failed", "exit_code": 0,
            "start_utc": "2026-10-02T00:00:00+00:00", "end_utc": "2026-10-02T01:00:00+00:00",
            "worker_wall_seconds": 3600., "recon_command_seconds": 3500.,
            "command": command, "launch_arguments": launch, "cpu_threads": 8,
            "freesurfer_version": {"returncode": 0, "stdout": "freesurfer official version fixture", "stderr": ""},
            "executable_sha256": cohort.sha256(executable), "setup_script_sha256": cohort.sha256(setup),
            "raw_input_provenance": copy.deepcopy(self.case["input_files"]),
            "input_verification": cohort.verify_inputs(self.case), "anatomy": copy.deepcopy(self.anatomy),
            "error": {"type": "RuntimeError", "message": SERIALIZATION_ERROR},
        }
        self.original_path = self.job / "recon_report.json"
        self.write_original(self.original)
        original_worker = self.root / "frozen_worker.py"
        original_worker.write_bytes(b"Original immutable worker identity fixture.")
        wall = self.root / "frozen_wall.py"
        wall.write_bytes(b"Original immutable raw-DWI wall identity fixture.")
        self.config.update(worker_script=cohort.__file__, wall_script=str(wall),
                           wall_script_sha256=cohort.sha256(wall),
                           sources={"baseline": str(self.root / "frozen_baseline")}, pilot=False)
        original_config = copy.deepcopy(self.config)
        original_config.update(worker_script=str(original_worker), worker_script_sha256=cohort.sha256(original_worker))
        self.snapshot = self.root / "origin_driver_snapshot.json"
        self.snapshot.write_text(json.dumps({"config": original_config,
                                             "fresh_namespace": {"status": "claimed_fresh_namespace", "path": str(self.run_root)}}))
        self.input_manifest = self.run_root / "input_manifest.json"
        self.input_manifest.write_text(json.dumps({"cases": [self.case]}))
        self.config.update(
            recovery_mode=recovery.RECOVERY_MODE, recovery_worker_script=recovery.__file__,
            recovery_worker_sha256=cohort.sha256(recovery.__file__),
            recovery_origin={"snapshot_path": str(self.snapshot), "snapshot_sha256": cohort.sha256(self.snapshot),
                             "input_manifest_sha256": cohort.sha256(self.input_manifest)},
        )

    def write_original(self, report):
        self.original_path.write_text(json.dumps(report, indent=2) + "\n")
        return cohort.sha256(self.original_path)

    def revalidate(self, geometry_effect=None):
        with patch.object(cohort, "validate_anatomy_child", return_value=self.geometry,
                          side_effect=geometry_effect) as validate, patch.object(cohort.subprocess, "run") as run:
            report = recovery.revalidate_worker({"config": self.config, "case": self.case, "version": "baseline"})
        run.assert_not_called()
        return report, validate

    def authorize_repair(self):
        self.config["revalidated_recon_reports"] = {
            "baseline/sub-CON03": str(self.job / "recon_report.revalidated.json")
        }


class OriginalReportGateTests(RecoveryFixture):
    def validate(self, report=None, config=None, case=None, version="baseline", job=None):
        return recovery.validate_original_recon(config or self.config, case or self.case, version,
                                                job or self.job, report or self.original)

    def test_only_documented_post_command_serialization_failure_is_eligible(self):
        self.assertIsNone(self.validate())

    def test_status_action_exit_and_error_type_are_not_generic_resume_gates(self):
        changes = (
            {"status": "completed"}, {"status": "running"}, {"action": "gpu"},
            {"exit_code": 1}, {"exit_code": None},
            {"error": {"type": "TypeError", "message": SERIALIZATION_ERROR}},
            {"error": {"type": "RuntimeError", "message": "official recon-all exited 1"}},
        )
        for change in changes:
            report = copy.deepcopy(self.original)
            report.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validate(report)

    def test_error_requires_anatomy_wrapper_traceback_and_exact_int32_json_tail(self):
        bad_messages = (
            "TypeError: Object of type int32 is not JSON serializable",
            SERIALIZATION_ERROR.replace("actual nibabel anatomy validation failed:", "unrelated step failed:"),
            SERIALIZATION_ERROR.replace("Traceback (most recent call last):", "no actual traceback:"),
            SERIALIZATION_ERROR.replace("int32", "float32"),
            SERIALIZATION_ERROR + "\nA different later failure occurred",
            "int32 JSON serializable", "",
        )
        for message in bad_messages:
            report = copy.deepcopy(self.original)
            report["error"]["message"] = message
            with self.subTest(message=message), self.assertRaises(ValueError):
                self.validate(report)

    def test_identity_and_raw_t1_declaration_must_match_original_cohort(self):
        for field, value in (("case_id", "sub-OTHER"), ("subject", "OTHER"), ("version", "candidate")):
            report = copy.deepcopy(self.original)
            report[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(report)
        for field in ("raw_input_provenance", "input_verification"):
            report = copy.deepcopy(self.original)
            report[field][0]["sha256"] = "f" * 64
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(report)

    def test_recon_command_must_be_the_original_fresh_import_with_fixed_threads(self):
        commands = []
        for flag in ("-i", "-all", "-openmp", "-sd", "-s"):
            command = self.original["command"].copy()
            index = command.index(flag)
            del command[index:index + (1 if flag == "-all" else 2)]
            commands.append(command)
        command = self.original["command"].copy()
        command[command.index("-i") + 1] = str(self.root / "different_T1w.nii.gz")
        commands.append(command)
        command = self.original["command"].copy()
        command[command.index("-openmp") + 1] = "16"
        commands.append(command)
        commands.append(self.original["command"] + ["-autorecon2"])
        for command in commands:
            report = copy.deepcopy(self.original)
            report["command"] = command
            with self.subTest(command=command), self.assertRaises(ValueError):
                self.validate(report)

    def test_namespace_cannot_be_a_previous_run_or_other_version(self):
        for job in (self.root / "old_run" / "baseline" / self.case["case_id"],
                    self.run_root / "candidate" / self.case["case_id"],
                    self.job.parent / "sub-OTHER"):
            with self.subTest(job=job), self.assertRaises(ValueError):
                self.validate(job=job)

    def test_namespace_symlinks_cannot_select_anatomy_from_outside_the_run(self):
        for index, component in enumerate((self.job, self.subject.parent, self.subject)):
            outside = self.root / f"outside_anatomy_{index}"
            component.rename(outside)
            component.symlink_to(outside, target_is_directory=True)
            try:
                with self.subTest(component=component), self.assertRaises(ValueError):
                    self.validate()
            finally:
                component.unlink()
                outside.rename(component)

    def test_recorded_anatomy_paths_cannot_select_old_official_files(self):
        report = copy.deepcopy(self.original)
        report["anatomy"]["mri/brain.mgz"]["path"] = str(self.root / "old_subject" / "mri/brain.mgz")
        with self.assertRaises(ValueError):
            self.validate(report)


class FreshDownstreamGateTests(RecoveryFixture):
    def test_absent_downstream_output_is_eligible(self):
        self.assertIsNone(recovery.assert_fresh_gpu(self.job))

    def test_even_empty_downstream_files_and_directory_forbid_recovery(self):
        for name in ("gpu_report.json", "raw_bids_wall.json", "raw_bids_wall.log", "connectome"):
            path = self.job / name
            if name == "connectome":
                path.mkdir()
            else:
                path.touch()
            with self.subTest(name=name), self.assertRaises(FileExistsError):
                recovery.assert_fresh_gpu(self.job)
            if path.is_dir():
                path.rmdir()
            else:
                path.unlink()


class NormalCohortGateTests(RecoveryFixture):
    def test_normal_gpu_worker_still_rejects_failed_recon_without_recovery_authorization(self):
        original_bytes = self.original_path.read_bytes()
        self.config["worker_script_sha256"] = cohort.sha256(cohort.__file__)
        with patch.object(cohort.subprocess, "run") as run:
            result = cohort.worker({"action": "gpu", "config": self.config,
                                    "case": self.case, "version": "baseline"})
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["type"], "RuntimeError")
        self.assertIn("has not completed", result["error"]["message"])
        self.assertEqual(self.original_path.read_bytes(), original_bytes)
        run.assert_not_called()

    def test_repair_file_alone_does_not_authorize_normal_cohort_loading(self):
        repaired, _ = self.revalidate()
        self.assertEqual(repaired["status"], "completed")
        with self.assertRaisesRegex(RuntimeError, "has not completed"):
            cohort.load_recon_for_gpu(self.config, self.case, "baseline", self.job)

    def test_explicit_mode_and_exact_per_case_repair_are_required(self):
        repaired, _ = self.revalidate()
        self.assertEqual(repaired["status"], "completed")
        self.authorize_repair()
        config = copy.deepcopy(self.config)
        del config["recovery_mode"]
        with self.assertRaisesRegex(RuntimeError, "explicit private recovery mode"):
            cohort.load_recon_for_gpu(config, self.case, "baseline", self.job)
        for path in (self.original_path, self.root / "other_report.json"):
            config = copy.deepcopy(self.config)
            config["revalidated_recon_reports"]["baseline/sub-CON03"] = str(path)
            with self.subTest(path=path), self.assertRaises(ValueError):
                cohort.load_recon_for_gpu(config, self.case, "baseline", self.job)

    def test_exact_bound_completed_repair_is_loaded_without_rewriting_original(self):
        self.revalidate()
        self.authorize_repair()
        original_bytes = self.original_path.read_bytes()
        loaded = cohort.load_recon_for_gpu(self.config, self.case, "baseline", self.job)
        self.assertEqual(loaded["status"], "completed")
        self.assertEqual(loaded["recovery"]["original_status"], "failed")
        self.assertEqual(loaded["recovery"]["original_failure"], self.original["error"])
        self.assertEqual(loaded["recon_command_seconds"], 3500.)
        self.assertEqual(self.original_path.read_bytes(), original_bytes)
        self.assertEqual(json.loads(self.original_path.read_text())["status"], "failed")

    def test_repair_binding_and_geometry_cannot_be_forged(self):
        report, _ = self.revalidate()
        self.authorize_repair()
        changes = (
            {"status": "failed"}, {"action": "recon"}, {"recon_all_rerun": True},
            {"raw_dwi_preprocessing_resumed": True}, {"case_id": "sub-OTHER"},
            {"subject": "OTHER"}, {"version": "candidate"},
            {"origin_snapshot_sha256": "f" * 64}, {"recovery_worker_sha256": "f" * 64},
            {"cohort_worker_sha256": "f" * 64},
            {"original_report": {"path": str(self.original_path), "sha256": "f" * 64}},
            {"original_recon_command_seconds": 1.},
            {"original_recon_start_utc": "2026-10-02T00:30:00+00:00"},
            {"original_failure": {"type": "RuntimeError", "message": "different failure"}},
            {"anatomy_geometry": {"status": "file_names_only"}},
            {"input_verification_after": []}, {"raw_t1_sha256_after": "f" * 64},
        )
        repair_path = self.job / "recon_report.revalidated.json"
        for change in changes:
            altered = copy.deepcopy(report)
            altered.update(change)
            repair_path.write_text(json.dumps(altered))
            with self.subTest(change=change), self.assertRaises(ValueError):
                cohort.load_recon_for_gpu(self.config, self.case, "baseline", self.job)


class RevalidationIOTests(RecoveryFixture):
    def test_repair_reads_current_bytes_and_writes_a_separate_immutable_bound_report(self):
        original_bytes = self.original_path.read_bytes()
        original_sha = cohort.sha256(self.original_path)
        protected = {path: path.read_bytes() for path in (self.snapshot, self.input_manifest,
                                                        self.root / "frozen_worker.py", self.root / "frozen_wall.py")}
        report, validate = self.revalidate()
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["action"], "revalidate_official_anatomy")
        self.assertIs(report["recon_all_rerun"], False)
        self.assertIs(report["raw_dwi_preprocessing_resumed"], False)
        self.assertEqual(report["original_report"], {"path": str(self.original_path), "sha256": original_sha})
        self.assertEqual(report["original_failure"], self.original["error"])
        self.assertEqual(report["input_verification"], cohort.verify_inputs(self.case))
        self.assertEqual(report["input_verification_after"], report["input_verification"])
        self.assertEqual(report["raw_t1_sha256_after"], cohort.sha256(self.t1))
        self.assertEqual(report["anatomy"], self.anatomy)
        validate.assert_called_once_with(self.subject, self.anatomy, self.config["anatomy_validation_python"])
        self.assertEqual(json.loads((self.job / "recon_report.revalidated.json").read_text()), report)
        self.assertEqual(self.original_path.read_bytes(), original_bytes)
        self.assertEqual(cohort.sha256(self.original_path), original_sha)
        self.assertEqual(json.loads(self.original_path.read_text())["status"], "failed")
        for path, contents in protected.items():
            self.assertEqual(path.read_bytes(), contents)

    def test_existing_repair_is_never_overwritten(self):
        path = self.job / "recon_report.revalidated.json"
        path.write_bytes(b"Existing repair must remain byte-identical.")
        contents = path.read_bytes()
        with self.assertRaises(FileExistsError):
            self.revalidate()
        self.assertEqual(path.read_bytes(), contents)
        self.assertFalse((self.job / "recovery_claim.json").exists())

    def test_existing_claim_is_not_resumed(self):
        claim = self.job / "recovery_claim.json"
        claim.write_bytes(b"Existing claim must remain byte-identical.")
        with self.assertRaises(FileExistsError):
            self.revalidate()
        self.assertEqual(claim.read_bytes(), b"Existing claim must remain byte-identical.")
        self.assertFalse((self.job / "recon_report.revalidated.json").exists())

    def test_changed_raw_t1_is_rejected_before_geometry_validation(self):
        self.t1.write_bytes(b"Changed raw T1 IO fixture.")
        report, validate = self.revalidate()
        self.assertEqual(report["status"], "failed")
        self.assertIn("raw input hash mismatch", report["error"]["message"])
        validate.assert_not_called()

    def test_missing_official_done_cannot_be_repaired(self):
        (self.subject / "scripts/recon-all.done").unlink()
        report, validate = self.revalidate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error"]["type"], "FileNotFoundError")
        validate.assert_not_called()

    def test_changed_official_executable_cannot_be_repaired(self):
        Path(self.config["recon_all"]).write_bytes(b"Changed official executable identity.")
        report, validate = self.revalidate()
        self.assertEqual(report["status"], "failed")
        self.assertIn("official recon-all executable changed", report["error"]["message"])
        validate.assert_not_called()

    def test_changed_official_setup_script_cannot_be_repaired(self):
        (Path(self.config["freesurfer_home"]) / "SetUpFreeSurfer.sh").write_bytes(b"Changed official setup identity.")
        report, validate = self.revalidate()
        self.assertEqual(report["status"], "failed")
        self.assertIn("official setup script changed", report["error"]["message"])
        validate.assert_not_called()

    def test_actual_anatomy_validation_failure_cannot_mark_the_repair_completed(self):
        original_bytes = self.original_path.read_bytes()
        report, validate = self.revalidate(RuntimeError("invalid real MGZ data/geometry"))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error"]["message"], "invalid real MGZ data/geometry")
        self.assertEqual(validate.call_count, 1)
        self.assertEqual(self.original_path.read_bytes(), original_bytes)

    def test_changed_official_anatomy_is_rejected_before_geometry_validation(self):
        (self.subject / "mri/brain.mgz").write_bytes(b"Changed anatomy IO fixture.")
        report, validate = self.revalidate()
        self.assertEqual(report["status"], "failed")
        self.assertIn("official anatomy changed", report["error"]["message"])
        validate.assert_not_called()

    def test_anatomy_changes_during_validation_cannot_complete(self):
        def change_anatomy(*_):
            (self.subject / "mri/brain.mgz").write_bytes(b"Concurrent changed anatomy IO fixture.")
            return self.geometry
        report, _ = self.revalidate(change_anatomy)
        self.assertEqual(report["status"], "failed")
        self.assertIn("anatomy changed while", report["error"]["message"])

    def test_raw_t1_changes_during_validation_cannot_complete(self):
        def change_t1(*_):
            self.t1.write_bytes(b"Concurrent changed raw T1 IO fixture.")
            return self.geometry
        report, _ = self.revalidate(change_t1)
        self.assertEqual(report["status"], "failed")
        self.assertIn("raw input hash mismatch", report["error"]["message"])

    def test_original_report_changes_during_validation_are_detected(self):
        def change_original(*_):
            changed = copy.deepcopy(self.original)
            changed["worker_wall_seconds"] += 1.
            self.write_original(changed)
            return self.geometry
        with self.assertRaisesRegex(ValueError, "original failure report changed"):
            self.revalidate(change_original)
        repair = json.loads((self.job / "recon_report.revalidated.json").read_text())
        self.assertEqual(repair["status"], "failed")

    def test_existing_gpu_outputs_prevent_even_anatomy_revalidation(self):
        (self.job / "connectome").mkdir()
        with self.assertRaises(FileExistsError):
            self.revalidate()
        self.assertFalse((self.job / "recovery_claim.json").exists())
        self.assertFalse((self.job / "recon_report.revalidated.json").exists())

    def test_original_snapshot_cannot_be_relabelled_or_modified(self):
        self.snapshot.write_bytes(self.snapshot.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "driver snapshot changed"):
            self.revalidate()

    def test_original_namespace_requires_a_successful_fresh_claim(self):
        snapshot = json.loads(self.snapshot.read_text())
        snapshot["fresh_namespace"]["status"] = "failed"
        self.snapshot.write_text(json.dumps(snapshot))
        self.config["recovery_origin"]["snapshot_sha256"] = cohort.sha256(self.snapshot)
        with self.assertRaisesRegex(ValueError, "fresh-namespace claim was not successful"):
            self.revalidate()
        self.assertFalse((self.job / "recovery_claim.json").exists())

    def test_original_frozen_worker_cannot_be_modified(self):
        (self.root / "frozen_worker.py").write_bytes(b"Changed original worker.")
        with self.assertRaisesRegex(ValueError, "original frozen worker changed"):
            self.revalidate()

    def test_original_frozen_wall_runner_cannot_be_modified(self):
        (self.root / "frozen_wall.py").write_bytes(b"Changed original wall runner.")
        with self.assertRaisesRegex(ValueError, "original frozen raw-DWI wall runner changed"):
            self.revalidate()

    def test_original_input_manifest_cannot_be_modified(self):
        self.input_manifest.write_bytes(self.input_manifest.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "input manifest changed"):
            self.revalidate()

    def test_recovery_cannot_change_original_threads_or_atlas_configuration(self):
        for field, changed in (("cpu_threads", 16), ("atlases", ["schaefer200+tian-s1"])):
            config = copy.deepcopy(self.config)
            config[field] = changed
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "changed original frozen configuration"):
                recovery.verify_origin(config)


class RecoveryTimingTests(unittest.TestCase):
    @staticmethod
    def arguments():
        return {
            "original_start_utc": "2026-10-02T00:00:00+00:00",
            "original_end_utc": "2026-10-02T01:00:00+00:00",
            "repaired_start_utc": "2026-10-02T01:20:00+00:00",
            "repaired_end_utc": "2026-10-02T01:21:00+00:00",
            "gpu_end_utc": "2026-10-02T01:30:00+00:00",
            "recon_seconds": 3500., "gpu_driver_queue": 90., "gpu_lock_queue": 30.,
        }

    def test_invalid_ordering_and_unmeasured_or_impossible_timers_are_rejected(self):
        changes = (
            {"original_start_utc": "not a UTC timestamp"},
            {"original_start_utc": "2026-10-02T00:00:00"},
            {"original_end_utc": "2026-10-01T23:59:00+00:00"},
            {"repaired_start_utc": "2026-10-02T00:59:00+00:00"},
            {"repaired_end_utc": "2026-10-02T01:19:00+00:00"},
            {"gpu_end_utc": "2026-10-02T01:20:30+00:00"},
            {"recon_seconds": -1.}, {"recon_seconds": float("nan")},
            {"recon_seconds": 3601.}, {"gpu_driver_queue": -1.},
            {"gpu_lock_queue": float("inf")}, {"gpu_driver_queue": 5400.},
        )
        for change in changes:
            arguments = self.arguments()
            arguments.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                recovery.recovery_timing(**arguments)

    def test_full_elapsed_includes_failure_repair_wait_validation_and_gpu_phase(self):
        measured = recovery.recovery_timing(**self.arguments())
        expected = {
            "recovered_full_elapsed_utc_seconds": 5400.,
            "recovered_full_elapsed_utc_excluding_gpu_queue_seconds": 5280.,
            "original_recon_command_seconds": 3500.,
            "original_failure_to_revalidation_gap_utc_seconds": 1200.,
            "revalidation_elapsed_utc_seconds": 60.,
            "post_revalidation_through_gpu_completion_utc_seconds": 540.,
            "gpu_driver_queue_seconds": 90., "gpu_lock_queue_seconds": 30.,
        }
        for field, value in expected.items():
            self.assertEqual(measured[field], value)
        self.assertIn("not a continuous monotonic cold-run timer", measured["scope"])


if __name__ == "__main__":
    unittest.main()
