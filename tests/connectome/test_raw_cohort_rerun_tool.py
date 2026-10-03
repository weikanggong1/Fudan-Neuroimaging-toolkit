"""Stdlib provenance tests; byte fixtures are not MRI accuracy benchmarks."""
import copy
from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import benchmark_connectome_raw_cohort as cohort
from tools import benchmark_connectome_raw_rerun as rerun


SERIALIZATION_ERROR = (
    "actual nibabel anatomy validation failed: Traceback (most recent call last):\n"
    '  File "cohort.py", line 1, in <module>\n'
    "TypeError: Object of type int32 is not JSON serializable"
)


class RerunFixture(unittest.TestCase):
    """Keep the original reconstruction and fresh GPU outputs in distinct roots."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.old_root = self.root / "original_raw_round"
        self.old_job = self.old_root / "baseline" / "sub-CON03"
        self.old_job.mkdir(parents=True)
        self.new_root = self.root / "common_v2_raw_dwi"
        self.new_root.mkdir()
        self.new_job = self.new_root / "baseline" / "sub-CON03"
        self.t1 = self.root / "raw" / "sub-CON03" / "ses-preop" / "anat" / "sub-CON03_ses-preop_T1w.nii.gz"
        self.t1.parent.mkdir(parents=True)
        self.t1.write_bytes(b"Fresh raw T1 provenance fixture; not an MRI image.")
        self.dwi = self.t1.parents[1] / "dwi" / "sub-CON03_ses-preop_dwi.nii.gz"
        self.dwi.parent.mkdir()
        self.dwi.write_bytes(b"Fresh raw DWI provenance fixture; not an MRI image.")
        self.case = {
            "case_id": "sub-CON03", "subject": "CON03", "session": "preop",
            "bids_root": str(self.root / "raw"), "t1w": str(self.t1),
            "input_files": [{"kind": kind, "path": str(path), "sha256": cohort.sha256(path)}
                            for kind, path in (("raw_t1w", self.t1), ("raw_dwi", self.dwi))],
        }
        executable = self.root / "official_fs" / "bin" / "recon-all"
        executable.parent.mkdir(parents=True)
        executable.write_bytes(b"Official executable byte identity; never executed.")
        setup = executable.parent.parent / "SetUpFreeSurfer.sh"
        setup.write_bytes(b"Official setup byte identity; never executed.")
        old_worker = self.root / "immutable_original_worker.py"
        old_worker.write_bytes(b"Immutable original cohort worker fixture.")
        old_wall = self.root / "immutable_original_wall.py"
        old_wall.write_bytes(b"Immutable original raw-DWI wall fixture.")
        self.original_source = self.root / "original_frozen_source"
        self.common_source = self.root / "common_compatible_v2"
        for source, core_bytes in ((self.original_source, b"Original common baseline identity fixture."),
                                   (self.common_source, b"Common MGH compatibility identity fixture.")):
            (source / "src/fnit/flirt").mkdir(parents=True)
            (source / "src/fnit/cli.py").write_bytes(b"Shared CLI source identity fixture.")
            (source / "src/fnit/flirt/core.py").write_bytes(core_bytes)
        self.original_config = {
            "run_root": str(self.old_root), "cpu_threads": 8, "gpu_cpu_threads": 8,
            "recon_all": str(executable), "freesurfer_home": str(executable.parent.parent),
            "atlases": ["fs-aparc"], "anatomy_validation_python": "/usr/bin/python3",
            "worker_script": str(old_worker), "worker_script_sha256": cohort.sha256(old_worker),
            "wall_script": str(old_wall), "wall_script_sha256": cohort.sha256(old_wall),
            "sources": {"baseline": str(self.original_source)}, "pilot": False,
            "frozen_sources": {"baseline": cohort.source_manifest(self.original_source)},
        }
        command, launch, self.subject = cohort.recon_command(self.original_config, self.case, self.old_job)
        for relative in (*cohort.ANATOMY, "scripts/recon-all.done"):
            path = self.subject / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("Official anatomy IO fixture: " + relative).encode())
        self.anatomy = cohort.check_anatomy(self.subject, self.original_config["atlases"])
        self.geometry = {"status": "actual_images_surfaces_annotations_read", "images": {},
                         "surfaces": {}, "annotations": {}}
        self.original_report = {
            "schema_version": cohort.SCHEMA_VERSION, "action": "recon", "case_id": "sub-CON03",
            "subject": "CON03", "version": "baseline", "status": "failed", "exit_code": 0,
            "start_utc": "2026-10-02T13:28:39.058731+00:00",
            "end_utc": "2026-10-02T14:39:23.003982+00:00",
            "worker_wall_seconds": 4243.918243408203, "recon_command_seconds": 4241.2467591241,
            "command": command, "launch_arguments": launch, "cpu_threads": 8,
            "freesurfer_version": {"returncode": 0, "stdout": "freesurfer official version fixture", "stderr": ""},
            "executable_sha256": cohort.sha256(executable), "setup_script_sha256": cohort.sha256(setup),
            "raw_input_provenance": copy.deepcopy(self.case["input_files"]),
            "input_verification": cohort.verify_inputs(self.case), "anatomy": copy.deepcopy(self.anatomy),
            "error": {"type": "RuntimeError", "message": SERIALIZATION_ERROR},
        }
        self.original_report_path = self.old_job / "recon_report.json"
        self.write_original(self.original_report)
        self.snapshot = {
            "config": copy.deepcopy(self.original_config),
            "fresh_namespace": {"status": "claimed_fresh_namespace", "path": str(self.old_root)},
            "cases": {"baseline/sub-CON03": {"start_utc": "2026-10-02T13:28:52.183833+00:00"}},
        }
        self.snapshot_path = self.root / "immutable_origin_snapshot.json"
        self.snapshot_path.write_text(json.dumps(self.snapshot))
        self.manifest_path = self.old_root / "input_manifest.json"
        self.manifest_path.write_text(json.dumps({"cases": [self.case]}))
        self.common_identity_path = self.root / "common_identity.json"
        self.common_identity_path.write_text(json.dumps({
            "performance_optimization": False,
            "core_sha256": cohort.sha256(self.common_source / "src/fnit/flirt/core.py"),
        }))
        self.resources_path = self.root / "resources.json"
        self.resources_path.write_text(json.dumps({"files": []}))
        self.config = copy.deepcopy(self.original_config)
        self.config.update(
            run_root=str(self.new_root), worker_script=cohort.__file__,
            worker_script_sha256=cohort.sha256(cohort.__file__), recovery_mode=rerun.MODE,
            rerun_worker_script=rerun.__file__, rerun_worker_sha256=cohort.sha256(rerun.__file__),
            sources={"baseline": str(self.common_source)},
            frozen_sources={"baseline": cohort.source_manifest(self.common_source)},
            common_identity={"path": str(self.common_identity_path),
                             "sha256": cohort.sha256(self.common_identity_path)},
            resources_manifest={"path": str(self.resources_path),
                                "sha256": cohort.sha256(self.resources_path)},
            rerun_origin={"snapshot_path": str(self.snapshot_path),
                          "snapshot_sha256": cohort.sha256(self.snapshot_path),
                          "input_manifest_sha256": cohort.sha256(self.manifest_path)},
        )

    def write_original(self, report):
        self.original_report_path.write_text(json.dumps(report, indent=2) + "\n")

    def original_patch(self):
        return patch.object(rerun, "origin_state", return_value=copy.deepcopy(self.snapshot))

    def revalidate(self, geometry_effect=None):
        with self.original_patch(), patch.object(rerun, "verify_common_source", return_value={}), \
                patch.object(rerun, "verify_resources", return_value=[]), \
                patch.object(cohort, "validate_anatomy_child", return_value=self.geometry,
                             side_effect=geometry_effect) as validate, \
                patch.object(cohort.subprocess, "run") as run:
            report = rerun.revalidate_worker({"config": self.config, "case": self.case, "version": "baseline"})
        run.assert_not_called()
        return report, validate

    def load(self, config=None, job=None):
        with self.original_patch(), patch.object(rerun, "verify_common_source", return_value={}), \
                patch.object(rerun, "verify_resources", return_value=[]):
            return rerun.load_anatomy(config or self.config, self.case, "baseline", job or self.new_job)


class OriginalAnatomySourceTests(RerunFixture):
    def test_source_comes_from_original_claimed_round_and_not_the_fresh_output_root(self):
        with self.original_patch():
            original_config, job, subject = rerun.original_paths(self.config, self.case)
        self.assertEqual(original_config, self.original_config)
        self.assertEqual(job, self.old_job)
        self.assertEqual(subject, self.subject)
        self.assertFalse(subject.is_relative_to(self.new_root))

    def test_original_exact_serialization_failure_remains_visible(self):
        contents = self.original_report_path.read_bytes()
        with self.original_patch():
            original, path, subject = rerun.validate_original(self.config, self.case)
        self.assertEqual(original["status"], "failed")
        self.assertEqual(original["error"]["message"], SERIALIZATION_ERROR)
        self.assertEqual(path, self.original_report_path)
        self.assertEqual(subject, self.subject)
        self.assertEqual(self.original_report_path.read_bytes(), contents)

    def test_unrelated_reconstruction_failures_do_not_authorize_a_rerun(self):
        for change in ({"exit_code": 1}, {"status": "running"},
                       {"error": {"type": "RuntimeError", "message": "official recon-all exited 1"}},
                       {"error": {"type": "RuntimeError", "message": SERIALIZATION_ERROR.replace("int32", "float32")}}):
            report = copy.deepcopy(self.original_report)
            report.update(change)
            self.write_original(report)
            with self.subTest(change=change), self.original_patch(), self.assertRaises(ValueError):
                rerun.validate_original(self.config, self.case)

    def test_original_subject_cannot_be_replaced_by_an_outside_symlink(self):
        outside = self.root / "different_round_subject"
        self.subject.rename(outside)
        self.subject.symlink_to(outside, target_is_directory=True)
        with self.original_patch(), self.assertRaises(ValueError):
            rerun.validate_original(self.config, self.case)

    def test_original_anatomy_record_cannot_point_to_another_subject(self):
        report = copy.deepcopy(self.original_report)
        report["anatomy"]["mri/brain.mgz"]["path"] = str(self.root / "old_subject" / "mri/brain.mgz")
        self.write_original(report)
        with self.original_patch(), self.assertRaises(ValueError):
            rerun.validate_original(self.config, self.case)

    def test_original_command_cannot_import_an_unrelated_t1(self):
        report = copy.deepcopy(self.original_report)
        report["command"][report["command"].index("-i") + 1] = str(self.root / "another_T1w.nii.gz")
        self.write_original(report)
        with self.original_patch(), self.assertRaises(ValueError):
            rerun.validate_original(self.config, self.case)


class FreshRawDWIRevalidationTests(RerunFixture):
    def test_reread_writes_only_a_new_report_and_preserves_original_attempts(self):
        original_bytes = self.original_report_path.read_bytes()
        (self.old_job / "connectome").mkdir()
        partial = self.old_job / "connectome" / "partial_topup.txt"
        partial.write_bytes(b"Keep this failed attempt for audit.")
        old_gpu = self.old_job / "gpu_report.json"
        old_gpu.write_bytes(b"Keep original GPU failure byte-for-byte.")
        report, validate = self.revalidate()
        self.assertEqual(report["status"], "completed")
        validate.assert_called_once_with(self.subject, self.anatomy, self.config["anatomy_validation_python"])
        self.assertEqual(self.original_report_path.read_bytes(), original_bytes)
        self.assertEqual(partial.read_bytes(), b"Keep this failed attempt for audit.")
        self.assertEqual(old_gpu.read_bytes(), b"Keep original GPU failure byte-for-byte.")
        self.assertEqual(json.loads((self.new_job / "anatomy_origin.json").read_text()), report)
        self.assertFalse((self.new_job / "connectome").exists())
        self.assertFalse((self.new_job / "freesurfer").exists())

    def test_even_an_empty_existing_new_case_directory_is_not_resumed(self):
        self.new_job.mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            self.revalidate()
        self.assertFalse((self.new_job / "anatomy_origin.json").exists())

    def test_an_existing_gpu_output_in_the_new_namespace_is_not_resumed(self):
        for name in ("connectome", "gpu_report.json", "raw_bids_wall.json", "raw_bids_wall.log"):
            self.new_job.mkdir(parents=True)
            path = self.new_job / name
            path.mkdir() if name == "connectome" else path.touch()
            with self.subTest(name=name), self.assertRaises(FileExistsError):
                self.revalidate()
            path.rmdir() if path.is_dir() else path.unlink()
            self.new_job.rmdir()

    def test_changed_raw_t1_and_dwi_are_rejected_before_array_reading(self):
        for path in (self.t1, self.dwi):
            contents = path.read_bytes()
            path.write_bytes(contents + b"changed")
            report, validate = self.revalidate()
            self.assertEqual(report["status"], "failed")
            self.assertIn("raw input hash mismatch", report["error"]["message"])
            validate.assert_not_called()
            path.write_bytes(contents)
            for artifact in self.new_job.iterdir():
                artifact.unlink()
            self.new_job.rmdir()

    def test_changed_official_anatomy_cannot_be_adopted(self):
        (self.subject / "mri/brain.mgz").write_bytes(b"Changed original anatomy.")
        report, validate = self.revalidate()
        self.assertEqual(report["status"], "failed")
        validate.assert_not_called()

    def test_actual_array_validation_error_cannot_complete(self):
        report, validate = self.revalidate(RuntimeError("MGZ geometry is invalid"))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error"]["message"], "MGZ geometry is invalid")
        self.assertEqual(validate.call_count, 1)

    def test_concurrent_anatomy_changes_cannot_complete(self):
        def change_anatomy(*_):
            (self.subject / "mri/brain.mgz").write_bytes(b"Concurrent anatomy change.")
            return self.geometry
        report, _ = self.revalidate(change_anatomy)
        self.assertEqual(report["status"], "failed")

    def test_concurrent_raw_input_changes_cannot_complete(self):
        def change_dwi(*_):
            self.dwi.write_bytes(b"Concurrent raw DWI change.")
            return self.geometry
        report, _ = self.revalidate(change_dwi)
        self.assertEqual(report["status"], "failed")


class LoadedAnatomyGateTests(RerunFixture):
    def test_loading_uses_the_bound_original_subject_and_keeps_original_failure(self):
        report, _ = self.revalidate()
        self.assertEqual(report["status"], "completed")
        original_bytes = self.original_report_path.read_bytes()
        loaded = self.load()
        self.assertEqual(loaded["status"], "completed")
        self.assertEqual(Path(loaded["rerun"]["anatomy_subject_dir"]), self.subject)
        self.assertEqual(loaded["recon_command_seconds"], self.original_report["recon_command_seconds"])
        self.assertEqual(self.original_report_path.read_bytes(), original_bytes)
        self.assertEqual(json.loads(self.original_report_path.read_text())["status"], "failed")

    def test_original_report_change_after_revalidation_is_rejected(self):
        self.revalidate()
        altered = copy.deepcopy(self.original_report)
        altered["worker_wall_seconds"] += 1.
        self.write_original(altered)
        with self.assertRaises(ValueError):
            self.load()

    def test_anatomy_change_after_revalidation_is_rejected(self):
        self.revalidate()
        (self.subject / "mri/brain.mgz").write_bytes(b"Anatomy changed after validation.")
        with self.assertRaises(ValueError):
            self.load()

    def test_raw_dwi_change_after_revalidation_is_rejected(self):
        self.revalidate()
        self.dwi.write_bytes(b"Raw DWI changed after validation.")
        with self.assertRaises(ValueError):
            self.load()

    def test_loading_rejects_a_report_from_another_gpu_namespace(self):
        self.revalidate()
        with self.assertRaises(ValueError):
            self.load(job=self.root / "unrelated_round" / "baseline" / self.case["case_id"])

    def test_failed_revalidation_never_authorizes_gpu(self):
        self.revalidate(RuntimeError("Actual MRI arrays failed validation"))
        with self.assertRaises(ValueError):
            self.load()

    def test_private_cohort_gate_loads_before_looking_for_a_new_recon_report(self):
        self.revalidate()
        self.assertFalse((self.new_job / "recon_report.json").exists())
        with self.original_patch(), patch.object(rerun, "verify_common_source", return_value={}), \
                patch.object(rerun, "verify_resources", return_value=[]):
            loaded = cohort.load_recon_for_gpu(self.config, self.case, "baseline", self.new_job)
        self.assertEqual(Path(loaded["rerun"]["anatomy_subject_dir"]), self.subject)
        self.assertFalse((self.new_job / "recon_report.json").exists())

    def test_absent_private_mode_does_not_authorize_external_original_anatomy(self):
        self.revalidate()
        config = copy.deepcopy(self.config)
        del config["recovery_mode"]
        with self.original_patch(), patch.object(rerun, "verify_resources", return_value=[]), \
                self.assertRaises((RuntimeError, ValueError, FileNotFoundError)):
            cohort.load_recon_for_gpu(config, self.case, "baseline", self.new_job)


class FrozenOriginTests(RerunFixture):
    def test_current_origin_and_only_the_declared_common_compatibility_diff_are_accepted(self):
        self.assertEqual(rerun.origin_state(self.config), self.snapshot)
        identity = rerun.verify_common_source(self.config)
        self.assertEqual(identity["changed_files"], ["src/fnit/flirt/core.py"])

    def test_explicit_mode_is_required_even_with_a_valid_snapshot(self):
        config = copy.deepcopy(self.config)
        del config["recovery_mode"]
        with self.assertRaises(ValueError):
            rerun.origin_state(config)

    def test_snapshot_bytes_cannot_be_replaced(self):
        self.snapshot_path.write_bytes(self.snapshot_path.read_bytes() + b"\n")
        with self.assertRaises(ValueError):
            rerun.origin_state(self.config)

    def test_original_worker_and_wall_are_never_patched_in_place(self):
        for key in ("worker_script", "wall_script"):
            path = Path(self.original_config[key])
            contents = path.read_bytes()
            path.write_bytes(contents + b"changed")
            with self.subTest(key=key), self.assertRaises(ValueError):
                rerun.origin_state(self.config)
            path.write_bytes(contents)

    def test_original_frozen_source_cannot_change(self):
        (self.original_source / "src/fnit/flirt/core.py").write_bytes(b"Altered frozen original source.")
        with self.assertRaises(ValueError):
            rerun.origin_state(self.config)

    def test_original_raw_manifest_cannot_change(self):
        self.manifest_path.write_bytes(self.manifest_path.read_bytes() + b"\n")
        with self.assertRaises(ValueError):
            rerun.origin_state(self.config)

    def test_compatibility_source_cannot_include_an_unrelated_change(self):
        (self.common_source / "src/fnit/cli.py").write_bytes(b"Unrelated numerical or CLI change.")
        self.config["frozen_sources"]["baseline"] = cohort.source_manifest(self.common_source)
        with self.assertRaises(ValueError):
            rerun.verify_common_source(self.config)

    def test_common_baseline_cannot_be_declared_a_performance_optimization(self):
        identity = json.loads(self.common_identity_path.read_text())
        identity["performance_optimization"] = True
        self.common_identity_path.write_text(json.dumps(identity))
        self.config["common_identity"]["sha256"] = cohort.sha256(self.common_identity_path)
        with self.assertRaises(ValueError):
            rerun.verify_common_source(self.config)

    def test_scientific_and_runtime_settings_remain_bound_to_the_original_cohort(self):
        for key, value in (("cpu_threads", 16), ("gpu_cpu_threads", 16),
                           ("atlases", ["schaefer200+tian-s1"]), ("n_seeds", 1)):
            config = copy.deepcopy(self.config)
            config[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                rerun.origin_state(config)

    def test_new_raw_dwi_namespace_cannot_equal_the_original_output_namespace(self):
        config = copy.deepcopy(self.config)
        config["run_root"] = str(self.old_root)
        with self.assertRaises(ValueError):
            rerun.origin_state(config)

    def test_changed_private_helper_bytes_are_rejected_at_gpu_load(self):
        self.revalidate()
        actual_sha256 = cohort.sha256
        def changed_helper_sha256(path):
            if Path(path).resolve() == Path(rerun.__file__).resolve():
                return "f" * 64
            return actual_sha256(path)
        with patch.object(rerun, "verify_resources", return_value=[]), \
                patch.object(cohort, "sha256", side_effect=changed_helper_sha256), \
                self.assertRaises(ValueError):
            rerun.load_anatomy(self.config, self.case, "baseline", self.new_job)


class CommonWallEvaluatorTests(RerunFixture):
    """Evaluator repairs have an explicit scope separate from runtime changes."""

    WALL_SCOPE = (
        "common evaluator: mode-isolated diagnostic export, reserved-memory "
        "budget and allocator-environment reporting"
    )

    def setUp(self):
        super().setUp()
        self.new_wall = self.root / "new_common_evaluator.py"
        self.new_wall.write_bytes(b"Common evaluator identity fixture; never executed.")
        self.new_wall_sha256 = cohort.sha256(self.new_wall)
        self.config.update(
            wall_script=str(self.new_wall), wall_script_sha256=self.new_wall_sha256,
            wall_change={"scope": self.WALL_SCOPE,
                         "original_sha256": self.original_config["wall_script_sha256"],
                         "new_sha256": self.new_wall_sha256},
        )
        self.update_identity("wall_script_sha256", self.new_wall_sha256)

    def update_identity(self, field, value):
        identity = json.loads(self.common_identity_path.read_text())
        identity.pop("wall_script_sha256", None)
        identity.pop("wall_sha256", None)
        if field:
            identity[field] = value
        self.common_identity_path.write_text(json.dumps(identity))
        self.config["common_identity"]["sha256"] = cohort.sha256(self.common_identity_path)

    def test_explicit_common_evaluator_update_accepts_either_supported_identity_key(self):
        for field in ("wall_script_sha256", "wall_sha256"):
            self.update_identity(field, self.new_wall_sha256)
            with self.subTest(field=field):
                result = rerun.verify_common_source(self.config)
            self.assertEqual(result["wall_change"], self.config["wall_change"])
            self.assertEqual(result["changed_files"], ["src/fnit/flirt/core.py"])

    def test_new_evaluator_without_an_explicit_change_declaration_is_rejected(self):
        del self.config["wall_change"]
        with self.assertRaises(ValueError):
            rerun.verify_common_source(self.config)

    def test_missing_or_unapproved_evaluator_scope_is_rejected(self):
        for scope in (None, "", "unrestricted benchmark changes", self.WALL_SCOPE + "; change runtime math"):
            config = copy.deepcopy(self.config)
            if scope is None:
                config["wall_change"].pop("scope")
            else:
                config["wall_change"]["scope"] = scope
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                rerun.verify_common_source(config)

    def test_original_and_new_evaluator_hash_declarations_must_both_match(self):
        for field in ("original_sha256", "new_sha256"):
            config = copy.deepcopy(self.config)
            config["wall_change"][field] = "f" * 64
            with self.subTest(field=field), self.assertRaises(ValueError):
                rerun.verify_common_source(config)

    def test_common_identity_must_explicitly_bind_the_new_evaluator_bytes(self):
        for field, value in ((None, None), ("wall_script_sha256", "f" * 64),
                             ("wall_sha256", "f" * 64)):
            self.update_identity(field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                rerun.verify_common_source(self.config)

    def test_new_evaluator_must_match_its_current_frozen_config_hash(self):
        self.config["wall_script_sha256"] = "f" * 64
        with self.assertRaises(ValueError):
            rerun.verify_common_source(self.config)

    def test_common_update_never_allows_modifying_the_original_evaluator_in_place(self):
        original_wall = Path(self.original_config["wall_script"])
        original_wall.write_bytes(original_wall.read_bytes() + b"changed after freezing")
        with self.assertRaises(ValueError):
            rerun.verify_common_source(self.config)

    def test_current_new_evaluator_bytes_cannot_change_after_authorization(self):
        self.new_wall.write_bytes(self.new_wall.read_bytes() + b"changed after freezing")
        with self.assertRaises(ValueError):
            rerun.verify_common_source(self.config)

    def test_evaluator_exception_does_not_allow_an_unrelated_runtime_source_change(self):
        (self.common_source / "src/fnit/cli.py").write_bytes(b"Unrelated runtime change.")
        self.config["frozen_sources"]["baseline"] = cohort.source_manifest(self.common_source)
        with self.assertRaises(ValueError):
            rerun.verify_common_source(self.config)


class ResourceManifestTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.resource = self.root / "atlas.annot"
        self.resource.write_bytes(b"Atlas resource identity fixture; not anatomical labels.")
        self.manifest_path = self.root / "resources.json"
        self.manifest = {"files": [{"role": "native_atlas", "path": str(self.resource),
                                    "size_bytes": self.resource.stat().st_size,
                                    "sha256": cohort.sha256(self.resource)}]}
        self.config = {}
        self.write_manifest()

    def write_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.config["resources_manifest"] = {"path": str(self.manifest_path),
                                               "sha256": cohort.sha256(self.manifest_path)}

    def verify(self):
        with patch.object(rerun, "required_resource_paths", return_value=[("native_atlas", self.resource)]):
            return rerun.verify_resources(self.config)

    def test_current_required_resource_bytes_are_verified(self):
        self.verify()

    def test_manifest_bytes_cannot_change_after_freezing(self):
        self.manifest_path.write_bytes(self.manifest_path.read_bytes() + b"\n")
        with self.assertRaises(ValueError):
            self.verify()

    def test_manifest_cannot_omit_a_required_file(self):
        self.manifest["files"] = []
        self.write_manifest()
        with self.assertRaises(ValueError):
            self.verify()

    def test_missing_resource_is_rejected(self):
        self.resource.unlink()
        with self.assertRaises((FileNotFoundError, ValueError)):
            self.verify()

    def test_changed_resource_sha_is_rejected_even_if_size_is_unchanged(self):
        contents = self.resource.read_bytes()
        self.resource.write_bytes(b"X" + contents[1:])
        with self.assertRaises(ValueError):
            self.verify()

    def test_declared_wrong_size_is_rejected_even_with_the_correct_sha(self):
        self.manifest["files"][0]["size_bytes"] += 1
        self.write_manifest()
        with self.assertRaises(ValueError):
            self.verify()

    def test_manifest_cannot_substitute_an_unrelated_atlas(self):
        other = self.root / "other_atlas.annot"
        other.write_bytes(self.resource.read_bytes())
        self.manifest["files"][0]["path"] = str(other)
        self.write_manifest()
        with self.assertRaises(ValueError):
            self.verify()

    def test_one_executable_can_have_three_distinct_required_roles(self):
        record = self.manifest["files"][0]
        self.manifest["files"] = [{**record, "role": role} for role in
                                  ("gpu_python", "cpu_python", "anatomy_validation_python")]
        self.write_manifest()
        required = [(role, self.resource) for role in
                    ("gpu_python", "cpu_python", "anatomy_validation_python")]
        with patch.object(rerun, "required_resource_paths", return_value=required):
            rerun.verify_resources(self.config)

    def test_duplicate_records_cannot_cover_one_required_role_twice(self):
        self.manifest["files"].append(copy.deepcopy(self.manifest["files"][0]))
        self.write_manifest()
        with self.assertRaises(ValueError):
            self.verify()


class CrossHostTimingTests(unittest.TestCase):
    @staticmethod
    def record():
        return {
            "original_driver_start_utc": "2026-10-02T13:28:52.183833+00:00",
            "original_recon_start_utc": "2026-10-02T13:28:39.058731+00:00",
            "original_recon_end_utc": "2026-10-02T14:39:23.003982+00:00",
            "original_recon_command_seconds": 4241.2467591241,
            "original_recon_worker_wall_seconds": 4243.918243408203,
            "revalidation_report": {"start_utc": "2026-10-02T15:11:10.726352+00:00",
                                    "end_utc": "2026-10-02T15:11:13.481565+00:00",
                                    "revalidation_wall_seconds": 2.7552409172058105},
        }

    @staticmethod
    def gpu(status="failed"):
        report = {
            "status": status, "start_utc": "2026-10-02T15:22:49.434127+00:00",
            "end_utc": "2026-10-02T15:30:01.133929+00:00",
            "worker_wall_seconds": 431.67106582038105, "gpu_lock_queue_seconds": 410.9364620167762,
            "gpu_command_wall_seconds": 19.750317370984703,
        }
        if status == "failed":
            report["error"] = {"type": "RuntimeError", "message": "raw-BIDS runner exited 1"}
        return report

    def timing(self, record=None, gpu=None, head_end="2026-10-02T15:30:14.133929+00:00", queue=0.):
        return rerun.rerun_timing(record or self.record(), gpu or self.gpu(), head_end, queue)

    def test_measured_cross_host_start_offset_does_not_shorten_official_node_timer(self):
        record = self.record()
        original = copy.deepcopy(record)
        measured = self.timing(record)
        same_node = (datetime.fromisoformat(record["original_recon_end_utc"])
                     - datetime.fromisoformat(record["original_recon_start_utc"])).total_seconds()
        head_interval = (datetime.fromisoformat("2026-10-02T15:30:14.133929+00:00")
                         - datetime.fromisoformat(record["original_driver_start_utc"])).total_seconds()
        self.assertAlmostEqual(measured["original_recon_same_node_elapsed_utc_seconds"], same_node, places=5)
        self.assertAlmostEqual(measured["head_driver_full_elapsed_utc_seconds"], head_interval, places=5)
        self.assertAlmostEqual(measured["driver_start_minus_recon_worker_start_utc_seconds"], 13.125102, places=5)
        self.assertEqual(record, original)
        # The earlier faulty cross-host bound was smaller than this real
        # command duration. No timestamp widening is needed for a valid timer.
        wrong_bound = (datetime.fromisoformat(record["original_recon_end_utc"])
                       - datetime.fromisoformat(record["original_driver_start_utc"])).total_seconds()
        self.assertGreater(record["original_recon_command_seconds"], wrong_bound)

    def test_official_monotonic_timer_must_fit_both_original_node_intervals(self):
        for change in ({"original_recon_command_seconds": 4250.},
                       {"original_recon_worker_wall_seconds": 4200.},
                       {"original_recon_end_utc": "2026-10-02T13:28:00+00:00"}):
            record = self.record()
            record.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.timing(record)

    def test_missing_offset_and_invalid_original_timers_are_rejected(self):
        for field, value in (("original_recon_start_utc", "2026-10-02T13:28:39"),
                             ("original_recon_command_seconds", -1.),
                             ("original_recon_command_seconds", float("nan")),
                             ("original_recon_worker_wall_seconds", float("inf")),
                             ("original_recon_command_seconds", True)):
            record = self.record()
            record[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.timing(record)

    def test_head_end_must_follow_head_start_without_shifting_either_timestamp(self):
        with self.assertRaises(ValueError):
            self.timing(head_end="2026-10-02T12:28:52.183833+00:00")

    def test_collection_retains_actual_gpu_failure_when_timing_also_fails(self):
        record = self.record()
        gpu = self.gpu()
        with patch.object(rerun, "rerun_timing", side_effect=ValueError("independent timer invalid")):
            rerun.collect_gpu_result(record, gpu, 0., "2026-10-02T15:30:14.133929+00:00")
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["gpu_report"], gpu)
        self.assertEqual(record["error"], gpu["error"])
        self.assertEqual(record["timing_error"], {"type": "ValueError", "message": "independent timer invalid"})

    def test_collection_does_not_relabel_a_successful_gpu_run_after_timing_error(self):
        record = self.record()
        gpu = self.gpu("completed")
        with patch.object(rerun, "rerun_timing", side_effect=ValueError("independent timer invalid")):
            rerun.collect_gpu_result(record, gpu, 0., "2026-10-02T15:30:14.133929+00:00")
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["gpu_report"], gpu)
        self.assertEqual(record["timing_error"]["message"], "independent timer invalid")


class SavedTimingCSVTests(unittest.TestCase):
    def test_export_uses_controlled_host_timing_and_keeps_unknown_empty(self):
        import csv
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case_times.csv"
            records = {"baseline/sub-CON03": {"version": "baseline", "case_id": "sub-CON03", "subject": "CON03",
                       "status": "completed", "timing": {"head_driver_full_elapsed_utc_seconds": 12000.5,
                       "original_recon_same_node_elapsed_utc_seconds": 4243.945251},
                       "gpu_report": {"raw_dwi_cli_total_runtime_seconds": 2500., "memory_budget": {"status": "observed_below_budget"}}},
                       "baseline/sub-CON04": {"version": "baseline", "case_id": "sub-CON04", "subject": "CON04", "status": "waiting_original_cpu_report"}}
            rerun.atomic_cases_csv(path, records)
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["head_driver_full_elapsed_utc_seconds"], "12000.5")
            self.assertEqual(rows[0]["original_recon_same_node_elapsed_utc_seconds"], "4243.945251")
            self.assertEqual(rows[0]["raw_dwi_cli_total_runtime_seconds"], "2500.0")
            self.assertEqual(rows[1]["head_driver_full_elapsed_utc_seconds"], "")
            self.assertEqual(rows[1]["memory_budget_status"], "")

    def test_saved_export_preserves_original_state_and_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "status.json"
            state.write_text(json.dumps({"config": {"recovery_mode": rerun.MODE}, "cases": {}}))
            before = state.read_bytes()
            output = root / "new.csv"
            args = ["export-csv", "--status-json", str(state), "--output", str(output)]
            self.assertEqual(rerun.main(args), 0)
            self.assertEqual(state.read_bytes(), before)
            with self.assertRaises(FileExistsError):
                rerun.main(args)


if __name__ == "__main__":
    unittest.main()
