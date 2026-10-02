"""Protocol/IO unit tests only: these fixtures are not MRI benchmarks."""
import copy
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from tools import benchmark_connectome_raw_cohort as cohort


def manifest(count=10):
    cases = []
    for index in range(count):
        subject = f"S{index:02d}"
        root = Path("/shared/new-raw-bids")
        t1 = root / f"sub-{subject}/ses-one/anat/sub-{subject}_ses-one_T1w.nii.gz"
        dwi = root / f"sub-{subject}/ses-one/dwi/sub-{subject}_ses-one_dir-AP_dwi.nii.gz"
        cases.append({"case_id": f"C{index:02d}", "subject": subject, "session": "one", "bids_root": str(root), "t1w": str(t1),
                      "input_files": [{"kind": kind, "path": str(path), "sha256": "a" * 64} for kind, path in (
                          ("raw_t1w", t1), ("raw_dwi", dwi), ("bval", dwi.with_name(dwi.name[:-7] + ".bval")),
                          ("bvec", dwi.with_name(dwi.name[:-7] + ".bvec")), ("dwi_json", dwi.with_name(dwi.name[:-7] + ".json")),
                          ("dataset_description", root / "dataset_description.json"))]})
    return {"dataset": "public-test-schema", "snapshot": "1.0.0", "license": "CC0", "source_url": "https://example.org/dataset",
            "download_completed_utc": "2026-10-02T00:00:00+00:00", "downloaded_new": True, "cases": cases}


def config():
    return {"sources": {"baseline": "/shared/frozen baseline", "candidate": "/shared/frozen candidate"},
            "recon_all": "/shared/free'surfer/bin/recon-all", "freesurfer_home": "/shared/free'surfer",
            "cpu_threads": 8, "gpu_cpu_threads": 8, "device": "cuda:0", "n_seeds": 100000, "seed": 0,
            "eddy_gp_seed": 12345, "atlases": ["fs-aparc", "schaefer200+tian-s1"],
            "atlas_options": ["--mni-template", "/shared/MNI template.nii.gz"], "cpu_jobs": 2, "pilot": True,
            "run_root": "/shared/fresh-namespace", "worker_script_sha256": cohort.sha256(cohort.__file__)}


def wall_report():
    return {"mode": "wall", "status": "completed", "exit_code": 0, "total_runtime_seconds": 2.,
            "initial_output_state": {"output_directory_existed": False, "preexisting_state_files": [], "preexisting_run_state": False},
            "preprocessing": [{"topup": "no_reverse_pe", "eddy": "completed", "recon_all": "supplied"}],
            "actual_eddy_gp_seeds": [12345], "official_recon_all_calls": [], "outputs": {"status": "complete"},
            "gpu_process_memory": {"peak_process_tree_bytes": 4000000000},
            "cuda_allocator": {"allocated_bytes": 3000000000, "reserved_bytes": 3500000000}}


class ManifestTests(unittest.TestCase):
    def test_formal_ten_distinct_subjects(self):
        self.assertEqual(len(cohort.validate_manifest(manifest())), 10)

    def test_small_cohort_requires_explicit_pilot(self):
        with self.assertRaisesRegex(ValueError, "at least 10"):
            cohort.validate_manifest(manifest(2))
        self.assertEqual(len(cohort.validate_manifest(manifest(2), pilot=True)), 2)

    def test_duplicate_subject_rejected(self):
        data = manifest()
        data["cases"][1]["subject"] = data["cases"][0]["subject"]
        with self.assertRaises(ValueError):
            cohort.validate_manifest(data)

    def test_download_declaration_and_source_are_required(self):
        for key in ("downloaded_new", "source_url", "license", "snapshot", "download_completed_utc"):
            data = manifest()
            del data[key]
            with self.assertRaises(ValueError):
                cohort.validate_manifest(data)

    def test_incomplete_raw_hash_record_rejected(self):
        data = manifest()
        data["cases"][0]["input_files"][0]["sha256"] = "unknown"
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            cohort.validate_manifest(data)

    def test_processed_t1_and_mismatched_session_rejected(self):
        for name in ("sub-S00_ses-other_T1w.nii.gz", "sub-S00_ses-one_brain.nii.gz"):
            data = manifest()
            case = data["cases"][0]
            path = str(Path(case["bids_root"]) / "sub-S00/ses-other/anat" / name)
            case["t1w"] = case["input_files"][0]["path"] = path
            with self.assertRaises(ValueError):
                cohort.validate_manifest(data)

    def test_formal_subset_cannot_be_reported_as_ten_cases(self):
        with self.assertRaisesRegex(ValueError, "at least 10"):
            cohort.validate_manifest(manifest(), ["C00", "C01"])
        with self.assertRaisesRegex(ValueError, "unknown"):
            cohort.validate_manifest(manifest(), ["MISSING"], pilot=True)


class CommandAndFreshnessTests(unittest.TestCase):
    def test_official_environment_does_not_inherit_another_installation(self):
        cfg = config()
        with patch.dict(cohort.os.environ, {"FREESURFER_HOME": "/other/install", "CUDA_VISIBLE_DEVICES": "1"}):
            env = cohort.recon_environment(cfg, Path("/shared/new/subjects/sub-S00"))
        self.assertEqual(env["FREESURFER_HOME"], cfg["freesurfer_home"])
        self.assertEqual(env["SUBJECTS_DIR"], "/shared/new/subjects")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(env["OMP_NUM_THREADS"], "8")

    def test_source_freeze_includes_scientific_binary_and_label_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            package = source / "src/fnit"
            package.mkdir(parents=True)
            (package / "cli.py").write_text("# source identity fixture\n")
            (package / "nodes.tsv").write_text("index\tlabel\n1\t1001\n")
            resource = package / "sphere.npz"
            resource.write_bytes(b"resource identity fixture")
            before = cohort.source_manifest(source)
            self.assertIn("src/fnit/nodes.tsv", before["source_sha256"])
            self.assertIn("src/fnit/sphere.npz", before["source_sha256"])
            resource.write_bytes(b"changed")
            self.assertNotEqual(before["source_fingerprint"], cohort.source_manifest(source)["source_fingerprint"])

    def test_safe_ssh_argument_roundtrip(self):
        adversarial = "space ' quote; $(touch /tmp/do-not-run) `uname` $SECRET"
        command = ["/shared/python", "worker.py", adversarial]
        actual = cohort.ssh_command("gongwk@nodecw10", 22, "/tmp/socket with space", command)
        self.assertEqual(shlex.split(actual[-1]), command)
        self.assertIn("BatchMode=yes", actual)
        self.assertIn("ControlMaster=no", actual)
        for host in ("-oProxyCommand=evil", "host with space"):
            with self.assertRaises(ValueError):
                cohort.ssh_command(host, None, None, command)

    def test_official_recon_has_raw_import_and_fixed_threads(self):
        case = manifest()["cases"][0]
        case["t1w"] = "/shared/raw image $(do-not-run)' T1w.nii.gz"
        command, launch, anatomy = cohort.recon_command(config(), case, "/shared/new job")
        self.assertEqual(command[command.index("-i") + 1], case["t1w"])
        self.assertEqual(command[-3:], ["-all", "-openmp", "8"])
        self.assertEqual(launch[4:], [config()["freesurfer_home"], *command])
        self.assertEqual(str(anatomy), "/shared/new job/freesurfer/sub-S00_ses-one")
        self.assertNotIn(case["t1w"], launch[2])

    def test_raw_cli_has_no_preprocessed_bypass_or_overwrite(self):
        words = cohort.cli_command(config(), manifest()["cases"][0], "/shared/fresh")
        self.assertEqual(words[0], "UKBConnectome_pipeline")
        for forbidden in ("--corrected-dwi", "--rotated-bvecs", "--fa-map", "--overwrite", "--compile-arc"):
            self.assertNotIn(forbidden, words)
        self.assertEqual(words[words.index("--n-seeds") + 1], "100000")
        self.assertIn("--freesurfer-subject-dir", words)

    def test_atomic_fresh_namespace_rejects_empty_previous_run(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run"
            cohort.require_fresh(path)
            with self.assertRaises(FileExistsError):
                cohort.require_fresh(path)

    def test_existing_worker_report_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = config()
            cfg["run_root"] = directory
            case = manifest()["cases"][0]
            report = Path(directory) / "baseline/C00/recon_report.json"
            report.parent.mkdir(parents=True)
            report.write_text('{"status":"old-result"}\n')
            result = cohort.worker({"action": "recon", "config": cfg, "case": case, "version": "baseline"})
            self.assertEqual(result["status"], "failed")
            self.assertEqual(json.loads(report.read_text())["status"], "old-result")

    def test_input_hash_checks_real_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data"
            path.write_bytes(b"IO unit-test bytes, not MRI data")
            case = {"input_files": [{"path": str(path), "sha256": cohort.sha256(path), "kind": "metadata"}]}
            self.assertEqual(cohort.verify_inputs(case)[0]["size_bytes"], path.stat().st_size)
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                cohort.verify_inputs(case)

    def test_actual_anatomy_child_requires_explicit_success_report(self):
        with patch.object(cohort.subprocess, "run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, '{"status":"actual_images_surfaces_annotations_read"}', "")
            result = cohort.validate_anatomy_child("/shared/fs", {"mri/brain.mgz": {}}, "/shared/conda/python")
            self.assertEqual(result["status"], "actual_images_surfaces_annotations_read")
            self.assertEqual(run.call_args.args[0][0], "/shared/conda/python")
            self.assertEqual(json.loads(run.call_args.kwargs["input"])["files"], ["mri/brain.mgz"])
            run.return_value = subprocess.CompletedProcess([], 1, "", "unreadable real MGZ")
            with self.assertRaisesRegex(RuntimeError, "nibabel anatomy"):
                cohort.validate_anatomy_child("/shared/fs", {}, "/shared/conda/python")

    def test_remote_payload_uses_stdin_not_long_shell_argument(self):
        cfg = config()
        cfg.update(cpu_host="nodecw10", cpu_python="/shared/python", worker_script="/shared/worker.py")
        payload = {"raw_provenance": "x" * 200000}
        with tempfile.TemporaryDirectory() as directory, patch.object(cohort.subprocess, "run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, '{"status":"ok"}', "")
            result = cohort.remote(cfg, "cpu", payload, Path(directory) / "stderr.log")
            words, = run.call_args.args
            self.assertNotIn("x" * 100, words[-1])
            self.assertEqual(json.loads(run.call_args.kwargs["input"]), payload)
            self.assertEqual(result["status"], "ok")


class EntryGateTests(unittest.TestCase):
    @staticmethod
    def arguments():
        return ["run", "--manifest", "/shared/manifest.json", "--run-root", "/shared/new-run",
                "--report-dir", "/shared/new-reports", "--baseline-source", "/shared/baseline",
                "--cpu-python", "/usr/bin/python3", "--gpu-python", "/shared/conda/python",
                "--worker-script", "/shared/worker.py", "--wall-script", "/shared/wall.py",
                "--freesurfer-home", "/shared/freesurfer", "--recon-all", "/shared/freesurfer/bin/recon-all"]

    def test_baseline_only_can_start_without_candidate_ready(self):
        options = cohort.parse_options(self.arguments() + ["--versions", "baseline"])
        self.assertEqual(options.versions, ["baseline"])
        self.assertFalse(options.candidate_ready)
        self.assertIsNone(options.candidate_source)

    def test_candidate_is_gated_until_explicitly_ready(self):
        with patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit):
            cohort.parse_options(self.arguments() + ["--candidate-source", "/shared/candidate"])
        options = cohort.parse_options(self.arguments() + ["--candidate-source", "/shared/candidate", "--candidate-ready"])
        self.assertTrue(options.candidate_ready)


class ReportTests(unittest.TestCase):
    def test_incomplete_memory_sampling_cannot_pass_budget(self):
        for fields in ({"failed_samples": 1}, {"errors": ["query failed"]},
                       {"unresolved_device_samples": 2},
                       {"sample_interval_seconds": .5, "max_observed_interval_seconds": 9.}):
            report = wall_report()
            report["gpu_process_memory"].update(fields)
            self.assertEqual(cohort.memory_budget(report)["status"], "not_fully_measured")

    def test_fresh_seeded_wall_report_is_accepted(self):
        cohort.check_wall_report(wall_report(), config(), manifest()["cases"][0])

    def test_reuse_wrong_seed_wrong_mode_and_missing_outputs_fail(self):
        reports = []
        report = wall_report(); report["mode"] = "diagnostic"; reports.append(report)
        report = wall_report(); report["initial_output_state"]["output_directory_existed"] = True; reports.append(report)
        report = wall_report(); report["preprocessing"][0]["eddy"] = "skipped"; reports.append(report)
        report = wall_report(); report["actual_eddy_gp_seeds"] = [42]; reports.append(report)
        report = wall_report(); report["preprocessing"][0]["recon_all"] = "skipped"; reports.append(report)
        report = wall_report(); report["total_runtime_seconds"] = None; reports.append(report)
        report = wall_report(); report["outputs"]["status"] = "incomplete"; reports.append(report)
        for report in reports:
            with self.subTest(report=report), self.assertRaises(RuntimeError):
                cohort.check_wall_report(report, config(), manifest()["cases"][0])

    def test_reverse_pe_requires_actual_completed_topup(self):
        case = manifest()["cases"][0]
        case["input_files"].append({"kind": "reverse_pe"})
        with self.assertRaisesRegex(RuntimeError, "TOPUP"):
            cohort.check_wall_report(wall_report(), config(), case)
        report = wall_report()
        report["preprocessing"][0]["topup"] = "completed"
        cohort.check_wall_report(report, config(), case)

    def test_actual_cli_primary_acquisition_and_json_hashes_must_match(self):
        case = manifest()["cases"][0]
        names = {"raw_dwi": "raw/image", "bval": "raw/bval", "bvec": "raw/bvec", "dataset_description": "raw/dataset_description"}
        inputs = {}
        for item in case["input_files"]:
            if item["kind"] in names:
                inputs[names[item["kind"]]] = {"path": item["path"], "sha256": item["sha256"]}
            elif item["kind"] == "dwi_json":
                inputs["raw_metadata/sidecar.json"] = {"path": item["path"], "sha256": item["sha256"]}
        cohort.check_selected_inputs({"inputs": inputs}, case)
        inputs["raw/image"]["path"] = "/shared/other-subject_dwi.nii.gz"
        with self.assertRaisesRegex(RuntimeError, "selection/hash"):
            cohort.check_selected_inputs({"inputs": inputs}, case)

    def test_twenty_decimal_gb_is_strict_and_missing_memory_is_unknown(self):
        self.assertEqual(cohort.memory_budget(wall_report())["status"], "observed_below_budget")
        report = wall_report()
        report["cuda_allocator"]["reserved_bytes"] = 20_000_000_000
        self.assertEqual(cohort.memory_budget(report)["status"], "exceeded")
        report = wall_report()
        report["gpu_process_memory"] = {}
        self.assertEqual(cohort.memory_budget(report)["status"], "not_fully_measured")
        self.assertEqual(cohort.memory_budget({})["status"], "not_fully_measured")

    def test_parent_wall_subtracts_only_measured_gpu_queues(self):
        result = cohort.parent_timing(10, 40, 4, 6, cpu_driver_queue=100)
        self.assertEqual(result["parent_full_wall_seconds"], 30)
        self.assertEqual(result["parent_full_wall_excluding_gpu_queue_seconds"], 20)
        self.assertEqual(result["cpu_driver_queue_seconds"], 100)
        with self.assertRaises(ValueError):
            cohort.parent_timing(10, 12, 2, 3)

    def test_missing_anatomy_and_cli_outputs_cannot_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                cohort.check_anatomy(directory, ["fs-aparc"])
            with self.assertRaises(RuntimeError):
                cohort.check_outputs(directory, ["fs-aparc"], False)

    def test_self_connections_are_retained_and_asymmetry_is_rejected(self):
        # CSV protocol fixture, not a neuroimaging or accuracy benchmark.
        with tempfile.TemporaryDirectory() as directory:
            for path in cohort.expected_outputs(directory, ["fs-aparc"], False):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture\n")
            atlas = Path(directory) / "atlases/fs-aparc"
            (atlas / "nodes.tsv").write_text("index\tname\n1\tleft\n2\tright\n")
            (atlas / "region_labels.csv").write_text("1\n2\n")
            for name in cohort.MATRICES:
                (atlas / f"connectome_{name}.csv").write_text("2,1\n1,3\n")
            result = cohort.check_outputs(directory, ["fs-aparc"], False)
            self.assertEqual(result["atlas_node_counts"], {"fs-aparc": 2})
            self.assertIn("retained", result["self_connection_policy"])
            (atlas / "connectome_count.csv").write_text("2,1\n0,3\n")
            with self.assertRaisesRegex(RuntimeError, "asymmetric"):
                cohort.check_outputs(directory, ["fs-aparc"], False)

    def test_case_completion_is_saved_before_other_recon_finishes(self):
        # Only scheduling/report protocol is mocked, never neuroimaging computations.
        data = manifest(2)
        cfg = config(); cfg["sources"] = {"baseline": "/shared/baseline"}
        first_finished, release_second = threading.Event(), threading.Event()
        errors = []
        def fake_remote(config, host, payload, log):
            if payload["action"] == "claim":
                return {"frozen_sources": {}, "wall_script_sha256": "b" * 64}
            if payload["action"] == "recon":
                if payload["case"]["case_id"] == "C01":
                    release_second.wait(5)
                return {"status": "completed", "recon_command_seconds": 1.}
            if payload["case"]["case_id"] == "C00":
                first_finished.set()
            return {"status": "completed", "gpu_lock_queue_seconds": 0., "raw_dwi_cli_total_runtime_seconds": 1.}
        with tempfile.TemporaryDirectory() as directory, patch.object(cohort, "remote", fake_remote), patch("sys.stdout", new_callable=io.StringIO):
            report_dir = Path(directory) / "reports"
            def launch():
                try:
                    cohort.run_cohort(cfg, data, data["cases"], report_dir)
                except Exception as error:
                    errors.append(error)
            thread = threading.Thread(target=launch)
            thread.start()
            self.assertTrue(first_finished.wait(5))
            # Wait for the thread's atomic write, not an MRI timing measurement.
            for _ in range(1000):
                state = json.loads((report_dir / "status.json").read_text())
                if state["cases"]["baseline/C00"]["status"] == "completed":
                    break
                threading.Event().wait(0.001)
            self.assertEqual(state["cases"]["baseline/C00"]["status"], "completed")
            self.assertEqual(state["cases"]["baseline/C01"]["status"], "recon_running")
            release_second.set()
            thread.join(5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            saved = json.loads((report_dir / "status.json").read_text())
            self.assertEqual(saved["scope"], "diagnostic_subset")
            self.assertFalse(saved["comparison_ready"])
            self.assertEqual(saved["scientific_parity"], "not_assessed")


if __name__ == "__main__":
    unittest.main()
