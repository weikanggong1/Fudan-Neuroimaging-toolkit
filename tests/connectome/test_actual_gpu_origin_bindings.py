"""Stdlib origin fixtures only; no MRI, performance or recovery run is faked."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/reference"))
import benchmark_connectome_cohort_compare as compare
import connectome_actual_gpu_origins as origins


class ActualOriginTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup); self.root = Path(self.temp.name)
        self.options = argparse.Namespace(baseline_root=self.root / "old-baseline", candidate_root=self.root / "old-candidate",
            baseline_anatomy_root=self.root / "old-anatomy", baseline_driver=self.root / "old-baseline-driver/status.json",
            candidate_driver=self.root / "old-candidate-driver/status.json")
        self.cases = [{"case_id": f"sub-{index:02d}", "subject": str(index), "t1w": str(self.root / f"raw/{index}/T1w.nii.gz"),
            "input_files": [{"kind": "raw_t1w", "path": str(self.root / f"raw/{index}/T1w.nii.gz"), "sha256": "0" * 64}]} for index in range(10)]
        self.manifest = {"dataset": "metadata-fixture-only", "snapshot": "fixture", "license": "fixture", "cases": self.cases}
        self.source_root = self.root / "actual-fixture-source"
        for name in ("src/fnit/cli.py", "src/fnit/__init__.py", "environment.yml", "pyproject.toml"):
            path = self.source_root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("fixture-only " + name)
        hashes = {str(path.relative_to(self.source_root)): compare.anatomy.sha(path) for path in self.source_root.rglob("*") if path.is_file()}
        self.source = {"directory": str(self.source_root), "source_sha256": hashes,
                       "source_fingerprint": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}
        self.wall_script = self.root / "wall.py"; self.wall_script.write_text("fixture-only")
        self.original_config = {"run_root": str(self.options.candidate_root), "frozen_sources": {"candidate": self.source},
            "gpu_python": sys.executable, "wall_script": str(self.wall_script), "wall_script_sha256": compare.anatomy.sha(self.wall_script),
            "atlases": ["fixture-atlas"], "n_seeds": 100000, "seed": 0, "eddy_gp_seed": 12345, "device": "cuda:0",
            "atlas_options": [], "gpu_cpu_threads": 8, "cuda_visible_devices": "1",
            "gpu_lock": "/tmp/fixture-only-global.lock", "gpu_host": "fixture-host"}
        self.original_GPU = {"status": "completed", "exit_code": 0, "case_id": "sub-00", "version": "candidate",
            "raw_input_provenance": self.cases[0]["input_files"], "source_before": self.source, "source_after": self.source,
            "identity": {"python": sys.executable, "python_executable_sha256": compare.anatomy.sha(sys.executable), "python_version": sys.version, "hostname": "fixture-host"},
            "input_verification": [{**self.cases[0]["input_files"][0], "actual_sha256": "0" * 64}],
            "input_verification_after": [{**self.cases[0]["input_files"][0], "actual_sha256": "0" * 64}],
            "gpu_memory": {"process": {"backend": "nvidia-smi", "failed_samples": 1, "unresolved_device_samples": 0, "peak_process_tree_bytes": 19_000_000_000,
                "errors": ["TimeoutExpired: Command '['nvidia-smi', 'fixture']' timed out after 3 seconds"]},
                "allocator": {"allocated_bytes": 14_000_000_000, "reserved_bytes": 17_000_000_000}},
            "memory_budget": {"limit_bytes": 20_000_000_000, "status": "not_fully_measured", "monitor_issues": ["failed_samples", "errors"],
                "measurements": {"allocated_bytes": 14_000_000_000, "reserved_bytes": 17_000_000_000, "process_tree": 19_000_000_000}}}
        self.subject_dir = self.root / "actual-fixture-fresh-FS"
        self.original_wall = {"status": "completed", "exit_code": 0, "outputs": {"status": "complete", "files": {}}, "total_runtime_seconds": 1.,
            "initial_output_state": {"output_directory_existed": False, "preexisting_run_state": False, "preexisting_state_files": []},
            "preprocessing": [{"topup": "completed", "eddy": "completed", "recon_all": "supplied"}],
            "actual_eddy_gp_seeds": [12345],
            "cli_arguments": ["UKBConnectome_pipeline", "--n-seeds", "100000", "--output-dir", str(self.options.candidate_root / "candidate/sub-00/connectome")],
            "inputs": {"raw/image": {"sha256": "0" * 64}, "prepared/dwi": {"path": "old-actual-output"}}, "provenance": {},
            "selected_inputs": {"freesurfer_subject_dir": str(self.subject_dir)}}
        old_job = self.options.candidate_root / "candidate/sub-00"
        old_wall = self.write(old_job / "raw_bids_wall.json", self.original_wall)
        self.original_GPU["wall_report"] = str(old_wall)
        old_GPU = self.write(old_job / "gpu_report.json", self.original_GPU)
        old_config = self.write(self.options.candidate_root / "staged_gpu_config.json", self.original_config)
        old_driver = self.write(self.root / "preserved-driver-snapshot.json", {"config": self.original_config,
            "cases": {"candidate/sub-00": {"status": "failed_gpu_eligibility", "gpu_report": self.original_GPU}}})
        self.new_root = self.root / "new-actual-run"; self.write(self.new_root / "input_manifest.json", self.manifest)
        self.new_config = copy.deepcopy(self.original_config); self.new_config["run_root"] = str(self.new_root)
        new_config = self.write(self.new_root / "staged_gpu_config.json", self.new_config)
        modules = {}
        for name in ("torch", "torch._C", "numpy", "numpy.core._multiarray_umath", "nibabel"):
            path = self.root / "original-env-module" / name; path.parent.mkdir(exist_ok=True); path.write_text("fixture " + name)
            record = {"path": str(path), "sha256": compare.anatomy.sha(path), "version": "fixture-only"}
            modules[name] = record
        nvml_module = self.root / "actual-fixture-pynvml.py"; nvml_module.write_text("fixture-only NVML module")
        runtime = {"gpu_python": sys.executable, "python_binary": str(Path(sys.executable).resolve()),
            "python_binary_sha256": compare.anatomy.sha(sys.executable), "python_version": sys.version, "modules": modules, "CUDA_initialized": False}
        self.preflight = {"scope": "optional_NVML_monitor_environment", "gpu_python": sys.executable, "science_modules_unchanged": True,
            "CUDA_initialized": False, "original_runtime": runtime, "new_runtime": copy.deepcopy(runtime),
            "NVML": {"package": "nvidia-ml-py", "CUDA_initialized_before": False, "CUDA_initialized_after": False,
                "module_path": str(nvml_module), "module_sha256": compare.anatomy.sha(nvml_module),
                "original_wall_helper": {"path": str(self.wall_script), "sha256": compare.anatomy.sha(self.wall_script)}}}
        preflight = self.write(self.root / "new-env-preflight.json", self.preflight)
        self.declaration = {"arm": "candidate", "case_id": "sub-00", "reason": "monitor_incomplete", "original": {
            "GPU_report": self.identity(old_GPU), "wall_report": self.identity(old_wall), "configuration": self.identity(old_config), "driver_snapshot": self.identity(old_driver)},
            "replacement": {"root": str(self.new_root), "driver_status": str(self.root / "new-driver/status.json"),
                "configuration": self.identity(new_config), "runtime_preflight": self.identity(preflight)}}
        self.value = {"schema_version": 1, "scope": "explicit_actual_GPU_monitor_recovery", "bindings": [self.declaration]}
        self.path = self.write(self.root / "GPU-bindings.json", self.value)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value)); return path

    def identity(self, path): return {"path": str(path), "sha256": compare.anatomy.sha(path)}

    def load(self):
        self.write(self.path, self.value)
        return origins.load_bindings(self.path, self.cases, self.options)

    def test_known_failure_explicit_case_mapping(self):
        mapping, _ = self.load(); self.assertEqual(set(mapping), {("candidate", "sub-00")})
        self.assertEqual(origins.selected_origin(self.options, "candidate", "sub-00", mapping)[:2],
            (self.new_root, Path(self.declaration["replacement"]["driver_status"])))
        self.assertEqual(origins.selected_origin(self.options, "candidate", "sub-01", mapping)[:2],
            (self.options.candidate_root, self.options.candidate_driver))

    def test_unknown_wrong_error_is_rejected(self):
        value = copy.deepcopy(self.original_GPU); value["gpu_memory"]["process"]["errors"] = ["CUDA OOM"]
        with self.assertRaises(ValueError): origins.incomplete_monitor(value)

    def test_original_measured_over_budget_not_monitor_failure(self):
        value = copy.deepcopy(self.original_GPU); value["memory_budget"]["measurements"]["reserved_bytes"] = 20_000_000_000
        with self.assertRaises(ValueError): origins.incomplete_monitor(value)

    def test_original_case_reports_or_rawSHA_change_refused(self):
        path = Path(self.declaration["original"]["GPU_report"]["path"]); self.write(path, {})
        with self.assertRaises(ValueError): self.load()

    def test_duplicate_or_unknown_case_refused(self):
        self.value["bindings"].append(self.declaration)
        with self.assertRaises(ValueError): self.load()
        self.value["bindings"] = [copy.deepcopy(self.declaration)]; self.value["bindings"][0]["case_id"] = "not-in-manifest"
        with self.assertRaises(ValueError): self.load()

    def test_shared_or_old_namespace_refused(self):
        self.declaration["replacement"]["root"] = str(self.options.candidate_root)
        with self.assertRaises(ValueError): self.load()

    def test_new_frozen_config_cannot_change_science(self):
        self.new_config["n_seeds"] = 90000
        path = self.write(Path(self.declaration["replacement"]["configuration"]["path"]), self.new_config)
        self.declaration["replacement"]["configuration"] = self.identity(path)
        with self.assertRaises(ValueError): self.load()

    def test_replacement_canonical_raw_manifest_change_refused(self):
        value = copy.deepcopy(self.manifest); value["cases"][0]["input_files"][0]["sha256"] = "1" * 64
        self.write(self.new_root / "input_manifest.json", value)
        with self.assertRaises(ValueError): self.load()

    def test_preflight_cannot_claim_module_equality_after_file_change(self):
        Path(self.preflight["original_runtime"]["modules"]["torch"]["path"]).write_text("changed")
        with self.assertRaises(ValueError): self.load()

    def test_preflight_has_actual_module_paths_and_SHA(self):
        self.preflight["original_runtime"].pop("modules")
        path = self.write(Path(self.declaration["replacement"]["runtime_preflight"]["path"]), self.preflight)
        self.declaration["replacement"]["runtime_preflight"] = self.identity(path)
        with self.assertRaises(ValueError): self.load()

    def replacement(self):
        mapping, _ = self.load(); binding = mapping["candidate", "sub-00"]
        GPU = copy.deepcopy(self.original_GPU); GPU["gpu_memory"]["process"]["backend"] = "pynvml"
        GPU["gpu_memory"]["process"].update(status="measured", failed_samples=0, errors=[], samples=10, sample_interval_seconds=.5, max_observed_interval_seconds=.5)
        GPU["memory_budget"].update(status="observed_below_budget", monitor_issues=[])
        wall = copy.deepcopy(self.original_wall); wall["cli_arguments"][-1] = str(self.new_root / "candidate/sub-00/connectome")
        wall["inputs"]["prepared/dwi"]["path"] = "new-actual-output"
        wall["selected_inputs"].update(dwi=str(self.new_root / "candidate/sub-00/connectome/preproc/eddy/data.nii.gz"),
            bvecs=str(self.new_root / "candidate/sub-00/connectome/preproc/eddy/data.eddy_rotated_bvecs"))
        return binding, GPU, wall

    def test_monitor_runtime_is_explicit_and_original_time_retained(self):
        binding, GPU, wall = self.replacement()
        result = origins.verify_replacement(binding, GPU, wall, self.cases[0], self.subject_dir)
        self.assertEqual(result["original_CLI_seconds"], 1.)
        self.assertEqual(result["original_memory_budget"]["status"], "not_fully_measured")

    def test_replacement_actual_CLI_input_or_FS_change_refused(self):
        binding, GPU, wall = self.replacement(); wall["cli_arguments"][2] = "50000"
        with self.assertRaises(ValueError): origins.verify_replacement(binding, GPU, wall, self.cases[0], self.subject_dir)
        binding, GPU, wall = self.replacement(); wall["inputs"]["raw/image"]["sha256"] = "1" * 64
        with self.assertRaises(ValueError): origins.verify_replacement(binding, GPU, wall, self.cases[0], self.subject_dir)
        binding, GPU, wall = self.replacement(); wall["selected_inputs"]["freesurfer_subject_dir"] = str(self.root / "other-FS")
        with self.assertRaises(ValueError): origins.verify_replacement(binding, GPU, wall, self.cases[0], self.subject_dir)

    def test_fallback_monitor_cannot_qualify_replacement(self):
        binding, GPU, wall = self.replacement(); GPU["gpu_memory"]["process"]["backend"] = "nvidia-smi"
        with self.assertRaises(ValueError): origins.verify_replacement(binding, GPU, wall, self.cases[0], self.subject_dir)

    def test_replacement_monitor_failures_and_exact20GB_still_fail(self):
        binding, GPU, wall = self.replacement(); GPU["gpu_memory"]["process"]["failed_samples"] = 1
        with self.assertRaises(ValueError): origins.verify_replacement(binding, GPU, wall, self.cases[0], self.subject_dir)
        binding, GPU, wall = self.replacement(); GPU["gpu_memory"]["allocator"]["reserved_bytes"] = 20_000_000_000
        GPU["memory_budget"]["measurements"]["reserved_bytes"] = 20_000_000_000
        with self.assertRaises(ValueError): origins.verify_replacement(binding, GPU, wall, self.cases[0], self.subject_dir)

    def undispatched(self):
        self.declaration["reason"] = "original_not_dispatched"
        for key in ("GPU_report", "wall_report"):
            Path(self.declaration["original"][key]["path"]).unlink(); self.declaration["original"][key] = None
        path = Path(self.declaration["original"]["driver_snapshot"]["path"])
        self.write(path, {"config": self.original_config, "end_utc": "2026-10-02T19:00:00+00:00", "dispatch_paused": True,
            "status": "failed_or_incomplete_staged_raw_cohort", "cases": {"candidate/sub-00": {"status": "waiting_original_preparation_report"}}})
        self.declaration["original"]["driver_snapshot"] = self.identity(path)

    def test_original_unstarted_case_requires_stopped_actual_driver_and_absence(self):
        self.undispatched(); mapping, _ = self.load()
        self.assertIsNone(mapping["candidate", "sub-00"]["original_GPU"])
        (self.options.candidate_root / "candidate/sub-00/connectome").mkdir()
        with self.assertRaises(ValueError): self.load()

    def test_original_still_queued_or_driver_running_cannot_be_replaced(self):
        self.undispatched(); path = Path(self.declaration["original"]["driver_snapshot"]["path"])
        value = json.loads(path.read_text()); value["cases"]["candidate/sub-00"]["status"] = "gpu_queued"
        self.write(path, value); self.declaration["original"]["driver_snapshot"] = self.identity(path)
        with self.assertRaises(ValueError): self.load()

    def test_unstarted_case_has_no_invented_original_CLI_timer(self):
        self.undispatched(); mapping, _ = self.load(); binding = mapping["candidate", "sub-00"]
        GPU = copy.deepcopy(self.original_GPU); GPU["gpu_memory"]["process"].update(backend="pynvml", status="measured", errors=[], failed_samples=0,
            samples=10, sample_interval_seconds=.5, max_observed_interval_seconds=.5)
        GPU["memory_budget"].update(status="observed_below_budget", monitor_issues=[])
        wall = copy.deepcopy(self.original_wall)
        wall["cli_arguments"] = ["UKBConnectome_pipeline", "--n-seeds", "100000", "--seed", "0", "--device", "cuda:0", "--atlas", "fixture-atlas",
            "--output-dir", str(self.new_root / "candidate/sub-00/connectome")]
        wall["selected_inputs"].update(dwi=str(self.new_root / "candidate/sub-00/connectome/preproc/eddy/data.nii.gz"),
            bvecs=str(self.new_root / "candidate/sub-00/connectome/preproc/eddy/data.eddy_rotated_bvecs"))
        wall["provenance"]["torch_version"] = "fixture-only"
        actual = origins.verify_replacement(binding, GPU, wall, self.cases[0], self.subject_dir)
        self.assertIsNone(actual["original_CLI_seconds"]); self.assertIsNone(actual["original_memory_budget"])

    def queued_stop(self):
        self.declaration["reason"] = "original_queue_stopped_before_compute"
        Path(self.declaration["original"]["wall_report"]["path"]).unlink(); self.declaration["original"]["wall_report"] = None
        path = Path(self.declaration["original"]["GPU_report"]["path"])
        self.stopped_GPU = {key: self.original_GPU[key] for key in ("case_id", "version", "identity", "raw_input_provenance", "source_before")}
        self.stopped_GPU.update(status="failed", error={"type": "RuntimeError", "message": "STOP_DISPATCH prevents starting the queued raw-DWI computation"},
            worker_wall_seconds=8., gpu_lock_queue_seconds=7.)
        self.write(path, self.stopped_GPU); self.declaration["original"]["GPU_report"] = self.identity(path)
        snapshot = Path(self.declaration["original"]["driver_snapshot"]["path"])
        self.write(snapshot, {"config": self.original_config, "end_utc": "2026-10-02T19:00:00+00:00", "dispatch_paused": True,
            "status": "failed_or_incomplete_staged_raw_cohort", "cases": {"candidate/sub-00": {"status": "failed", "gpu_report": self.stopped_GPU}}})
        self.declaration["original"]["driver_snapshot"] = self.identity(snapshot)

    def test_queue_stop_preserves_failed_worker_report(self):
        self.queued_stop(); mapping, _ = self.load()
        self.assertEqual(mapping["candidate", "sub-00"]["original_GPU"]["worker_wall_seconds"], 8.)
        self.assertIsNone(mapping["candidate", "sub-00"]["original_wall"])

    def test_actual_queue_stop_driver_failed_gpu_execution_is_explicit(self):
        self.queued_stop(); path = Path(self.declaration["original"]["driver_snapshot"]["path"])
        state = json.loads(path.read_text()); state["cases"]["candidate/sub-00"]["status"] = "failed_gpu_execution"
        self.write(path, state); self.declaration["original"]["driver_snapshot"] = self.identity(path)
        mapping, _ = self.load(); self.assertEqual(mapping["candidate", "sub-00"]["original_GPU"]["error"], self.stopped_GPU["error"])

    def test_queue_stop_other_error_or_actual_science_start_refused(self):
        self.queued_stop(); path = Path(self.declaration["original"]["GPU_report"]["path"])
        for update in ({"error": {"type": "RuntimeError", "message": "other failure"}}, {"command": ["actual-CLI-was-started"]}):
            value = {**self.stopped_GPU, **update}; self.write(path, value); self.declaration["original"]["GPU_report"] = self.identity(path)
            snapshot = Path(self.declaration["original"]["driver_snapshot"]["path"]); driver = json.loads(snapshot.read_text())
            driver["cases"]["candidate/sub-00"]["gpu_report"] = value; self.write(snapshot, driver); self.declaration["original"]["driver_snapshot"] = self.identity(snapshot)
            with self.assertRaises(ValueError): self.load()

    def test_queue_stop_cannot_hide_saved_wall_or_images(self):
        self.queued_stop(); self.write(self.options.candidate_root / "candidate/sub-00/raw_bids_wall.json", {"status": "actually started"})
        with self.assertRaises(ValueError): self.load()


if __name__ == "__main__": unittest.main()
