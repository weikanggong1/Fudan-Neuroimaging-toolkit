"""轻量编排契约测试；替身子进程不是影像或性能 benchmark。"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[2] / "tools/benchmark_recon_native_speed_pair.py"
SPEC = importlib.util.spec_from_file_location("fnit_native_speed_pair", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class NativeSpeedPairTest(unittest.TestCase):
    def _inputs(self, root):
        cases = []
        for index in range(2):
            path = root / f"raw-{index}.nii.gz"
            path.write_bytes(f"unit-test input contract {index}".encode())
            cases.append({"case": f"case{index}", "t1": str(path),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        (root / "cases.json").write_text(json.dumps(cases))
        for arm in ("baseline", "candidate"):
            (root / arm / "tools").mkdir(parents=True)
            (root / arm / "src/fnit/recon_all").mkdir(parents=True)
            for relative in (MODULE.SUPERVISOR, MODULE.BENCHMARK, "src/fnit/recon_all/example.py"):
                (root / arm / relative).write_text(f"# frozen {arm}\n")
        for name in ("weights", "assets", "bin", "deps"):
            (root / name).mkdir()
        (root / "deps.json").write_text(json.dumps([str(root / "deps")]))
        (root / "common.json").write_text(json.dumps(["--hemisphere-execution", "groups"]))
        return cases

    def _run(self, root):
        return MODULE.run_pair(
            cases_json=root / "cases.json", output_root=root / "new-pair",
            baseline_source=root / "baseline", baseline_version="baseline-commit",
            candidate_source=root / "candidate", candidate_version="candidate-commit",
            weights_dir=root / "weights", assets_dir=root / "assets", native_bin_dir=root / "bin",
            dependency_pythonpath_json=root / "deps.json", common_arguments_json=root / "common.json",
            python_interpreter=Path(sys.executable))

    def _process(self, calls, *, fail_at=None, mutate_source=None):
        class Process:
            def __init__(self, command, **kwargs):
                index = len(calls)
                env = kwargs["env"]
                plan = Path(command[command.index("--cases-json") + 1])
                case = json.loads(plan.read_text())[0]
                output = Path(command[command.index("--output-root") + 1])
                if output.exists():
                    raise AssertionError("each one-case supervisor needs a fresh root")
                calls.append({"command": command, "env": env, "case": case})
                for key in ("NUMBA_CACHE_DIR", "TRITON_CACHE_DIR"):
                    directory = Path(env[key])
                    marker = directory / "cache-state-unit-test"
                    seen_arm = sum(1 for call in calls[:-1] if call["env"][key] == str(directory))
                    if marker.exists() != bool(seen_arm):
                        raise AssertionError("cache must start empty per arm and persist across cases")
                    marker.write_text("compiled unit-test marker")
                output.mkdir(parents=True)
                benchmark_dir = output / case["case"]
                benchmark_dir.mkdir()
                (benchmark_dir / "benchmark.json").write_text(json.dumps({
                    "execution": "complete", "output_completeness": "complete", "grid_quality": "passed",
                    "cli_wall_seconds": 10. + index, "full_harness_seconds": 11. + index,
                    "process_memory": {"ownership": "not_assessed"}}))
                self.exit_code = 9 if fail_at == index else 0
                (output / "cases.private.json").write_text(json.dumps({
                    "status": "complete" if not self.exit_code else "completed_with_failures",
                    "cases": [{"case": case["case"], "input_sha256": case["sha256"],
                               "outer_case_wall_seconds": 12. + index, "exit_code": self.exit_code}]}))
                if mutate_source is not None and index == 0:
                    mutate_source.write_text("# source changed during run\n")

            def wait(self):
                return self.exit_code
        return Process

    def test_case_block_order_two_source_paths_and_fresh_per_arm_caches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._inputs(root)
            calls = []
            inherited = {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "1",
                         "CUDA_VISIBLE_DEVICES": "GPU-explicit-target",
                         "FS_LICENSE": "/private/path/not-for-reports", "PYTHONPATH": "/old/untrusted/source"}
            with patch.dict(os.environ, inherited), patch.object(MODULE.subprocess, "Popen", self._process(calls)):
                report = self._run(root)
            self.assertEqual(report["status"], "complete")
            self.assertEqual([(row["case"], row["arm"]) for row in report["runs"]],
                             [("case0", "baseline"), ("case0", "candidate"),
                              ("case1", "candidate"), ("case1", "baseline")])
            for row, call in zip(report["runs"], calls):
                command, env = call["command"], call["env"]
                expected_source = root / row["arm"]
                self.assertEqual(env["PYTHONPATH"], os.pathsep.join(map(str,
                    (root / "deps", expected_source / "src", expected_source / "tools"))))
                self.assertEqual(env["OMP_NUM_THREADS"], "4")
                self.assertEqual(env["MKL_NUM_THREADS"], "1")
                extras = command[command.index("--") + 1:]
                self.assertEqual(extras[:2], ["--hemisphere-execution", "groups"])
                self.assertEqual("--sphere-normals-backend" in extras, row["arm"] == "candidate")
                self.assertEqual(row["benchmark_cli_wall_seconds"], 10. + report["runs"].index(row))
                self.assertGreaterEqual(row["outer_supervisor_wall_seconds"], 0.)
                self.assertTrue(Path(row["benchmark_path"]).is_file())
            saved = (root / "new-pair/speed_pair.private.json").read_text()
            self.assertNotIn(inherited["FS_LICENSE"], saved)
            self.assertNotIn("/old/untrusted/source", saved)
            self.assertTrue(all(report["integrity_after_runs"]["source_sha256_unchanged"].values()))

    def test_failed_child_preserved_no_retry_and_final_noncomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._inputs(root)
            calls = []
            with patch.object(MODULE.subprocess, "Popen", self._process(calls, fail_at=1)):
                report = self._run(root)
            self.assertEqual(len(calls), 4)
            self.assertEqual(report["status"], "completed_with_failures")
            self.assertEqual([row["exit_code"] for row in report["runs"]], [0, 9, 0, 0])
            self.assertEqual(report["runs"][1]["status"], "failed")
            self.assertTrue(Path(report["runs"][1]["supervisor_report"]).is_file())

    def test_source_mutation_is_not_accepted_as_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._inputs(root)
            calls = []
            changed = root / "candidate/src/fnit/recon_all/example.py"
            with patch.object(MODULE.subprocess, "Popen", self._process(calls, mutate_source=changed)):
                report = self._run(root)
            self.assertEqual(report["status"], "completed_with_failures")
            self.assertFalse(report["integrity_after_runs"]["source_sha256_unchanged"]["candidate"])

    def test_all_inputs_validate_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = self._inputs(root)
            cases[1]["sha256"] = "0" * 64
            (root / "cases.json").write_text(json.dumps(cases))
            with patch.object(MODULE.subprocess, "Popen") as process:
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    self._run(root)
                process.assert_not_called()
            self.assertFalse((root / "new-pair").exists())

    def test_common_profile_override_and_existing_output_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._inputs(root)
            (root / "common.json").write_text(json.dumps(["--sphere-normals-backend=torch"]))
            with self.assertRaisesRegex(ValueError, "cannot override"):
                self._run(root)
            self.assertFalse((root / "new-pair").exists())
            (root / "new-pair").mkdir()
            with self.assertRaises(FileExistsError):
                self._run(root)


if __name__ == "__main__":
    unittest.main()
