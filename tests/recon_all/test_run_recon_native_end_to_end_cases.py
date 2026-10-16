"""原生阶段整例监督脚本的空目录、哈希、配方和报告边界测试。"""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[2] / "tools/run_recon_native_end_to_end_cases.py"
SPEC = importlib.util.spec_from_file_location("fnit_native_cases_supervisor", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class NativeCasesSupervisorTest(unittest.TestCase):
    def _inputs(self, root: Path):
        t1a, t1b = root / "public-a.nii.gz", root / "public-b.nii.gz"
        t1a.write_bytes(b"public real-input contract A")
        t1b.write_bytes(b"public real-input contract B")
        cases = [{"case": identifier, "t1": str(path),
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                 for identifier, path in (("a", t1a), ("b", t1b))]
        script = root / "benchmark.py"
        script.write_text("# mocked benchmark\n")
        license_file = root / "private.license"
        license_file.write_text("test placeholder, not a real license\n")
        for name in ("weights", "assets", "bin"):
            (root / name).mkdir()
        return cases, script, license_file

    def _run(self, root: Path, cases: list[dict], script: Path, **kwargs):
        return MODULE.run_cases(
            cases=cases, output_root=root / "new-run", benchmark_script=script,
            weights_dir=root / "weights", assets_dir=root / "assets",
            native_bin_dir=root / "bin", code_version="tested-commit",
            python_interpreter="/declared/conda/bin/python", **kwargs)

    def test_native_recipe_sequential_exit_and_private_license(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases, script, license_file = self._inputs(root)
            calls = []

            class Process:
                def __init__(self, command, **kwargs):
                    calls.append(command)
                    case_dir = Path(command[command.index("--output-root") + 1])
                    case_dir.mkdir()
                    (case_dir / "benchmark.json").write_text(json.dumps({
                        "execution": "complete" if len(calls) == 1 else "failed",
                        "output_completeness": "complete" if len(calls) == 1 else "not_assessed",
                        "grid_quality": "passed" if len(calls) == 1 else "failed",
                        "full_harness_seconds": 1.5,
                    }))

                def wait(self):
                    return 0 if len(calls) == 1 else 5

            with patch.dict(os.environ, {"FS_LICENSE": str(license_file),
                                      "FNIT_RECON_FORBID_ORIGINAL_STAGES": "0"}), \
                 patch.object(MODULE.subprocess, "Popen", Process):
                result = self._run(root, cases, script,
                                   benchmark_arguments=("--hemisphere-execution", "persistent"))
            self.assertEqual(result["status"], "completed_with_failures")
            self.assertEqual([row["exit_code"] for row in result["cases"]], [0, 5])
            self.assertTrue(all(row["outer_case_wall_seconds"] >= 0 for row in result["cases"]))
            self.assertEqual(result["recipe"], {
                "backend": "native", "topology_backend": "native",
                "topology_finish_backend": "conda-cpp",
                "surface_placement_backend": "native",
                "n4_backend": "simpleitk", "n4_execution": "in-process",
            })
            for command in calls:
                self.assertEqual(command[command.index("--backend") + 1], "native")
                self.assertEqual(command[command.index("--topology-backend") + 1], "native")
                self.assertEqual(command[command.index("--topology-finish-backend") + 1], "conda-cpp")
                self.assertEqual(command[command.index("--surface-placement-backend") + 1], "native")
                self.assertEqual(command[command.index("--native-bin-dir") + 1], str(root / "bin"))
                self.assertEqual(command[command.index("--device") + 1], "cuda:0")
                self.assertEqual(command[command.index("--threads") + 1], "4")
            saved = (root / "new-run/cases.private.json").read_text()
            self.assertNotIn(str(license_file), saved)
            self.assertNotIn("FS_LICENSE", saved)

    def test_input_hash_mismatch_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases, script, license_file = self._inputs(root)
            cases[1]["sha256"] = "0" * 64
            with patch.dict(os.environ, {"FS_LICENSE": str(license_file),
                                      "FNIT_RECON_FORBID_ORIGINAL_STAGES": "0"}):
                with self.assertRaisesRegex(ValueError, "raw T1 hash mismatch"):
                    self._run(root, cases, script)
            self.assertFalse((root / "new-run").exists())

    def test_existing_output_and_unsafe_profile_fail_without_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases, script, license_file = self._inputs(root)
            (root / "new-run").mkdir()
            with patch.dict(os.environ, {"FS_LICENSE": str(license_file)}, clear=True):
                with self.assertRaises(FileExistsError):
                    self._run(root, cases, script)
            (root / "new-run").rmdir()
            with patch.dict(os.environ, {"FS_LICENSE": str(license_file),
                                      "FNIT_RECON_FORBID_ORIGINAL_STAGES": "1"}):
                with self.assertRaisesRegex(RuntimeError, "Python-only"):
                    self._run(root, cases, script)
            with patch.dict(os.environ, {"FS_LICENSE": str(license_file),
                                      "FNIT_RECON_FORBID_ORIGINAL_STAGES": "0"}):
                with self.assertRaisesRegex(ValueError, "cannot override"):
                    self._run(root, cases, script,
                              benchmark_arguments=("--topology-finish-backend=python",))
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "FS_LICENSE"):
                    self._run(root, cases, script)
            self.assertFalse((root / "new-run").exists())


if __name__ == "__main__":
    unittest.main()
