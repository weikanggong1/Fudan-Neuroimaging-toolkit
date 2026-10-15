"""真实benchmark参数解析；在缺失T1预检处停止，不冒充影像验证。"""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from fnit.recon_all.python_profile import python_profile_options


class BenchmarkGeneticChoiceTest(unittest.TestCase):
    def test_explicit_round2_profile_reaches_python_recipe(self):
        tool = Path(__file__).resolve().parents[2] / "tools/benchmark_recon_torch_end_to_end.py"
        spec = importlib.util.spec_from_file_location("fnit_e2e_round2_contract", tool)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            command = [str(tool), "--t1", str(root / "missing.nii.gz"),
                "--output-root", str(root / "new-run"),
                "--weights-dir", str(root / "weights"),
                "--assets-dir", str(root / "assets"),
                "--backend", "python", "--device", "cuda:0", "--threads", "4",
                "--topology-backend", "python-ga",
                "--hemisphere-execution", "persistent",
                "--gca-reuse-atlas-context", "--gca-candidate-chunk", "4096",
                "--topology-distance-backend", "gpu_defect_signed_distance",
                "--topology-state-backend", "packed", "--code-version", "contract-only"]
            with patch.object(sys, "argv", command), \
                    patch("fnit.recon_all.python_profile.python_profile_options",
                          wraps=python_profile_options) as resolve, \
                    self.assertRaises(FileNotFoundError):
                module.main()
            settings = resolve.call_args.kwargs
            self.assertEqual(settings["hemisphere_execution"], "persistent")
            self.assertTrue(settings["gca_reuse_atlas_context"])
            self.assertEqual(settings["gca_candidate_chunk"], 4096)
            self.assertEqual(settings["topology_distance_backend"], "gpu_defect_signed_distance")
            self.assertEqual(settings["topology_state_backend"], "packed")
            self.assertFalse((root / "new-run").exists())

    def test_actual_main_preserves_genetic_choice_and_default_digital_recipe(self):
        tool = Path(__file__).resolve().parents[2] / "tools/benchmark_recon_torch_end_to_end.py"
        spec = importlib.util.spec_from_file_location("fnit_e2e_choice_contract", tool)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for choice, expected in (("python-ga", "python-ga"), ("python", "python"), ("native", "python")):
            with self.subTest(choice=choice), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                command = [str(tool), "--t1", str(root / "missing.nii.gz"),
                    "--output-root", str(root / "new-run"),
                    "--weights-dir", str(root / "weights"),
                    "--assets-dir", str(root / "assets"),
                    "--backend", "python", "--device", "cuda:1", "--threads", "4",
                    "--topology-backend", choice, "--code-version", "contract-only"]
                with patch.object(sys, "argv", command), \
                        patch("fnit.recon_all.python_profile.python_profile_options",
                              wraps=python_profile_options) as resolve, \
                        self.assertRaises(FileNotFoundError):
                    module.main()
                self.assertEqual(resolve.call_args.kwargs["topology_backend"], expected)
                self.assertEqual(resolve.call_args.kwargs["n4_backend"], "simpleitk")
                self.assertEqual(resolve.call_args.kwargs["gca_candidate_chunk"], 1024)
                self.assertFalse((root / "new-run").exists())


if __name__ == "__main__":
    unittest.main()
