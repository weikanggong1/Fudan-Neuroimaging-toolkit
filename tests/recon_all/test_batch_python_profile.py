"""batch配方预检与真实CLI解析合同；不运行MRI或初始化CUDA。"""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

from fnit.recon_all.batch import run_recon_all_python_batch
from fnit.recon_all.native_free import main
from fnit.recon_all.python_profile import python_profile_options


class BatchPythonProfileTest(unittest.TestCase):
    def _cli_fixture(self, **options):
        cuda_initialized = torch.cuda.is_initialized()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            weights, assets = root / "weights", root / "assets"
            weights.mkdir(); assets.mkdir()
            source = root / "input.nii.gz"
            source.write_bytes(b"CLI fixture only; not an MRI benchmark")
            captured = []
            commands = []

            def pipeline(**kwargs):
                captured.append(kwargs)
                Path(kwargs["subject_dir"]).mkdir()
                return {"status": "complete", "total_seconds": 0.0,
                        "fixture_only": True}

            def child(command, **kwargs):
                commands.append(command)
                with patch("fnit.recon_all.native_free._run_recon_all_python",
                           side_effect=pipeline), redirect_stdout(io.StringIO()):
                    main(command[3:])
                return type("Completed", (), {"returncode": 0})()

            with patch("fnit.recon_all.batch.subprocess.run", side_effect=child):
                reports = run_recon_all_python_batch(
                    jobs=[{"t1": source, "subject_dir": root / "subject"}],
                    weights_dir=weights, assets_dir=assets,
                    devices=("cuda:1",), threads=4, **options)
            self.assertTrue(reports[0]["fixture_only"])
            self.assertEqual(torch.cuda.is_initialized(), cuda_initialized)
            return captured[0], commands[0]

    def _assert_recipe(self, captured, *, n4_backend="simpleitk",
                       n4_execution="in-process", input_execution="serial"):
        expected = python_profile_options(device="cuda:1", threads=4,
            n4_backend=n4_backend, n4_execution=n4_execution, input_execution=input_execution)
        for key, value in expected.items():
            with self.subTest(key=key):
                self.assertEqual(captured.get(key, "in-process"), value)

    def test_default_python_batch_cli_consumes_same_single_recipe(self):
        captured, command = self._cli_fixture(backend="python")
        self._assert_recipe(captured)
        self.assertEqual(command[command.index("--n4-backend") + 1], "simpleitk")

    def test_covered_parameters_resolve_before_preflight(self):
        # 历史batch以native WM/单worker校验，错误拒绝；单例始终先解析配方。
        captured, _ = self._cli_fixture(backend="python", wm_execution="isolated",
            inflate_backend="torch", sphere_finish_backend="torch",
            native_optimizations="original", gca_candidate_chunk=17)
        self._assert_recipe(captured)

    def test_explicit_torch_isolated_n4_survives_batch_and_cli(self):
        captured, _ = self._cli_fixture(backend="python", n4_backend="torch",
                                      n4_execution="isolated")
        self._assert_recipe(captured, n4_backend="torch", n4_execution="isolated")

    def test_parallel_simpleitk_choice_survives_cli(self):
        captured, _ = self._cli_fixture(backend="python", input_execution="parallel")
        self._assert_recipe(captured, input_execution="parallel")
        self.assertEqual(captured["input_execution"], "parallel")

    def test_unsupported_n4_combinations_fail_without_dispatch(self):
        for options in ({"n4_backend": "torch", "input_execution": "parallel"},
                        {"n4_backend": "simpleitk", "n4_execution": "isolated"}):
            with self.subTest(options=options), patch("fnit.recon_all.batch.subprocess.run") as child:
                with self.assertRaises(ValueError):
                    run_recon_all_python_batch([], "unused", "unused",
                        backend="python", devices=("cuda:1",), **options)
                child.assert_not_called()

    def test_native_branch_keeps_original_invalid_combination_checks(self):
        for options in ({"wm_execution": "isolated"}, {"inflate_backend": "torch"}):
            with self.subTest(options=options), patch("fnit.recon_all.batch.subprocess.run") as child:
                with self.assertRaises(ValueError):
                    run_recon_all_python_batch([], "unused", "unused",
                        backend="native", devices=("cuda:1",), **options)
                child.assert_not_called()


if __name__ == "__main__":
    unittest.main()
