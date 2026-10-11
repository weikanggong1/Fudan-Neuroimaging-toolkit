"""纯Python配方的N4选择契约；不替代真实N4或整例benchmark。"""
import os
import unittest
from unittest.mock import patch

import torch

from fnit.recon_all.input_n4_chain import run_n4_stage
from fnit.recon_all.python_profile import python_profile_options


class PythonProfileN4SelectionTest(unittest.TestCase):
    def test_default_retains_complete_simpleitk_recipe(self):
        default = python_profile_options(device="cuda:1", threads=4)
        explicit = python_profile_options(device="cuda:1", threads=4,
                                          n4_backend="simpleitk", n4_execution="in-process")
        self.assertEqual(default, explicit)
        self.assertEqual(len(default), 22)
        self.assertEqual(default['input_execution'], 'serial')
        self.assertEqual(default["n4_backend"], "simpleitk")
        self.assertEqual(default["n4_execution"], "in-process")

    def test_opt_in_changes_only_two_n4_keys(self):
        baseline = python_profile_options(device="cuda:1", threads=4)
        for execution in ("in-process", "isolated"):
            with self.subTest(execution=execution):
                candidate = python_profile_options(device="cuda:1", threads=4,
                    n4_backend="torch", n4_execution=execution)
                expected = dict(baseline, n4_backend="torch", n4_execution=execution)
                self.assertEqual(candidate, expected)
                self.assertEqual(set(candidate), set(baseline))

    def test_native_and_invalid_backends_are_rejected(self):
        for backend in ("native", "blur", "SimpleITK", None, True, ["torch"]):
            with self.subTest(backend=backend), self.assertRaisesRegex(ValueError, "N4 backend"):
                python_profile_options(n4_backend=backend)

    def test_invalid_execution_is_rejected(self):
        for execution in ("parallel", "native", None, False, ["isolated"]):
            with self.subTest(execution=execution), self.assertRaisesRegex(ValueError, "N4 execution"):
                python_profile_options(n4_execution=execution)
        with self.assertRaisesRegex(ValueError, "isolated N4 requires"):
            python_profile_options(n4_backend="simpleitk", n4_execution="isolated")

    def test_selector_does_not_initialize_cuda_or_mutate_precision_environment(self):
        environment = dict(os.environ)
        initialized = torch.cuda.is_initialized()
        precision = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        with patch.object(torch.cuda, "init", side_effect=AssertionError("selector initialized CUDA")), \
             patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("selector initialized CUDA")):
            for backend, execution in (("simpleitk", "in-process"), ("torch", "in-process"), ("torch", "isolated")):
                python_profile_options(device="cuda:7", threads=2,
                                       n4_backend=backend, n4_execution=execution)
        self.assertEqual(dict(os.environ), environment)
        self.assertEqual(torch.cuda.is_initialized(), initialized)
        self.assertEqual((torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32), precision)

    def test_selected_simpleitk_enters_existing_library_route(self):
        options = python_profile_options()
        # 路由合同使用明确的依赖fixture；真实SimpleITK版本/库哈希另作现场验证。
        with patch("fnit.recon_all.input_n4_chain.require_n4_runtime_dependencies",
                   return_value={"backend": "simpleitk", "required": True}) as dependency, \
             patch("fnit.recon_all.input_n4_chain.correct_volume", return_value={"total_seconds": 1.0}) as simpleitk, \
             patch("fnit.recon_all.n4_itk_torch_experimental.correct_volume") as candidate:
            result = run_n4_stage(input_path="orig.mgz", output_path="nu0.mgz", device="cuda:0",
                threads=4, n4_backend=options["n4_backend"], n4_execution=options["n4_execution"])
        dependency.assert_called_once_with(n4_backend="simpleitk")
        simpleitk.assert_called_once_with(input_file="orig.mgz", output_file="nu0.mgz",
            backend="simpleitk", reconstruction_threads=4, profile_path=None)
        candidate.assert_not_called()
        self.assertFalse(result["used_native_programs"])

    def test_selected_torch_enters_existing_full_in_process_route(self):
        options = python_profile_options(n4_backend="torch")
        with patch("fnit.recon_all.n4_itk_torch_experimental.correct_volume", return_value={"iterations": 200}) as candidate, \
             patch("fnit.recon_all.input_n4_chain.correct_volume") as other:
            result = run_n4_stage(input_path="orig.mgz", output_path="nu0.mgz", device="cuda:3", threads=4,
                n4_backend=options["n4_backend"], n4_execution=options["n4_execution"], profile=True)
        candidate.assert_called_once_with(input_path="orig.mgz", output_path="nu0.mgz", device="cuda:3", profile=True)
        other.assert_not_called()
        self.assertEqual(result["iterations"], 200)

    def test_selected_torch_enters_existing_full_isolated_route(self):
        options = python_profile_options(n4_backend="torch", n4_execution="isolated")
        with patch("fnit.recon_all.n4_torch_worker.run_isolated_n4", return_value={"iterations": 200}) as worker, \
             patch("fnit.recon_all.input_n4_chain.correct_volume") as other:
            result = run_n4_stage(input_path="orig.mgz", output_path="nu0.mgz", device="cuda:3", threads=2,
                n4_backend=options["n4_backend"], n4_execution=options["n4_execution"], report_path="worker.json")
        worker.assert_called_once_with(input_path="orig.mgz", output_path="nu0.mgz",
            report_path="worker.json", device="cuda:3", threads=2, profile=False)
        other.assert_not_called()
        self.assertEqual(result["iterations"], 200)

    def test_parallel_input_rejects_torch_without_silent_switch(self):
        from fnit.recon_all.native_free import _validate_input_execution
        for execution in ("in-process", "isolated"):
            with self.subTest(execution=execution), self.assertRaisesRegex(ValueError, "parallel input requires simpleitk"):
                _validate_input_execution("parallel", n4_backend="torch",
                    n4_execution=execution, device="cuda:0", threads=4)


if __name__ == "__main__":
    unittest.main()
