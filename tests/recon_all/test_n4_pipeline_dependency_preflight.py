"""缺失已声明N4依赖必须在模型、分配器和输出前失败。"""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from fnit.recon_all.native_free import _run_recon_all_python


class N4PipelineDependencyPreflightTest(unittest.TestCase):
    def test_missing_simpleitk_fails_before_allocator_program_or_subject(self):
        with tempfile.TemporaryDirectory() as folder:
            subject = Path(folder) / "new-subject"
            with patch("fnit.recon_all.input_n4_chain.require_n4_runtime_dependencies",
                       side_effect=ModuleNotFoundError("SimpleITK is required")) as dependency, \
                    patch("fnit.recon_all.profiling.configure_cuda_allocator") as allocator, \
                    patch("fnit.recon_all.native_free._native_bin_directory") as programs, \
                    patch("fnit.recon_all.input_talairach_chain.run_input_talairach_chain") as input_stage, \
                    self.assertRaises(ModuleNotFoundError):
                _run_recon_all_python(t1="missing-input", subject_dir=subject,
                    weights_dir="missing-weights", assets_dir="missing-assets",
                    device="cuda:0", threads=4, backend="native", n4_backend="simpleitk")
            dependency.assert_called_once_with(n4_backend="simpleitk")
            allocator.assert_not_called()
            programs.assert_not_called()
            input_stage.assert_not_called()
            self.assertFalse(subject.exists())

    def test_benchmark_missing_dependency_fails_before_output_or_child(self):
        tool = Path(__file__).resolve().parents[2] / "tools/benchmark_recon_torch_end_to_end.py"
        spec = importlib.util.spec_from_file_location("fnit_n4_e2e_preflight", tool)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "new-run"
            command = [str(tool), "--t1", "missing-input", "--output-root", str(output),
                "--weights-dir", "missing-weights", "--assets-dir", "missing-assets",
                "--backend", "python", "--topology-backend", "python-ga",
                "--device", "cuda:0", "--threads", "4", "--code-version", "contract-only"]
            with patch.object(sys, "argv", command), \
                    patch("fnit.recon_all.input_n4_chain.require_n4_runtime_dependencies",
                          side_effect=ModuleNotFoundError("SimpleITK is required")) as dependency, \
                    patch.object(module.subprocess, "Popen") as child, \
                    self.assertRaises(ModuleNotFoundError):
                module.main()
            dependency.assert_called_once_with(n4_backend="simpleitk")
            child.assert_not_called()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
