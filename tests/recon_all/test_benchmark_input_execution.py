"""真实包装器选路及失败边界合同；不运行MRI、GPU或外部程序。"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


class FinishedFailedChild:
    """仅模拟子进程失败，以核对实际命令/报告而不伪造完成产物。"""
    pid = 123456
    returncode = 7

    def poll(self):
        return self.returncode


class NoDeviceSampler:
    def __init__(self, *, device, parent_pid, interval):
        self.device = device

    def report(self):
        return {"own_tree_peak_bytes": None, "scope": "contract fixture only"}


class BenchmarkInputExecutionTest(unittest.TestCase):
    @staticmethod
    def module():
        tool = Path(__file__).resolve().parents[2] / "tools/benchmark_recon_torch_end_to_end.py"
        spec = importlib.util.spec_from_file_location("fnit_input_harness_contract", tool)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def arguments(module, root, *, execution, n4="simpleitk", threads=4):
        return [module.__file__, "--t1", str(root / "input.contract"),
                "--output-root", str(root / "new-run"),
                "--weights-dir", str(root / "weights"),
                "--assets-dir", str(root / "assets"),
                "--backend", "python", "--device", "cuda:1",
                "--threads", str(threads), "--n4-backend", n4,
                "--input-execution", execution, "--code-version", "contract-only"]

    def test_requested_mode_reaches_real_command_and_report(self):
        module = self.module()
        for execution in ("serial", "parallel"):
            with self.subTest(execution=execution), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                # 仅供SHA预检的字节fixture，子进程不启动，不读取影像。
                (root / "input.contract").write_bytes(b"not an MRI")
                with patch.object(sys, "argv", self.arguments(module, root, execution=execution)), \
                     patch("fnit.recon_all.input_n4_chain.require_n4_runtime_dependencies",
                           return_value={"scope": "dependency fixture"}), \
                     patch.object(module.subprocess, "Popen", return_value=FinishedFailedChild()) as launch, \
                     patch.object(module, "ProcessTreeDeviceSampler", NoDeviceSampler):
                    self.assertEqual(module.main(), 7)
                command = launch.call_args.args[0]
                self.assertEqual(command[command.index("--input-execution") + 1], execution)
                report = json.loads((root / "new-run/benchmark.json").read_text())
                self.assertEqual(report["input_execution"], execution)
                self.assertEqual(report["execution"], "failed")
                self.assertEqual(report["output_completeness"], "not_assessed")
                self.assertFalse(report["execution_audit"]["passed"])

    def test_parallel_torch_is_rejected_before_dependency_or_output(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(sys, "argv", self.arguments(module, root, execution="parallel", n4="torch")), \
                 patch("fnit.recon_all.input_n4_chain.require_n4_runtime_dependencies") as dependency, \
                 patch.object(module.subprocess, "Popen") as launch, \
                 self.assertRaisesRegex(ValueError, "parallel Python input requires"):
                module.main()
            dependency.assert_not_called()
            launch.assert_not_called()
            self.assertFalse((root / "new-run").exists())

    def test_invalid_parallel_budget_is_not_silently_serialized(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(sys, "argv", self.arguments(module, root, execution="parallel", threads=6)), \
                 patch("fnit.recon_all.input_n4_chain.require_n4_runtime_dependencies") as dependency, \
                 patch.object(module.subprocess, "Popen") as launch, \
                 self.assertRaisesRegex(ValueError, "total threads=4"):
                module.main()
            dependency.assert_not_called()
            launch.assert_not_called()
            self.assertFalse((root / "new-run").exists())
