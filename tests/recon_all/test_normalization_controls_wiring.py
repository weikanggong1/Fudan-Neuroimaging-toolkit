"""归一化邻域候选的公开入口契约；真实文件与逐轮回归另有冻结报告。"""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fnit.recon_all import native_free
from fnit.recon_all.batch import run_recon_all_python_batch


class NormalizationControlsWiringTest(unittest.TestCase):
    def test_invalid_controls_fail_before_thread_setup_or_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for backend, device in (("invalid", "cuda:0"), ("torch", "cpu"),
                                    ("torch", "cuda"), ("torch", "cuda:-1")):
                subject = root / (backend + "_" + device.replace(":", "_"))
                with self.subTest(backend=backend, device=device), \
                        patch("fnit.recon_all.thread_budget.thread_budget") as budget:
                    with self.assertRaises((ValueError, RuntimeError)):
                        native_free.run_recon_all_python(
                            t1="raw.nii.gz", subject_dir=subject, weights_dir="weights",
                            assets_dir="assets", device=device,
                            normalization_controls_backend=backend)
                    budget.assert_not_called()
                    self.assertFalse(subject.exists())

    def test_public_api_forwards_explicit_backend_and_retains_default(self):
        with tempfile.TemporaryDirectory() as directory:
            subject = Path(directory) / "subject"
            for backend in ("cpu", "torch"):
                with patch("fnit.recon_all.native_free._run_recon_all_python",
                           return_value={"total_seconds": 0.}) as run:
                    native_free.run_recon_all_python(
                        t1="raw.nii.gz", subject_dir=subject, weights_dir="weights",
                        assets_dir="assets", device="cuda:1",
                        normalization_controls_backend=backend)
                self.assertEqual(run.call_args.kwargs["device"], "cuda:1")
                self.assertEqual(run.call_args.kwargs.get("normalization_controls_backend", "cpu"), backend)
                self.assertFalse(subject.exists())

    def test_cli_forwards_nondefault_backend_without_changing_device(self):
        with patch("fnit.recon_all.native_free.run_recon_all_python",
                   return_value={"status": "complete"}) as run, \
                contextlib.redirect_stdout(io.StringIO()):
            native_free.main(["raw.nii.gz", "subject", "--weights-dir", "weights",
                "--assets-dir", "assets", "--device", "cuda:1",
                "--normalization-controls-backend", "torch"])
        self.assertEqual(run.call_args.kwargs["normalization_controls_backend"], "torch")
        self.assertEqual(run.call_args.kwargs["device"], "cuda:1")

    def test_batch_gpu_backend_reaches_cli_and_cpu_is_rejected_before_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights, assets = root / "weights", root / "assets"
            weights.mkdir(); assets.mkdir()
            raw = root / "raw.nii.gz"; raw.write_bytes(b"interface contract")
            subject = root / "subject"

            def launch(command, **kwargs):
                subject.mkdir()
                (subject / "fnit-native-free-run.json").write_text(json.dumps({"status": "complete"}))
                return SimpleNamespace(returncode=0)

            with patch("fnit.recon_all.batch.subprocess.run", side_effect=launch) as run:
                run_recon_all_python_batch(
                    jobs=[{"t1": raw, "subject_dir": subject}], weights_dir=weights,
                    assets_dir=assets, devices=("cuda:1",), normalization_controls_backend="torch")
                command = run.call_args.args[0]
                self.assertEqual(command[command.index("--normalization-controls-backend") + 1], "torch")
                self.assertEqual(command[command.index("--device") + 1], "cuda:1")
                with self.assertRaisesRegex(ValueError, "explicit cuda:N"):
                    run_recon_all_python_batch(
                        jobs=[], weights_dir=weights, assets_dir=assets, devices=("cpu",),
                        normalization_controls_backend="torch")
                self.assertEqual(run.call_count, 1)

    def test_benchmark_guard_and_command_record_match(self):
        script = Path(__file__).resolve().parents[2] / "tools/benchmark_recon_torch_end_to_end.py"
        spec = importlib.util.spec_from_file_location("normalization_benchmark_contract", script)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("weights", "assets", "native"):
                (root / name).mkdir()
            raw = root / "raw.nii.gz"; raw.write_bytes(b"interface contract")
            output = root / "output"
            arguments = [str(script), "--t1", str(raw), "--output-root", str(output),
                "--weights-dir", str(root / "weights"), "--assets-dir", str(root / "assets"),
                "--native-bin-dir", str(root / "native"), "--code-version", "contract",
                "--normalization-controls-backend", "torch"]
            with patch.object(sys, "argv", arguments + ["--device", "cpu"]):
                with self.assertRaisesRegex(ValueError, "explicit cuda:N"):
                    module.main()
            self.assertFalse(output.exists())

            class Sampler:
                def __init__(self, **kwargs): pass
                def report(self): return {}

            with patch.object(sys, "argv", arguments + ["--device", "cuda:1"]), \
                    patch.object(module, "ProcessTreeDeviceSampler", Sampler), \
                    patch.object(module, "sha", return_value="contract-hash"), \
                    patch.object(module.subprocess, "Popen",
                        return_value=SimpleNamespace(pid=123, returncode=0, poll=lambda: 0)) as launch:
                self.assertEqual(module.main(), 0)
            command = launch.call_args.args[0]
            self.assertEqual(command[command.index("--normalization-controls-backend") + 1], "torch")
            report = json.loads((output / "benchmark.json").read_text())
            self.assertEqual(report["normalization_controls_backend"], "torch")
            self.assertEqual(report["device"], "cuda:1")


if __name__ == "__main__":
    unittest.main()
