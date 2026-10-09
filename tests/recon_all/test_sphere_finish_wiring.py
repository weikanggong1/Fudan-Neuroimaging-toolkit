"""既有完整dense GPU球面收尾的入口/worker契约；真实链单列验证。"""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fnit.recon_all import native_free
from fnit.recon_all.batch import run_recon_all_python_batch


class SphereFinishWiringTest(unittest.TestCase):
    def test_invalid_finish_fails_before_setup_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            for backend, device, workers in (("bad", "cuda:0", 2), ("torch", "cpu", 2),
                                             ("torch", "cuda", 2), ("torch", "cuda:0", 1)):
                subject = Path(directory) / "subject"
                with self.subTest(backend=backend, device=device, workers=workers), \
                        patch("fnit.recon_all.thread_budget.thread_budget") as budget:
                    with self.assertRaises(ValueError):
                        native_free.run_recon_all_python(t1="raw", subject_dir=subject,
                            weights_dir="weights", assets_dir="assets", device=device,
                            hemisphere_workers=workers, sphere_finish_backend=backend)
                    budget.assert_not_called(); self.assertFalse(subject.exists())
        native_free._validate_sphere_finish_backend("cpu", "cpu", 1)

    def test_leaf_reuses_full_standard_sphere_only_selecting_finish_device(self):
        with patch("fnit.recon_all.native_free._run_native_sphere_step", return_value=1.) as inflate, \
                patch("fnit.recon_all.sphere_standard_run.run_standard_sphere",
                      return_value={"total_seconds_including_io": 2.}) as sphere:
            times, report = native_free._run_accurate_sphere_pair(
                inflate_binary=Path("inflate"), subject=Path("subject"), hemi="lh",
                assets=Path("assets"), device="cuda:1", sphere_finish_backend="torch")
        inflate.assert_called_once()
        self.assertEqual(sphere.call_args.kwargs, {"finish_device": "cuda:1", "averaging_device": "cuda:1"})
        self.assertEqual(times, {"inflate": 1., "sphere": 2.})
        self.assertEqual(report["inflate_runtime"]["backend"], "native")

    def test_exec_operation_forwards_finish_to_surface_leaf(self):
        with patch("fnit.recon_all.native_free._surface_pair", return_value={}) as leaf:
            native_free._hemisphere_operation(subject="subject", hemi="lh", device="cuda:1",
                threads=2, operation="surface", assets="assets", sphere_finish_backend="torch",
                binaries={name: "bin/" + name for name in ("topology", "inflate", "intersection", "metrics")})
        self.assertEqual(leaf.call_args.kwargs["sphere_finish_backend"], "torch")
        self.assertTrue(leaf.call_args.kwargs["defer_defects"])

    def test_api_and_cli_forward_explicit_finish(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("fnit.recon_all.native_free._run_recon_all_python", return_value={}) as run:
                native_free.run_recon_all_python(t1="raw", subject_dir=Path(directory),
                    weights_dir="weights", assets_dir="assets", device="cuda:1",
                    hemisphere_workers=2, sphere_finish_backend="torch")
            self.assertEqual(run.call_args.kwargs["sphere_finish_backend"], "torch")
        with patch("fnit.recon_all.native_free.run_recon_all_python", return_value={}) as run, \
                contextlib.redirect_stdout(io.StringIO()):
            native_free.main(["raw", "subject", "--weights-dir", "weights", "--assets-dir", "assets",
                "--sphere-finish-backend", "torch", "--device", "cuda:1", "--hemisphere-workers", "2"])
        self.assertEqual(run.call_args.kwargs["sphere_finish_backend"], "torch")

    def test_batch_forwards_finish_and_correct_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("weights", "assets"):
                (root / name).mkdir()
            raw, subject = root / "raw", root / "subject"; raw.write_bytes(b"entry contract")
            def launch(command, **kwargs):
                subject.mkdir()
                (subject / "fnit-native-free-run.json").write_text(json.dumps({"status": "complete"}))
                return SimpleNamespace(returncode=0)
            launch = Mock(side_effect=launch)
            with patch("fnit.recon_all.batch.subprocess", SimpleNamespace(run=launch)):
                run_recon_all_python_batch(jobs=[{"t1": raw, "subject_dir": subject}],
                    weights_dir=root / "weights", assets_dir=root / "assets", devices=("cuda:1",),
                    hemisphere_workers=2, sphere_finish_backend="torch")
            command = launch.call_args.args[0]
            self.assertEqual(command[command.index("--sphere-finish-backend") + 1], "torch")
            self.assertEqual(command[command.index("--device") + 1], "cuda:1")

    def test_benchmark_guards_and_records_actual_finish(self):
        script = Path(__file__).resolve().parents[2] / "tools/benchmark_recon_torch_end_to_end.py"
        spec = importlib.util.spec_from_file_location("sphere_finish_contract", script)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("weights", "assets", "native"):
                (root / name).mkdir()
            raw = root / "raw"; raw.write_bytes(b"entry contract"); output = root / "output"
            args = [str(script), "--t1", str(raw), "--output-root", str(output),
                "--weights-dir", str(root / "weights"), "--assets-dir", str(root / "assets"),
                "--native-bin-dir", str(root / "native"), "--code-version", "entry-contract",
                "--sphere-finish-backend", "torch"]
            with patch.object(sys, "argv", args + ["--device", "cpu"]):
                with self.assertRaisesRegex(ValueError, "explicit cuda:N"):module.main()
            self.assertFalse(output.exists())
            class Sampler:
                def __init__(self, **kwargs):pass
                def report(self):return {}
            with patch.object(sys, "argv", args + ["--device", "cuda:1"]), \
                    patch.object(module, "ProcessTreeDeviceSampler", Sampler), \
                    patch.object(module, "sha", return_value="entry-contract"), \
                    patch.object(module.subprocess, "Popen",
                        return_value=SimpleNamespace(pid=123, returncode=0, poll=lambda: 0)) as run:
                self.assertEqual(module.main(), 0)
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--sphere-finish-backend") + 1], "torch")
            self.assertEqual(json.loads((output / "benchmark.json").read_text())["sphere_finish_backend"], "torch")


if __name__ == "__main__":unittest.main()
