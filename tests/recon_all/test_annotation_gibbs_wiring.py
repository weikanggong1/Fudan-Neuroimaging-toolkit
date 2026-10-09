"""完整有序 GCSA 和 remesh 存储候选的入口契约；不充当真实影像 benchmark。"""

import contextlib
import importlib.util
import inspect
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from fnit.recon_all import native_free
from fnit.recon_all.batch import run_recon_all_python_batch


class TestAnnotationGibbsWiring(unittest.TestCase):
    def test_defaults_and_early_rejection(self):
        for function in (native_free.run_recon_all_python,
                         native_free._run_recon_all_python,
                         run_recon_all_python_batch):
            self.assertEqual(inspect.signature(function).parameters[
                "annotation_gibbs_backend"].default, "python")
            self.assertEqual(inspect.signature(function).parameters[
                "remesh_scalar_storage"].default, "numpy")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "subject"
            with patch.object(native_free.torch.cuda, "is_initialized") as initialized:
                with self.assertRaisesRegex(ValueError, "annotation_gibbs_backend"):
                    native_free.run_recon_all_python(
                        t1="raw", subject_dir=output, weights_dir="weights",
                        assets_dir="assets", device="cpu", annotation_gibbs_backend="invalid")
                initialized.assert_not_called()
            self.assertFalse(output.exists())
            with patch.object(native_free.torch.cuda, "is_initialized") as initialized:
                with self.assertRaisesRegex(ValueError, "remesh_scalar_storage"):
                    native_free.run_recon_all_python(
                        t1="raw", subject_dir=output, weights_dir="weights",
                        assets_dir="assets", device="cpu", remesh_scalar_storage="invalid")
                initialized.assert_not_called()
            self.assertFalse(output.exists())

    def test_worker_uses_existing_cache_and_all_three_atlases(self):
        cache = object()
        with patch("fnit.recon_all.gcsa_label_python.GCSAFeatureCache", return_value=cache) as prepare, \
                patch("fnit.recon_all.gcsa_label_python.label_surface", return_value={}) as label:
            native_free._hemisphere_operation(
                subject="subject", hemi="rh", device="cpu", threads=2,
                operation="annotation", assets="assets", annotation_gibbs_backend="numba")
        self.assertEqual(prepare.call_count, 1)
        self.assertEqual(label.call_count, 3)
        for call in label.call_args_list:
            self.assertIs(call.kwargs["prepared"], cache)
            self.assertEqual(call.kwargs["gibbs_backend"], "numba")
            self.assertEqual(call.kwargs["device"], "cpu")

    def test_surface_worker_forwards_existing_remesh_backend(self):
        with patch.object(native_free, "_surface_pair", return_value={}) as surface:
            native_free._hemisphere_operation(
                subject="subject", hemi="lh", device="cpu", threads=2,
                operation="surface", assets="assets", remesh_scalar_storage="python",
                binaries={name: name for name in ("topology", "inflate", "intersection", "metrics")})
        self.assertEqual(surface.call_args.kwargs["remesh_scalar_storage"], "python")
        self.assertEqual(surface.call_args.kwargs["threads"], 2)

    def test_public_and_cli_forward_option(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(native_free, "_run_recon_all_python", return_value={}) as run:
                native_free.run_recon_all_python(
                    t1="raw", subject_dir=Path(directory), weights_dir="weights",
                    assets_dir="assets", device="cpu", annotation_gibbs_backend="numba",
                    remesh_scalar_storage="python")
            self.assertEqual(run.call_args.kwargs["annotation_gibbs_backend"], "numba")
        self.assertEqual(run.call_args.kwargs["remesh_scalar_storage"], "python")
        with patch.object(native_free, "run_recon_all_python", return_value={}) as run, \
                contextlib.redirect_stdout(io.StringIO()):
            native_free.main(["raw", "subject", "--weights-dir", "weights",
                              "--assets-dir", "assets", "--device", "cpu",
                              "--annotation-gibbs-backend", "numba", "--remesh-scalar-storage", "python"])
        self.assertEqual(run.call_args.kwargs["annotation_gibbs_backend"], "numba")
        self.assertEqual(run.call_args.kwargs["remesh_scalar_storage"], "python")

    def test_batch_forwards_option_without_changing_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("weights", "assets"):
                (root / name).mkdir()
            raw, subject = root / "raw", root / "subject"
            raw.write_bytes(b"entry contract")
            def launch(command, **kwargs):
                subject.mkdir()
                (subject / "fnit-native-free-run.json").write_text(json.dumps({"status": "complete"}))
                return SimpleNamespace(returncode=0)
            launch = Mock(side_effect=launch)
            with patch("fnit.recon_all.batch.subprocess", SimpleNamespace(run=launch)):
                run_recon_all_python_batch(
                    jobs=[{"t1": raw, "subject_dir": subject}], weights_dir=root / "weights",
                    assets_dir=root / "assets", devices=("cpu",), threads=4,
                    hemisphere_workers=2, annotation_gibbs_backend="numba",
                    remesh_scalar_storage="python")
            command = launch.call_args.args[0]
            self.assertEqual(command[command.index("--annotation-gibbs-backend") + 1], "numba")
            self.assertEqual(command[command.index("--remesh-scalar-storage") + 1], "python")
            self.assertEqual(command[command.index("--threads") + 1], "4")
            self.assertEqual(command[command.index("--hemisphere-workers") + 1], "2")

    def test_benchmark_records_actual_selection(self):
        script = Path(__file__).resolve().parents[2] / "tools/benchmark_recon_torch_end_to_end.py"
        spec = importlib.util.spec_from_file_location("annotation_contract", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("weights", "assets", "native"):
                (root / name).mkdir()
            raw = root / "raw"; raw.write_bytes(b"entry contract")
            output = root / "output"
            args = [str(script), "--t1", str(raw), "--output-root", str(output),
                    "--weights-dir", str(root / "weights"), "--assets-dir", str(root / "assets"),
                    "--native-bin-dir", str(root / "native"), "--code-version", "entry-contract",
                    "--device", "cpu", "--annotation-gibbs-backend", "numba",
                    "--remesh-scalar-storage", "python"]
            class Sampler:
                def __init__(self, **kwargs): pass
                def report(self): return {}
            with patch.object(sys, "argv", args), \
                    patch.object(module, "ProcessTreeDeviceSampler", Sampler), \
                    patch.object(module, "sha", return_value="entry-contract"), \
                    patch.object(module.subprocess, "Popen", return_value=SimpleNamespace(
                        pid=123, returncode=0, poll=lambda: 0)) as run:
                self.assertEqual(module.main(), 0)
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--annotation-gibbs-backend") + 1], "numba")
            self.assertEqual(command[command.index("--remesh-scalar-storage") + 1], "python")
            report = json.loads((output / "benchmark.json").read_text())
            self.assertEqual(report["annotation_gibbs_backend"], "numba")
            self.assertEqual(report["remesh_scalar_storage"], "python")


if __name__ == "__main__":
    unittest.main()
