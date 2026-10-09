"""标准inflate路由/设备/隔离缓存契约；真实完整链另在冻结输入验证。"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fnit.recon_all import native_free
from fnit.recon_all.batch import run_recon_all_python_batch


class InflateWiringTest(unittest.TestCase):
    def test_invalid_inflate_fails_before_thread_setup_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            for backend, device, workers in (("invalid", "cuda:0", 2), ("torch", "cpu", 2),
                                             ("torch", "cuda", 2), ("torch", "cuda:0", 1)):
                subject = Path(directory) / (backend + device.replace(":", "_") + str(workers))
                with self.subTest(backend=backend, device=device, workers=workers), \
                        patch("fnit.recon_all.thread_budget.thread_budget") as budget:
                    with self.assertRaises(ValueError):
                        native_free.run_recon_all_python(
                            t1="raw.nii.gz", subject_dir=subject, weights_dir="weights",
                            assets_dir="assets", device=device, hemisphere_workers=workers,
                            inflate_backend=backend)
                    budget.assert_not_called()
                    self.assertFalse(subject.exists())

    def test_standard_torch_inflate_reuses_existing_api_before_unchanged_sphere(self):
        with patch("fnit.recon_all.inflate_standard_run.run_standard_inflate",
                   return_value={"total_seconds_including_io": 1.}) as inflate, \
                patch("fnit.recon_all.sphere_standard_run.run_standard_sphere",
                      return_value={"total_seconds_including_io": 2.}) as sphere, \
                patch("fnit.recon_all.native_free._run_native_sphere_step") as native:
            times, report = native_free._run_accurate_sphere_pair(
                inflate_binary=Path("bin/inflate"), subject=Path("subject"),
                hemi="lh", assets=Path("assets"), device="cuda:1", inflate_backend="torch")
        self.assertEqual(inflate.call_args.kwargs, {
            "input_surface": Path("subject/surf/lh.smoothwm"),
            "inflated_output": Path("subject/surf/lh.inflated"),
            "sulc_output": Path("subject/surf/lh.sulc"), "backend": "torch",
            "device": "cuda:1", "profile": False})
        self.assertEqual(sphere.call_args.kwargs, {"finish_device": "cpu", "averaging_device": "cuda:1"})
        self.assertEqual((times["inflate"], times["sphere"]), (1., 2.))
        self.assertIn("inflate_runtime", report)
        native.assert_not_called()

    def test_cli_and_api_forward_explicit_inflate_and_keep_default_native(self):
        with patch("fnit.recon_all.native_free.run_recon_all_python", return_value={}) as run, \
                contextlib.redirect_stdout(io.StringIO()):
            native_free.main(["raw.nii.gz", "subject", "--weights-dir", "weights", "--assets-dir", "assets",
                              "--inflate-backend", "torch", "--hemisphere-workers", "2", "--device", "cuda:1"])
        self.assertEqual(run.call_args.kwargs["inflate_backend"], "torch")
        with tempfile.TemporaryDirectory() as directory:
            subject = Path(directory); subject.mkdir(exist_ok=True)
            with patch("fnit.recon_all.native_free._run_recon_all_python", return_value={}) as run:
                native_free.run_recon_all_python(t1="raw", subject_dir=subject, weights_dir="weights",
                                                assets_dir="assets", device="cuda:1", hemisphere_workers=2,
                                                inflate_backend="torch")
            self.assertEqual(run.call_args.kwargs["inflate_backend"], "torch")
        native_free._validate_inflate_backend("native", "cpu", 1)

    def test_exec_operation_reaches_surface_leaf(self):
        with patch("fnit.recon_all.native_free._surface_pair", return_value={}) as surface:
            native_free._hemisphere_operation(subject="subject", hemi="lh", device="cpu", threads=2,
                operation="surface", assets="assets", inflate_backend="torch",
                binaries={name: "bin/" + name for name in ("topology", "inflate", "intersection", "metrics")})
        self.assertEqual(surface.call_args.kwargs["inflate_backend"], "torch")
        self.assertTrue(surface.call_args.kwargs["defer_defects"])

    def test_batch_forwards_torch_inflate_to_declared_gpu_and_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("weights", "assets"):
                (root / name).mkdir()
            raw, subject = root / "raw.nii.gz", root / "subject"
            raw.write_bytes(b"entry contract")
            def launch(command, **kwargs):
                subject.mkdir()
                (subject / "fnit-native-free-run.json").write_text(json.dumps({"status": "complete"}))
                return SimpleNamespace(returncode=0)
            launch_mock = Mock(side_effect=launch)
            with patch("fnit.recon_all.batch.subprocess", SimpleNamespace(run=launch_mock)):
                run_recon_all_python_batch(jobs=[{"t1": raw, "subject_dir": subject}],
                    weights_dir=root / "weights", assets_dir=root / "assets",
                    devices=("cuda:1",), hemisphere_workers=2, inflate_backend="torch")
            command = launch_mock.call_args.args[0]
            self.assertEqual(command[command.index("--inflate-backend") + 1], "torch")
            self.assertEqual(command[command.index("--device") + 1], "cuda:1")
            self.assertEqual(command[command.index("--hemisphere-workers") + 1], "2")

    def test_batch_rejects_uncached_serial_torch_before_dispatch(self):
        with patch("fnit.recon_all.batch.subprocess") as process:
            with self.assertRaisesRegex(ValueError, "two cached"):
                run_recon_all_python_batch(jobs=[], weights_dir="weights", assets_dir="assets",
                    devices=("cuda:1",), hemisphere_workers=1, inflate_backend="torch")
        process.run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
