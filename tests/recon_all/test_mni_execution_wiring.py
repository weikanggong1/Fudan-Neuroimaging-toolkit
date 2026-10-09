"""末尾并行 MNI 的依赖、join 和入口契约；真实 MRI 回归单列。"""
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


class MNIExecutionWiringTest(unittest.TestCase):
    def test_invalid_execution_fails_before_setup_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            for execution, device, threads in (("bad", "cuda:0", 4),
                    ("parallel-late", "cpu", 4), ("parallel-late", "cuda", 4),
                    ("parallel-late", "cuda:0", 1), ("parallel-late", "cuda:0", True)):
                subject = Path(directory) / "subject"
                with self.subTest(execution=execution, device=device, threads=threads), \
                        patch("fnit.recon_all.thread_budget.thread_budget") as budget:
                    with self.assertRaises(ValueError):
                        native_free.run_recon_all_python(t1="raw", subject_dir=subject,
                            weights_dir="weights", assets_dir="assets", device=device,
                            threads=threads, mni_execution=execution)
                    budget.assert_not_called()
                    self.assertFalse(subject.exists())
        native_free._validate_mni_execution("in-process", "cpu", 1)

    def test_single_hemisphere_api_rejects_parent_autocast_before_thread_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            subject = Path(directory) / "subject"
            for active in ("cpu", "cuda"):
                with self.subTest(active=active), \
                        patch("fnit.recon_all.profiling.autocast_state",
                              side_effect=lambda kind: {"enabled": kind == active}), \
                        patch("fnit.recon_all.thread_budget.thread_budget") as budget:
                    with self.assertRaisesRegex(ValueError, "caller autocast disabled"):
                        native_free.run_recon_all_python(t1="raw", subject_dir=subject,
                            weights_dir="weights", assets_dir="assets", device="cuda:1",
                            threads=4, hemisphere_workers=1, mni_execution="parallel-late")
                    budget.assert_not_called()
                    self.assertFalse(subject.exists())

    def test_defer_preserves_affine_crop_and_finalsurfs(self):
        def stage(name, function, *args, **kwargs):
            return function(*args, **kwargs)
        for deferred in (False, True):
            with self.subTest(deferred=deferred), \
                    patch("fnit.recon_all.mni_aux_chain.run_mni_aux_chain",
                          return_value={"runtime": {"actual_precision": "FP32"}}) as auxiliary, \
                    patch("fnit.recon_all.mni_nonlinear_chain.run_mni_nonlinear_chain") as nonlinear, \
                    patch("fnit.recon_all.finalsurfs_python.run_finalsurfs") as final:
                value = native_free._run_white_mri_chain(
                    subject=Path("subject"), weights=Path("weights"), assets=Path("assets"),
                    threads=4, warp_binaries=(Path("convert"), Path("inverse"), Path("resample")),
                    stage=stage, device="cuda:1", defer_mni_nonlinear=deferred)
                auxiliary.assert_called_once()
                final.assert_called_once_with(Path("subject"), device="cuda:1")
                self.assertEqual(nonlinear.call_count, int(not deferred))
                self.assertEqual(value, {"actual_precision": "FP32"})

    def test_complete_output_checks_run_after_late_worker_join(self):
        with tempfile.TemporaryDirectory() as directory:
            subject = Path(directory)
            expected = ("mri/forward.nii.gz", "mri/inverse.nii.gz", "mri/check.mgz")
            mesh = {"status": "passed", "meshes": {"lh.orig": {"vertices": 12}}}
            group = {"status": "complete", "mesh_validation": mesh, "mni_nonlinear": {"real": True}}
            def complete(**kwargs):
                self.assertEqual(kwargs, {"subject": subject, "device": "cuda:1"})
                for name in expected:
                    path = subject / name; path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"worker output contract")
                return group
            def stage(name, function, *args, **kwargs):
                self.assertEqual(name, "mni_mesh_parallel")
                return function(*args, **kwargs)
            with patch("fnit.recon_all.expected_outputs.paths", return_value=expected), \
                    patch("fnit.recon_all.mni_mesh_parallel.run_mni_and_validate", side_effect=complete), \
                    patch("fnit.recon_all.native_free._validate_meshes") as duplicate:
                report = native_free._complete_output_validation(subject, stage,
                    late_mni_options={"subject": subject, "device": "cuda:1"})
            self.assertEqual(report["output_validation"]["present"], 3)
            self.assertEqual(report["output_validation"]["status"], "passed")
            self.assertIs(report["mesh_validation"], mesh)
            self.assertIs(report["mni_mesh_parallel"], group)
            duplicate.assert_not_called()

    def test_worker_failure_does_not_become_complete_or_missing_output(self):
        def stage(name, function, *args, **kwargs):
            return function(*args, **kwargs)
        with patch("fnit.recon_all.mni_mesh_parallel.run_mni_and_validate",
                   side_effect=RuntimeError("worker failed")), \
                patch("fnit.recon_all.expected_outputs.paths") as manifest:
            with self.assertRaisesRegex(RuntimeError, "worker failed"):
                native_free._complete_output_validation(Path("subject"), stage,
                                                        late_mni_options={"device": "cuda:1"})
            manifest.assert_not_called()

    def test_default_retains_mesh_check_without_second_mni(self):
        def stage(name, function, *args, **kwargs):
            self.assertEqual(name, "mesh_validation")
            return function(*args, **kwargs)
        with patch("fnit.recon_all.expected_outputs.paths", return_value=()), \
                patch("fnit.recon_all.native_free._validate_meshes", return_value={"status": "passed"}) as mesh, \
                patch("fnit.recon_all.mni_mesh_parallel.run_mni_and_validate") as mni:
            report = native_free._complete_output_validation(Path("subject"), stage)
        mesh.assert_called_once_with(Path("subject")); mni.assert_not_called()
        self.assertNotIn("mni_mesh_parallel", report)

    def test_api_and_cli_forward_explicit_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            subject = Path(directory)
            with patch("fnit.recon_all.native_free._run_recon_all_python", return_value={}) as run:
                native_free.run_recon_all_python(t1="raw", subject_dir=subject,
                    weights_dir="weights", assets_dir="assets", device="cuda:1", threads=4,
                    mni_execution="parallel-late")
            self.assertEqual(run.call_args.kwargs["mni_execution"], "parallel-late")
        with patch("fnit.recon_all.native_free.run_recon_all_python", return_value={}) as run, \
                contextlib.redirect_stdout(io.StringIO()):
            native_free.main(["raw", "subject", "--weights-dir", "weights", "--assets-dir", "assets",
                              "--mni-execution", "parallel-late", "--device", "cuda:1", "--threads", "4"])
        self.assertEqual(run.call_args.kwargs["mni_execution"], "parallel-late")

    def test_batch_forwards_device_and_budget_and_rejects_cpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("weights", "assets"):
                (root / name).mkdir()
            raw, subject = root / "raw", root / "subject"; raw.write_bytes(b"entry contract")
            def launch(command, **kwargs):
                subject.mkdir()
                (subject / "fnit-native-free-run.json").write_text(json.dumps({"status": "complete"}))
                return SimpleNamespace(returncode=0)
            run = Mock(side_effect=launch)
            with patch("fnit.recon_all.batch.subprocess", SimpleNamespace(run=run)):
                run_recon_all_python_batch(jobs=[{"t1": raw, "subject_dir": subject}],
                    weights_dir=root / "weights", assets_dir=root / "assets", devices=("cuda:1",),
                    threads=4, mni_execution="parallel-late")
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--mni-execution") + 1], "parallel-late")
            self.assertEqual(command[command.index("--device") + 1], "cuda:1")
            self.assertEqual(command[command.index("--threads") + 1], "4")
        with patch("fnit.recon_all.batch.subprocess") as run:
            with self.assertRaisesRegex(ValueError, "explicit cuda:N"):
                run_recon_all_python_batch(jobs=[], weights_dir="weights", assets_dir="assets",
                                          devices=("cpu",), mni_execution="parallel-late")
            run.run.assert_not_called()

    def test_benchmark_guards_and_records_actual_option(self):
        script = Path(__file__).resolve().parents[2] / "tools/benchmark_recon_torch_end_to_end.py"
        spec = importlib.util.spec_from_file_location("mni_benchmark_contract", script)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("weights", "assets", "native"):
                (root / name).mkdir()
            raw = root / "raw"; raw.write_bytes(b"entry contract"); output = root / "output"
            arguments = [str(script), "--t1", str(raw), "--output-root", str(output),
                "--weights-dir", str(root / "weights"), "--assets-dir", str(root / "assets"),
                "--native-bin-dir", str(root / "native"), "--code-version", "entry-contract",
                "--mni-execution", "parallel-late", "--threads", "4"]
            with patch.object(sys, "argv", arguments + ["--device", "cpu"]):
                with self.assertRaisesRegex(ValueError, "explicit cuda:N"):
                    module.main()
            self.assertFalse(output.exists())
            class Sampler:
                def __init__(self, **kwargs): pass
                def report(self): return {}
            with patch.object(sys, "argv", arguments + ["--device", "cuda:1"]), \
                    patch.object(module, "ProcessTreeDeviceSampler", Sampler), \
                    patch.object(module, "sha", return_value="entry-contract"), \
                    patch.object(module.subprocess, "Popen",
                        return_value=SimpleNamespace(pid=123, returncode=0, poll=lambda: 0)) as run:
                self.assertEqual(module.main(), 0)
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--mni-execution") + 1], "parallel-late")
            self.assertEqual(json.loads((output / "benchmark.json").read_text())["mni_execution"], "parallel-late")


if __name__ == "__main__":
    unittest.main()
