"""Check the Python batch API keeps subjects isolated and assigns one per device."""

import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from fnit.recon_all.batch import run_recon_all_python_batch


class BatchPythonTest(unittest.TestCase):
    def test_two_devices_run_concurrently_and_return_input_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights, assets = root / "weights", root / "assets"
            weights.mkdir()
            assets.mkdir()
            jobs = []
            for index in range(2):
                t1 = root / f"input{index}.nii.gz"
                t1.write_bytes(b"test")
                jobs.append({"t1": t1, "subject_dir": root / f"sub{index}"})
            native_dir = root / "native-bin"
            native_dir.mkdir()
            barrier = threading.Barrier(2)
            calls = []

            def fake_run(command, **kwargs):
                calls.append(command)
                barrier.wait(timeout=5)
                subject = Path(command[4])
                subject.mkdir()
                (subject / "fnit-native-free-run.json").write_text(
                    json.dumps({"subject_dir": str(subject), "device": command[10]}))
                return type("Completed", (), {"returncode": 0})()

            with patch("fnit.recon_all.batch.subprocess.run", side_effect=fake_run):
                reports = run_recon_all_python_batch(
                    jobs, weights, assets, devices=("cuda:0", "cuda:1"),
                    native_bin_dir=native_dir, native_optimizations="original",
                    hemisphere_workers=2)
            self.assertTrue(all(command[-2:] == ["--native-bin-dir", str(native_dir)]
                                for command in calls))
            self.assertEqual([row["subject_dir"] for row in reports],
                             [str(job["subject_dir"]) for job in jobs])
            self.assertEqual({command[10] for command in calls}, {"cuda:0", "cuda:1"})
            self.assertTrue(all(command[command.index("--native-optimizations") + 1] == "original"
                                for command in calls))
            self.assertTrue(all(command[command.index("--hemisphere-workers") + 1] == "2"
                                for command in calls))

    def test_rejects_overlapping_outputs_before_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "weights").mkdir()
            (root / "assets").mkdir()
            t1 = root / "input.nii.gz"
            t1.write_bytes(b"test")
            jobs = [{"t1": t1, "subject_dir": root / "subjects"},
                    {"t1": t1, "subject_dir": root / "subjects/sub02"}]
            with patch("fnit.recon_all.batch.subprocess.run") as runner:
                with self.assertRaises(ValueError):
                    run_recon_all_python_batch(jobs, root / "weights", root / "assets",
                                               native_bin_dir=root / "native-bin")
                runner.assert_not_called()

    def test_candidate_backends_are_passed_to_each_subject_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "weights").mkdir()
            (root / "assets").mkdir()
            t1 = root / "input.nii.gz"
            t1.write_bytes(b"test")
            subject = root / "subject"

            def fake_run(command, **kwargs):
                subject.mkdir()
                (subject / "fnit-native-free-run.json").write_text(json.dumps({"status": "complete"}))
                return type("Completed", (), {"returncode": 0})()

            with patch("fnit.recon_all.batch.subprocess.run", side_effect=fake_run) as runner:
                run_recon_all_python_batch(
                    [{"t1": t1, "subject_dir": subject}], root / "weights", root / "assets",
                    wm_backend="torch-optimized", gca_inverse_backend="torch",
                    gca_candidate_chunk=1024, gca_execution="isolated", fill_backend="numba")
            command = runner.call_args.args[0]
            for flag, value in (("--wm-backend", "torch-optimized"),
                                ("--gca-inverse-backend", "torch"),
                                ("--gca-candidate-chunk", "1024"), ("--gca-execution", "isolated"),
                                ("--fill-backend", "numba")):
                self.assertEqual(command[command.index(flag) + 1], value)


if __name__ == "__main__":
    unittest.main()
