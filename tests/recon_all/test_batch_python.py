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
                    native_bin_dir=native_dir)
            self.assertTrue(all(command[-2:] == ["--native-bin-dir", str(native_dir)]
                                for command in calls))
            self.assertEqual([row["subject_dir"] for row in reports],
                             [str(job["subject_dir"]) for job in jobs])
            self.assertEqual({command[10] for command in calls}, {"cuda:0", "cuda:1"})

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


if __name__ == "__main__":
    unittest.main()
