"""WM独立exec保存父进程精度/分配器，失败不回退或覆盖输出。"""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch

from fnit.recon_all.wm_torch_worker import run_isolated_segmentation, run_worker


class WMIsolationTests(unittest.TestCase):
    def test_exec_failure_preserves_initialized_parent_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            precision = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
            with patch.dict(os.environ, {"PYTORCH_NO_CUDA_MEMORY_CACHING": "1",
                                        "CUDA_VISIBLE_DEVICES": "GPU-test-mapping"}), \
                    patch("torch.cuda.is_initialized", return_value=True), \
                    patch("fnit.recon_all.wm_torch_worker.subprocess.run",
                          side_effect=subprocess.CalledProcessError(17, ["worker"])) as launch:
                with self.assertRaises(subprocess.CalledProcessError):
                    run_isolated_segmentation(source_path=root / "source.mgz",
                        output_path=root / "wm.mgz", report_path=root / "report.json",
                        device="cuda:1", threads=2)
                child = launch.call_args.kwargs["env"]
                self.assertNotIn("PYTORCH_NO_CUDA_MEMORY_CACHING", child)
                self.assertEqual(os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"], "1")
                self.assertEqual(child["CUDA_VISIBLE_DEVICES"], "GPU-test-mapping")
                self.assertEqual(child["NUMBA_NUM_THREADS"], "2")
                self.assertFalse((root / "wm.mgz").exists())
            self.assertEqual((torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32), precision)

    def test_worker_rejects_inherited_initialized_cuda(self):
        with patch("torch.cuda.is_initialized", return_value=True), \
                patch("fnit.recon_all.wm_torch_worker.segment_white_matter_mgz") as segment:
            with self.assertRaisesRegex(ValueError, "fresh exec"):
                run_worker(source_path=Path("source.mgz"), output_path=Path("new.mgz"),
                           report_path=Path("new.json"), code_version="test")
            segment.assert_not_called()

    def test_new_paths_and_explicit_device_before_launch(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch("fnit.recon_all.wm_torch_worker.subprocess.run") as launch:
            root = Path(temporary)
            for device in ("cpu", "cuda"):
                with self.assertRaises(ValueError):
                    run_isolated_segmentation(source_path=root / "source.mgz",
                        output_path=root / "new.mgz", report_path=root / "new.json", device=device)
            (root / "new.mgz").write_bytes(b"keep-existing-work")
            with self.assertRaises(FileExistsError):
                run_isolated_segmentation(source_path=root / "source.mgz",
                    output_path=root / "new.mgz", report_path=root / "new.json", device="cuda:1")
            self.assertEqual((root / "new.mgz").read_bytes(), b"keep-existing-work")
            launch.assert_not_called()

    def test_invalid_batches_and_threads_rejected(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch("fnit.recon_all.wm_torch_worker.subprocess.run") as launch:
            root = Path(temporary)
            for keywords in ({"histogram_batch_size": 0}, {"planar_batch_size": True},
                             {"threads": False}, {"threads": 0}):
                with self.assertRaises(ValueError):
                    run_isolated_segmentation(source_path=root / "source.mgz",
                        output_path=root / "new.mgz", report_path=root / "new.json",
                        device="cuda:1", **keywords)
            launch.assert_not_called()

    def test_success_argv_keeps_paths_separate_and_reports_full_exec_wall(self):
        with tempfile.TemporaryDirectory(prefix="WM path ") as temporary:
            root = Path(temporary)
            report_path = root / "worker report.json"

            def complete(command, **keywords):
                self.assertNotIn("shell", keywords)
                self.assertEqual(command[command.index("--source") + 1], str(root / "source input.mgz"))
                self.assertIn("--profile-stages", command)
                self.assertNotIn("--reference", command)
                report_path.write_text(json.dumps({"scope": "worker", "worker_pid": 12}))

            with patch("fnit.recon_all.wm_torch_worker.subprocess.run", side_effect=complete):
                result = run_isolated_segmentation(source_path=root / "source input.mgz",
                    output_path=root / "wm output.mgz", report_path=report_path,
                    device="cuda:1", threads=4, profile_stages=True)
            self.assertGreater(result["isolated_cli_wall_seconds"], 0)
            self.assertEqual(json.loads(report_path.read_text())["worker_pid"], 12)
            self.assertFalse(report_path.with_suffix(".pending").exists())


if __name__ == "__main__":
    unittest.main()
