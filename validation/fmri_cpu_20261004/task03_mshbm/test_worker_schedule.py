"""Controller-only dry run; it produces no scientific benchmark evidence."""

import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import gpu_regression_worker as worker


class SilentMonitor:
    def __init__(self, target, daemon):
        self.target = target

    def start(self):
        pass

    def join(self, timeout):
        pass


class ScheduleTest(unittest.TestCase):
    def test_abba_filenames_source_binding_and_unobserved_peak(self):
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = root / "config.private.json"
            cfg.write_text(json.dumps({"python": "test-python", "baseline_source": "old-source",
                "candidate_source": "new-source", "baseline_commit": "test-commit", "threads": 8,
                "cpu8": [2, 14, 18, 22, 26, 30, 34, 38], "binding": {"expected_frames": 490}}))

            def launch(command, env, stdout, stderr):
                slot = Path(command[command.index("--output-dir") + 1])
                slot.mkdir()
                binding = json.loads(Path(command[command.index("--binding") + 1]).read_text())
                report = {"history": [{"outer": 1, "em": 5, "kappa": 1.0, "cost": 2.0}],
                          "gpu": {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0}}
                (slot / "report.public.json").write_text(json.dumps(report))
                calls.append({"slot": slot.name, "clock": command[command.index("-o") + 1],
                              "source": binding["source_root"], "threads": env["OMP_NUM_THREADS"],
                              "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"]})
                return types.SimpleNamespace(pid=999999, wait=lambda: 0)

            argv = ["gpu_regression_worker.py", "--config", str(cfg), "--gpu-uuid", "test-uuid",
                    "--lock", str(root / "gpu.lock"), "--output-dir", str(root / "run")]
            with patch.object(sys, "argv", argv), patch.object(worker.subprocess, "Popen", launch), \
                 patch.object(worker.threading, "Thread", SilentMonitor), \
                 patch.object(worker, "compare_outputs", lambda left, right: {}), \
                 patch.dict(sys.modules, {"nibabel": types.ModuleType("nibabel"),
                                          "numpy": types.ModuleType("numpy")}):
                worker.main()
            expected = ["baseline_1", "candidate_1", "candidate_2", "baseline_2"]
            self.assertEqual([call["slot"] for call in calls], expected)
            self.assertEqual([call["source"] for call in calls],
                             ["old-source", "new-source", "new-source", "old-source"])
            self.assertEqual(len({call["clock"] for call in calls}), 4)
            self.assertTrue(all(call["threads"] == "8" for call in calls))
            self.assertTrue(all(call["cuda_visible_devices"] == "test-uuid" for call in calls))
            record = json.loads((root / "run/gpu_execution.public.json").read_text())["records"]
            self.assertEqual([row["name"] for row in record], expected)
            self.assertTrue(all(row["simultaneous_process_tree_peak_bytes"] is None for row in record))
            self.assertTrue(all(row["within_20gb"] is None for row in record))
            for slot in expected:
                for suffix in (".log", ".memory_samples.private.json", ".binding.private.json"):
                    self.assertTrue((root / "run" / (slot + suffix)).is_file())
            control = json.loads((root / "run/gpu_old_new_control.public.json").read_text())
            self.assertEqual(control["order"], expected)


if __name__ == "__main__":
    unittest.main()
