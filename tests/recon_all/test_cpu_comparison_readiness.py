"""A failed timed process must not be scored from stale pipeline metadata."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "validation/smri_cpu/task5/recon_compare.py"
SPEC = importlib.util.spec_from_file_location("cpu_recon_comparison", SCRIPT)
DRIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVER)


class ComparisonReadinessTests(unittest.TestCase):
    def test_failed_receipt_overrides_stale_running_and_complete_reports(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            receipt = root / "receipt.json"
            receipt.write_text(json.dumps({"status": "failed", "returncode": 1}))
            for pipeline_status in ("running", "complete"):
                (root / "fnit-native-free-run.json").write_text(json.dumps({"status": pipeline_status}))
                run, failure = DRIVER.candidate_state(root, receipt)
                self.assertIsNone(run)
                self.assertEqual(failure["runner_status"], "failed")
                self.assertEqual(failure["returncode"], 1)

    def test_complete_pipeline_waits_for_process_receipt(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            receipt = root / "receipt.json"
            (root / "fnit-native-free-run.json").write_text(json.dumps({"status": "complete"}))
            receipt.write_text(json.dumps({"status": "running"}))
            self.assertEqual(DRIVER.candidate_state(root, receipt), (None, None))
            receipt.write_text(json.dumps({"status": "complete", "returncode": 0}))
            run, failure = DRIVER.candidate_state(root, receipt)
            self.assertEqual(run["status"], "complete")
            self.assertIsNone(failure)
