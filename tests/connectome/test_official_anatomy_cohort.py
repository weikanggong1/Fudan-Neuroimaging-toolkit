"""Cohort freshness and independent official-origin contracts."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

FILE = Path(__file__).resolve().parents[2] / "tools/reference/benchmark_connectome_official_anatomy_cohort.py"
SPEC = importlib.util.spec_from_file_location("official_anatomy_cohort", FILE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class OfficialAnatomyCohort(unittest.TestCase):
    def test_baseline_subject_cannot_be_replaced_by_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = "sub-CON01"
            target = root / "baseline" / case
            target.mkdir(parents=True)
            candidate = root / "candidate" / case / "freesurfer" / f"{case}_ses-preop"
            candidate.mkdir(parents=True)
            (candidate / "scripts").mkdir()
            (candidate / "scripts/recon-all.done").write_text("actual done")
            report = {"case_id": case, "status": "completed", "exit_code": 0,
                "command": ["recon-all", "-sd", str(candidate.parent), "-s", candidate.name]}
            (target / "recon_report.json").write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "exact baseline"):
                MODULE.completed_baseline_origin(target, case)

    def test_running_fresh_fs_waits_instead_of_being_declared_done(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "recon_report.json").write_text('{"status":"running"}')
            self.assertIsNone(MODULE.completed_baseline_origin(root, "sub-CON01"))

    def test_unrelated_recon_failure_is_not_revalidated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "recon_report.json").write_text(json.dumps({"case_id": "sub-CON01",
                "status": "failed", "exit_code": 0, "error": {"type": "RuntimeError", "message": "bad anatomy"}}))
            with self.assertRaisesRegex(ValueError, "JSON int32"):
                MODULE.completed_baseline_origin(root, "sub-CON01")

    def test_consumer_refuses_an_incomplete_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text('{"case_id":"sub-CON01","state":"running"}')
            with self.assertRaisesRegex(ValueError, "completed same-case"):
                MODULE.consumer_contract("sub-CON01", {}, path, {})

    def test_existing_cohort_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SystemExit):
                MODULE.main(["--config-template", "/absent.json", "--baseline-root", "/baseline",
                    "--raw-root", "/raw", "--official-dwi-root", "/official",
                    "--validation-script", "/absent.py", "--tool-commit", "frozen", "--output", directory])


if __name__ == "__main__":
    unittest.main()
