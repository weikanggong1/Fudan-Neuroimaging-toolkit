"""整例监督/计时元数据的控制回归；不运行MRI或GPU算法。"""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "validation/recon_all/optimizations/20261002_parallel"


def module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


SUMMARY = module("summarize_performance")
SUPERVISOR = module("run_installed_candidate")


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


class WholeAutomationDiagnosticsTests(unittest.TestCase):
    def test_missing_or_zero_memory_is_not_a_budget_pass(self):
        run = {"cuda_allocator": {"torch_stats_known_valid": False, "torch_stats_known_unavailable": True},
               "stages": [], "gpu_peak_allocated_bytes": 8, "gpu_peak_reserved_bytes": 16}
        for peak, samples, failed in ((None, 0, 0), (None, 5, 5), (0, 5, 0)):
            result = SUMMARY.memory_record(run, {"peak_sampled_process_bytes": peak, "samples": samples,
                                                "failed_app_queries": failed})
            self.assertIsNone(result["peak_GB"])
            self.assertIsNone(result["under_requested_budget_at_samples"])
            self.assertEqual(result["allocated"], 8)  # 原始分量值不删除/改成0。

    def test_partial_samples_and_child_allocator_have_separate_scopes(self):
        run = {"cuda_allocator": {"torch_stats_known_valid": False, "torch_stats_known_unavailable": True},
               "stages": [{"name": "input_talairach", "talairach_child_gpu": {"gpu_peak_allocated_bytes": 8}}],
               "gpu_peak_allocated_bytes": 8, "gpu_peak_reserved_bytes": 16}
        result = SUMMARY.memory_record(run, {"peak_sampled_process_bytes": 12_000_000_000,
            "samples": 10, "failed_app_queries": 2, "monitor_thread_finished": True})
        self.assertEqual(result["peak_GB"], 12)
        self.assertEqual(result["sampling_status"], "partially_measured")
        self.assertTrue(result["under_requested_budget_at_samples"])
        self.assertTrue(result["parent_allocator_stats_known_unavailable"])
        self.assertEqual(result["allocator_components"][0]["process"], "Talairach child")

    def installed_fixture(self, root):
        installed = root / "candidate_install_8d750e2/installed"
        installed.mkdir(parents=True)
        wrapper = root / "candidate_runtime_8d750e2"
        wrapper.mkdir()
        (wrapper / "src").symlink_to(installed, target_is_directory=True)
        for case in ("sub01", "sub02"):
            write(root / "coordinator" / f"candidate_{case}_8d750e2.json", {
                "code_root": str(wrapper), "python": sys.executable,
                "code_commit": SUPERVISOR.CANDIDATE, "source_archive_sha256": SUPERVISOR.ARCHIVE})
        return {"status": "passed", "code_commit": SUPERVISOR.CANDIDATE,
                "compared_recon_all_py_files": 171, "mismatches": [],
                "installed_import_root": str(installed)}

    def test_installed_entry_requires_the_actual_queue_symlink(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            check = self.installed_fixture(root)
            self.assertEqual(len(SUPERVISOR.validate_installed_entry(root, Path(sys.executable), check)), 2)
            entry = root / "candidate_runtime_8d750e2/src"
            entry.unlink(); entry.mkdir()
            with self.assertRaisesRegex(ValueError, "frozen installation"):
                SUPERVISOR.validate_installed_entry(root, Path(sys.executable), check)

    def test_other_installed_commit_cannot_be_labeled_8d(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            check = self.installed_fixture(root)
            check["code_commit"] = "other commit"
            with self.assertRaisesRegex(ValueError, "source checks"):
                SUPERVISOR.validate_installed_entry(root, Path(sys.executable), check)

    def supervisor_argv(self, root):
        return ["run_installed_candidate.py", "--round", str(root), "--python", sys.executable,
                "--lock", str(root / "lock"), "--resource-script", str(root / "capture.py")]

    def test_supervisor_failure_is_archived_and_old_report_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root / "coordinator").mkdir()
            with patch.object(sys, "argv", self.supervisor_argv(root)), \
                    patch.object(SUPERVISOR, "run", side_effect=RuntimeError("unit-control failure")):
                with self.assertRaises(RuntimeError):
                    SUPERVISOR.main()
            path = root / "coordinator/candidate_ready_validation.json"
            result = json.loads(path.read_text())
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["error_type"], "RuntimeError")
            prior = path.read_bytes()
            with patch.object(sys, "argv", self.supervisor_argv(root)), patch.object(SUPERVISOR, "run") as run:
                with self.assertRaises(FileExistsError):
                    SUPERVISOR.main()
                run.assert_not_called()
            self.assertEqual(path.read_bytes(), prior)

    def timing_fixture(self, root):
        # 复制实际baseline元数据作为结构fixture；候选时间仅控制stub，不作benchmark。
        source = HERE / "whole/baseline_sub01"
        baseline = {name: json.loads((source / (name + ".json")).read_text())
                    for name in ("run", "launch", "completion", "monitor")}
        candidate = copy.deepcopy(baseline)
        consumed = {row["name"] for row in baseline["run"]["stages"]
                    if row["name"].startswith(("surface_", "register_", "avg_curv_", "annot_", "finish_surface_"))}
        candidate["run"]["stages"] = [row for row in candidate["run"]["stages"] if row["name"] not in consumed]
        candidate["run"]["stages"] += [{"name": name, "seconds": 10.0} for name in (
            "surface_hemisphere_group", "defects_lh", "defects_rh", "register_hemisphere_group",
            "annotation_hemisphere_group", "finish_surface_hemisphere_group", "finish_metrics_lh", "finish_metrics_rh")]
        candidate["run"]["native_optimizations"] = {}
        candidate["run"]["hemisphere_scheduling"] = {"workers": 2}
        candidate["launch"]["code_commit"] = candidate["completion"]["code_commit"] = SUMMARY.CANDIDATE
        for kind, data in (("baseline", baseline), ("candidate", candidate)):
            for name, value in data.items(): write(root / f"{kind}_sub02" / (name + ".json"), value)
            (root / f"{kind}_sub02/gpu_samples.csv").write_bytes((source / "gpu_samples.csv").read_bytes())
        return baseline, candidate

    def test_register_scope_contains_avg_curv_and_annotation_does_not(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); baseline, _ = self.timing_fixture(root)
            result = SUMMARY.summarize_case(root, "sub02")
            rows = {row["name"]: row for row in result["stages"]}
            self.assertEqual(set(rows["register_bilateral_scope"]["baseline_stage_names"]),
                             {"register_lh", "register_rh", "avg_curv_lh", "avg_curv_rh"})
            self.assertTrue(all(name.startswith("annot_") for name in rows["annotation_bilateral_scope"]["baseline_stage_names"]))
            expected = sum(row["seconds"] for row in baseline["run"]["stages"]
                           if row["name"].startswith(("register_", "avg_curv_")))
            self.assertEqual(rows["register_bilateral_scope"]["baseline_seconds"], expected)
            self.assertEqual(len(result["sources"]), 10)

    def test_another_run_input_cannot_be_paired_with_matching_launches(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); _, candidate = self.timing_fixture(root)
            candidate["run"]["input"] = "another T1"
            write(root / "candidate_sub02/run.json", candidate["run"])
            with self.assertRaisesRegex(ValueError, "same_raw_input"):
                SUMMARY.summarize_case(root, "sub02")


if __name__ == "__main__":
    unittest.main()
