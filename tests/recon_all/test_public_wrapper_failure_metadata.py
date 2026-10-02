"""公开入口失败元数据回归；提取实际函数并模拟调度，不导入 Torch 或使用 GPU。"""

import ast
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


class PublicWrapperFailureMetadataTests(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[2] / "src/fnit/recon_all/native_free.py"
        tree = ast.parse(source.read_text())
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "run_recon_all_python")
        self.clock = 100.0
        self.budget = {"restoration_complete": False, "torch": {"restored": None},
                       "numba": {"restored": None}}
        self.restore_error = self.setup_error = None
        self.engine = Mock()

        @contextmanager
        def fake_budget(*, threads):
            if self.setup_error is not None:
                raise self.setup_error
            self.clock += 0.5
            try:
                yield self.budget
            finally:
                self.clock += 1.5
                if self.restore_error is not None:
                    raise self.restore_error
                self.budget.update(restoration_complete=True)
                self.budget["torch"]["restored"] = 8
                self.budget["numba"]["restored"] = 128

        thread_module = types.ModuleType("fnit.recon_all.thread_budget")
        thread_module.thread_budget = fake_budget
        self.module_patch = patch.dict("sys.modules", {thread_module.__name__: thread_module})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        namespace = {"__package__": "fnit.recon_all", "Path": Path, "json": json,
                     "time": types.SimpleNamespace(perf_counter=lambda: self.clock),
                     "_run_recon_all_python": self.engine}
        unit = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
            ast.alias(name="annotations")], level=0), function], type_ignores=[])
        exec(compile(ast.fix_missing_locations(unit), str(source), "exec"), namespace)
        self.run = namespace["run_recon_all_python"]
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.subject = Path(temporary.name) / "subject"
        self.subject.mkdir()
        self.report_path = self.subject / "fnit-native-free-run.json"

    def call(self):
        return self.run(t1="raw.nii.gz", subject_dir=self.subject, weights_dir="weights",
                        assets_dir="assets", device="cuda:1", threads=4,
                        native_bin_dir="native", profile_stages=False,
                        cuda_allocator_cache="auto")

    def write_internal_report(self, *, status):
        self.clock += 5.0
        report = {"status": status, "total_seconds": 5.0, "timing": {"pipeline_seconds": 4.5}}
        if status == "failed":
            report["failed_stage"] = "white"
        self.report_path.write_text(json.dumps(report))
        # 内部函数最后写出报告在其 total_seconds 之后、公开入口退出之前。
        self.clock += 0.25
        return report

    def fail_engine(self, error):
        def run(**kwargs):
            self.write_internal_report(status="failed")
            raise error
        self.engine.side_effect = run

    def test_success_keeps_arguments_and_records_residual_scope(self):
        self.engine.side_effect = lambda **kwargs: self.write_internal_report(status="complete")
        report = self.call()
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["total_seconds"], 7.25)
        self.assertEqual(report["timing"]["thread_setup_and_restore_seconds"], 2.25)
        self.assertIn("internal final report write/return", report["timing"]["thread_setup_and_restore_scope"])
        self.assertTrue(report["thread_budget"]["restoration_complete"])
        self.assertEqual(json.loads(self.report_path.read_text()), report)
        self.engine.assert_called_once_with(t1="raw.nii.gz", subject_dir=self.subject,
            weights_dir="weights", assets_dir="assets", device="cuda:1", threads=4,
            native_bin_dir="native", profile_stages=False, cuda_allocator_cache="auto")

    def test_stage_failure_records_restored_budget_and_public_wall(self):
        original = RuntimeError("white failed")
        self.fail_engine(original)
        with self.assertRaises(RuntimeError) as raised:
            self.call()
        self.assertIs(raised.exception, original)
        report = json.loads(self.report_path.read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["failed_stage"], "white")
        self.assertEqual(report["total_seconds"], 7.25)
        self.assertTrue(report["thread_budget"]["restoration_complete"])
        self.assertEqual(report["thread_budget"]["numba"]["restored"], 128)

    def test_restore_failure_changes_written_complete_to_failed(self):
        self.engine.side_effect = lambda **kwargs: self.write_internal_report(status="complete")
        self.restore_error = RuntimeError("restore failed")
        with self.assertRaises(RuntimeError) as raised:
            self.call()
        self.assertIs(raised.exception, self.restore_error)
        report = json.loads(self.report_path.read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["failed_stage"], "thread_budget_restore")
        self.assertEqual(report["total_seconds"], 7.25)
        self.assertFalse(report["thread_budget"]["restoration_complete"])

    def test_stage_error_survives_an_additional_restore_error(self):
        original = RuntimeError("white failed")
        self.fail_engine(original)
        self.restore_error = RuntimeError("restore also failed")
        with self.assertRaises(RuntimeError) as raised:
            self.call()
        self.assertIs(raised.exception, original)
        self.assertIs(raised.exception.__cause__, self.restore_error)
        report = json.loads(self.report_path.read_text())
        self.assertEqual(report["failed_stage"], "white")
        self.assertIn("white failed", report["error"])
        self.assertIn("restore also failed", report["thread_budget_restoration_error"])

    def test_metadata_write_error_does_not_replace_stage_error(self):
        original = RuntimeError("white failed")
        self.fail_engine(original)
        actual_write = Path.write_text
        writes = []

        def reject_public_write(path, *args, **kwargs):
            writes.append(path)
            if len(writes) > 1:
                raise OSError("metadata disk failure")
            return actual_write(path, *args, **kwargs)

        with patch.object(Path, "write_text", reject_public_write):
            with self.assertRaises(RuntimeError) as raised:
                self.call()
        self.assertIs(raised.exception, original)
        self.assertTrue(self.budget["restoration_complete"])
        self.assertEqual(len(writes), 2)
        if hasattr(original, "add_note"):
            self.assertIn("metadata disk failure", " ".join(getattr(original, "__notes__", [])))

    def test_invalid_current_metadata_preserves_original_error(self):
        original = RuntimeError("stage failed")

        def fail(**kwargs):
            self.report_path.write_text("invalid-json")
            raise original

        self.engine.side_effect = fail
        with self.assertRaises(RuntimeError) as raised:
            self.call()
        self.assertIs(raised.exception, original)
        self.assertEqual(self.report_path.read_text(), "invalid-json")

    def test_validation_failure_preserves_prior_report_bytes(self):
        previous = '{"status": "complete", "prior_run": true}\n'
        self.report_path.write_text(previous)
        self.engine.side_effect = ValueError("subject_dir must be empty")
        with self.assertRaisesRegex(ValueError, "subject_dir must be empty"):
            self.call()
        self.assertEqual(self.report_path.read_text(), previous)
        self.assertTrue(self.budget["restoration_complete"])

    def test_budget_setup_failure_does_not_start_engine_or_create_report(self):
        self.setup_error = ValueError("thread capacity exceeded")
        with self.assertRaises(ValueError) as raised:
            self.call()
        self.assertIs(raised.exception, self.setup_error)
        self.engine.assert_not_called()
        self.assertFalse(self.report_path.exists())


if __name__ == "__main__":
    unittest.main()
