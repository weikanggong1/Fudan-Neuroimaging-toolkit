"""独立QA的证据契约；不运行影像计算，不作为真实benchmark。"""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[2] / "validation/recon_all/optimizations/20261017_sphere_normals/benchmark_standard_sphere.py"
SPEC = importlib.util.spec_from_file_location("fnit_sphere_saved_quality", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SavedQualityProcessTest(unittest.TestCase):
    def call(self, *, returncode=0, valid=True, mutate_surface=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            surface = root / "lh.sphere"
            surface.write_bytes(b"test file binding only; no simulated scientific measurement")
            payload = root / "payload.json"
            payload.write_text(json.dumps({"output": str(root), "row": {"hemisphere": "lh"}}))
            expected = MODULE.sha(payload)

            def child(command, *, env, **kwargs):
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
                self.assertEqual(env["OMP_NUM_THREADS"], "4")
                self.assertEqual(command[5], "quality-worker")
                report = {"status": "complete", "payload_sha256": expected,
                    "surface_sha256": MODULE.sha(surface), "tool_sha256": MODULE.sha(SCRIPT),
                    "cuda_visible_devices": "", "cuda_initialized": False,
                    "quality": {"passed": valid}}
                (root / "saved_quality.private.json").write_text(json.dumps(report))
                if mutate_surface:
                    surface.write_bytes(b"changed during child")
                return SimpleNamespace(returncode=returncode)

            with patch.object(MODULE.subprocess, "run", side_effect=child) as process:
                if returncode != 0 or not valid or mutate_surface:
                    with self.assertRaises(RuntimeError):
                        MODULE.inspect_surface_fresh_process(payload_path=payload,
                            payload_sha256=expected, python_interpreter="/python")
                    self.assertTrue((root / "saved_quality_receipt.private.json").is_file())
                else:
                    quality, receipt = MODULE.inspect_surface_fresh_process(payload_path=payload,
                        payload_sha256=expected, python_interpreter="/python")
                    self.assertTrue(quality["passed"])
                    self.assertEqual(receipt["exit_code"], 0)
                self.assertEqual(process.call_count, 1)  # 失败绝不自动重试

    def test_success_requires_original_quality_and_hides_cuda(self):
        self.call()

    def test_nonzero_child_exit_cannot_pass_even_with_success_json(self):
        self.call(returncode=-11)

    def test_false_quality_cannot_pass_with_exit_zero(self):
        self.call(valid=False)

    def test_saved_surface_mutation_is_rejected(self):
        self.call(mutate_surface=True)


if __name__ == "__main__":
    unittest.main()
