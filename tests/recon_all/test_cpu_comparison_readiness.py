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
            self.assertEqual(DRIVER.candidate_state(root, receipt), (None, None))
            receipt.write_text(json.dumps({"status": "running"}))
            self.assertEqual(DRIVER.candidate_state(root, receipt), (None, None))
            receipt.write_text(json.dumps({"status": "complete", "returncode": 0}))
            run, failure = DRIVER.candidate_state(root, receipt)
            self.assertEqual(run["status"], "complete")
            self.assertIsNone(failure)

    def test_ordered_topology_keeps_indexed_and_triangle_distances_separate(self):
        import nibabel.freesurfer.io as fsio
        import numpy as np

        source = SCRIPT.parents[2] / "recon_all/python_gpu_port/compare_surface_chain.py"
        spec = importlib.util.spec_from_file_location("surface_distance_test", source)
        surface = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(surface)
        vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32)
        faces = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
        with tempfile.TemporaryDirectory() as name:
            reference, candidate = Path(name) / "reference", Path(name) / "candidate"
            for root, shift in ((reference, 0.0), (candidate, 0.5)):
                (root / "surf").mkdir(parents=True)
                shifted = vertices + np.array([shift, 0, 0])
                for hemi in ("lh", "rh"):
                    for label in ("white", "pial"):
                        fsio.write_geometry(str(root / "surf" / f"{hemi}.{label}"), shifted, faces)
            result = DRIVER.final_surface_distances(surface, reference, candidate)
        for hemi in ("lh", "rh"):
            for label in ("white", "pial"):
                row = result["stages"][hemi][label]
                self.assertTrue(row["ordered_faces_equal"])
                self.assertEqual(row["indexed_vertex_distance"]["mean_mm"], 0.5)
                for direction in ("candidate_to_reference_triangle", "reference_to_candidate_triangle"):
                    self.assertEqual(row[direction]["mean_mm"], 0.25)
