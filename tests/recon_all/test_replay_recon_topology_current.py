"""私有拓扑阶段回放的文件与失败回执合同；不代替真实T1测试。"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


PATH = Path(__file__).resolve().parents[2] / "tools" / "replay_recon_topology_current.py"
SPEC = importlib.util.spec_from_file_location("replay_recon_topology_current", PATH)
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


class StageReplayContracts(unittest.TestCase):
    def inputs(self, root):
        files = {}
        for name in ("nofix", "qsphere", "brain", "wm"):
            path = root / (name + ".fake")
            path.write_bytes(name.encode())
            files[name] = path
        return dict(input_surface=files["nofix"], input_qsphere=files["qsphere"],
                    brain=files["brain"], wm=files["wm"], hemisphere="lh",
                    output_directory=root / "stage", threads=2,
                    distance_backend="serial", distance_device=None,
                    state_backend="legacy")

    def test_captures_source_cleaned_geometry_without_changing_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.inputs(Path(temporary))
            vertices = np.zeros((3, 3), np.float32)
            faces = np.array([[0, 1, 2]], np.int32)
            def cleaner(**kwargs):
                self.assertEqual(kwargs["marking_backend"], "source_numba")
                return vertices.copy(), {"intersecting_faces_after": 0}
            def stage(**kwargs):
                self.assertEqual(kwargs["face_detection_cache"], True)
                self.assertEqual(kwargs["blocker_scan_backend"], "numba")
                self.assertEqual(kwargs["remesh_scalar_storage"], "python")
                self.assertEqual(kwargs["distance_backend"], "serial")
                self.assertIsNone(kwargs["distance_device"])
                self.assertEqual(kwargs["state_backend"], "legacy")
                kwargs["output_directory"].mkdir()
                tool.genetic.repair_intersections(
                    vertices=vertices, faces=faces, ripped=np.zeros(3, bool),
                    ordered_neighbor_offsets=np.zeros(4, np.int64),
                    ordered_neighbors=np.empty(0, np.int32),
                    marking_backend="source_numba")
                return {"output_sha256": {"orig": "f" * 64}, "quality": {"status": "passed"}}
            with patch.object(tool.genetic, "repair_intersections", side_effect=cleaner), \
                    patch.object(tool, "run_genetic_topology_files", side_effect=stage):
                result = tool.run(**args)
            self.assertEqual(result["stage_execution"], "complete")
            self.assertEqual(result["quality"]["status"], "passed")
            self.assertEqual(json.loads((args["output_directory"] / "stage_replay.private.json").read_text())
                             ["algorithm_policy_requested"],
                             {"distance_backend": "serial", "distance_device": None,
                              "state_backend": "legacy"})
            self.assertTrue((args["output_directory"] / "source_cleaned_current.private.npz").exists())
            self.assertEqual(json.loads((args["output_directory"] / "stage_replay.private.json").read_text())
                             ["diagnostic"]["source_cleanup"]["intersecting_faces_after"], 0)

    def test_failure_preserves_current_and_never_claims_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.inputs(Path(temporary))
            def stage(**kwargs):
                kwargs["output_directory"].mkdir()
                kwargs["failure_state_callback"](
                    original_vertices=np.zeros((3, 3), np.float32),
                    current_cleanup_vertices=np.zeros((3, 3), np.float32),
                    faces=np.array([[0, 1, 2]], np.int32),
                    quality={"status": "failed"}, current_cleanup={"intersecting_faces_after": 2})
                raise RuntimeError("geometric residual")
            with patch.object(tool, "run_genetic_topology_files", side_effect=stage):
                with self.assertRaisesRegex(RuntimeError, "geometric residual"):
                    tool.run(**args)
            receipt = json.loads((args["output_directory"] / "stage_replay.private.json").read_text())
            self.assertEqual(receipt["stage_execution"], "failed")
            self.assertEqual(receipt["diagnostic"]["failed_quality"]["status"], "failed")
            self.assertTrue((args["output_directory"] / "failed_current.private.npz").exists())

    def test_rejects_implicit_or_incompatible_backend_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.inputs(Path(temporary))
            args["distance_backend"] = "gpu_defect_signed_distance"
            with self.assertRaisesRegex(ValueError, "explicit cuda:N"):
                tool.run(**args)
            self.assertFalse(args["output_directory"].exists())
            args["distance_backend"] = "serial"
            args["state_backend"] = "unknown"
            with self.assertRaisesRegex(ValueError, "state_backend"):
                tool.run(**args)
            self.assertFalse(args["output_directory"].exists())


if __name__ == "__main__":
    unittest.main()
