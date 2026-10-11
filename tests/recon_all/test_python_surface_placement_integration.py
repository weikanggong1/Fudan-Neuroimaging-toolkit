"""自有完整放置的生产接线契约；真实影像回归另存benchmark。"""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fnit.recon_all.native_free import (
    _finish_cortical_surface, _run_avg_curv, _run_curvature_stats, _validate_surface_placement_backend, main,
)
from fnit.recon_all.native_runtime_selection import select_native_optimizations
from fnit.recon_all.surface_placement_python import placement_options


class PythonSurfacePlacementIntegrationTest(unittest.TestCase):
    def test_cpu_preserves_compiled_ordered_core_and_full_source_cleanup(self):
        options = placement_options(device="cpu")
        self.assertEqual(options["candidate_backend"], "snapshot")
        self.assertEqual(options["retained_mht_backend"], "compiled")
        self.assertEqual(options["cleanup_marking_backend"], "source_numba")
        self.assertEqual(options["sampling_backend"], "cpu")
        self.assertEqual(options["regularization_backend"], "cpu")

    def test_cuda_only_moves_complete_candidates_and_cleanup(self):
        options = placement_options(device="cuda:1")
        self.assertEqual(options["device"], "cuda:1")
        self.assertEqual(options["candidate_backend"], "torch_snapshot")
        self.assertEqual(options["cleanup_marking_backend"], "source_torch")
        self.assertEqual(options["candidate_grid_cells_per_axis"], 3)
        self.assertEqual(options["cleanup_candidate_grid_cells_per_axis"], 3)
        self.assertEqual(options["sampling_backend"], "cpu")
        for value in ("cuda", "mps"):
            with self.assertRaises(ValueError):
                placement_options(device=value)

    def test_final_white_then_pial_without_any_placement_program(self):
        with tempfile.TemporaryDirectory() as directory:
            subject = Path(directory)
            (subject / "surf").mkdir()
            calls = []
            def white(**kwargs):
                calls.append(("white", kwargs))
                (subject / "surf/lh.white").write_bytes(b"complete-white")
                return {"native_programs": [], "seconds": 1}
            def pial(**kwargs):
                self.assertEqual((subject / "surf/lh.white").read_bytes(), b"complete-white")
                calls.append(("pial", kwargs))
                (subject / "surf/lh.pial.T1").write_bytes(b"complete-pial")
                return {"native_programs": [], "seconds": 2}
            with patch("fnit.recon_all.surface_placement_python.run_final_white_python", side_effect=white), \
                 patch("fnit.recon_all.surface_placement_python.run_pial_python", side_effect=pial), \
                 patch("fnit.recon_all.final_white_conda.run_final_white", side_effect=AssertionError("native white called")), \
                 patch("fnit.recon_all.native_free._run_native_pial", side_effect=AssertionError("native pial called")):
                report = _finish_cortical_surface(subject, "lh", None, subject,
                    device="cpu", threads=2, defer_metrics=True,
                    surface_placement_backend="python")
            self.assertEqual([item[0] for item in calls], ["white", "pial"])
            self.assertEqual((subject / "surf/lh.pial").read_bytes(), b"complete-pial")
            self.assertFalse(report["placement_pending"])
            self.assertTrue(report["metrics_pending"])

    def test_python_placement_skips_native_white_capability_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "mris_place_surface_white_fast").write_bytes(b"must not execute")
            with patch("fnit.recon_all.native_runtime_selection.subprocess.run",
                       side_effect=AssertionError("native executable started")):
                value = select_native_optimizations(root, threads=4, mode="auto",
                    device="cuda:0", include_white=False)
            self.assertEqual(value["em_backend"], "torch")
            self.assertIsNone(value["white_capabilities"])

    def test_cli_forwards_python_placement(self):
        with patch("fnit.recon_all.native_free.run_recon_all_python", return_value={"status": "complete"}) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                main(["input.nii.gz", "subject", "--weights-dir", "weights", "--assets-dir", "assets",
                      "--surface-placement-backend", "python"])
        self.assertEqual(run.call_args.kwargs["surface_placement_backend"], "python")

    def test_unknown_backend_rejected(self):
        for value in ("torch", "approximate", ""):
            with self.assertRaises(ValueError):
                _validate_surface_placement_backend(value)

    def test_complete_curvature_uses_source_discrete_profile_without_program(self):
        with patch("fnit.recon_all.curvature_stats_python.write_curvature_stats", return_value={"outputs": {}}) as run, \
             patch("fnit.recon_all.native_free.subprocess.run", side_effect=AssertionError("original program called")):
            _run_curvature_stats(None, Path("subject"), "lh", Path("assets"), device="cuda:1", python_only=True)
        self.assertEqual(run.call_args.kwargs["angle_backend"], "source_libm")
        self.assertEqual(run.call_args.kwargs["averages"], 0)
        self.assertEqual(run.call_args.kwargs["device"], "cuda:1")
        self.assertTrue(run.call_args.kwargs["overwrite"])

    def test_paint_uses_validated_source_cpu_profile_without_program(self):
        with patch("fnit.recon_all.mrisp_paint_python.run_paint_atlas", return_value={"output": "avg_curv"}) as run, \
             patch("fnit.recon_all.native_free.subprocess.run", side_effect=AssertionError("original program called")):
            _run_avg_curv(None, Path("subject"), "rh", Path("atlas.tif"), Path("assets"), python_only=True)
        self.assertEqual(run.call_args.kwargs["frame"], 6)
        self.assertEqual(run.call_args.kwargs["averages"], 5)
        self.assertEqual(run.call_args.kwargs["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
