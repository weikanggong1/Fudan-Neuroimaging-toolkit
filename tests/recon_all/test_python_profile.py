"""独立Python完整配方/候选拓扑接线；真实影像精度另行验证。"""
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from fnit.recon_all.native_free import _prepare_native_topology, run_recon_all_python
from fnit.recon_all.python_profile import python_profile_options


class PythonProfileTest(unittest.TestCase):
    def test_profile_allows_cpu_kernels_and_avoids_upstream_stage_backends(self):
        value = python_profile_options(device="cuda:1", threads=4)
        self.assertEqual(value["n4_backend"], "simpleitk")
        self.assertEqual(value["wm_edit_backend"], "python")
        self.assertEqual(value["topology_backend"], "python")
        self.assertEqual(value["hemisphere_workers"], 2)
        self.assertEqual(value["gca_execution"], "isolated")
        self.assertEqual(value["gca_inverse_backend"], "torch")
        self.assertEqual(value["gca_candidate_chunk"], 1024)
        self.assertEqual(value["normalization_controls_backend"], "torch")
        self.assertEqual(value["normalization_initial_bias_backend"], "torch")
        self.assertEqual(value["sphere_finish_backend"], "torch")
        self.assertEqual(value["mni_execution"], "parallel-late")
        self.assertNotIn("native", value.values())
        for device in ("cpu", "cuda", "mps"):
            with self.assertRaises(ValueError):
                python_profile_options(device=device, threads=4)
        for threads in (True, 1, 1.5):
            with self.assertRaises(ValueError):
                python_profile_options(device="cuda:0", threads=threads)

    def test_public_api_passes_complete_recipe_to_pipeline(self):
        with tempfile.TemporaryDirectory() as folder:
            subject = Path(folder) / "subject"
            subject.mkdir()
            def complete(**kwargs):
                return {"status": "complete", "total_seconds": 0.0}
            with patch("fnit.recon_all.native_free._run_recon_all_python", side_effect=complete) as run:
                result = run_recon_all_python(t1="input", subject_dir=subject,
                    weights_dir="weights", assets_dir="assets", device="cuda:1",
                    threads=4, backend="python")
            for key, value in python_profile_options(device="cuda:1", threads=4).items():
                self.assertEqual(run.call_args.kwargs.get(key, "in-process"), value)
            self.assertEqual(run.call_args.kwargs["backend"], "python")
            self.assertEqual(result["status"], "complete")

    def test_topology_candidate_preserves_nofix_and_requires_full_remesh_cleanup(self):
        with tempfile.TemporaryDirectory() as folder:
            subject = Path(folder)
            (subject / "surf").mkdir()
            (subject / "scripts").mkdir()
            surf = subject / "surf"
            (surf / "lh.orig.nofix").write_bytes(b"own-input")
            calls = []
            def copy_stage(source, output, **kwargs):
                calls.append(Path(output).name)
                if Path(source) != Path(output):
                    shutil.copyfile(source, output)
            def inflate(**kwargs):
                self.assertFalse(kwargs["save_sulc"])
                self.assertIsNone(kwargs["sulc_output"])
                self.assertNotIn("niterations", kwargs)
                shutil.copyfile(kwargs["input_surface"], kwargs["inflated_output"])
                return {"total_seconds_including_io": 0.1}
            def topology(**kwargs):
                self.assertEqual(kwargs["source_surface"], surf / "lh.orig.nofix")
                self.assertEqual(kwargs["output_defect_labels"], surf / "lh.defect_labels")
                Path(kwargs["output_surface"]).write_bytes(b"complete-repair")
            with patch("fnit.recon_all.smooth_surface_python.smooth_surface", side_effect=copy_stage), \
                 patch("fnit.recon_all.inflate_standard_run.run_standard_inflate", side_effect=inflate), \
                 patch("fnit.recon_all.sphere_quick_python.write_quick_sphere", side_effect=copy_stage), \
                 patch("fnit.recon_all.topology_volume_python.run_volume_topology", side_effect=topology), \
                 patch("fnit.recon_all.mris_remesh_python.remesh_surface", side_effect=copy_stage), \
                 patch("fnit.recon_all.mris_remove_intersection_python.remove_intersection_surface", side_effect=copy_stage), \
                 patch("fnit.recon_all.native_free.subprocess.run", side_effect=AssertionError("stage program called")):
                value = _prepare_native_topology(None, subject, "lh", subject,
                    "cpu", None, None, python_auxiliary=True,
                    topology_backend="python", pretess_file=subject / "pretess.mgz")
            self.assertEqual((surf / "lh.orig.nofix").read_bytes(), b"own-input")
            self.assertEqual((surf / "lh.orig").read_bytes(), b"complete-repair")
            self.assertEqual(value[1], 0.0)
            self.assertEqual(calls[-2:], ["lh.orig", "lh.orig"])


if __name__ == "__main__":
    unittest.main()
