"""固定单 T1 流程的入口和输出契约。"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fnit.recon_all.expected_outputs import paths
from fnit.recon_all.native_free import (
    _project_exvivo_annotations, _run_defects_volume, _run_torch_wm_edit,
    _run_accurate_sphere_pair, _run_torch_wm_segment, main, run_recon_all_python,
)


class StandardReconWiringTest(unittest.TestCase):
    def test_fixed_output_manifest_has_138_unique_paths(self):
        expected = paths()
        self.assertEqual(len(expected), 138)
        self.assertEqual(len(set(expected)), 138)

    def test_no_partial_surface_switches(self):
        with patch("fnit.recon_all.native_free.run_recon_all_python",
                   return_value={"status": "complete"}) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                main(["t1.nii.gz", "subject", "--weights-dir", "weights",
                      "--assets-dir", "assets", "--native-bin-dir", "bin"])
            self.assertEqual(run.call_args.kwargs["native_bin_dir"], Path("bin"))
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main(["t1.nii.gz", "subject", "--weights-dir", "weights",
                          "--assets-dir", "assets", "--native-topology"])

    def test_missing_full_chain_assets_fail_before_subject_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "t1.nii.gz").write_bytes(b"image")
            for name in ("weights", "assets", "bin"):
                (root / name).mkdir()
            with self.assertRaisesRegex(FileNotFoundError,
                                        "synthmorph.affine.2.h5"):
                run_recon_all_python(root / "t1.nii.gz", root / "subject",
                                     root / "weights", root / "assets",
                                     native_bin_dir=root / "bin")
            self.assertFalse((root / "subject").exists())

    def test_defect_volume_uses_two_hemisphere_reference_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            subject = Path(directory) / "subject"
            for folder in ("mri", "surf", "label", "scripts"):
                (subject / folder).mkdir(parents=True)
            (subject / "mri/orig.mgz").write_bytes(b"orig")
            for hemi in ("lh", "rh"):
                (subject / f"surf/{hemi}.orig.nofix").write_bytes(b"surf")
                (subject / f"surf/{hemi}.defect_labels").write_bytes(b"defects")
                (subject / f"label/{hemi}.nofix.cortex.label").write_bytes(b"cortex")

            def run(args, **kwargs):
                Path(args[-2]).write_bytes(b"volume")

            with patch("fnit.recon_all.native_free.subprocess.run", side_effect=run) as call:
                _run_defects_volume(Path("/bin/mri_label2vol"), subject, "lh", Path("/assets"))
                _run_defects_volume(Path("/bin/mri_label2vol"), subject, "rh", Path("/assets"))
            left, right = [item.args[0] for item in call.call_args_list]
            self.assertEqual(left[1], "--defects")
            self.assertEqual(left[4], str(subject / "mri/orig.mgz"))
            self.assertEqual(left[5:7], ["1000", "0"])
            self.assertEqual(right[4], str(subject / "mri/surface.defects.mgz"))
            self.assertEqual(right[5:7], ["2000", "1"])

    def test_exvivo_annotation_keeps_official_tie_order(self):
        with patch("fnit.recon_all.label2label_surface_python.SurfaceLabelMapper"), \
                patch("fnit.recon_all.label2annot_python.write_label_annotation") as annot:
            _project_exvivo_annotations(Path("/subject"), Path("/assets"))
        left_ba = [path.name for path in annot.call_args_list[0].args[2]]
        self.assertEqual(left_ba, [
            "lh.BA1_exvivo.label", "lh.BA2_exvivo.label",
            "lh.BA3a_exvivo.label", "lh.BA3b_exvivo.label",
            "lh.BA4a_exvivo.label", "lh.BA4p_exvivo.label",
            "lh.BA6_exvivo.label", "lh.BA44_exvivo.label",
            "lh.BA45_exvivo.label", "lh.V1_exvivo.label",
            "lh.V2_exvivo.label", "lh.MT_exvivo.label",
            "lh.perirhinal_exvivo.label", "lh.entorhinal_exvivo.label",
        ])

    def test_torch_defects_preserves_merge_order_without_native_command(self):
        with tempfile.TemporaryDirectory() as directory:
            subject = Path(directory)
            for folder in ("mri", "surf", "label"):
                (subject / folder).mkdir()
            (subject / "mri/orig.mgz").write_bytes(b"orig")
            for hemi in ("lh", "rh"):
                (subject / f"surf/{hemi}.defect_labels").write_bytes(b"defects")
                (subject / f"label/{hemi}.nofix.cortex.label").write_bytes(b"cortex")
            def project(**kwargs):
                kwargs["output_file"].write_bytes(b"volume")
                return {"device": kwargs["device"]}
            with patch("fnit.recon_all.defects_label_volume_torch.defects_to_volume",
                       side_effect=project) as call, \
                    patch("fnit.recon_all.native_free.subprocess.run") as command:
                for hemi in ("lh", "rh"):
                    result = _run_defects_volume(None, subject, hemi, Path("/assets"),
                                                 backend="torch", device="cuda:1")
                    self.assertEqual(result["device"], "cuda:1")
            left, right = [item.kwargs for item in call.call_args_list]
            self.assertEqual(left["template_file"], subject / "mri/orig.mgz")
            self.assertEqual((left["offset"], left["merge"]), (1000, False))
            self.assertEqual(right["template_file"], subject / "mri/surface.defects.mgz")
            self.assertEqual((right["offset"], right["merge"]), (2000, True))
            command.assert_not_called()

    def test_torch_defects_cli_option_reaches_api(self):
        with patch("fnit.recon_all.native_free.run_recon_all_python",
                   return_value={"status": "complete"}) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                main(["t1.nii.gz", "subject", "--weights-dir", "weights",
                      "--assets-dir", "assets", "--defects-backend", "torch",
                      "--wm-edit-backend", "torch-hybrid", "--sphere-normals-backend", "torch"])
        self.assertEqual(run.call_args.kwargs["defects_backend"], "torch")
        self.assertEqual(run.call_args.kwargs["wm_edit_backend"], "torch-hybrid")
        self.assertEqual(run.call_args.kwargs["sphere_normals_backend"], "torch")

    def test_torch_wm_edit_uses_only_candidate_inputs(self):
        with patch("fnit.recon_all.edit_wm_aseg_torch.write_wm_asegedit_hybrid_diagnostic",
                   return_value={"device": "cuda:1"}) as edit:
            _run_torch_wm_edit(Path("/subject/mri"), device="cuda:1")
        self.assertEqual(edit.call_args.kwargs, {
            "wm_file": Path("/subject/mri/wm.seg.mgz"),
            "brain_file": Path("/subject/mri/brain.mgz"),
            "aseg_file": Path("/subject/mri/aseg.presurf.mgz"),
            "entowm_file": Path("/subject/mri/entowm.mgz"),
            "output_file": Path("/subject/mri/wm.asegedit.mgz"),
            "device": "cuda:1", "fill_seg_wm": True})

    def test_sphere_normals_backend_is_explicit_and_keeps_finish_cpu(self):
        with patch("fnit.recon_all.native_free._run_native_sphere_step", return_value=1.), \
                patch("fnit.recon_all.sphere_standard_run.run_standard_sphere",
                      return_value={"total_seconds_including_io": 2.}) as sphere:
            _run_accurate_sphere_pair(Path("/bin/inflate"), Path("/subject"), "lh",
                                      Path("/assets"), device="cuda:1", normals_backend="torch")
        self.assertEqual(sphere.call_args.kwargs, {
            "finish_device": "cpu", "averaging_device": "cuda:1", "normals_device": "cuda:1"})

    def test_performance_candidates_reach_cli_api_without_changing_defaults(self):
        with patch("fnit.recon_all.native_free.run_recon_all_python",
                   return_value={"status": "complete"}) as run, \
                contextlib.redirect_stdout(io.StringIO()):
            main(["t1.nii.gz", "subject", "--weights-dir", "weights", "--assets-dir", "assets",
                  "--wm-backend", "torch-optimized", "--gca-inverse-backend", "torch",
                  "--gca-candidate-chunk", "1024", "--gca-execution", "isolated", "--fill-backend", "numba"])
        self.assertEqual(run.call_args.kwargs["wm_backend"], "torch-optimized")
        self.assertEqual(run.call_args.kwargs["gca_inverse_backend"], "torch")
        self.assertEqual(run.call_args.kwargs["gca_candidate_chunk"], 1024)
        self.assertEqual(run.call_args.kwargs["gca_execution"], "isolated")
        self.assertEqual(run.call_args.kwargs["fill_backend"], "numba")

    def test_optimized_wm_reuses_existing_kernels_and_candidate_inputs(self):
        with patch("fnit.recon_all.mri_segment.segment_white_matter_mgz",
                   return_value={"device": "cuda:1"}) as segment:
            _run_torch_wm_segment(Path("/subject/mri"), device="cuda:1", optimized=True)
        self.assertEqual(segment.call_args.args, (
            Path("/subject/mri/antsdn.brain.mgz"), Path("/subject/mri/wm.seg.mgz")))
        self.assertEqual(segment.call_args.kwargs, {
            "device": "cuda:1", "histogram_backend": "torch", "histogram_batch_size": 2048,
            "planar_backend": "cached", "planar_batch_size": 256})

    def test_native_gca_cannot_silently_ignore_torch_candidate_options(self):
        with tempfile.TemporaryDirectory() as directory:
            subject = Path(directory) / "subject"
            with self.assertRaisesRegex(ValueError, "CUDA Torch GCA"):
                run_recon_all_python("missing.nii.gz", subject, "weights", "assets",
                                     native_optimizations="original", gca_inverse_backend="torch")
            self.assertFalse(subject.exists())


if __name__ == "__main__":
    unittest.main()
