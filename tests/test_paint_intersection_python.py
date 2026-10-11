"""自有paint/清理文件接口的空间、失败与无原生调用契约。"""

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import nibabel.freesurfer.io as fsio
import numpy as np
import tifffile

from fnit.recon_all.mrisp_paint_python import paint_atlas_values, run_paint_atlas
from fnit.recon_all.mris_remove_intersection_python import remove_intersection_surface, run_remove_intersection,mark_intersections
from fnit.recon_all.inflate_standard_run import run_standard_inflate


class OwnSurfaceFileContracts(unittest.TestCase):
    def setUp(self):
        self.vertices = np.asarray([[100, 0, 0], [-100, 0, 0], [0, 100, 0],
                                    [0, -100, 0], [0, 0, 100], [0, 0, -100]], np.float32)
        self.faces = np.asarray([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
                                 [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]], np.int32)

    def test_paint_constant_and_translated_sphere(self):
        frame = np.full((16, 8), .25, np.float32)
        baseline = paint_atlas_values(vertices=self.vertices, faces=self.faces, atlas_frame=frame)
        translated = paint_atlas_values(vertices=self.vertices + [10, 20, 30], faces=self.faces, atlas_frame=frame)
        np.testing.assert_allclose(baseline, .25, rtol=0, atol=1e-7)
        np.testing.assert_array_equal(translated, baseline)

    def test_paint_integer_tiff_bits_and_frame_six(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            sphere, atlas, output = (folder / name for name in ("sphere", "atlas.tif", "avg_curv"))
            fsio.write_geometry(str(sphere), self.vertices, self.faces)
            images = np.empty((7, 16, 8), np.float32)
            for i in range(7):
                images[i].fill(i * .25)
            tifffile.imwrite(atlas, images.view(np.int32), photometric="minisblack")
            report = run_paint_atlas(atlas_path=atlas, sphere_path=sphere, output_path=output,
                                    frame=6, averages=5, device="cpu")
            np.testing.assert_allclose(fsio.read_morph_data(str(output)), 1.5, rtol=0, atol=1e-6)
            self.assertEqual(report["frame_zero_based"], 6)
            self.assertFalse(report["native_program_called"])

    def test_paint_rejects_bad_frame_before_output(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            output = folder / "output"
            with self.assertRaises(ValueError):
                run_paint_atlas(atlas_path=folder / "absent", sphere_path=folder / "absent",
                                output_path=output, frame=-1)
            self.assertFalse(output.exists())

    def test_paint_rejects_face_and_iteration_contract(self):
        for faces, averages in ((np.asarray([[0, 1, 9]]), 5), (self.faces, .5)):
            with self.assertRaises(ValueError):
                paint_atlas_values(vertices=self.vertices, faces=faces,
                                   atlas_frame=np.ones((16, 8)), averages=averages)

    def test_zero_intersection_preserves_bytes_and_ignores_native_path(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source, output = folder / "orig", folder / "result"
            fsio.write_geometry(str(source), self.vertices, self.faces)
            with patch("subprocess.run", side_effect=AssertionError("native execution forbidden")):
                result = remove_intersection_surface(input_path=source, output_path=output,
                                                     binary="/unavailable/reference-only",
                                                     assets_dir="/unavailable")
            self.assertEqual(result, (0, 0))
            self.assertEqual(source.read_bytes(), output.read_bytes())

    def test_residual_failure_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source, output = folder / "orig", folder / "result"
            fsio.write_geometry(str(source), self.vertices, self.faces)
            output.write_bytes(b"preserve existing output")
            mock_report = {"intersecting_faces_before": 2, "intersecting_faces_after": 1,
                           "marked_vertices": 3}
            with patch("fnit.recon_all.place_surface_final_cleanup.repair_intersections",
                       return_value=(self.vertices.copy(), mock_report)), patch(
                           "fnit.recon_all.place_surface_intersection_marking.mark_source_intersections",
                           return_value=(np.ones(6, bool), 1)):
                with self.assertRaisesRegex(RuntimeError, "1 source"):
                    run_remove_intersection(input_path=source, output_path=output)
            self.assertEqual(output.read_bytes(), b"preserve existing output")

    def test_no_sulc_rejects_ambiguous_options_before_input_read(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            for save_sulc, sulc in ((False, folder/"sulc"), (True, None), ("false", None)):
                with self.assertRaises(ValueError):
                    run_standard_inflate(input_surface=folder/"missing", inflated_output=folder/"inflated",
                                         sulc_output=sulc, save_sulc=save_sulc)
            self.assertEqual(list(folder.iterdir()), [])

    def test_no_sulc_preserves_complete_integration_and_default_iterations(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source, output = folder/"smoothwm.nofix", folder/"inflated.nofix"
            fsio.write_geometry(str(source), self.vertices, self.faces)
            observed = {}
            def integrate(**kwargs):
                observed.update(kwargs)
                return kwargs["xyz"].copy()
            with patch("fnit.recon_all.inflate_standard_run.inflate_updates", side_effect=integrate), patch(
                    "fnit.recon_all.inflate_standard_run.postprocess", side_effect=lambda xyz, *_: xyz), patch(
                    "fnit.recon_all.inflate_standard_run.write_standard_sphere_surface"), patch(
                    "fnit.recon_all.inflate_standard_run.fsio.write_morph_data", side_effect=AssertionError("must not write sulc")):
                result = run_standard_inflate(input_surface=source, inflated_output=output,
                                               sulc_output=None, save_sulc=False, backend="numpy", device="cpu")
            self.assertEqual(observed["niterations"], 10)
            self.assertEqual(observed["sulc"].shape, (len(self.vertices),))
            self.assertIsNone(result["sulc_output"])
            self.assertFalse(result["save_sulc"])

if __name__ == "__main__":
    unittest.main()
