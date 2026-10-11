"""球面图谱frame/坐标/顶点顺序契约；真实benchmark另见公开收据。"""
import tempfile
from pathlib import Path
import unittest
import nibabel.freesurfer.io as fsio
import numpy as np
import tifffile
from fnit.recon_all.mrisp_paint_python import paint_atlas_values, run_paint_atlas


class AtlasPaintContracts(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
