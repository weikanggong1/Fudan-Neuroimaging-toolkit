"""完整曲率入口的数学来源、顺序累计和文件/掩膜契约；真实benchmark另列。"""
from pathlib import Path
import ctypes
import ctypes.util
import tempfile
import unittest
from unittest.mock import patch

import nibabel.freesurfer.io as fsio
import numpy as np

from fnit.recon_all.discrete_curvature_torch import _source_acosf
from fnit.recon_all.curvature_stats_python import (_ordered_statistics,
    _average_texture, write_curvature_stats)


class FullCurvatureStatsTests(unittest.TestCase):
    def test_numba_acosf_matches_system_source_math(self):
        library = ctypes.CDLL(ctypes.util.find_library("m"))
        library.acosf.argtypes = [ctypes.c_float]
        library.acosf.restype = ctypes.c_float
        values = np.linspace(-1, 1, 257, dtype=np.float32)
        expected = np.array([library.acosf(float(v)) for v in values], np.float32)
        np.testing.assert_array_equal(_source_acosf(values), expected)

    def test_ordered_statistics_keeps_source_precision_and_sign_groups(self):
        values = np.array([-2, 0, 4], np.float32)
        areas = np.array([1, 2, 3], np.float32)
        out = _ordered_statistics(values, areas, np.zeros(3, np.bool_))
        self.assertEqual(out[4:7], (0, 2, 3))
        np.testing.assert_array_equal(out[7][:, 0], [10, 14, 12, 2])
        np.testing.assert_array_equal(out[7][:, 1], [3, 3, 2, 1])
        # 源函数少于两点不计算mean/areaNorm，负组只含一个顶点。
        np.testing.assert_array_equal(out[7][3, 3:], [0, 0])

    def test_ripped_first_vertex_still_initializes_native_minmax(self):
        out = _ordered_statistics(np.array([-99, 1, 2], np.float32),
            np.ones(3, np.float32), np.array([1, 0, 0], np.bool_))
        self.assertEqual(float(out[2]), -99)
        self.assertEqual(out[4], 0)
        self.assertEqual(out[6], 2)

    def test_average_texture_preserves_ripped_state(self):
        values = np.array([1, 3, 99], np.float32)
        neighbors = np.array([[1, 2], [0, 2], [0, 1]], np.int32)
        degree = np.full(3, 2, np.int32)
        out = _average_texture(values, neighbors, degree, np.array([0, 0, 1], np.bool_), 1)
        np.testing.assert_array_equal(out, [2, 2, 99])

    def fixture(self, root):
        (root / "surf").mkdir()
        xyz = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], np.float32)
        faces = np.array([[0,2,4],[2,1,4],[1,3,4],[3,0,4],[2,0,5],[1,2,5],[3,1,5],[0,3,5]], np.int32)
        fsio.write_geometry(str(root / "surf/lh.smoothwm"), xyz, faces)
        fsio.write_morph_data(str(root / "surf/lh.curv"), np.arange(6, dtype=np.float32)-2)
        fsio.write_morph_data(str(root / "surf/lh.sulc"), np.arange(6, dtype=np.float32)-3)

    def test_complete_eight_maps_and_ten_stat_sections_without_commands(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            with patch("subprocess.run", side_effect=AssertionError("external command forbidden")):
                report = write_curvature_stats(subject_dir=root, hemisphere="lh", device="cpu")
            self.assertEqual(len(report["outputs"]), 9)
            for name, path in report["outputs"].items():
                self.assertTrue(Path(path).is_file())
                if name != "stats":
                    self.assertEqual(fsio.read_morph_data(path).shape, (6,))
            text = Path(report["outputs"]["stats"]).read_text()
            self.assertEqual(text.count("<mean> +- <std>"), 10)
            self.assertEqual(text.count("AreaNorm Negative Surface Integral:"), 10)
            self.assertIn("Mean across 10 curvatures:", text)
            with self.assertRaises(FileExistsError):
                write_curvature_stats(subject_dir=root, hemisphere="lh", device="cpu")

    def test_input_alias_and_invalid_parameters_fail_before_write(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            with self.assertRaises(ValueError):
                write_curvature_stats(subject_dir=root, hemisphere="lh", device="cpu",
                                      stats_file=root / "surf/lh.curv")
            with self.assertRaises(ValueError):
                write_curvature_stats(subject_dir=root, hemisphere="lh", device="cpu", averages=-1)
            with self.assertRaises(ValueError):
                write_curvature_stats(subject_dir=root, hemisphere="lh", device="cpu", low_pass=float("nan"))
            label = root/'lh.cortex.label'
            label.write_text('# test\n1\n0 1 0 0 0\n')
            with self.assertRaises(ValueError):
                write_curvature_stats(subject_dir=root, hemisphere="lh", device="cpu", label_file=label,
                                      stats_file=label,overwrite=True)
            with self.assertRaises(ValueError):
                write_curvature_stats(subject_dir=root, hemisphere="lh", device="cpu", low_pass=0)


if __name__ == "__main__":
    unittest.main()
