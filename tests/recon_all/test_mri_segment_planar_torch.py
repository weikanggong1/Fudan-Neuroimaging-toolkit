"""固定平面几何与逐点标签更新回归；模拟输入不替代真实WM benchmark。"""
import os
import unittest

import numpy as np
import torch

from fnit.recon_all.mri_segment import _fill_planar_holes, _plane_bases, _plane_positions, _sample_trilinear
from fnit.recon_all.mri_segment_planar_torch import fill_planar_holes_cached, plane_indices_torch, _sample_ray


class PlanarHolesTests(unittest.TestCase):
    def test_geometry_rounding_clipping_and_batch_independence(self):
        _, first, second = _plane_bases()
        offsets = _plane_positions(np.zeros((22, 3), np.float32), first, second, 5)
        points = np.asarray([[0,0,0],[1,2,3],[10,9,8],[15,14,13]], np.int64)
        shape = (16,15,14)
        positions = points[:,None,None,None,:].astype(np.float32) + offsets[None]
        expected = np.floor(positions.astype(np.float64) + .5).astype(np.int64)
        expected = np.clip(expected, 0, np.asarray(shape)-1)
        linear = ((expected[...,0]*shape[1]+expected[...,1])*shape[2]+expected[...,2]).reshape(-1,22,25)
        for batch in (1, 3, 256):
            np.testing.assert_array_equal(plane_indices_torch(points, offsets, shape, batch_size=batch), linear)

    def test_ray_interpolation_source_order(self):
        rng = np.random.default_rng(95473)
        binary = (rng.random((13,14,15)) < .4).astype(np.uint8)
        positions = (rng.random((400,3))*np.asarray(binary.shape)-.5).astype(np.float32)
        expected = _sample_trilinear(binary, positions)
        actual = np.asarray([_sample_ray(binary, *point) for point in positions],np.float32)
        np.testing.assert_array_equal(actual, expected)

    def test_complete_ordered_fill_on_irregular_masks(self):
        rng = np.random.default_rng(8317)
        for repetition in range(3):
            strand = np.zeros((19,18,17), np.uint8)
            strand[4:15,4:14,4:13] = np.where(rng.random((11,10,9))<.75,200,0).astype(np.uint8)
            result = strand.copy()
            result[8:10,8:10,7:9] = 0
            before = strand.copy()
            expected, actual = result.copy(), result.copy()
            _fill_planar_holes(expected, strand)
            report = fill_planar_holes_cached(actual, strand, batch_size=73)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(strand, before)
            self.assertGreater(report["candidates"],0)

    def test_empty_and_invalid_inputs(self):
        empty = np.zeros((9,10,11),np.uint8)
        actual = empty.copy()
        self.assertEqual(fill_planar_holes_cached(actual,empty)["added_voxels"],0)
        np.testing.assert_array_equal(actual,empty)
        with self.assertRaises(ValueError):
            fill_planar_holes_cached(actual.astype(np.int32),empty)
        offsets = np.zeros((22, 5, 5, 3), np.float32)
        points = np.zeros((1, 3), np.int64)
        with self.assertRaises(ValueError):
            plane_indices_torch(points, offsets, (9.0, 10, 11))
        with self.assertRaises(ValueError):
            plane_indices_torch(points - 1, offsets, (9, 10, 11))
        offsets[0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            plane_indices_torch(points, offsets, (9, 10, 11))

    def test_crop_preserves_physical_boundary_and_disjoint_components(self):
        strand = np.zeros((19, 18, 17), np.uint8)
        strand[0:5, 5:10, 5:10] = 200
        strand[10:14, 8:12, 8:12] = 200
        result = strand.copy()
        for axis in range(3):
            last = [slice(None)] * 3
            last[axis] = -1
            result[tuple(last)] = 200
        expected, actual = result.copy(), result.copy()
        _fill_planar_holes(expected, strand)
        report = fill_planar_holes_cached(actual, strand)
        np.testing.assert_array_equal(actual, expected)
        self.assertLessEqual(report["geometry_candidates"], report["candidates"])

    @unittest.skipUnless(os.environ.get("FNIT_TEST_GPU_DEVICE"),"explicit GPU window not requested")
    def test_gpu_static_geometry_identical_cpu(self):
        _, first, second = _plane_bases()
        offsets = _plane_positions(np.zeros((22,3),np.float32),first,second,5)
        points = np.random.default_rng(281).integers(0,20,size=(193,3),dtype=np.int64)
        cpu = plane_indices_torch(points,offsets,(20,20,20),device="cpu",batch_size=32)
        gpu = plane_indices_torch(points,offsets,(20,20,20),device=os.environ["FNIT_TEST_GPU_DEVICE"],batch_size=71)
        np.testing.assert_array_equal(cpu,gpu)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
