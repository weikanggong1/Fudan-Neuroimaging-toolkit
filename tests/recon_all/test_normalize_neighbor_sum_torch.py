"""GPU邻域算子的同输入回归；小网格不替代真实归一化benchmark。"""
import os
import unittest

import numpy as np
import torch

from fnit.recon_all.normalization.normalize_3d_controls import controls_3d, _neighbor_sum
from fnit.recon_all.normalization.normalize_neighbor_sum_torch import NeighborSumTorch


class NeighborGuardTests(unittest.TestCase):
    def test_explicit_backend_and_device_required(self):
        image = np.ones((4, 4, 4), dtype=np.float32)
        for kwargs in ({"neighbor_backend": "other"}, {"neighbor_backend": "torch"},
                       {"neighbor_backend": "torch", "device": "cpu"}):
            with self.assertRaises(ValueError):
                controls_3d(source=image, **kwargs)
        with self.assertRaises(ValueError):
            NeighborSumTorch(image, np.ones((3, 3, 3)), (slice(0, 4),) * 3, device="cpu")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for exact neighborhood regression")
class NeighborCudaTests(unittest.TestCase):
    device = os.environ.get("FNIT_TEST_GPU_DEVICE", "cuda:0")

    def test_initialized_cuda0_target_other_device_preserves_parent(self):
        target = torch.device(self.device)
        if torch.cuda.device_count() < 2 or target.index in (None, 0):
            self.skipTest("requires explicit second CUDA device")
        previous = torch.cuda.current_device()
        try:
            torch.cuda.set_device(0)
            parent_tensor = torch.full((1,), 19., device="cuda:0")
            torch.cuda.synchronize("cuda:0")
            image = np.arange(8 * 7 * 6, dtype=np.float32).reshape(8, 7, 6) / 3
            control = image > 30
            kernel = np.ones((3, 3, 3), dtype=np.int16)
            region = (slice(0, 8), slice(0, 7), slice(0, 6))
            context = NeighborSumTorch(image=image, kernel=kernel, region=region, device=self.device)
            self.assertEqual(torch.cuda.current_device(), 0)
            expected = _neighbor_sum(image=image, control=control, kernel=kernel, region=region)
            actual = context(control=control)
            np.testing.assert_array_equal(expected[0], actual[0])
            np.testing.assert_array_equal(expected[1], actual[1])
            self.assertEqual(torch.cuda.current_device(), 0)
            self.assertEqual(parent_tensor.item(), 19.)
        finally:
            torch.cuda.set_device(previous)

    def test_six_cube_and_nearest_boundaries(self):
        rng = np.random.default_rng(93)
        shape = (8, 9, 7)
        image = (rng.random(shape) * 255).astype(np.float32)
        control = rng.random(shape) > .45
        original_image, original_control = image.copy(), control.copy()
        six = np.zeros((3, 3, 3), dtype=np.int16)
        six[0, 1, 1] = six[2, 1, 1] = 1
        six[1, 0, 1] = six[1, 2, 1] = 1
        six[1, 1, 0] = six[1, 1, 2] = 1
        for kernel in (six, np.ones((3, 3, 3), dtype=np.int16)):
            for region in ((slice(0, 8), slice(0, 9), slice(0, 7)),
                           (slice(0, 2), slice(6, 9), slice(0, 1)),
                           (slice(3, 6), slice(1, 5), slice(2, 6))):
                context = NeighborSumTorch(image=image, kernel=kernel, region=region, device=self.device)
                expected = _neighbor_sum(image=image, control=control, kernel=kernel, region=region)
                actual = context(control=control)
                np.testing.assert_array_equal(expected[0], actual[0])
                np.testing.assert_array_equal(expected[1], actual[1])
                pointer = actual[0].__array_interface__["data"][0]
                alternative = ~control
                expected = _neighbor_sum(image=image, control=alternative, kernel=kernel, region=region)
                second = context(control=alternative)
                self.assertEqual(pointer, second[0].__array_interface__["data"][0])
                np.testing.assert_array_equal(expected[0], second[0])
                np.testing.assert_array_equal(expected[1], second[1])
        np.testing.assert_array_equal(image, original_image)
        np.testing.assert_array_equal(control, original_control)

    def test_full_control_feedback_and_details(self):
        image = np.zeros((32, 31, 30), dtype=np.float32)
        image[6:15, 6:15, 6:15] = 110.
        image[14:27, 9:12, 9:12] = 105.25
        reference, details = controls_3d(source=image, wm_peak=110., gm_peak=75.)
        candidate, gpu_details = controls_3d(source=image, wm_peak=110., gm_peak=75.,
            neighbor_backend="torch", device=self.device)
        np.testing.assert_array_equal(reference, candidate)
        self.assertEqual(details, gpu_details)
        self.assertGreater(details["three_added"] + details["six_added"], 0)


if __name__ == "__main__":
    unittest.main()
