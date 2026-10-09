"""标准库unittest回归；模拟输入只验证算子，不代替真实T1 benchmark。"""
import os
import unittest

import numpy as np
import torch

from fnit.recon_all.mri_segment import (
    _histogram_kernel, _last_peak, _first_valley,
    histogram_segmentation, intensity_segmentation,
    segment_white_matter,
)
from fnit.recon_all.mri_segment_histogram_torch import (
    _batch_histograms, _smooth_source_order, _classify_histograms,
    histogram_segmentation_torch,
)


def fixture():
    rng = np.random.default_rng(74031)
    array = np.clip(rng.normal(76, 3, (25, 25, 25)), 0, 255).astype(np.uint8)
    array[13:] = np.clip(rng.normal(110, 5, array[13:].shape), 0, 255).astype(np.uint8)
    for x in (7, 10, 13, 16):
        for y in (7, 12, 17):
            for z in (7, 12, 17):
                array[x, y, z] = rng.integers(80, 99)
    image = torch.from_numpy(array)
    labels = intensity_segmentation(image, wm_low=79, wm_hi=125, gray_hi=99)
    return image, labels


def source_smooth(counts, low, high):
    kernel = _histogram_kernel()
    output = np.zeros(counts.shape, np.float32)
    for row in range(len(counts)):
        for center in range(int(low[row]), int(high[row]) + 1):
            total = norm = np.float32(0)
            for index, weight in enumerate(kernel):
                neighbor = center + index - 12
                if low[row] <= neighbor <= high[row]:
                    norm = np.float32(norm + weight)
                    total = np.float32(total + np.float32(weight * counts[row, neighbor]))
            output[row, center] = np.float32(total / norm)
    return output


class HistogramTorchTests(unittest.TestCase):
    def test_integer_counts_preserve_cropped_border_window(self):
        image = torch.arange(7 * 8 * 9).reshape(7, 8, 9).remainder(256).to(torch.uint8)
        points = torch.tensor([[0, 0, 0], [3, 4, 5], [6, 7, 8]], dtype=torch.int32)
        axis = torch.arange(-2, 3, dtype=torch.int32)
        offsets = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
        counts, low, high = _batch_histograms(image, points, offsets)
        for index, (x, y, z) in enumerate(points.tolist()):
            values = image[max(0, x - 2):x + 3, max(0, y - 2):y + 3, max(0, z - 2):z + 3].numpy()
            np.testing.assert_array_equal(counts[index].numpy(), np.bincount(values.ravel(), minlength=256))
            self.assertEqual(low[index].item(), int(values.min()))
            self.assertEqual(high[index].item(), int(values.max()))

    def test_smoothing_preserves_fixed_float32_source_order(self):
        rng = np.random.default_rng(17)
        counts = rng.integers(0, 130, (4, 256), dtype=np.int32)
        low = np.array([0, 31, 75, 123]); high = np.array([255, 160, 75, 255])
        for row in range(4):
            counts[row, :low[row]] = counts[row, high[row] + 1:] = 0
        actual, _ = _smooth_source_order(torch.from_numpy(counts), torch.from_numpy(low),
                                        torch.from_numpy(high), torch.from_numpy(_histogram_kernel()))
        np.testing.assert_array_equal(actual.numpy(), source_smooth(counts, low, high))

    def test_complete_histogram_matches_existing_cpu_on_decisive_candidates(self):
        image, labels = fixture()
        expected = histogram_segmentation(image, labels, wm_low=79, wm_hi=125, gray_hi=99)
        actual = histogram_segmentation_torch(image, labels, wm_low=79, wm_hi=125,
                                              gray_hi=99, batch_size=73)
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())
        self.assertGreater(torch.count_nonzero(actual != labels).item(), 0)

    def test_batch_size_keeps_candidate_set_and_does_not_modify_inputs(self):
        image, labels = fixture()
        original_image, original_labels = image.clone(), labels.clone()
        expected = histogram_segmentation_torch(image, labels, wm_low=79, wm_hi=125,
                                                gray_hi=99, batch_size=128)
        for batch in (1, 17):
            actual = histogram_segmentation_torch(image, labels, wm_low=79, wm_hi=125,
                                                  gray_hi=99, batch_size=batch)
            np.testing.assert_array_equal(actual.numpy(), expected.numpy())
        self.assertTrue(torch.equal(image, original_image))
        self.assertTrue(torch.equal(labels, original_labels))

    def test_binary_lifting_matches_scalar_bimodal_peak_chain(self):
        counts = np.zeros((4, 256), np.int32)
        bins = np.arange(256)
        for row, peak_centers in enumerate(((76, 110), (76, 96, 116),
                                           (72, 88, 104, 120), (73, 85, 97, 109, 121))):
            for peak in peak_centers:
                counts[row] += np.rint(160 * np.exp(-((bins - peak) / 1.8) ** 2)).astype(np.int32)
        low, high = np.full(4, 50), np.full(4, 140)
        smooth = source_smooth(counts, low, high)
        center = np.array([84, 87, 81, 90], np.uint8)
        expected = []
        for row in range(4):
            local = smooth[row, 50:141]
            white = _last_peak(local, 50, 79, 118)
            gray = _last_peak(local, 50, 72, white - 10)
            while gray > 90 and white >= 0:
                white = gray; gray = _last_peak(local, 50, 72, white - 10)
            valley = _first_valley(local, 50, gray + 2, white - 2) if min(white, gray) >= 0 else -1
            expected.append(128 if valley < 0 or valley >= 90 or abs(int(center[row]) - valley) <= 2
                            else (255 if center[row] >= valley else 1))
        actual = _classify_histograms(torch.from_numpy(counts), torch.from_numpy(low),
            torch.from_numpy(high), torch.from_numpy(center), torch.from_numpy(_histogram_kernel()),
            wm_low=79, wm_hi=125, gray_hi=90)
        np.testing.assert_array_equal(actual.numpy(), expected)

    def test_empty_candidates_and_invalid_parameters(self):
        image = torch.zeros((3, 4, 5), dtype=torch.uint8)
        labels = torch.ones_like(image)
        self.assertTrue(torch.equal(histogram_segmentation_torch(image, labels,
            wm_low=79, wm_hi=125, gray_hi=99), labels))
        for kwargs in ({"window": 2}, {"batch_size": 0}, {"wm_low": float("nan")},
                       {"gray_hi": -1}, {"wm_low": 129}):
            params = {"wm_low": 79, "wm_hi": 125, "gray_hi": 99}; params.update(kwargs)
            with self.assertRaises(ValueError):
                histogram_segmentation_torch(image, labels, **params)

    def test_complete_wm_backend_rejects_invalid_options_before_computing(self):
        image = torch.zeros((3, 4, 5), dtype=torch.uint8)
        for kwargs in ({"histogram_backend": "native"}, {"histogram_batch_size": 0}):
            with self.assertRaises(ValueError):
                segment_white_matter(image, **kwargs)

    @unittest.skipUnless(os.environ.get("FNIT_TEST_GPU_DEVICE"), "explicit GPU window not requested")
    def test_explicit_gpu_matches_cpu_without_half_precision(self):
        image, labels = fixture()
        expected = histogram_segmentation_torch(image, labels, wm_low=79, wm_hi=125, gray_hi=99)
        device = torch.device(os.environ["FNIT_TEST_GPU_DEVICE"])
        actual = histogram_segmentation_torch(image.to(device), labels.to(device),
            wm_low=79, wm_hi=125, gray_hi=99, batch_size=73)
        np.testing.assert_array_equal(actual.cpu().numpy(), expected.numpy())
        self.assertEqual(actual.device, device)
        self.assertEqual(actual.dtype, torch.uint8)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
