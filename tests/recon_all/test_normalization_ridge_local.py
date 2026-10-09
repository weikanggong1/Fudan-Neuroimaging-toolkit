"""限域消费完整性的回归；真实输入由独立完整API benchmark核验。"""
import importlib.util
import os
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.ndimage import maximum_filter

from fnit.recon_all.normalization import normalize_aseg_ridge as reference


if os.environ.get("FNIT_RIDGE_TEST_OVERLAY"):
    # 保留reference旧模块对象，仅候选明确两文件从新目录加载。
    root = Path(os.environ["FNIT_RIDGE_TEST_OVERLAY"])
    for name in ("normalize_aseg_ridge", "normalize_ridge_local"):
        full_name = "fnit.recon_all.normalization." + name
        spec = importlib.util.spec_from_file_location(full_name, root / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)

from fnit.recon_all.normalization.normalize_ridge_local import (
    medial_ridge_local, ridge_required_distance,
)


class RidgeRequiredDistanceTests(unittest.TestCase):
    def assert_same_consumer(self, labels):
        mask = (labels == 2) | (labels == 41)
        full, _ = reference.signed_distance(mask)
        local, details = ridge_required_distance(aseg=labels)
        query = maximum_filter(mask, size=3, mode="nearest") != 0
        np.testing.assert_array_equal(local[query], full[query])
        ridge, _ = medial_ridge_local(aseg=labels)
        np.testing.assert_array_equal(ridge, reference._nonmax(full))
        self.assertEqual(local.dtype, np.float32)
        self.assertEqual(details["outside_query_voxels"], int(np.count_nonzero(query & ~mask)))

    def test_random_non_cubic_and_float_storage(self):
        rng = np.random.default_rng(82)
        for shape in ((5, 7, 6), (8, 6, 9), (12, 11, 10)):
            self.assert_same_consumer(np.where(rng.random(shape) > .64, 2, 0).astype(np.float32))

    def test_boundary_disconnected_and_corner_sampling(self):
        labels = np.zeros((13, 12, 11), np.int32)
        labels[0:5, 0:5, 0:5] = 2
        labels[-4:, -5:, -3:] = 41
        labels[5:10, 4:9, 3:8] = 2
        labels[6:8, 5:7, 4:6] = 0
        self.assert_same_consumer(labels)

    def test_all_zero_and_all_wm(self):
        for label in (0, 2):
            self.assert_same_consumer(np.full((6, 7, 8), label, np.int16))

    def test_input_unchanged_and_invalid(self):
        labels = np.zeros((6, 7, 8), np.int32)
        labels[1:5, 1:6, 2:7] = 41
        before = labels.copy()
        medial_ridge_local(aseg=labels)
        np.testing.assert_array_equal(labels, before)
        for invalid in (np.zeros((1, 4, 5)), np.zeros((5, 6)), np.zeros((4, 5, 6), complex)):
            with self.assertRaises(ValueError):
                medial_ridge_local(aseg=invalid)


if __name__ == "__main__":
    unittest.main()
