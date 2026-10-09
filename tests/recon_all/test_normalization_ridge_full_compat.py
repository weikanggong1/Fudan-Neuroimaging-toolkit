"""完整signed_distance默认接口保持完整场、顺序和计数的回归。"""
import importlib.util
import os
from pathlib import Path
import sys
import unittest

import numpy as np

from fnit.recon_all.normalization import normalize_aseg_ridge as reference


if os.environ.get("FNIT_RIDGE_TEST_OVERLAY"):
    path = Path(os.environ["FNIT_RIDGE_TEST_OVERLAY"]) / "normalize_aseg_ridge.py"
    name = "fnit.recon_all.normalization.normalize_aseg_ridge"
    spec = importlib.util.spec_from_file_location(name, path)
    candidate = importlib.util.module_from_spec(spec)
    sys.modules[name] = candidate
    spec.loader.exec_module(candidate)
else:
    candidate = reference


class FullDistanceCompatibilityTests(unittest.TestCase):
    def test_legacy_full_field_and_counts(self):
        rng = np.random.default_rng(872)
        for shape in ((5, 7, 9), (12, 13, 14), (19, 18, 17)):
            for mask in (rng.random(shape) > .65, np.zeros(shape, bool), np.ones(shape, bool)):
                expected, counts = reference.signed_distance(mask=mask)
                actual, actual_counts = candidate.signed_distance(mask=mask)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual_counts, counts)


if __name__ == "__main__":
    unittest.main()
