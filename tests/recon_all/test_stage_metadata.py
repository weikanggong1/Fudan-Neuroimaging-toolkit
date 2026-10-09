"""防止剖析丢失嵌套归一化计时或误把控制点/非法数值计成秒数。"""
import copy
import unittest

from fnit.recon_all.stage_metadata import extract_algorithm_seconds


class StageMetadataTests(unittest.TestCase):
    def test_normalization_nested_and_overlapping_scopes_remain_separate(self):
        value = {"total_seconds": 20.0, "ridge_seconds": 2.0,
                 "steps": {"smoothing_seconds": 3.0},
                 "completion": {"steps": {"smoothing_seconds": 4.0,
                     "three_d_1_controls": {"count": 42}}}}
        original = copy.deepcopy(value)
        self.assertEqual(extract_algorithm_seconds(value), {
            "total_seconds": 20.0, "ridge_seconds": 2.0,
            "steps/smoothing_seconds": 3.0,
            "completion/steps/smoothing_seconds": 4.0})
        self.assertEqual(value, original)

    def test_invalid_metadata_cannot_publish_negative_or_nan_timings(self):
        for value in (None, 12, [1], {"steps": None}, {"completion": []}):
            self.assertEqual(extract_algorithm_seconds(value), {})
        self.assertEqual(extract_algorithm_seconds({"steps": {
            "truth_seconds": True, "nan_seconds": float("nan"),
            "infinite_seconds": float("inf"), "negative_seconds": -1,
            "text_seconds": "1.0", "controls": 4, "zero_seconds": 0}}),
            {"steps/zero_seconds": 0.0})


if __name__ == "__main__":
    unittest.main()
