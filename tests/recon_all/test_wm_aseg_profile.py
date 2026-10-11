"""标准WM选项的有序CPU契约；真实数据由独立benchmark验证。"""
import tempfile
from pathlib import Path
import unittest

import nibabel as nib
import numpy as np

from fnit.recon_all.edit_wm_aseg_core_python import edit_wm_aseg_profile, write_wm_asegedit_profile


class WMProfileTests(unittest.TestCase):
    def arrays(self):
        shape = (16, 17, 18)
        wm = np.zeros(shape, np.uint8)
        brain = np.full(shape, 110, np.uint8)
        aseg = np.zeros(shape, np.int32)
        aseg[4:12, 4:12, 4:12] = 2
        ento = np.zeros(shape, np.int32)
        return wm, brain, aseg, ento

    def test_fill_is_optional_and_inputs_are_immutable(self):
        wm, brain, aseg, ento = self.arrays()
        initial = [a.copy() for a in (wm, brain, aseg, ento)]
        filled, report = edit_wm_aseg_profile(wm, brain, aseg, ento, fill_seg_wm=True)
        unfilled, _ = edit_wm_aseg_profile(wm, brain, aseg, ento, fill_seg_wm=False)
        self.assertEqual(int(filled[7, 7, 7]), 250)
        self.assertEqual(int(unfilled[7, 7, 7]), 0)
        self.assertFalse(report["used_native_programs"])
        for a, b in zip((wm, brain, aseg, ento), initial):
            np.testing.assert_array_equal(a, b)

    def test_keepin_and_ento_are_applied_after_core(self):
        wm, brain, aseg, ento = self.arrays()
        wm[7, 7, 7] = 1
        ento[8, 8, 8] = 3006
        output, _ = edit_wm_aseg_profile(wm, brain, aseg, ento)
        self.assertEqual(int(output[7, 7, 7]), 1)
        self.assertEqual(int(output[8, 8, 8]), 255)

    def test_active_byte_path_without_seed_completes(self):
        wm, brain, aseg, ento = self.arrays()
        aseg[2, 7, 7] = 17
        output, report = edit_wm_aseg_profile(wm, brain, aseg, ento)
        self.assertEqual(output.dtype, np.uint8)
        self.assertEqual(report["paths_filled"], 0)
        self.assertFalse(report["used_native_programs"])

    def test_bad_array_labels_and_bool_option_are_rejected(self):
        wm, brain, aseg, ento = self.arrays()
        with self.assertRaises(ValueError):
            edit_wm_aseg_profile(wm, brain, aseg.astype(np.int16), ento)
        with self.assertRaises(ValueError):
            edit_wm_aseg_profile(wm, brain, aseg, ento, fill_seg_wm=1)

    def test_file_rejects_original_small_storage_before_conversion(self):
        wm, brain, aseg, ento = self.arrays()
        with tempfile.TemporaryDirectory() as temporary:
            paths = [Path(temporary)/name for name in ("wm.mgz", "brain.mgz", "aseg.mgz", "ento.mgz", "out.mgz")]
            for array, path in zip((wm, brain, aseg.astype(np.int16), ento), paths):
                nib.save(nib.MGHImage(array, np.eye(4)), str(path))
            with self.assertRaisesRegex(ValueError, "original int32 or float32"):
                write_wm_asegedit_profile(*paths)
            self.assertFalse(paths[-1].exists())


if __name__ == "__main__":
    unittest.main()
