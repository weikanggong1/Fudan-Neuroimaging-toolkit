"""亮非白质整数阈值契约；人工图只测边界，不替代真实WM benchmark。"""
import unittest

import numpy as np
import torch

from fnit.recon_all.mri_segment import remove_bright_nonwhite


class BrightNonwhiteTests(unittest.TestCase):
    def inputs(self, count):
        image = np.zeros((11, 11, 11), np.uint8)
        wm = np.zeros_like(image)
        image[5, 5, 5] = 130
        neighbors = [(x, y, z) for z in range(4, 7) for y in range(4, 7)
                     for x in range(4, 7) if (x, y, z) != (5, 5, 5)]
        for point in neighbors[:count]:
            image[point] = 110
            wm[point] = 100
        return torch.from_numpy(image), torch.from_numpy(wm)

    def test_thirteen_neighbors_do_not_start_bright_mask(self):
        image, wm = self.inputs(13)
        before_image, before_wm = image.clone(), wm.clone()
        torch.testing.assert_close(remove_bright_nonwhite(image, wm), wm, rtol=0, atol=0)
        torch.testing.assert_close(image, before_image, rtol=0, atol=0)
        torch.testing.assert_close(wm, before_wm, rtol=0, atol=0)

    def test_twelve_neighbors_start_mask_and_remove_adjacent_bright_wm(self):
        image, wm = self.inputs(12)
        output = remove_bright_nonwhite(image, wm)
        self.assertEqual(int(torch.count_nonzero(output)), 0)
        self.assertEqual(int(torch.count_nonzero(wm)), 12)

    def test_invalid_grid_or_dtype_is_rejected(self):
        image, wm = self.inputs(13)
        for invalid in (wm.float(), wm[0], wm[:, :, :-1]):
            with self.assertRaises(ValueError):
                remove_bright_nonwhite(image, invalid)


if __name__ == "__main__":
    unittest.main()
