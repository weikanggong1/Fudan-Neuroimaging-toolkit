"""MGH 输入经 FNIT NIfTI 容器及 MGZ 写回后保持空间。"""

import nibabel as nib
import numpy as np

from fnit._nib import new_image


def test_mgh_new_image_preserves_oblique_affine_on_mgz_save(tmp_path):
    affine = np.array([[0, -1, 0, 12], [0, 0, 1, -8], [-1, 0, 0, 19],
                       [0, 0, 0, 1]], dtype=np.float64)
    source = nib.MGHImage(np.ones((4, 5, 6), dtype=np.uint8), affine)
    output = new_image(np.zeros(source.shape, dtype=np.uint8), source)
    np.testing.assert_allclose(output.affine, source.affine, atol=1e-6)
    path = tmp_path / "mask.mgz"
    output.save(path)
    np.testing.assert_allclose(nib.load(str(path)).affine, source.affine, atol=1e-6)
