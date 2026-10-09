"""余子式批量迁移保留原顺序、病态输入与平移，不改用不同求逆算法。"""
import unittest
import numpy as np
import torch

from fnit.recon_all.mri_em_register import _vnl_affine_inverse
from fnit.recon_all.mri_em_register_score_gpu import vnl_affine_inverse_tensor


class GCAInverseTest(unittest.TestCase):
    def test_same_cofactor_bits_for_well_and_ill_conditioned_affines(self):
        rng = np.random.default_rng(147)
        matrices = np.repeat(np.eye(4, dtype=np.float32)[None], 128, axis=0)
        matrices[:, :3, :3] += rng.normal(0, .08, (128, 3, 3)).astype(np.float32)
        matrices[:, :3, 3] = rng.normal(0, 50, (128, 3)).astype(np.float32)
        matrices[-1, :3, 0] *= np.float32(.0001)
        expected = np.stack([_vnl_affine_inverse(matrix) for matrix in matrices])
        actual = vnl_affine_inverse_tensor(torch.as_tensor(matrices)).numpy()
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

    def test_rejects_dtype_and_shape_without_device_mutation(self):
        for invalid in (torch.eye(4), torch.eye(4)[None].double()):
            with self.assertRaises(ValueError):
                vnl_affine_inverse_tensor(invalid)


if __name__ == "__main__":
    unittest.main()
