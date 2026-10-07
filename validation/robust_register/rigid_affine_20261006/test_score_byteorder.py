"""Scorer read boundary only; no MRI, registration or numerical benchmark."""
import numpy as np
import torch


def test_saved_mgh_float32_big_endian_values_and_mask_are_exact():
    stored = np.array([0.0, -1.0, 1.0, np.finfo(np.float32).tiny,
                       np.nextafter(np.float32(0), np.float32(1)), 1000.5], dtype=">f4")
    assert not stored.dtype.isnative
    converted = np.array(stored, dtype=np.float32, copy=True)
    assert converted.dtype.isnative
    np.testing.assert_array_equal(converted, stored)
    np.testing.assert_array_equal(torch.from_numpy(converted).numpy(), stored)
    np.testing.assert_array_equal((torch.from_numpy(converted) > 0).numpy(), stored > 0)
