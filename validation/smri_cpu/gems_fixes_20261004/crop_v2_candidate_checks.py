"""Working images use their declared cropped domain before interpolation."""

import nibabel as nib
import numpy as np

from fnit.gems.context import SubregionContext
from fnit.gems.recipes.base import _cropped_cubic_image, working_image


def test_crop_boundary_does_not_consume_intensity_outside_declared_crop():
    coarse = np.zeros((48, 48, 48), np.int32)
    coarse[24, 24, 24] = 10
    data = np.full(coarse.shape, 80, np.float32)
    context = SubregionContext(nib.Nifti1Image(data, np.eye(4)), data, coarse, None, None)
    first, _, details = working_image(context, (10,), .5)
    lower, upper = details['crop_start_native'], details['crop_stop_native']
    inside = np.zeros(coarse.shape, bool)
    inside[tuple(slice(a, b) for a, b in zip(lower, upper))] = True
    context.data[~inside] = 5000
    second, _, _ = working_image(context, (10,), .5)
    assert np.array_equal(first.affine, second.affine)
    assert np.array_equal(first.dataobj, second.dataobj)


def test_rounded_support_respects_both_crop_size_parities():
    for size, outside_last in [(4, True), (5, False)]:
        data = np.full((size, size, size), 80, np.float32)
        result = _cropped_cubic_image(data, np.full(3, .5), np.zeros(3), (2 * size,) * 3)
        assert np.allclose(result[:-1, :-1, :-1], 80, atol=1e-5)
        assert np.all(result[-1] == 0) if outside_last else np.allclose(result[-1], 80, atol=1e-5)


def test_nonnegative_input_cannot_acquire_negative_spline_ringing():
    positive = np.zeros((5, 5, 5), np.float32)
    positive[2, 2, 2] = 100
    step, origin, shape = np.full(3, .5), np.zeros(3), (10, 10, 10)
    clipped = _cropped_cubic_image(positive, step, origin, shape)
    signed = _cropped_cubic_image(positive - 1, step, origin, shape)
    assert clipped.min() >= 0
    assert signed.min() < 0
    assert clipped[4, 4, 4] == 100
