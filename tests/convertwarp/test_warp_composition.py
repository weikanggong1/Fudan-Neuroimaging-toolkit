"""Check FSL pull direction through composition, inversion and resampling."""

import nibabel as nib
import numpy as np

from fnit.applywarp import TorchApplyWarp
from fnit.convertwarp import TorchConvertWarp
from fnit.invwarp import TorchInvWarp
from fnit.mmorf import apply_mmorf_warp


def _image(data, affine=None):
    return nib.Nifti1Image(np.asarray(data), np.diag([-1, 1, 1, 1])
                           if affine is None else affine)


def test_composite_matches_applywarp_premat_and_inverse_recovers_mask():
    shape = (12, 10, 8)
    data = np.zeros(shape, dtype=np.uint8)
    data[4:7, 3:6, 2:5] = 1
    mask = _image(data)
    field = np.zeros((*shape, 3), dtype=np.float32)
    field[..., 0] = 0.2
    nonlinear = _image(field)
    nonlinear.header["intent_code"] = 2006
    premat = np.eye(4)
    premat[0, 3] = 1.0

    composite = TorchConvertWarp("cpu")(
        reference=mask, warp1=nonlinear, premat=premat)
    assert composite.image.shape == (*shape, 3)
    assert int(composite.image.header["intent_code"]) == 0
    np.testing.assert_allclose(np.asarray(composite.image.dataobj)[..., 0], -0.8,
                               atol=1e-6)
    direct = TorchApplyWarp("cpu")(
        input=mask, reference=mask, warp=nonlinear, premat=premat,
        interpolation="nearest")
    composed = TorchApplyWarp("cpu")(
        input=mask, reference=mask, warp=composite.image,
        warp_convention="relative", interpolation="nearest")
    np.testing.assert_array_equal(np.asarray(direct.image.dataobj),
                                  np.asarray(composed.image.dataobj))

    inverse = TorchInvWarp("cpu")(
        reference=mask, warp=composite.image, warp_convention="relative")
    assert int(inverse.image.header["intent_code"]) == 2006
    np.testing.assert_allclose(np.asarray(inverse.image.dataobj)[..., 0], 0.8,
                               atol=1e-4)
    recovered = TorchApplyWarp("cpu")(
        input=composed.image, reference=mask, warp=inverse.image,
        warp_convention="relative", interpolation="nearest")
    np.testing.assert_array_equal(np.asarray(recovered.image.dataobj), data)


def test_inverse_of_smooth_nonlinear_field_has_small_pull_residual():
    shape = (18, 14, 10)
    reference = _image(np.zeros(shape, dtype=np.float32))
    x, y, z = np.indices(shape, dtype=np.float32)
    field = np.stack((0.2 * np.sin(y / 5), 0.1 * np.sin(z / 4),
                      0.1 * np.sin(x / 6)), axis=-1).astype(np.float32)
    warp = _image(field)
    warp.header["intent_code"] = 2006
    inverse = TorchInvWarp("cpu")(reference, warp, iterations=20)
    assert inverse.valid_fraction > 0.75
    assert inverse.qc["median_residual_mm_in_field"] < 0.01


def test_mmorf_field_conversion_matches_mmorf_sampler_interior():
    shape = (12, 10, 8)
    x, y, z = np.indices(shape, dtype=np.float32)
    source = _image(x + 2 * y + 3 * z)
    reference = _image(np.zeros(shape, dtype=np.float32))
    data = np.zeros((*shape, 3), dtype=np.float32)
    data[..., 0] = 0.35
    data[..., 1] = -0.2
    mmorf = _image(data)
    affine = np.eye(4)
    affine[0, 3] = 1.0
    converted = TorchConvertWarp("cpu").from_mmorf(
        reference=reference, source=source, mmorf_warp=mmorf, affine=affine)
    actual = TorchApplyWarp("cpu")(
        input=source, reference=reference, warp=converted.image,
        warp_convention="relative").image
    expected = apply_mmorf_warp(
        source, reference, mmorf, affine=affine, device="cpu")
    np.testing.assert_allclose(np.asarray(actual.dataobj)[2:-2, 2:-2, 2:-2],
                               np.asarray(expected.dataobj)[2:-2, 2:-2, 2:-2],
                               atol=1e-4)
