"""SynthStrip exports its model and inference API."""

from fnit.synthstrip import ConvBlock, StripModel, StripResult, SynthStrip, extend_sdt
from fnit.synthstrip import model, pipeline


def test_feature_exports_are_original_objects():
    assert ConvBlock is model.ConvBlock
    assert StripModel is model.StripModel
    assert SynthStrip is pipeline.SynthStrip
    assert StripResult is pipeline.StripResult
    assert extend_sdt is pipeline.extend_sdt


def test_geometry_helpers_return_nifti_images():
    import nibabel as nib
    import numpy as np

    from fnit.synthstrip.pipeline import _crop_nonzero, _largest_filled_component

    data = np.zeros((7, 8, 9), dtype=np.float32)
    data[2:5, 3:7, 1:6] = 1
    affine = np.diag([1.2, 1.3, 1.4, 1.0])
    cropped = _crop_nonzero(nib.Nifti1Image(data, affine))
    assert isinstance(cropped, nib.Nifti1Image)
    assert cropped.shape == (3, 4, 5)
    np.testing.assert_allclose(cropped.affine[:3, 3], affine[:3, :3] @ [2, 3, 1])

    mask = np.zeros((7, 8, 9), dtype=bool)
    mask[1:6, 1:7, 1:8] = True
    mask[3, 3, 3] = False
    mask[0, 0, 0] = True
    kept = _largest_filled_component(mask)
    assert kept.dtype == np.uint8
    assert kept[3, 3, 3] == 1
    assert kept[0, 0, 0] == 0
