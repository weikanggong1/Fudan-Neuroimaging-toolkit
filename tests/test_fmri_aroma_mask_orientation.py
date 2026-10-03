"""Original ICA-AROMA mask storage changes retain exact physical support."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from fnit.fmri.end_to_end import _classification_masks_on_template
from fnit.fmri import end_to_end


def _original_masks():
    root = Path(end_to_end.__file__).parent / "assets"
    return tuple(root / f"mask_{name}.nii.gz" for name in ("csf", "edge", "out"))


def test_original_las_masks_keep_existing_paths_without_writes(tmp_path):
    paths = _original_masks()
    output = tmp_path / "masks"
    assert _classification_masks_on_template(nib.load(paths[0]), paths, output) == paths
    assert not output.exists()


def test_real_original_masks_reorient_exactly_to_ras_and_back(tmp_path):
    paths = _original_masks()
    template = nib.as_closest_canonical(nib.load(paths[0]))
    outputs = _classification_masks_on_template(template, paths, tmp_path / "masks")
    for source, output in zip(paths, outputs):
        original, aligned = nib.load(source), nib.load(output)
        assert aligned.shape == template.shape
        np.testing.assert_allclose(aligned.affine, template.affine, rtol=0, atol=1e-4)
        transform = nib.orientations.ornt_transform(nib.orientations.io_orientation(aligned.affine),
                                                    nib.orientations.io_orientation(original.affine))
        restored = aligned.as_reoriented(transform)
        assert restored.get_data_dtype() == original.get_data_dtype()
        np.testing.assert_array_equal(np.asanyarray(restored.dataobj), np.asanyarray(original.dataobj))
        np.testing.assert_allclose(restored.affine, original.affine, rtol=0, atol=1e-4)


def test_physical_grid_shift_still_rejected(tmp_path):
    paths = _original_masks()
    image = nib.as_closest_canonical(nib.load(paths[0]))
    shifted = image.affine.copy()
    shifted[0, 3] += 2
    template = nib.Nifti1Image(np.asanyarray(image.dataobj), shifted)
    with pytest.raises(ValueError, match="mask grid"):
        _classification_masks_on_template(template, paths, tmp_path / "masks")
