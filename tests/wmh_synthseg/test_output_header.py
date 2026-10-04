"""Saved WMH images have the original new-grid header without scanner XML."""

import nibabel as nib
import numpy as np

from fnit.wmh_synthseg.pipeline import _output_image


def test_new_grid_labels_and_probabilities_use_fresh_float32_header(tmp_path):
    affine = np.array([[-1, 0, 0, 98.3], [0, 0, 1, -123.7],
                       [0, -1, 0, 77.8], [0, 0, 0, 1]])
    for name, data in [('segmentation', np.full((3, 4, 5), 77, dtype=np.int64)),
                       ('probability', np.full((3, 4, 5), .25, dtype=np.float32))]:
        image = _output_image(data, affine)
        path = tmp_path / (name + '.nii.gz')
        image.save(path)
        restored = nib.load(path)
        assert restored.get_data_dtype() == np.dtype(np.float32)
        assert np.array_equal(np.asanyarray(restored.dataobj), data.astype(np.float32))
        assert np.allclose(restored.affine, affine, rtol=0, atol=5e-6)
        assert int(restored.header['qform_code']) == 0
        assert int(restored.header['sform_code']) == 2
        assert restored.header.get_zooms() == (1., 1., 1.)
        assert not restored.header.extensions
