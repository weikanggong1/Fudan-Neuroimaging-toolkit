from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates
import torch

from fnit.space_conversion import _sample_volume


def test_real_t1_world_coordinates_match_trilinear_reference():
    image = nib.load(str(Path(__file__).resolve().parents[1] /
                         "examples/data/sub-01_T1w.nii.gz"))
    data = np.asarray(image.dataobj, dtype=np.float32)
    candidates = np.argwhere(data > np.percentile(data[data > 0], 75))
    voxels = candidates[np.linspace(0, len(candidates) - 1, 100, dtype=int)].astype(float) + 0.25
    ras = nib.affines.apply_affine(image.affine, voxels)
    got = _sample_volume(image, ras, torch.device("cpu"), nearest=False)[:, 0]
    expected = map_coordinates(data, voxels.T, order=1, mode="constant", cval=0)
    assert np.max(np.abs(got - expected)) < 0.05


def test_real_t1_world_coordinates_match_nearest_reference():
    image = nib.load(str(Path(__file__).resolve().parents[1] /
                         "examples/data/sub-01_T1w.nii.gz"))
    data = np.asarray(image.dataobj, dtype=np.float32)
    voxels = np.array([[20.2, 30.1, 40.4], [55.8, 60.9, 70.7], [90.1, 80.2, 60.3]])
    ras = nib.affines.apply_affine(image.affine, voxels)
    got = _sample_volume(image, ras, torch.device("cpu"), nearest=True)[:, 0]
    expected = map_coordinates(data, voxels.T, order=0, mode="constant", cval=0)
    np.testing.assert_array_equal(got, expected)
