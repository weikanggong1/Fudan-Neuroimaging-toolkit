"""Independent SciPy oracle for composed fMRIPrep-style volume resampling."""

import nibabel as nib
import numpy as np
import pytest
from scipy.ndimage import map_coordinates
import torch

from fnit.fmri.normalization import resample_world


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_grid_constant_spline_and_frame_motion_match_scipy(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    data = np.random.default_rng(87).normal(size=(9, 8, 7, 2)).astype(np.float32)
    source = nib.Nifti1Image(data, np.eye(4))
    source.header.set_zooms((1, 1, 1, .735))
    source.header.set_xyzt_units(t="sec")
    nib.save(source, tmp_path / "source.nii.gz")
    shape = (10, 7, 6)
    nib.save(nib.Nifti1Image(np.zeros(shape, np.float32), np.eye(4)),
             tmp_path / "reference.nii.gz")
    fixed = np.eye(4)
    fixed[:3, 3] = (-.7, .61, -.3)
    motion = np.repeat(np.eye(4)[None], 2, axis=0)
    motion[1, :3, 3] = (.25, -.38, .41)
    coordinates = np.indices(shape, dtype=np.float64)
    expected = np.stack([
        map_coordinates(data[..., frame], coordinates +
                        (fixed[:3, 3] + motion[frame, :3, 3])[:, None, None, None],
                        order=3, mode="grid-constant", cval=0, prefilter=True)
        for frame in range(2)
    ], axis=-1)
    for batch_size in (1, 2):
        result = resample_world(
            tmp_path / "source.nii.gz", tmp_path / "reference.nii.gz", fixed,
            tmp_path / f"output-{batch_size}.nii.gz", interpolation="spline",
            motion_pull_world=motion, boundary="grid-constant",
            batch_size=batch_size, spatial_chunk_size=113, device=device,
        )
        actual = nib.load(result)
        np.testing.assert_allclose(np.asarray(actual.dataobj), expected,
                                   rtol=3e-5, atol=6e-6)
        assert actual.header.get_xyzt_units()[1] == "sec"


def test_composes_noncommuting_motion_after_fixed_registration(tmp_path):
    data = np.indices((11, 12, 13)).sum(axis=0).astype(np.float32)[..., None]
    nib.save(nib.Nifti1Image(data, np.eye(4)), tmp_path / "source.nii.gz")
    shape = (5, 6, 7)
    nib.save(nib.Nifti1Image(np.zeros(shape, np.float32), np.eye(4)),
             tmp_path / "reference.nii.gz")
    fixed = np.eye(4)
    fixed[:3, 3] = (2, 1, 1)
    motion = np.eye(4)
    motion[:2, :2] = ((0, -1), (1, 0))
    motion[0, 3] = 10
    coordinates = np.indices(shape, dtype=np.float64).reshape(3, -1)
    composed = motion @ fixed
    query = (composed[:3, :3] @ coordinates + composed[:3, 3:4]).reshape(3, *shape)
    expected = map_coordinates(data[..., 0], query, order=1, mode="grid-constant")
    result = resample_world(tmp_path / "source.nii.gz", tmp_path / "reference.nii.gz",
                            fixed, tmp_path / "result.nii.gz",
                            motion_pull_world=motion[None], device="cpu")
    np.testing.assert_allclose(np.asarray(nib.load(result).dataobj)[..., 0],
                               expected, atol=2e-5)
