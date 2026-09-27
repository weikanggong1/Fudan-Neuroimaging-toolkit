"""FastVBM registration input and displacement-QC regression tests."""

import nibabel as nib
import numpy as np
import pytest

from fnit._nib import new_image
from fnit._transforms import DenseWarp

from fnit.fast_vbm.registration import (
    _displacement_qc,
    _register_gm,
    _validate_pull_warp,
)


def _volume(shape=(6, 7, 8), affine=None):
    if affine is None:
        affine = np.eye(4)
    data = np.ones(shape, dtype=np.float32)
    return new_image(data, nib.Nifti1Image(data, affine))


def test_internal_registration_rejects_empty_volume():
    shape = (6, 7, 8)
    empty_data = np.zeros(shape, dtype=np.float32)
    positive_data = np.ones(shape, dtype=np.float32)
    empty = new_image(empty_data, nib.Nifti1Image(empty_data, np.eye(4)))
    positive = new_image(positive_data, nib.Nifti1Image(positive_data, np.eye(4)))

    with pytest.raises(ValueError, match="GM image is empty"):
        _register_gm(empty, positive)

def test_displacement_qc_separates_affine_and_nonlinear_components():
    shape = (9, 7, 5)
    vox2world = np.array(
        [
            [1.5, 0.2, 0, -4],
            [0, 2, 0.1, 3],
            [0, 0, -2.5, 8],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    fixed = _volume(shape=shape, affine=vox2world)
    pull_affine = np.array(
        [
            [1.1, 0.05, 0, 2],
            [0, 0.9, 0.02, -1],
            [0, 0, 1.05, 0.5],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    indices = np.stack(
        np.meshgrid(*[np.arange(size) for size in shape], indexing="ij"), axis=-1
    )
    world = np.einsum("ab,...b->...a", vox2world[:3, :3], indices)
    world += vox2world[:3, 3]
    affine_source = np.einsum("ab,...b->...a", pull_affine[:3, :3], world)
    affine_source += pull_affine[:3, 3]
    residual = np.array([0.25, -0.5, 0.75])
    displacement = affine_source - world + residual
    pull = new_image(displacement.astype(np.float32), fixed)

    qc = _displacement_qc(pull, fixed, pull_affine)

    expected_total = np.linalg.norm(displacement, axis=-1).max()
    assert qc["maximum_total_pull_displacement_mm"] == pytest.approx(expected_total)
    assert qc["maximum_nonlinear_displacement_mm"] == pytest.approx(
        np.linalg.norm(residual), abs=1e-6
    )


def test_dense_pull_source_geometry_is_checked():
    moving = _volume()
    fixed = _volume()
    wrong_source = _volume(shape=(7, 7, 8))
    pull = DenseWarp(
        np.zeros((*fixed.shape[:3], 3), dtype=np.float32),
        source=wrong_source,
        target=fixed,
    )

    with pytest.raises(ValueError, match="source geometry"):
        _validate_pull_warp(pull, moving, fixed)
