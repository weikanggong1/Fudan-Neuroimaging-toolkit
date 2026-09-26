import numpy as np
import torch
from fnit.eddy.core import EDDYConfig, _grid, _poly_basis, _qspace_weights, _rigid_grid


def test_batched_identity_rigid_grid():
    grid = _grid((5, 6, 7), "cpu")
    parameters = torch.zeros((3, 6))
    pulled = _rigid_grid(grid, parameters, (2, 2, 2))
    torch.testing.assert_close(pulled, grid[None].expand(3, -1, -1, -1, -1))


def test_quadratic_field_has_fsl_ten_parameters():
    basis, derivative = _poly_basis((5, 6, 7), (2, 2, 2), "cpu", 1)
    assert basis.shape == (10, 5, 6, 7) and derivative.shape == basis.shape
    torch.testing.assert_close(derivative[1], torch.full((5, 6, 7), 2.0))


def test_qspace_predictor_excludes_self_and_normalizes():
    b = np.array([0, 0, 1000, 1000, 1000])
    g = np.array(
        [[0, 0, 1, 0, 1 / np.sqrt(2)], [0, 0, 0, 1, 1 / np.sqrt(2)], [0, 0, 0, 0, 0.0]]
    )
    weights = _qspace_weights(b, g, k=3)
    np.testing.assert_allclose(weights.sum(1), 1)
    np.testing.assert_allclose(np.diag(weights), 0)


def test_default_seed_is_fixed():
    assert EDDYConfig().seed == 0


def test_outlier_squared_map_is_part_of_core_output(tmp_path):
    import nibabel as nib
    from fnit.eddy import EDDYResult

    image = nib.Nifti1Image(np.zeros((2, 2, 2, 1), np.float32), np.eye(4))
    result = EDDYResult(
        image,
        np.zeros((3, 1)),
        np.zeros((1, 16)),
        np.zeros((1, 2)),
        np.zeros((1, 2)),
        np.zeros((1, 2)),
        np.array([[2.0, -3.0]]),
        {},
    )
    paths = result.save(tmp_path / "eddy")
    squared = np.loadtxt(tmp_path / "eddy.eddy_outlier_n_sqr_stdev_map")
    np.testing.assert_allclose(squared, [4, 9])
    assert tmp_path / "eddy.eddy_outlier_n_sqr_stdev_map" in paths


def test_schedule_lengths_must_match():
    import pytest

    with pytest.raises(ValueError, match="same non-empty schedule"):
        EDDYConfig(fwhm=(1, 0), subsampling=(1,))
