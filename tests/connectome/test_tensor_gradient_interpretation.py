"""Gradient interpretation regressions; real accuracy is benchmarked separately."""
import numpy as np
import pytest
import torch

from fnit.connectome.response import _mrtrix_interpret_tensor_gradients


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_ten_digit_export_directions_are_normalised_without_bvalue_scaling(device):
    # A genuine ten-digit MRtrix text-export direction from public CON03.
    gradient = torch.tensor([[0., 0., 0., 0.],
                             [.6940924111, -.6589524964, .2898574345, 700.]],
                            dtype=torch.float64, device=device)
    original = gradient.clone()
    interpreted = _mrtrix_interpret_tensor_gradients(gradient)
    expected = original.cpu().numpy().copy()
    expected[1, :3] /= np.sqrt(np.sum(expected[1, :3] ** 2))
    assert torch.equal(gradient, original)
    np.testing.assert_allclose(interpreted.cpu().numpy(), expected, rtol=1e-15, atol=0)
    assert torch.equal(interpreted[:, 3], original[:, 3])
    assert not torch.equal(interpreted[1, :3], original[1, :3])


@pytest.mark.parametrize('log_norm_squared,scales', [(.009, False), (.011, True), (-.011, True)])
def test_auto_bvalue_scaling_uses_global_absolute_log_norm_threshold(log_norm_squared, scales):
    magnitude = np.exp(log_norm_squared / 2)
    gradient = torch.tensor([[magnitude, 0., 0., 1000.], [0., 0., 0., 0.],
                             [0., 1., 0., 2000.]], dtype=torch.float64)
    interpreted = _mrtrix_interpret_tensor_gradients(gradient)
    np.testing.assert_allclose(interpreted[0, :3].numpy(), [1., 0., 0.], rtol=1e-15)
    expected_bvalue = 1000 * magnitude * magnitude if scales else 1000
    assert float(interpreted[0, 3]) == pytest.approx(expected_bvalue, rel=1e-15)
    assert float(interpreted[2, 3]) == 2000
    assert not interpreted[1].any()


def test_all_zero_directions_are_retained_without_empty_reduction_or_signal_repair():
    gradient = torch.tensor([[0., 0., 0., 0.], [0., 0., 0., 1000.]], dtype=torch.float64)
    assert torch.equal(_mrtrix_interpret_tensor_gradients(gradient), gradient)


def test_auto_scaling_zero_direction_is_bzero_when_another_row_requires_scaling():
    # MRtrix's Auto decision applies globally; it also scales zero directions.
    gradient = torch.tensor([[2., 0., 0., 700.], [0., 0., 0., 1000.]],
                            dtype=torch.float64)
    interpreted = _mrtrix_interpret_tensor_gradients(gradient)
    assert torch.equal(interpreted, torch.tensor([[1., 0., 0., 2800.],
                                                  [0., 0., 0., 0.]], dtype=torch.float64))


def test_fsl_import_normalizes_affine_columns_before_double_polar_rotation(tmp_path):
    from fnit.connectome.pipeline import _gradients

    bvals = tmp_path / 'dwi.bval'; bvecs = tmp_path / 'dwi.bvec'
    np.savetxt(bvals, [[0., 700.]])
    np.savetxt(bvecs, [[0., 0.], [0., 2.], [0., 0.]])
    affine = torch.tensor([[2., .6, 0., 0.], [0., 3., 0., 0.],
                           [0., 0., 4., 0.], [0., 0., 0., 1.]], dtype=torch.float64)
    actual_bvals, actual_bvecs = _gradients(bvals, bvecs, 2, affine, torch.device('cpu'))
    # Analytic 2D polar angle of column-normalised shear; the scaled affine
    # would yield a different rotation. Nonunit input must trigger Auto.
    angle = np.arctan2(.2, np.sqrt(1 + .2 ** 2) + 1)
    expected = torch.tensor([np.sin(angle), np.cos(angle), 0.], dtype=torch.float64)
    assert actual_bvals.dtype == actual_bvecs.dtype == torch.float64
    assert torch.equal(actual_bvals, torch.tensor([0., 2800.], dtype=torch.float64))
    torch.testing.assert_close(actual_bvecs[1], expected, rtol=1e-14, atol=1e-15)
    assert not actual_bvecs[0].any()
