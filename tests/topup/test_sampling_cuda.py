"""CUDA arithmetic contracts for TOPUP sampling, not a real-image benchmark."""

from importlib.util import find_spec

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or find_spec("triton") is None,
    reason="CUDA and Triton are required for fused TOPUP sampling",
)


def _inputs(shape=(7, 9, 5), grid=(3, 5, 9)):
    generator = torch.Generator(device="cuda").manual_seed(70813)
    coefficients = torch.randn(shape, generator=generator, device="cuda", dtype=torch.float32)
    coordinates = torch.rand((3, *grid), generator=generator, device="cuda")
    for axis, size in enumerate(shape):
        coordinates[axis] *= size + 40
        coordinates[axis] -= 20
        coordinates[axis].reshape(-1)[:6] = torch.tensor(
            [0.0, size - 1.0, -0.25, size + 0.25, -2 * size - 0.125, 2 * size + 0.875],
            device="cuda",
        )
    return coefficients, coordinates


@pytest.mark.parametrize("phase_encode_axis", [0, 1, 2])
@pytest.mark.parametrize("shape", [(7, 9, 5), (1, 5, 2)])
def test_default_official_periodic_forward_and_analytic_gradient(shape, phase_encode_axis):
    from fnit.topup._sampling_cuda import sample_cubic_with_derivatives_cuda
    coefficients, coordinates = _inputs(shape)
    reference_coordinates = coordinates.detach().clone().requires_grad_(True)
    expected, _, expected_valid = _official_reference(coefficients, reference_coordinates, phase_encode_axis)
    # Each sampled voxel depends on only its own coordinate, so grad(sum) gives
    # its three partial derivatives without a voxel-by-coordinate Jacobian.
    expected_gradient = torch.autograd.grad(expected.sum(), reference_coordinates)[0]
    actual, gradient, valid = sample_cubic_with_derivatives_cuda(coefficients, coordinates, phase_encode_axis)
    assert torch.equal(valid, expected_valid)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(gradient, expected_gradient, rtol=3e-6, atol=3e-6)
    assert gradient.shape == coordinates.shape
    assert not actual.requires_grad and not gradient.requires_grad


@pytest.mark.parametrize("phase_encode_axis", [0, 1])
def test_coordinate_autograd_matches_weighted_official_tensor_oracle(phase_encode_axis):
    from fnit.topup._sampling_cuda import sample_cubic_cuda
    coefficients, coordinates = _inputs()
    oracle_coordinates = coordinates.detach().clone().requires_grad_(True)
    fused_coordinates = coordinates.detach().clone().requires_grad_(True)
    expected, _, _ = _official_reference(coefficients, oracle_coordinates, phase_encode_axis)
    weights = torch.linspace(-1, 2, expected.numel(), device="cuda").reshape(expected.shape)
    expected_gradient = torch.autograd.grad((expected * weights).sum(), oracle_coordinates)[0]
    actual, valid = sample_cubic_cuda(coefficients, fused_coordinates, phase_encode_axis)
    actual_gradient = torch.autograd.grad((actual * weights).sum(), fused_coordinates)[0]
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=4e-6, atol=4e-6)
    assert valid.dtype == torch.bool and not valid.requires_grad


def test_noncontiguous_inputs_current_stream_and_retained_outputs():
    from fnit.topup._sampling_cuda import sample_cubic_cuda
    coefficients, coordinates = _inputs()
    coefficients = coefficients.transpose(0, 1)
    coordinates = coordinates.transpose(1, 2)
    expected, _, expected_valid = _official_reference(coefficients, coordinates, 1)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual, valid = sample_cubic_cuda(coefficients, coordinates, 1)
        retained = actual.clone()
        second, _ = sample_cubic_cuda(coefficients + 3, coordinates, 1)
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    assert torch.equal(valid, expected_valid)
    assert torch.equal(actual, retained)
    assert actual.data_ptr() != second.data_ptr()


def test_rejects_trainable_volume_invalid_shapes_and_non_float32():
    from fnit.topup._sampling_cuda import sample_cubic_cuda
    coefficients, coordinates = _inputs()
    with pytest.raises(ValueError, match="fixed inputs"):
        sample_cubic_cuda(coefficients.clone().requires_grad_(True), coordinates, 1)
    with pytest.raises(ValueError, match="float32"):
        sample_cubic_cuda(coefficients.double(), coordinates, 1)
    with pytest.raises(ValueError, match="coordinates"):
        sample_cubic_cuda(coefficients, coordinates[:2], 1)
    with pytest.raises(ValueError, match="phase_encode_axis"):
        sample_cubic_cuda(coefficients, coordinates, 3)


def _official_reference(coefficients, coordinates, phase_encode_axis):
    """Double tensor oracle for Splinterpolator's piecewise order-3 code."""
    flat = coordinates.double().reshape(3, -1)
    indices, weights, derivative_weights = [], [], []
    valid = torch.ones(flat.shape[1], dtype=torch.bool, device=flat.device)
    offsets = torch.arange(4, device=flat.device)
    for axis, (coordinate, size) in enumerate(zip(flat, coefficients.shape)):
        if axis != phase_encode_axis:
            valid &= (coordinate + 1e-8 >= 0) & (coordinate <= size - 1 + 1e-8)
        nearest = torch.trunc(coordinate + 0.5).long()
        start = torch.where(nearest < coordinate, nearest - 1, nearest - 2)
        locations = start[:, None] + offsets
        distance = coordinate[:, None] - locations
        absolute = distance.abs()
        remainder = 2 - absolute
        weight = torch.where(
            absolute < 1,
            2.0 / 3.0 + 0.5 * absolute * absolute * (absolute - 2),
            torch.where(absolute < 2, (1.0 / 6.0) * (remainder * remainder * remainder), 0.0),
        )
        sign = torch.where(distance < 0, -1.0, 1.0)
        derivative = torch.where(
            absolute < 1,
            sign * (1.5 * absolute * absolute - 2.0 * absolute),
            torch.where(absolute < 2, sign * -0.5 * remainder * remainder, 0.0),
        )
        indices.append(torch.remainder(locations, size))
        weights.append(weight)
        derivative_weights.append(derivative)
    result = flat.new_zeros(flat.shape[1])
    gradient = flat.new_zeros(flat.shape)
    values = coefficients.double().reshape(-1)
    sy, sz = coefficients.shape[1:]
    for iz in range(4):
        wz = weights[2][:, iz]
        dwz = derivative_weights[2][:, iz]
        for iy in range(4):
            wy = weights[1][:, iy]
            dwy = derivative_weights[1][:, iy]
            wzy, dwzy, wzdy = wz * wy, dwz * wy, wz * dwy
            for ix in range(4):
                address = indices[0][:, ix] * (sy * sz) + indices[1][:, iy] * sz + indices[2][:, iz]
                coefficient = values[address]
                wx, dwx = weights[0][:, ix], derivative_weights[0][:, ix]
                coefficient_x = coefficient * wx
                result = result + coefficient_x * wzy
                gradient[0] = gradient[0] + coefficient * dwx * wzy
                gradient[1] = gradient[1] + coefficient_x * wzdy
                gradient[2] = gradient[2] + coefficient_x * dwzy
    return (result.float().reshape(coordinates.shape[1:]),
            gradient.float().reshape(coordinates.shape),
            valid.reshape(coordinates.shape[1:]))


@pytest.mark.parametrize("phase_encode_axis", [0, 1, 2])
@pytest.mark.parametrize("shape", [(7, 9, 5), (1, 5, 2)])
def test_official_double_weights_tap_starts_and_accumulation(shape, phase_encode_axis):
    from fnit.topup._sampling_cuda import sample_cubic_cuda, sample_cubic_with_derivatives_cuda
    coefficients, coordinates = _inputs(shape)
    # get_start_indicies intentionally differs from floor-1 at these negative
    # points; include them explicitly rather than relying on random coverage.
    coordinates[phase_encode_axis].reshape(-1)[6:9] = torch.tensor(
        [-1.25, -2.5, -3.125], device="cuda"
    )
    expected, expected_gradient, expected_valid = _official_reference(coefficients, coordinates, phase_encode_axis)
    actual, gradient, valid = sample_cubic_with_derivatives_cuda(
        coefficients, coordinates, phase_encode_axis, official_precision=True
    )
    forward_only, _ = sample_cubic_cuda(coefficients, coordinates, phase_encode_axis, official_precision=True)
    assert torch.equal(valid, expected_valid)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(gradient, expected_gradient, rtol=1e-6, atol=1e-6)
    assert torch.equal(forward_only, actual)


def test_official_precision_coordinate_backward_uses_returned_derivatives():
    from fnit.topup._sampling_cuda import sample_cubic_cuda, sample_cubic_with_derivatives_cuda
    coefficients, coordinates = _inputs()
    coordinates.requires_grad_(True)
    expected, derivative, expected_valid = sample_cubic_with_derivatives_cuda(
        coefficients, coordinates, 1, official_precision=True
    )
    actual, valid = sample_cubic_cuda(coefficients, coordinates, 1, official_precision=True)
    weights = torch.linspace(-1, 2, actual.numel(), device="cuda").reshape(actual.shape)
    gradient = torch.autograd.grad((actual * weights).sum(), coordinates)[0]
    assert torch.equal(actual, expected) and torch.equal(valid, expected_valid)
    assert torch.equal(gradient, derivative * weights.unsqueeze(0))


def test_official_non_pe_validity_tolerance_matches_cpu_reference():
    from fnit.topup._sampling_cuda import sample_cubic_with_derivatives_cuda
    coordinates = torch.tensor([[-5e-9, -2e-8, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], device="cuda")
    coefficients = torch.ones((4, 4, 4), device="cuda")
    expected, expected_gradient, expected_valid = _official_reference(coefficients, coordinates, 1)
    values, gradient, valid = sample_cubic_with_derivatives_cuda(coefficients, coordinates, 1, official_precision=True)
    assert torch.equal(valid, expected_valid)
    assert valid.tolist() == [True, False, True]
    torch.testing.assert_close(values, expected)
    torch.testing.assert_close(gradient, expected_gradient)


@pytest.mark.parametrize("function_name", ["sample_cubic_cuda", "sample_cubic_with_derivatives_cuda"])
def test_rejects_removed_float32_algorithm(function_name):
    from fnit.topup import _sampling_cuda
    coefficients, coordinates = _inputs()
    function = getattr(_sampling_cuda, function_name)
    with pytest.raises(ValueError, match="official_precision=True"):
        function(coefficients, coordinates, 1, official_precision=False)
