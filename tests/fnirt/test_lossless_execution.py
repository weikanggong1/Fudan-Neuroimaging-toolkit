"""Unused work must not change FNIRT samples, normal equations or decisions."""

import pytest
import torch

import fnit.fnirt.registration as registration
from fnit.fnirt.spline import BendingOperator, fsl_control_shape, spline_bases


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"
))]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_value_only_trilinear_keeps_sample_and_boundary_bits(device, dtype):
    generator = torch.Generator().manual_seed(73)
    volume = torch.randn((5, 6, 7), generator=generator, dtype=dtype).to(device)
    coordinates = torch.rand((3, 41), generator=generator, dtype=dtype).to(device)
    coordinates *= torch.tensor([5, 6, 7], device=device, dtype=dtype)[:, None]
    coordinates -= 0.5
    coordinates[:, :5] = torch.tensor([
        [0, 4, -5e-9, -2e-8, 4 + 2e-8],
        [0, 5, 0, 0, 5],
        [0, 6, 0, 0, 6],
    ], device=device, dtype=dtype)
    expected, expected_valid, gradient = registration._trilinear_sample(volume, coordinates)
    actual, actual_valid, omitted = registration._trilinear_sample(
        volume, coordinates, derivatives=False
    )
    bits = torch.int32 if dtype == torch.float32 else torch.int64
    assert torch.equal(actual.view(bits), expected.view(bits))
    assert torch.equal(actual_valid, expected_valid)
    assert gradient.shape == coordinates.shape
    assert omitted is None


def _case(device):
    shape, spacing, bias_spacing = (6, 7, 5), (3, 3, 3), (4, 4, 4)
    axes = torch.meshgrid(*[torch.arange(n, device=device) for n in shape], indexing="ij")
    fixed = (30 + 6 * axes[0] + 4 * axes[1] + 2 * axes[2]).to(torch.float32)
    moving = (8 + 0.9 * fixed + 0.001 * fixed.square()).to(torch.float32)
    mask = torch.ones_like(fixed)
    mask[0] = 0
    dtype = torch.float64
    bases = spline_bases(shape, spacing, (1, 1, 1), device=device, dtype=dtype)
    bending = BendingOperator(shape, spacing, (1, 1, 1), device=device, dtype=dtype)
    affine = torch.eye(4, dtype=dtype, device=device)
    affine[0, 3] = 0.25
    level = registration._LevelSystem(
        moving, fixed, None, mask, torch.eye(4, device=device),
        torch.stack(axes).to(torch.float32), torch.eye(4, device=device),
        bases, bending, 0.1, False, False, affine,
    )
    coefficients = torch.zeros((3, *fsl_control_shape(shape, spacing)), dtype=dtype, device=device)
    joint = registration._JointT1System(
        level, spline_bases(shape, bias_spacing, (1, 1, 1), device=device, dtype=dtype),
        BendingOperator(shape, bias_spacing, (1, 1, 1), device=device, dtype=dtype), 10.0, 5,
    )
    bias = torch.ones((1, *fsl_control_shape(shape, bias_spacing)), dtype=dtype, device=device)
    polynomial = torch.tensor([0.1, 0.9, 0.02, 0, 0], dtype=dtype, device=device)
    return level, coefficients, joint, polynomial, bias


def _same_state(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        if isinstance(actual[key], torch.Tensor):
            assert torch.equal(actual[key], expected[key]), key
        else:
            assert actual[key] == expected[key], key


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("derivatives", [False, True])
def test_level_skips_unused_mask_and_cost_gradients_without_changing_state(device, derivatives, monkeypatch):
    level, coefficients, _, _, _ = _case(device)
    sample = registration._trilinear_sample
    flags = []

    def record(volume, coordinates, *, derivatives=True):
        flags.append(derivatives)
        return sample(volume, coordinates, derivatives=derivatives)

    monkeypatch.setattr(registration, "_trilinear_sample", record)
    actual = level.evaluate(coefficients, coefficients.new_ones(()), derivatives=derivatives)
    assert flags == [derivatives, False]
    monkeypatch.setattr(registration, "_trilinear_sample", lambda volume, coordinates, **_: sample(volume, coordinates))
    expected = level.evaluate(coefficients, coefficients.new_ones(()), derivatives=derivatives)
    _same_state(actual, expected)


@pytest.mark.parametrize("device", DEVICES)
def test_joint_linearization_reuses_mapping_and_keeps_normal_equations(device, monkeypatch):
    _, coefficients, joint, polynomial, bias = _case(device)
    expand = registration.expand_coefficients
    bias_calls = []
    deformation_calls = []

    def record(value, bases):
        if value is bias:
            bias_calls.append(1)
        if value.shape == coefficients.shape:
            deformation_calls.append(1)
        return expand(value, bases)

    monkeypatch.setattr(registration, "expand_coefficients", record)
    actual = joint.linearize(coefficients, polynomial, bias, fit_intensity=True)
    assert len(bias_calls) == 1
    mapping = joint._mapping

    def repeated_mapping(poly, bias_coefficients):
        mapped, _, _ = mapping(poly, bias_coefficients)
        _, global_map, mapped_bias = mapping(poly, bias_coefficients)
        return mapped, global_map, mapped_bias

    linearize = joint.deformation.linearize

    def repeated_expansion(*args, **kwargs):
        state, gradient, matvec, diagonal = linearize(*args, **kwargs)

        def product(vector, **unused):
            # Frozen path: the deformation block performs its own double
            # expansion rather than consuming the joint block's expansion.
            return matvec(vector)

        return state, gradient, product, diagonal

    monkeypatch.setattr(joint, "_mapping", repeated_mapping)
    monkeypatch.setattr(joint.deformation, "linearize", repeated_expansion)
    expected = joint.linearize(coefficients, polynomial, bias, fit_intensity=True)
    _same_state(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
    assert torch.equal(actual[3], expected[3])
    vector = torch.linspace(-0.05, 0.05, actual[1].numel(), dtype=coefficients.dtype, device=device)
    deformation_calls.clear()
    actual_product = actual[2](vector)
    assert len(deformation_calls) == 1
    deformation_calls.clear()
    expected_product = expected[2](vector)
    assert len(deformation_calls) == 2
    assert torch.equal(actual_product.view(torch.int64), expected_product.view(torch.int64))
