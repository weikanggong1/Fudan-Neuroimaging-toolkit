"""Numeric kernel regressions; synthetic fixtures are not speed benchmarks."""
import numba
import numpy as np
import pytest
import torch
from itertools import permutations

from fnit.fnirt._normal_cpu import SpatialNormalCPU
import fnit.fnirt._normal_cpu as cpu_normal_module


def reference(field, weights, cross, scale, count):
    dense = torch.zeros_like(field)
    for row in range(3):
        for column in range(3):
            dense[row] = dense[row] + weights[row][column] * field[column]
        if scale is not None:
            dense[row] = dense[row] + cross[row] * scale
    return dense / count


@pytest.mark.parametrize("threads", [1, 8])
@pytest.mark.parametrize("fit_scale", [False, True])
@pytest.mark.parametrize("layout", ["contiguous", "transposed", "spline"])
def test_fused_fp64_normal_is_bitwise_equal_and_reuses_scratch(threads, fit_scale, layout, monkeypatch):
    torch_threads, numba_threads = torch.get_num_threads(), numba.get_num_threads()
    try:
        torch.set_num_threads(threads)
        numba.set_num_threads(threads)
        monkeypatch.setattr(cpu_normal_module.config, "NUMBA_NUM_THREADS", threads)
        generator = torch.Generator().manual_seed(1234)
        field = torch.randn(3, 4, 5, 6, dtype=torch.float64, generator=generator)
        if layout == "transposed":
            field = field.transpose(1, 3)
        elif layout == "spline":
            field = field.permute(3, 0, 1, 2).contiguous().permute(1, 2, 3, 0)
        weights = tuple(tuple(torch.randn(field.shape[1:], dtype=torch.float64, generator=generator) for _ in range(3)) for _ in range(3))
        cross = tuple(torch.randn(field.shape[1:], dtype=torch.float64, generator=generator) for _ in range(3)) if fit_scale else None
        scale = torch.tensor(-0.123456789, dtype=torch.float64) if fit_scale else None
        normal = SpatialNormalCPU(weights, cross, 123.0)
        before = torch.get_num_threads(), numba.get_num_threads()
        actual = normal(field, scale)
        expected = reference(field, weights, cross, scale, 123.0)
        assert torch.equal(actual.contiguous().view(torch.int64), expected.contiguous().view(torch.int64))
        pointer = actual.data_ptr()
        assert normal(field, scale).data_ptr() == pointer
        assert (torch.get_num_threads(), numba.get_num_threads()) == before
    finally:
        torch.set_num_threads(torch_threads)
        numba.set_num_threads(numba_threads)


def test_signed_zero_and_cancellation_keep_original_rounding():
    field = torch.tensor([0.0, -0.0, 1e200, -1e200, 1e-200, -1e-200], dtype=torch.float64).repeat(3).reshape(3, 1, 2, 3)
    weights = tuple(tuple(torch.full(field.shape[1:], (-1.0) ** (row + column), dtype=torch.float64) for column in range(3)) for row in range(3))
    normal = SpatialNormalCPU(weights, None, 3.0)
    actual = normal(field, None)
    expected = reference(field, weights, None, None, 3.0)
    assert torch.equal(actual.contiguous().view(torch.int64), expected.contiguous().view(torch.int64))


def test_fixed_weight_layout_is_bitwise_and_owned_by_each_linearization():
    field = torch.arange(3 * 4 * 5 * 6, dtype=torch.float64).reshape(3, 4, 5, 6)
    field = field.permute(3, 0, 1, 2).contiguous().permute(1, 2, 3, 0)
    values = torch.arange(4 * 5 * 6, dtype=torch.float64).reshape(4, 5, 6)
    weights = tuple(tuple((values + row * 3 + column).clone() for column in range(3)) for row in range(3))
    cross = tuple((values - row).clone() for row in range(3))
    normal = SpatialNormalCPU(weights, cross, 117.0)
    actual = normal(field, torch.tensor(0.125, dtype=torch.float64))
    assert torch.equal(actual.contiguous().view(torch.int64), reference(field, weights, cross, 0.125, 117.0).contiguous().view(torch.int64))
    assert normal.layout_copy_bytes == 12 * values.numel() * 8
    pointers = [value.ctypes.data for value in (*normal.weights, *normal.cross)]
    for original, packed in zip([value for row in weights for value in row] + list(cross), (*normal.weights, *normal.cross)):
        assert np.array_equal(original.numpy().view(np.int64), packed.view(np.int64))
        assert packed.strides[1] == 8
    normal(field * 2, torch.tensor(0.25, dtype=torch.float64))
    assert pointers == [value.ctypes.data for value in (*normal.weights, *normal.cross)]
    changed = tuple(tuple(value + 1 for value in row) for row in weights)
    next_normal = SpatialNormalCPU(changed, cross, 117.0)
    next_result = next_normal(field, torch.tensor(0.125, dtype=torch.float64))
    assert torch.equal(next_result.contiguous().view(torch.int64), reference(field, changed, cross, 0.125, 117.0).contiguous().view(torch.int64))
    assert next_normal.weights[0].ctypes.data != normal.weights[0].ctypes.data


def test_already_matching_weight_layout_needs_no_copy():
    field = torch.ones((3, 4, 5, 6), dtype=torch.float64)
    weights = tuple(tuple(torch.ones(field.shape[1:], dtype=torch.float64) for _ in range(3)) for _ in range(3))
    normal = SpatialNormalCPU(weights, None, 120.0)
    normal(field, None)
    assert normal.layout_copy_bytes == 0
    assert all(np.shares_memory(value.numpy(), normal.weights[row * 3 + column]) for row, items in enumerate(weights) for column, value in enumerate(items))


def test_cached_layout_checks_geometry_before_numba_and_accepts_new_strides():
    weights = tuple(tuple(torch.ones((4, 5, 6), dtype=torch.float64) for _ in range(3)) for _ in range(3))
    normal = SpatialNormalCPU(weights, None, 120.0)
    field = torch.ones((3, 4, 5, 6), dtype=torch.float64)
    normal(field, None)
    # All retain the same fastest-to-slowest axes as the cached layout.
    for changed in [torch.ones((3, 4, 5, 7), dtype=torch.float64), torch.ones((2, 4, 5, 6), dtype=torch.float64)]:
        with pytest.raises(ValueError, match="field geometry changed"):
            normal(changed, None)
    strided = torch.arange(3 * 4 * 5 * 7, dtype=torch.float64).reshape(3, 4, 5, 7)[..., :6]
    actual = normal(strided, None)
    assert actual.stride() == torch.empty_like(strided).stride()
    assert torch.equal(actual.contiguous().view(torch.int64), reference(strided, weights, None, None, 120.0).contiguous().view(torch.int64))


def test_constructor_rejects_mismatched_cross_or_weight_grids():
    weights = tuple(tuple(torch.ones((4, 5, 6), dtype=torch.float64) for _ in range(3)) for _ in range(3))
    cross = [torch.ones((4, 5, 6), dtype=torch.float64) for _ in range(3)]
    cross[1] = torch.ones((4, 5, 7), dtype=torch.float64)
    with pytest.raises(ValueError, match="weight geometry"):
        SpatialNormalCPU(weights, cross, 120.0)


def test_oversized_numba_pool_does_not_expand_torch_thread_budget():
    torch_threads, numba_threads = torch.get_num_threads(), numba.get_num_threads()
    try:
        torch.set_num_threads(1)
        numba.set_num_threads(8)
        field = torch.ones((3, 2, 2, 2), dtype=torch.float64)
        weights = tuple(tuple(torch.ones(field.shape[1:], dtype=torch.float64) for _ in range(3)) for _ in range(3))
        normal = SpatialNormalCPU(weights, None, 7.0)
        assert torch.equal(normal(field, None).view(torch.int64), reference(field, weights, None, None, 7.0).view(torch.int64))
        assert torch.get_num_threads() == 1
        assert numba.get_num_threads() == 8
    finally:
        torch.set_num_threads(torch_threads)
        numba.set_num_threads(numba_threads)


def test_differentiable_weights_are_not_accepted_by_nondifferentiable_kernel():
    weights = tuple(tuple(torch.ones((2, 2, 2), dtype=torch.float64, requires_grad=True) for _ in range(3)) for _ in range(3))
    with pytest.raises(ValueError, match="non-differentiable"):
        SpatialNormalCPU(weights, None, 1)


def test_oversized_default_pool_is_not_initialized(monkeypatch):
    monkeypatch.setattr(cpu_normal_module.config, "NUMBA_NUM_THREADS", torch.get_num_threads() + 1)
    def unexpected_pool_initialization():
        raise AssertionError("Oversized Numba pool should not be initialized")
    monkeypatch.setattr(cpu_normal_module, "get_num_threads", unexpected_pool_initialization)
    field = torch.ones((3, 2, 2, 2), dtype=torch.float64)
    weights = tuple(tuple(torch.ones(field.shape[1:], dtype=torch.float64) for _ in range(3)) for _ in range(3))
    normal = SpatialNormalCPU(weights, None, 7.0)
    assert torch.equal(normal(field, None).view(torch.int64), reference(field, weights, None, None, 7.0).view(torch.int64))


@pytest.mark.parametrize("threads", [1, 8])
@pytest.mark.parametrize("mode", ["deformation", "scale", "joint_intensity"])
def test_complete_linearized_matvec_keeps_original_bits(threads, mode, monkeypatch):
    from fnit.fnirt.registration import _JointT1System, _LevelSystem
    from fnit.fnirt.spline import BendingOperator, fsl_control_shape, spline_bases

    torch_threads, numba_threads = torch.get_num_threads(), numba.get_num_threads()
    try:
        torch.set_num_threads(threads)
        numba.set_num_threads(threads)
        monkeypatch.setattr(cpu_normal_module.config, "NUMBA_NUM_THREADS", threads)
        shape, spacing = (9, 8, 7), (2, 2, 2)
        dtype = torch.float64
        axes = torch.stack(torch.meshgrid(*(torch.arange(n, dtype=torch.float32) for n in shape), indexing="ij"))
        fixed = 10.0 + 0.9 * axes[0] + torch.sin(axes[1]) + 0.3 * axes[2]
        moving = fixed + 0.05 * axes[0].square()
        bases = spline_bases(shape, spacing, (1.0, 1.0, 1.0), device="cpu", dtype=dtype)
        bending = BendingOperator(shape, spacing, (1.0, 1.0, 1.0), device="cpu", dtype=dtype, execution="optimized")
        system = _LevelSystem(moving, fixed, None, None, torch.eye(4), axes, torch.eye(4), bases, bending, 0.15, True, mode == "scale")
        generator = torch.Generator().manual_seed(913)
        coefficients = torch.randn((3, *fsl_control_shape(shape, spacing)), generator=generator, dtype=dtype) * 0.01
        if mode == "joint_intensity":
            joint = _JointT1System(system, bases, bending, 0.01, 3)
            polynomial = torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
            bias = torch.ones((1, *coefficients.shape[1:]), dtype=dtype)
            linearize = lambda: joint.linearize(coefficients, polynomial, bias, fit_intensity=True)
        else:
            linearize = lambda: system.linearize(coefficients, torch.tensor(1.125, dtype=dtype))

        actual_state, actual_gradient, actual_matvec, actual_diagonal = linearize()

        def original_normal(weights, cross, count):
            return lambda field, scale: reference(field, weights, cross, scale, count)
        monkeypatch.setattr(cpu_normal_module, "SpatialNormalCPU", original_normal)
        expected_state, expected_gradient, expected_matvec, expected_diagonal = linearize()
        for actual, expected in ((actual_state["cost"], expected_state["cost"]), (actual_gradient, expected_gradient), (actual_diagonal, expected_diagonal)):
            assert torch.equal(actual.contiguous().view(torch.int64), expected.contiguous().view(torch.int64))
        retained = None
        for _ in range(3):
            vector = torch.randn(actual_gradient.shape, generator=generator, dtype=dtype)
            actual, expected = actual_matvec(vector), expected_matvec(vector)
            assert torch.equal(actual.contiguous().view(torch.int64), expected.contiguous().view(torch.int64))
            if retained is not None:
                # Reused dense scratch must not alias a returned solver vector.
                assert torch.equal(retained[0], retained[1])
            retained = actual, actual.clone()
    finally:
        torch.set_num_threads(torch_threads)
        numba.set_num_threads(numba_threads)


@pytest.mark.parametrize("threads", [1, 8])
@pytest.mark.parametrize("fit_scale", [False, True])
def test_dense_physical_axis_permutations_preserve_bits_views_and_alias(threads, fit_scale, monkeypatch):
    """All channel positions use physical views, including scratch input reuse."""
    torch_threads, numba_threads = torch.get_num_threads(), numba.get_num_threads()
    try:
        torch.set_num_threads(threads)
        numba.set_num_threads(threads)
        monkeypatch.setattr(cpu_normal_module.config, "NUMBA_NUM_THREADS", threads)
        generator = torch.Generator().manual_seed(731)
        field = torch.randn((3, 12, 11, 10), generator=generator, dtype=torch.float64)
        weights = tuple(tuple(torch.randn(field.shape[1:], generator=generator, dtype=torch.float64) for _ in range(3)) for _ in range(3))
        cross = tuple(torch.randn(field.shape[1:], generator=generator, dtype=torch.float64) for _ in range(3)) if fit_scale else None
        scale = -0.812345678901 if fit_scale else None
        normal = SpatialNormalCPU(weights, cross, 193.0)
        for axes in permutations(range(4)):
            inverse = tuple(np.argsort(axes))
            current = field.permute(axes).contiguous().permute(inverse)
            before = current.clone()
            expected = reference(before, weights, cross, scale, 193.0)
            actual = normal(current, scale)
            assert actual.stride() == current.stride()
            assert torch.equal(actual.contiguous().view(torch.int64), expected.contiguous().view(torch.int64)), axes
            assert all(np.shares_memory(flat, packed) for flat, packed in zip(normal.flat_weights, normal.weights))
            assert all(np.shares_memory(flat, packed) for flat, packed in zip(normal.flat_cross, normal.cross))
            pointers = tuple(value.ctypes.data for value in (*normal.weights, *normal.cross))
            expected_alias = reference(actual.clone(), weights, cross, scale, 193.0)
            alias = normal(actual, scale)
            assert alias.data_ptr() == actual.data_ptr()
            assert torch.equal(alias.contiguous().view(torch.int64), expected_alias.contiguous().view(torch.int64)), axes
            assert pointers == tuple(value.ctypes.data for value in (*normal.weights, *normal.cross))
            assert (torch.get_num_threads(), numba.get_num_threads()) == (threads, threads)
    finally:
        torch.set_num_threads(torch_threads)
        numba.set_num_threads(numba_threads)


@pytest.mark.parametrize("threads", [1, 8])
def test_non_dense_field_never_enters_view_kernel(threads, monkeypatch):
    torch_threads, numba_threads = torch.get_num_threads(), numba.get_num_threads()
    try:
        torch.set_num_threads(threads)
        numba.set_num_threads(threads)
        monkeypatch.setattr(cpu_normal_module.config, "NUMBA_NUM_THREADS", threads)
        def unexpected_flat(*args, **kwargs):
            raise AssertionError("Strided field must use the original 3D kernel")
        monkeypatch.setattr(cpu_normal_module, "_flat_serial", unexpected_flat)
        monkeypatch.setattr(cpu_normal_module, "_flat_parallel", unexpected_flat)
        field = torch.arange(3 * 12 * 11 * 21, dtype=torch.float64).reshape(3, 12, 11, 21)[..., ::2]
        weights = tuple(tuple(torch.full(field.shape[1:], 0.2 + row - column, dtype=torch.float64) for column in range(3)) for row in range(3))
        normal = SpatialNormalCPU(weights, None, 23.0)
        actual = normal(field, None)
        expected = reference(field, weights, None, None, 23.0)
        assert torch.equal(actual.contiguous().view(torch.int64), expected.contiguous().view(torch.int64))
    finally:
        torch.set_num_threads(torch_threads)
        numba.set_num_threads(numba_threads)


@pytest.mark.parametrize("threads", [1, 8])
@pytest.mark.parametrize("fit_scale", [False, True])
@pytest.mark.parametrize("layout", ["contiguous", "spline"])
def test_vector_blocks_match_scalar_for_extremes_and_nonfinite_payloads(threads, fit_scale, layout, monkeypatch):
    torch_threads, numba_threads = torch.get_num_threads(), numba.get_num_threads()
    try:
        torch.set_num_threads(threads)
        numba.set_num_threads(threads)
        monkeypatch.setattr(cpu_normal_module.config, "NUMBA_NUM_THREADS", threads)
        bits = np.array([0, 1 << 63, 1, (1 << 63) + 1, 0x7ff0000000000000,
                         0xfff0000000000000, 0x7ff8000000000081, 0x7ff0000000000019], dtype=np.uint64)
        values = np.concatenate([np.array([1e308, -1e308, 1e-308, -1e-308, 1e200, -1e200, 1e-200, -1e-200]),
                                 bits.view(np.float64)])
        field = torch.from_numpy(np.tile(values, 3 * 12 * 11).reshape(3, 12, 11, 16).copy())
        if layout == "spline":
            field = field.permute(3, 0, 1, 2).contiguous().permute(1, 2, 3, 0)
        weights = tuple(tuple(torch.full(field.shape[1:], (-1.0) ** (row + column) * 1e200, dtype=torch.float64) for column in range(3)) for row in range(3))
        cross = tuple(torch.full(field.shape[1:], (-1.0) ** row * 1e200, dtype=torch.float64) for row in range(3)) if fit_scale else None
        scale = -1e200 if fit_scale else None
        normal = SpatialNormalCPU(weights, cross, 3.1)
        actual = normal(field, scale).clone()
        # Call the scalar 3D expression directly, including its original
        # exceptional-value propagation and payloads.
        expected = torch.empty_like(field)
        cpu_normal_module._serial(field.numpy(), normal.weights, normal.cross,
                                  0.0 if scale is None else scale, fit_scale, 3.1, expected.numpy(), 1 if layout == "spline" else 2)
        assert torch.equal(actual.contiguous().view(torch.int64), expected.contiguous().view(torch.int64))
    finally:
        torch.set_num_threads(torch_threads)
        numba.set_num_threads(numba_threads)
