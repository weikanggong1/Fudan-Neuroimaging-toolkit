"""Prepared field coordinates keep matrix, exceptional and gradient semantics."""
import numpy as np
import pytest
import torch

from fnit.applywarp import core as apply
from fnit.convertwarp import core


def _bits(value):
    return value.contiguous().view(torch.int64)


def _original(query, matrix):
    return (matrix[:3, :3] @ query.reshape(3, -1) + matrix[:3, 3:4]).reshape(query.shape)


@pytest.mark.parametrize("threads", [1, 8])
@pytest.mark.parametrize("scales", [(.5, .5, .5), (1/3, 1/5, 1/7), (-.5, .5, 2.)])
def test_finite_queries_match_all_fp64_bits(threads, scales):
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        matrix = torch.diag(torch.tensor([*scales, 1.], dtype=torch.float64))
        rng = np.random.default_rng(771)
        values = np.frombuffer(rng.bytes(3*16384*8), dtype=np.uint64).copy().view(np.float64)
        values[~np.isfinite(values)] = 0
        values[:8] = [-0., 0., np.nextafter(0., 1.), -np.nextafter(0., 1.),
                      np.finfo(np.float64).tiny, -np.finfo(np.float64).tiny,
                      np.finfo(np.float64).max, -np.finfo(np.float64).max]
        query = torch.from_numpy(values.reshape(3, 16, 32, 32))
        actual = core._diagonal_scaled_coordinates_cpu(query, matrix)
        assert actual is not None
        assert torch.equal(_bits(actual), _bits(_original(query, matrix)))
        assert actual.data_ptr() != query.data_ptr()
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_queries_keep_zero_times_nonfinite_matrix_terms(value):
    matrix = torch.eye(4, dtype=torch.float64)
    query = torch.zeros((3, 2, 3, 5), dtype=torch.float64)
    query[0, 0, 0, 0] = value
    assert core._diagonal_scaled_coordinates_cpu(query, matrix) is None
    # Zero times NaN/Inf affects the two orthogonal rows; the diagonal
    # row retains Inf for an infinite input and NaN for a NaN input.
    transformed = _original(query, matrix)[:, 0, 0, 0]
    assert torch.isnan(transformed[1:]).all()
    if np.isnan(value):
        assert torch.isnan(transformed[0])
    else:
        assert transformed[0].item() == value


@pytest.mark.parametrize("kind", ["shear", "translation", "negative_zero_translation", "nonfinite_diagonal"])
def test_other_matrices_keep_original_product(kind):
    matrix = torch.eye(4, dtype=torch.float64)
    if kind == "shear":
        matrix[0, 1] = .01
    elif kind == "translation":
        matrix[0, 3] = .1
    elif kind == "negative_zero_translation":
        matrix[0, 3] = -0.
    else:
        matrix[0, 0] = float("inf")
    query = torch.arange(3*4*5, dtype=torch.float64).reshape(3, 4, 5)
    assert core._diagonal_scaled_coordinates_cpu(query, matrix) is None


@pytest.mark.parametrize("kind", ["transpose", "stride", "negative_view", "empty", "float32", "query_grad", "matrix_grad", "matrix_negative_view"])
def test_other_layouts_dtypes_and_gradients_keep_original_product(kind):
    matrix = torch.eye(4, dtype=torch.float64)
    query = torch.arange(3*4*5, dtype=torch.float64).reshape(3, 4, 5)
    if kind == "transpose": query = query.transpose(1, 2)
    elif kind == "stride": query = query[:, ::2]
    elif kind == "negative_view": query = torch._neg_view(query)
    elif kind == "empty": query = query[:, :0]
    elif kind == "float32": query = query.float()
    elif kind == "query_grad": query.requires_grad_()
    elif kind == "matrix_grad": matrix.requires_grad_()
    else: matrix = torch._neg_view(matrix)
    assert core._diagonal_scaled_coordinates_cpu(query, matrix) is None


def _field(values):
    field = object.__new__(core._PullField)
    field.values = values
    field.scaled_inverse = torch.diag(torch.tensor([.5, .5, .5, 1.], dtype=torch.float64))
    field.embedded_inverse = None
    field.convention = "relative"
    return field


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("values_grad", [False, True])
def test_sampling_dispatch_and_values_gradient(monkeypatch, prepared, values_grad):
    values = torch.randn((3, 7, 9, 11), dtype=torch.float64,
                         generator=torch.Generator().manual_seed(671)).requires_grad_(values_grad)
    field = _field(values)
    query = torch.randn((3, 5, 7, 9), dtype=torch.float64,
                        generator=torch.Generator().manual_seed(672)) * 2 + 4
    source = values[None].contiguous(memory_format=torch.channels_last_3d) if prepared else None
    sampled, valid = apply._sample_linear(values, _original(query, field.scaled_inverse),
                                         prepared_source=source)
    expected = query + sampled
    calls = []
    original = core._diagonal_scaled_coordinates_cpu
    def capture(*args):
        calls.append(True)
        return original(*args)
    monkeypatch.setattr(core, "_diagonal_scaled_coordinates_cpu", capture)
    actual, actual_valid = field.sample(query, prepared_source=source)
    assert torch.equal(_bits(actual), _bits(expected))
    assert torch.equal(valid, actual_valid)
    assert bool(calls) == (prepared and not values_grad)
    if values_grad:
        actual.sum().backward()
        assert values.grad is not None and torch.isfinite(values.grad).all()


def test_explicit_cpu_query_ignores_global_default_meta_device():
    query = torch.arange(3*4*5, dtype=torch.float64, device="cpu").reshape(3, 4, 5)
    matrix = torch.eye(4, dtype=torch.float64, device="cpu")
    expected = _original(query, matrix)
    previous = torch.get_default_device()
    try:
        torch.set_default_device("meta")
        actual = core._diagonal_scaled_coordinates_cpu(query, matrix)
        assert actual.device.type == "cpu"
        assert torch.equal(_bits(actual), _bits(expected))
    finally:
        torch.set_default_device(previous)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_keeps_original_product():
    query = torch.arange(3*4*5, dtype=torch.float64, device="cuda").reshape(3, 4, 5)
    matrix = torch.eye(4, dtype=torch.float64, device="cuda")
    assert core._diagonal_scaled_coordinates_cpu(query, matrix) is None
