import numpy as np
import pytest
import torch
from numba import get_num_threads, set_num_threads

from fnit.fnirt import _sampling_cpu as fused
from fnit.fnirt import registration


def original(volume, coordinates, *, derivatives=True):
    # Force the unchanged tensor fallback as an independent reference, even
    # after CPU lazy dispatch is enabled. This changes only autograd metadata.
    return registration._trilinear_sample(
        volume.detach().requires_grad_(True), coordinates,
        derivatives=derivatives)



def assert_bits(actual, expected):
    assert torch.equal(actual.contiguous().view(torch.int32), expected.contiguous().view(torch.int32))


@pytest.mark.parametrize("threads", [1, 8])
@pytest.mark.parametrize("derivatives", [False, True])
@pytest.mark.parametrize("shape", [(1, 1, 1), (1, 4, 1), (7, 9, 11), (4096, 2, 3), (4097, 2, 3)])
@pytest.mark.parametrize("kind", ["normal", "signed_zero", "finite_raw_bits"])
def test_all_value_valid_and_gradient_bits(shape, kind, derivatives, threads):
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        rng = np.random.default_rng(782)
        if kind == "normal":
            image = rng.normal(0, 50, shape).astype(np.float32)
        elif kind == "signed_zero":
            image = np.zeros(shape, np.float32)
            image.reshape(-1)[::2] = np.float32(-0.)
        else:
            bits = np.frombuffer(rng.bytes(np.prod(shape).item() * 4), np.uint32).copy()
            # Keep finite mantissas/signs and a broad exponent range, including
            # subnormals, without forcing every call into overflow fallback.
            exponent = rng.integers(0, 210, bits.size, dtype=np.uint32)
            bits = (bits & np.uint32(0x807fffff)) | (exponent << np.uint32(23))
            image = bits.view(np.float32).reshape(shape)
        query = (rng.random((3, 67)) * (np.asarray(shape)[:, None] + 3) - 1.5).astype(np.float32)
        for axis, size in enumerate(shape):
            high = np.float32(size - 1)
            query[axis, :12] = np.array([
                -0., 0., -5e-9, -2e-8, high,
                np.nextafter(high, np.float32(np.inf)),
                np.nextafter(high, np.float32(-np.inf)),
                np.nextafter(np.float32(0), np.float32(-np.inf)),
                np.nextafter(np.float32(0), np.float32(np.inf)),
                -1000., size + 1000., .5,
            ], np.float32)
        volume, coordinates = torch.from_numpy(image), torch.from_numpy(query)
        expected = original(volume, coordinates, derivatives=derivatives)
        actual = fused.try_sample_cpu(volume, coordinates, derivatives=derivatives)
        assert actual is not None
        assert_bits(actual[0], expected[0])
        assert torch.equal(actual[1], expected[1])
        if derivatives:
            assert_bits(actual[2], expected[2])
        else:
            assert actual[2] is None
        assert actual[0].data_ptr() not in (volume.data_ptr(), coordinates.data_ptr())
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("kind", ["volume_nan", "volume_inf", "query_nan", "query_inf",
                                  "float64_volume", "float64_query", "volume_grad", "query_grad",
                                  "strided_volume", "strided_query", "negative_volume", "negative_query",
                                  "query_large", "empty_query", "scalar_query"])
def test_unsupported_inputs_return_original_fallback(kind):
    volume = torch.ones((4, 5, 6), dtype=torch.float32)
    query = torch.ones((3, 17), dtype=torch.float32)
    if kind == "volume_nan": volume[1, 2, 3] = float("nan")
    elif kind == "volume_inf": volume[1, 2, 3] = float("inf")
    elif kind == "query_nan": query[1, 0] = float("nan")
    elif kind == "query_inf": query[1, 0] = float("inf")
    elif kind == "float64_volume": volume = volume.double()
    elif kind == "float64_query": query = query.double()
    elif kind == "volume_grad": volume.requires_grad_()
    elif kind == "query_grad": query.requires_grad_()
    elif kind == "strided_volume": volume = volume.transpose(0, 1)
    elif kind == "strided_query": query = query[:, ::2]
    elif kind == "negative_volume": volume = torch._neg_view(volume)
    elif kind == "negative_query": query = torch._neg_view(query)
    elif kind == "query_large": query[0, 0] = 2**30
    elif kind == "empty_query": query = query[:, :0]
    else: query = query[:, 0]
    assert fused.try_sample_cpu(volume, query) is None


def test_intermediate_overflow_keeps_tensor_nan_payload_fallback():
    volume = torch.full((2, 2, 2), np.finfo(np.float32).max, dtype=torch.float32)
    volume[1] *= -1
    query = torch.full((3, 5), .5, dtype=torch.float32)
    assert fused.try_sample_cpu(volume, query) is None


@pytest.mark.parametrize("argument", ["volume", "coordinates"])
def test_forward_mode_tangents_keep_tensor_fallback(argument):
    volume = torch.ones((4, 5, 6), dtype=torch.float32)
    query = torch.ones((3, 17), dtype=torch.float32)
    with torch.autograd.forward_ad.dual_level():
        if argument == "volume":
            volume = torch.autograd.forward_ad.make_dual(volume, torch.ones_like(volume))
        else:
            query = torch.autograd.forward_ad.make_dual(query, torch.ones_like(query))
        assert fused.try_sample_cpu(volume, query) is None


def test_vmap_tensor_wrapper_keeps_original_tensor_fallback():
    volume = torch.ones((4, 5, 6), dtype=torch.float32)
    queries = torch.ones((2, 3, 17), dtype=torch.float32)
    def check(query):
        assert fused.try_sample_cpu(volume, query) is None
        return query
    assert torch.equal(torch.vmap(check)(queries), queries)


def test_inplace_mutation_is_observed_without_cache():
    volume = torch.arange(4*5*6, dtype=torch.float32).reshape(4, 5, 6)
    query = torch.full((3, 7), .5, dtype=torch.float32)
    before = fused.try_sample_cpu(volume, query)
    volume.add_(10)
    actual = fused.try_sample_cpu(volume, query)
    expected = original(volume, query)
    assert_bits(actual[0], expected[0])
    assert_bits(actual[2], expected[2])
    assert not torch.equal(actual[0], before[0])
    volume[0, 0, 0] = float("nan")
    assert fused.try_sample_cpu(volume, query) is None


def test_thread_mask_restores_on_kernel_failure(monkeypatch):
    previous_numba, previous_torch = get_num_threads(), torch.get_num_threads()
    try:
        set_num_threads(8)
        torch.set_num_threads(2)
        def fail(*args):
            assert get_num_threads() == 2
            raise RuntimeError("kernel failure")
        monkeypatch.setattr(fused, "_parallel", fail)
        with pytest.raises(RuntimeError, match="kernel failure"):
            fused.try_sample_cpu(torch.ones((4, 5, 6)), torch.ones((3, 7)))
        assert get_num_threads() == 8
    finally:
        set_num_threads(previous_numba)
        torch.set_num_threads(previous_torch)


def test_explicit_cpu_inputs_ignore_default_meta_device():
    volume = torch.arange(4*5*6, device="cpu", dtype=torch.float32).reshape(4, 5, 6)
    query = torch.full((3, 7), .5, device="cpu", dtype=torch.float32)
    expected = original(volume, query)
    previous = torch.get_default_device()
    try:
        torch.set_default_device("meta")
        actual = fused.try_sample_cpu(volume, query)
        assert actual[0].device.type == "cpu"
        assert_bits(actual[0], expected[0])
        assert_bits(actual[2], expected[2])
    finally:
        torch.set_default_device(previous)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_never_enters_numpy_sampler():
    assert fused.try_sample_cpu(torch.ones((4, 5, 6), device="cuda"),
                                torch.ones((3, 7), device="cuda")) is None
