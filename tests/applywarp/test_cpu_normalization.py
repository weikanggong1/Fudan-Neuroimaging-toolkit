"""Fused normalization keeps the existing coordinate and thread contract."""
import numpy as np
import pytest
import torch
import numba

from fnit.applywarp import core
from fnit.applywarp import _normalization_cpu as fused


def _assert_bits(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    bits = torch.int32 if actual.dtype == torch.float32 else torch.int64
    assert torch.equal(actual.contiguous().view(bits), expected.contiguous().view(bits))


@pytest.mark.parametrize("threads", [1, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(91, 109, 91), (1, 109, 91), (91, 1, 1), (2, 3, 5)])
def test_float32_grid_baseline_bits_boundaries_and_thread_settings(threads, dtype, shape):
    previous = torch.get_num_threads(), numba.get_num_threads()
    try:
        torch.set_num_threads(threads)
        numba.set_num_threads(min(threads, numba.config.NUMBA_NUM_THREADS))
        coordinates = torch.randn((3, 17, 19, 23), dtype=dtype,
                                  generator=torch.Generator().manual_seed(713)) * 110
        for axis, size in enumerate(shape):
            coordinates[axis].view(-1)[:8] = torch.tensor(
                [-0.0, 0.0, -.5, .5, size - 1., size - .5, size, -1.], dtype=dtype)
        expected = core._to_grid(coordinates, shape).float()
        before = torch.get_num_threads(), numba.get_num_threads()
        actual = core._to_float32_grid_cpu(coordinates, shape)
        _assert_bits(actual, expected)
        assert (torch.get_num_threads(), numba.get_num_threads()) == before
    finally:
        torch.set_num_threads(previous[0])
        numba.set_num_threads(previous[1])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_nonfinite_payloads_overflow_signed_zero_and_subnormals(dtype):
    # NaN payloads, signs and signaling bits exercise the exact store path.
    payloads = ([0x7fc12345, 0xffc23456, 0x7f812345, 0xff812345]
                if dtype == torch.float32 else
                [0x7ff8123456789abc, 0xfff823456789abcd,
                 0x7ff0123456789abc, 0xfff0123456789abc])
    unsigned, floating = ((np.uint32, np.float32) if dtype == torch.float32
                          else (np.uint64, np.float64))
    finfo = torch.finfo(dtype)
    values = torch.tensor([-0., 0., float("inf"), -float("inf"),
                           finfo.max, -finfo.max, finfo.tiny, -finfo.tiny,
                           finfo.tiny * finfo.eps, -finfo.tiny * finfo.eps], dtype=dtype)
    values = torch.cat((values, torch.from_numpy(np.array(payloads, dtype=unsigned).view(floating))))
    coordinates = values[None].expand(3, -1).clone()
    for shape in ((1, 13, 17), (11, 13, 17)):
        _assert_bits(core._to_float32_grid_cpu(coordinates, shape),
                     core._to_grid(coordinates, shape).float())
        if dtype == torch.float64:
            _assert_bits(core._to_float64_grid_cpu(coordinates, shape),
                         core._to_grid(coordinates, shape))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("denominator", [3, 5, 7, 15, 90, 108, 255])
def test_non_power_of_two_scalar_division_random_bits_and_boundary_ulps(dtype, denominator):
    rng = np.random.default_rng(177 + denominator)
    # Sample the whole exponent/sign/payload range; compare all uint32 stores,
    # including NaN bits. This is a unit test rather than a speed benchmark.
    unsigned, floating = ((np.uint32, np.float32) if dtype == torch.float32
                          else (np.uint64, np.float64))
    bits = np.frombuffer(rng.bytes(49152 * np.dtype(unsigned).itemsize), dtype=unsigned).copy()
    values = bits.view(floating)
    boundaries = np.array([-.5, 0., .5, denominator - .5, denominator,
                           denominator + .5], dtype=floating)
    neighbors = np.concatenate((np.nextafter(boundaries, floating(-np.inf)),
                                 boundaries,
                                 np.nextafter(boundaries, floating(np.inf))))
    values = np.concatenate((values, neighbors))
    coordinates = torch.from_numpy(values.copy())[None].expand(3, -1).clone()
    shape = (denominator + 1,) * 3
    _assert_bits(core._to_float32_grid_cpu(coordinates, shape),
                 core._to_grid(coordinates, shape).float())
    if dtype == torch.float64:
        _assert_bits(core._to_float64_grid_cpu(coordinates, shape),
                     core._to_grid(coordinates, shape))


@pytest.mark.parametrize("threads", [1, 8])
@pytest.mark.parametrize("shape", [(91, 109, 91), (1, 109, 91), (91, 1, 1), (2, 3, 5)])
def test_fp64_grid_keeps_all_fp64_bits_and_thread_settings(threads, shape):
    previous = torch.get_num_threads(), numba.get_num_threads()
    try:
        torch.set_num_threads(threads)
        numba.set_num_threads(min(threads, numba.config.NUMBA_NUM_THREADS))
        generator = torch.Generator().manual_seed(2917)
        coordinates = torch.randn((3, 17, 19, 23), dtype=torch.float64, generator=generator) * 110
        coordinates[:, 0, 0, :8] = torch.tensor([-0., 0., -.5, .5, float("nan"),
                                                float("inf"), -float("inf"), 1e300], dtype=torch.float64)
        before = torch.get_num_threads(), numba.get_num_threads()
        _assert_bits(core._to_float64_grid_cpu(coordinates, shape), core._to_grid(coordinates, shape))
        assert (torch.get_num_threads(), numba.get_num_threads()) == before
    finally:
        torch.set_num_threads(previous[0])
        numba.set_num_threads(previous[1])


def test_contiguous_storage_offset_uses_zero_copy_views(monkeypatch):
    coordinates = torch.arange(3 * 20 + 7, dtype=torch.float64)[7:].reshape(3, 4, 5)
    captured = {}
    original = fused.normalize_into
    def capture(source, shape, target):
        captured["source"] = source.__array_interface__["data"][0]
        return original(source, shape, target)
    monkeypatch.setattr(fused, "normalize_into", capture)
    actual = core._to_float64_grid_cpu(coordinates, (31, 23, 29))
    assert captured["source"] == coordinates.data_ptr()
    _assert_bits(actual, core._to_grid(coordinates, (31, 23, 29)))


@pytest.mark.parametrize("layout", ["transpose", "stride", "broadcast", "negative_view"])
def test_other_layouts_keep_tensor_path(monkeypatch, layout):
    coordinates = torch.arange(3 * 5 * 7 * 9, dtype=torch.float64).reshape(3, 5, 7, 9)
    if layout == "transpose":
        coordinates = coordinates.transpose(1, 2)
    elif layout == "stride":
        coordinates = coordinates[:, ::2]
    elif layout == "broadcast":
        coordinates = coordinates[:, :1].expand(-1, 5, -1, -1)
    else:
        coordinates = torch._neg_view(coordinates)
    def unexpected(*arguments):
        raise AssertionError("Other layouts must keep the tensor path")
    monkeypatch.setattr(fused, "normalize_into", unexpected)
    _assert_bits(core._to_float32_grid_cpu(coordinates, (31, 23, 29)),
                 core._to_grid(coordinates, (31, 23, 29)).float())
    _assert_bits(core._to_float64_grid_cpu(coordinates, (31, 23, 29)),
                 core._to_grid(coordinates, (31, 23, 29)))


def test_autograd_and_unsupported_dtype_keep_tensor_path(monkeypatch):
    def unexpected(*arguments):
        raise AssertionError("Autograd/other dtypes must keep the tensor path")
    monkeypatch.setattr(fused, "normalize_into", unexpected)
    for dtype, requires_grad in ((torch.float32, True), (torch.float64, True),
                                 (torch.float16, False), (torch.int64, False)):
        coordinates = torch.arange(3 * 4 * 5, dtype=dtype).reshape(3, 4, 5)
        coordinates.requires_grad_(requires_grad)
        actual = core._to_float32_grid_cpu(coordinates, (31, 23, 29))
        _assert_bits(actual, core._to_grid(coordinates, (31, 23, 29)).float())
        if requires_grad:
            actual.sum().backward()
            expected = torch.tensor([2 / 30, 2 / 22, 2 / 28], dtype=dtype)[:, None, None].expand_as(coordinates)
            assert torch.equal(coordinates.grad, expected)
        _assert_bits(core._to_float64_grid_cpu(coordinates, (31, 23, 29)),
                     core._to_grid(coordinates, (31, 23, 29)).double())


def test_oversized_default_pool_is_not_initialized(monkeypatch):
    monkeypatch.setattr(fused.config, "NUMBA_NUM_THREADS", torch.get_num_threads() + 1)
    def unexpected():
        raise AssertionError("Oversized default pool must remain uninitialized")
    monkeypatch.setattr(fused, "get_num_threads", unexpected)
    coordinates = torch.arange(3 * 4 * 5, dtype=torch.float64).reshape(3, 4, 5)
    _assert_bits(core._to_float32_grid_cpu(coordinates, (31, 23, 29)),
                 core._to_grid(coordinates, (31, 23, 29)).float())
    _assert_bits(core._to_float64_grid_cpu(coordinates, (31, 23, 29)),
                 core._to_grid(coordinates, (31, 23, 29)))


def test_explicit_cpu_grid_ignores_global_default_meta_device():
    coordinates = torch.arange(3 * 4 * 5, dtype=torch.float64, device="cpu").reshape(3, 4, 5)
    expected = core._to_grid(coordinates, (31, 23, 29))
    previous = torch.get_default_device()
    try:
        torch.set_default_device("meta")
        actual = core._to_float64_grid_cpu(coordinates, (31, 23, 29))
        assert actual.device.type == "cpu"
        _assert_bits(actual, expected)
    finally:
        torch.set_default_device(previous)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_keeps_original_tensor_path(monkeypatch):
    def unexpected(*arguments):
        raise AssertionError("CUDA must keep the tensor path")
    monkeypatch.setattr(fused, "normalize_into", unexpected)
    coordinates = torch.arange(3 * 4 * 5, dtype=torch.float64, device="cuda").reshape(3, 4, 5)
    _assert_bits(core._to_float32_grid_cpu(coordinates, (31, 23, 29)),
                 core._to_grid(coordinates, (31, 23, 29)).float())
    _assert_bits(core._to_float64_grid_cpu(coordinates, (31, 23, 29)),
                 core._to_grid(coordinates, (31, 23, 29)))


@pytest.mark.parametrize("threads", [1, 8])
def test_one_shot_fp64_sampling_keeps_tensor_normalization(monkeypatch, threads):
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        data = torch.randn((3, 5, 7, 9), dtype=torch.float64,
                           generator=torch.Generator().manual_seed(871))
        coordinates = torch.randn((3, 9, 7, 5), dtype=torch.float64,
                                  generator=torch.Generator().manual_seed(872)) * 3 + 2
        expected_grid = core._to_grid(coordinates, data.shape[1:]).to(data.dtype)
        expected = core._sample_linear_cpu(data, expected_grid)
        def unexpected(*arguments):
            raise AssertionError("One-shot FP64 sampling must not import/JIT the fused helper")
        monkeypatch.setattr(core, "_to_float64_grid_cpu", unexpected)
        actual, valid = core._sample_linear(data, coordinates)
        _assert_bits(actual, expected)
        assert torch.equal(valid, core._inside(coordinates, data.shape[1:]))
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("threads", [1, 8])
def test_prepared_fp64_sampling_uses_fused_normalization(monkeypatch, threads):
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        data = torch.randn((3, 5, 7, 9), dtype=torch.float64,
                           generator=torch.Generator().manual_seed(891))
        coordinates = torch.randn((3, 9, 7, 5), dtype=torch.float64,
                                  generator=torch.Generator().manual_seed(892)) * 3 + 2
        source = data[None].contiguous(memory_format=torch.channels_last_3d)
        expected_grid = core._to_grid(coordinates, data.shape[1:]).to(data.dtype)
        expected = core._sample_linear_cpu(data, expected_grid, prepared_source=source)
        calls = []
        original = core._to_float64_grid_cpu
        def capture(coordinates, shape):
            calls.append((coordinates.dtype, tuple(shape)))
            return original(coordinates, shape)
        monkeypatch.setattr(core, "_to_float64_grid_cpu", capture)
        actual, valid = core._sample_linear(data, coordinates, prepared_source=source)
        _assert_bits(actual, expected)
        assert len(calls) == 1
        assert torch.equal(valid, core._inside(coordinates, data.shape[1:]))
    finally:
        torch.set_num_threads(previous)
