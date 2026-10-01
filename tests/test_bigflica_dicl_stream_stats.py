"""Bounded statistics and SVD-driver regressions, not imaging benchmarks."""

import h5py
import numpy as np
import pytest
import torch
from sklearn.utils.extmath import randomized_svd

import fnit.bigflica.dicl_torch as dicl
from fnit.bigflica.dicl_torch import (_preload_standardized_projection,
                                      _randomized_svd_dictionary,
                                      _streaming_numpy_axis0_stats)


def assert_bits_equal(left, right):
    assert left.dtype == right.dtype == np.float64
    np.testing.assert_array_equal(left.view(np.uint64), right.view(np.uint64))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("block_size", [1, 7, 64, 2048])
def test_streaming_stats_match_full_c_order_float64_reference(dtype, block_size):
    generator = np.random.default_rng(382)
    values = np.ascontiguousarray(generator.normal(size=(257, 8)), dtype=dtype)
    values[:, 0] += 1e6
    values[:, 1] *= 1e-5
    values[:, 2] = 13.
    # Float32 storage is explicitly promoted before the reference arithmetic.
    reference = np.asarray(values, dtype=np.float64, order="C")
    mean, std = _streaming_numpy_axis0_stats(values, block_size)
    assert_bits_equal(mean, reference.mean(axis=0))
    assert_bits_equal(std, reference.std(axis=0))
    assert std[2] == 0


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_streaming_stats_only_read_bounded_hdf5_windows(tmp_path, dtype):
    values = np.ascontiguousarray(np.random.default_rng(3).normal(size=(211, 7)), dtype=dtype)
    reference = values.astype(np.float64)
    with h5py.File(tmp_path / "projected.h5", "w") as handle:
        stored = handle.create_dataset("data", data=values)

        class Windowed:
            shape, dtype = stored.shape, stored.dtype

            def __init__(self):
                self.reads = []

            def __getitem__(self, window):
                assert isinstance(window, slice) and window.step is None
                assert 0 <= window.start < window.stop <= len(values)
                assert window.stop - window.start <= 17
                self.reads.append((window.start, window.stop))
                return stored[window]

        projected = Windowed()
        mean, std = _streaming_numpy_axis0_stats(projected, 17)
        assert len(projected.reads) == 2 * int(np.ceil(len(values) / 17))
    assert_bits_equal(mean, reference.mean(axis=0))
    assert_bits_equal(std, reference.std(axis=0))


def test_streaming_stats_centered_variance_survives_large_offsets():
    values = np.empty((257, 3), dtype=np.float64)
    values[:, 0] = 1e12 + np.arange(257) / 257
    values[:, 1] = 1e5 + (np.arange(257) % 9) * 1e-4
    values[:, 2] = 13.
    mean, std = _streaming_numpy_axis0_stats(values, 17)
    assert_bits_equal(mean, values.mean(axis=0))
    assert_bits_equal(std, values.std(axis=0))
    assert .28 < std[0] < .30
    assert 1e-4 < std[1] < 3e-4
    assert std[2] == 0


def test_streaming_stats_reject_different_reduction_scope():
    with pytest.raises(ValueError, match="C-order"):
        _streaming_numpy_axis0_stats(np.ones((8, 3), order="F"), 4)
    with pytest.raises(ValueError, match="two PCs"):
        _streaming_numpy_axis0_stats(np.ones((8, 1)), 4)


@pytest.mark.parametrize("backend", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"))])
def test_randomized_svd_uses_gesvd_only_on_cuda(tmp_path, monkeypatch, backend):
    values = np.ascontiguousarray(np.random.default_rng(41).normal(size=(64, 16)))
    mean, std = _streaming_numpy_axis0_stats(values, 17)
    samples = (values - mean) / std
    _, singular_values, right = randomized_svd(
        samples, n_components=4, n_oversamples=10, n_iter=4,
        power_iteration_normalizer="QR", random_state=0,
        transpose=False, flip_sign=True)
    expected = singular_values[:, None] * right
    original = torch.linalg.svd
    observed = []

    def observe_svd(matrix, *args, **kwargs):
        observed.append((matrix.device.type, dict(kwargs)))
        return original(matrix, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "svd", observe_svd)
    with h5py.File(tmp_path / "projected.h5", "w") as handle:
        projected = handle.create_dataset("data", data=values)
        actual = _randomized_svd_dictionary(projected,
            torch.as_tensor(samples, device=backend, dtype=torch.float64),
            torch.as_tensor(mean, device=backend, dtype=torch.float64),
            torch.as_tensor(std, device=backend, dtype=torch.float64),
            4, np.random.RandomState(0), 17)
    assert len(observed) == 1
    if backend == "cuda":
        assert observed[0][1]["driver"] == "gesvd"
    else:
        assert "driver" not in observed[0][1]
    np.testing.assert_allclose(actual.cpu().numpy(), expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("backend", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"))])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_projection_preload_reads_windows_and_matches_old_expression_bitwise(tmp_path, backend, dtype):
    values = np.ascontiguousarray(np.random.default_rng(573).normal(size=(65, 9)), dtype=dtype)
    values[:, 0] += 1e5
    values[:, 1] *= 1e-5
    before = values.copy()
    mean, std = _streaming_numpy_axis0_stats(values, 7)
    mean_device = torch.as_tensor(mean, device=backend)
    std_device = torch.as_tensor(std, device=backend)
    expected = (torch.as_tensor(values, device=backend, dtype=torch.float64) - mean_device) / std_device
    with h5py.File(tmp_path / "projected.h5", "w") as handle:
        stored = handle.create_dataset("data", data=values)

        class Windowed:
            shape, dtype = stored.shape, stored.dtype

            def __init__(self):
                self.reads = []

            def __getitem__(self, window):
                assert isinstance(window, slice) and window.step is None
                assert window.start is not None and window.stop is not None
                assert 0 <= window.start < window.stop <= len(values)
                assert window.stop - window.start <= 7
                self.reads.append((window.start, window.stop))
                return stored[window]

        projected = Windowed()
        actual = _preload_standardized_projection(projected, mean_device, std_device, 7)
        assert len(projected.reads) == int(np.ceil(len(values) / 7))
    assert actual.is_contiguous()
    torch.testing.assert_close(actual.view(torch.int64), expected.view(torch.int64), atol=0, rtol=0)
    np.testing.assert_array_equal(values, before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_projection_preload_has_no_full_device_normalization_temporary():
    values = np.ascontiguousarray(np.random.default_rng(74).normal(size=(513, 17)), dtype=np.float32)
    mean, std = _streaming_numpy_axis0_stats(values, 7)
    mean_device, std_device = torch.as_tensor(mean, device="cuda"), torch.as_tensor(std, device="cuda")
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    actual = _preload_standardized_projection(values, mean_device, std_device, 7)
    torch.cuda.synchronize()
    added_peak = torch.cuda.max_memory_allocated() - baseline
    # Cache + at most two conversion windows and allocator alignment; the
    # former expression needed additional full-size subtraction/division.
    assert added_peak <= actual.numel() * actual.element_size() + 2 * 7 * 17 * 8 + 4096


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_dicl_entry_preloads_without_full_hdf5_reads(tmp_path, monkeypatch):
    values = np.ascontiguousarray(np.random.default_rng(926).normal(size=(64, 12)))
    with h5py.File(tmp_path / "vbm_projected.h5", "w") as handle:
        handle.create_dataset("data", data=values)
    original_file = dicl.h5py.File
    reads = []

    class Windowed:
        def __init__(self, dataset):
            self.dataset = dataset
            self.shape, self.dtype = dataset.shape, dataset.dtype

        def __getitem__(self, window):
            assert isinstance(window, slice) and window.step is None
            assert window.start is not None and window.stop is not None
            assert 0 <= window.start < window.stop <= len(values)
            assert window.stop - window.start <= 7
            reads.append((window.start, window.stop))
            return self.dataset[window]

    class BoundedFile:
        def __init__(self, *args, **kwargs):
            self.handle = original_file(*args, **kwargs)

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __getitem__(self, key):
            assert key == "data"
            return Windowed(self.handle[key])

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

    monkeypatch.setattr(dicl.h5py, "File", BoundedFile)
    actual = dicl.fit_dicl_gpu_streaming(tmp_path, ["vbm"], 4, max_iter=1,
        feature_block=7, batch_size=32, sparse_iterations=1000)["vbm"]
    # Two normalization passes and one preloading pass; initialization and
    # minibatches consume the GPU cache and cause no additional HDF5 reads.
    assert len(reads) == 3 * int(np.ceil(len(values) / 7))
    assert actual.shape == (4, 12) and np.isfinite(actual).all()
