import numpy as np
import pytest
import torch

from fnit.gems.rasterize import BlockIndex
from validation.subregions.benchmark_dense_layout import legacy_dense_batches


@pytest.fixture(params=("cpu", "cuda"))
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return torch.device(request.param)


def _candidates(shape=(37, 27, 9), block_size=4):
    nblocks = tuple((n + block_size - 1) // block_size for n in shape)
    candidates = tuple(np.arange(1 + (block * 17) % 137, dtype=np.int64)
                       if block % 11 else np.empty((0,), np.int64) for block in range(np.prod(nblocks)))
    return shape, block_size, candidates


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dense_packed_buffers_preserve_candidate_points_and_write_indices(device, dtype):
    expected = legacy_dense_batches(BlockIndex(*_candidates()), device, dtype)
    actual = BlockIndex(*_candidates()).device_batches(device, dtype)
    assert len(actual) == len(expected)
    for got, wanted in zip(actual, expected):
        for x, y in zip(got, wanted):
            if isinstance(x, tuple):
                for a, b in zip(x, y):
                    assert torch.equal(a, b) and a.stride() == b.stride()
            else:
                assert torch.equal(x, y) and x.shape == y.shape and x.dtype == y.dtype
    for column in (0, 1, 2, 3):
        assert len({batch[column].untyped_storage().data_ptr() for batch in actual}) == 1
    assert len({xyz.untyped_storage().data_ptr() for batch in actual for xyz in batch[4]}) == 1


def test_dense_normal_route_does_not_create_per_block_cuda_tensors(device, monkeypatch):
    def forbidden_blocks(*args, **kwargs):
        raise AssertionError("normal dense raster must not build old per-block GPU arrays")
    monkeypatch.setattr(BlockIndex, "device_blocks", forbidden_blocks)
    index = BlockIndex(*_candidates())
    actual = index.device_batches(device, torch.float32)
    assert actual and index.device_batches(device, torch.float32) is actual


def test_dense_empty_candidates_keep_empty_batch_protocol(device):
    shape, size, candidates = _candidates()
    empty = BlockIndex(shape, size, tuple(np.empty((0,), np.int64) for _ in candidates))
    assert empty.device_batches(device, torch.float32) == ()

