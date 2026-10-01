import numpy as np
import pytest
import torch

from fnit.gems.rasterize import BlockIndex
from validation.subregions.benchmark_compact_layout import legacy_compact_batches


@pytest.fixture(params=("cpu", "cuda"))
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return torch.device(request.param)


def _layout_input(device, mask_kind):
    shape, block_size = (37, 27, 9), 4
    nblocks = tuple((n + block_size - 1) // block_size for n in shape)
    candidates = tuple(np.arange(1 + (block * 17) % 137, dtype=np.int64)
                       if block % 11 else np.empty((0,), np.int64)
                       for block in range(np.prod(nblocks)))
    index = BlockIndex(shape, block_size, candidates)
    linear = torch.arange(np.prod(shape), device=device).reshape(shape)
    mask = (torch.ones_like(linear, dtype=torch.bool) if mask_kind == "full" else
            linear % 3 != 0 if mask_kind == "sparse" else torch.zeros_like(linear, dtype=torch.bool))
    return index, mask


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mask_kind", ["full", "sparse", "empty"])
def test_packed_layout_is_bitwise_equal_to_legacy_batches(device, dtype, mask_kind):
    index, mask = _layout_input(device, mask_kind)
    expected = legacy_compact_batches(index, mask, device, dtype)
    actual = index.device_compact_batches(mask, device, dtype)
    assert len(actual[0]) == len(expected[0])
    assert torch.equal(actual[1], expected[1])
    for got, wanted in zip(actual[0], expected[0]):
        for x, y in zip(got, wanted):
            assert torch.equal(x, y)
            assert x.dtype == y.dtype and x.shape == y.shape and x.is_contiguous()
    # All batch views share one buffer per type; no per-batch H2D copies.
    for column in (0, 1, 2, 3, 4):
        assert len({batch[column].untyped_storage().data_ptr() for batch in actual[0]}) <= 1
    assert index.device_compact_batches(mask, device, dtype)[0] is actual[0]


def test_packed_layout_tracks_inplace_mask_changes_and_inference_masks(device):
    index, mask = _layout_input(device, "sparse")
    first = index.device_compact_batches(mask, device, torch.float32)
    mask[0] = False
    updated = index.device_compact_batches(mask, device, torch.float32)
    assert updated[0] is not first[0]
    expected = legacy_compact_batches(index, mask, device, torch.float32)
    assert torch.equal(updated[1], expected[1])
    with torch.inference_mode():
        inferred = mask.clone()
        result = index.device_compact_batches(inferred, device, torch.float32)
        assert torch.equal(result[1], expected[1])

