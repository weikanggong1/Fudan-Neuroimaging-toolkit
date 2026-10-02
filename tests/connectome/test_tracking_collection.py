"""Exact path collection checks for the tracking dataflow optimization."""

import pytest
import torch

from fnit.connectome.tracking import Tractogram, _collect_tracks, _sample


def _original_collection(forward, backward, forward_counts, backward_counts,
                         one_way, keep, total_lengths, seeds, *, fa_image,
                         fod_inverse):
    """Freeze the previous production loop as a differential reference."""
    paths, endpoints, lengths, accepted_seeds, fa_means = [], [], [], [], []
    for index in keep.tolist():
        path = (forward[index, :forward_counts[index]] if bool(one_way[index]) else
                torch.cat((backward[index, :backward_counts[index]].flip(0),
                           forward[index, 1:forward_counts[index]]), dim=0))
        paths.append(path)
        endpoints.append(torch.stack((path[0], path[-1])))
        lengths.append(total_lengths[index])
        accepted_seeds.append(seeds[index])
        if fa_image is not None:
            fa_means.append(_sample(fa_image, path, fod_inverse).mean())
    return Tractogram(
        paths=tuple(paths),
        endpoints=torch.stack(endpoints) if endpoints else forward.new_empty((0, 2, 3)),
        lengths_mm=torch.stack(lengths) if lengths else forward.new_empty(0),
        mean_fa=(torch.stack(fa_means) if fa_means else
                 forward.new_empty(0) if fa_image is not None else None),
        seeds_attempted=seeds.shape[0],
        accepted_seeds=(torch.stack(accepted_seeds) if accepted_seeds else
                        seeds.new_empty((0, 3))),
    )


def _buffers(device):
    seeds = torch.tensor([[6., 4., 2.], [7., 5., 3.], [8., 6., 4.], [9., 7., 5.]],
                         device=device)
    offsets = torch.arange(6, device=device, dtype=torch.float32)
    delta = torch.stack((offsets, offsets / 4, offsets / 8), dim=-1)
    forward = seeds[:, None] + delta[None]
    backward = seeds[:, None] - delta[None]
    forward_counts = torch.tensor([5, 4, 1, 3], device=device)
    backward_counts = torch.tensor([2, 3, 4, 1], device=device)
    one_way = torch.tensor([True, False, True, False], device=device)
    keep = torch.tensor([3, 0, 1, 2], device=device)
    lengths = torch.tensor([4.123, 5.234, 0., 2.345], device=device)
    return (forward, backward, forward_counts, backward_counts, one_way, keep,
            lengths, seeds)


def _assert_equal(actual, expected):
    assert isinstance(actual.paths, tuple)
    assert len(actual.paths) == len(expected.paths)
    assert all(torch.equal(left, right) for left, right in zip(actual.paths, expected.paths))
    for field in ("endpoints", "lengths_mm", "accepted_seeds"):
        assert torch.equal(getattr(actual, field), getattr(expected, field)), field
    if expected.mean_fa is None:
        assert actual.mean_fa is None
    else:
        assert torch.equal(actual.mean_fa, expected.mean_fa)
    assert actual.seeds_attempted == expected.seeds_attempted


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("include_fa", [False, True])
def test_collection_preserves_all_values_order_and_view_semantics(device, include_fa):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    buffers = _buffers(device)
    fa_image = (torch.arange(16**3, dtype=torch.float32, device=device).reshape(16, 16, 16, 1)
                / 16**3 if include_fa else None)
    inverse = torch.eye(4, dtype=torch.float64, device=device)
    expected = _original_collection(*buffers, fa_image=fa_image, fod_inverse=inverse)
    rng_before = (torch.cuda.get_rng_state(device) if device == "cuda" else
                  torch.random.get_rng_state())
    actual = _collect_tracks(*buffers, fa_image=fa_image, fod_inverse=inverse)
    rng_after = (torch.cuda.get_rng_state(device) if device == "cuda" else
                 torch.random.get_rng_state())
    _assert_equal(actual, expected)
    assert torch.equal(rng_before, rng_after)

    forward, backward, _, _, one_way, keep, _, _ = buffers
    for path, index in zip(actual.paths, keep.tolist()):
        if bool(one_way[index]):
            assert path.untyped_storage().data_ptr() == forward.untyped_storage().data_ptr()
        else:
            assert path.untyped_storage().data_ptr() != forward.untyped_storage().data_ptr()
            assert path.untyped_storage().data_ptr() != backward.untyped_storage().data_ptr()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("include_fa", [False, True])
def test_collection_keeps_empty_shapes_and_optional_fa(device, include_fa):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    buffers = list(_buffers(device))
    buffers[5] = torch.empty(0, dtype=torch.long, device=device)
    fa_image = torch.zeros((16, 16, 16, 1), device=device) if include_fa else None
    inverse = torch.eye(4, dtype=torch.float64, device=device)
    expected = _original_collection(*buffers, fa_image=fa_image, fod_inverse=inverse)
    actual = _collect_tracks(*buffers, fa_image=fa_image, fod_inverse=inverse)
    _assert_equal(actual, expected)
    assert actual.endpoints.shape == (0, 2, 3)
    assert actual.lengths_mm.shape == (0,)
    assert actual.accepted_seeds.shape == (0, 3)
    assert actual.endpoints.device == buffers[0].device
