"""Analytic geometry checks for MRtrix-style SIFT2 fixel mapping."""

import pytest
import torch

from fnit.connectome.sift2_mapping import _upsample_tracks, map_streamlines_to_fixels


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA absent"))])
def test_no_upsampling_preserves_point_bits_and_track_order(device: str):
    """Ratio one preserves uneven, noncontiguous paths without interpolation."""
    first = torch.tensor([
        [-0., .125, -3.25], [17.5, 21., 22.], [8.25, -4., 6.5],
        [23., 24., 25.], [1.5, 9., -2.],
    ], device=device)[::2]
    second = torch.tensor([[.25, 1.25, -0.], [2.5, 3.5, 4.5]], device=device)
    points, track, starts = _upsample_tracks([first, second], 1)
    assert torch.equal(points.view(torch.int32), torch.cat((first, second)).view(torch.int32))
    assert track.tolist() == [0, 0, 0, 1, 1]
    assert starts.tolist() == [0, 3]
    assert torch.equal(points[starts], torch.stack((first[0], second[0])))


def _one_fixel_lookup(device: str, shape: tuple[int, int, int]):
    nvox = shape[0] * shape[1] * shape[2]
    dirs = torch.zeros((1281, 3), device=device, dtype=torch.float32)
    dirs[0, 0] = 1
    dirs[1:, 1] = 1
    return (
        torch.arange(nvox, device=device),
        torch.arange(1, nvox + 1, device=device),
        torch.ones(nvox, device=device, dtype=torch.uint8),
        torch.zeros((nvox, 1281), device=device, dtype=torch.uint8),
        dirs,
    )


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA absent"))])
def test_straight_line_crosses_three_voxels(device: str):
    """A 2 mm straight path deposits ~0.5, 1, 0.5 mm into voxels 0, 1, 2."""
    shape = (3, 1, 1)
    ids, first, count, lut, dirs = _one_fixel_lookup(device, shape)
    path = torch.tensor([[0., 0., 0.], [2., 0., 0.]], device=device)
    mapping = map_streamlines_to_fixels(
        [path], torch.eye(4, device=device), shape, ids, first, count, lut, dirs,
        step_size_mm=1., n_fixels=4,
    )
    assert mapping.track_index.tolist() == [0, 0, 0]
    assert mapping.fixel_index.tolist() == [1, 2, 3]
    torch.testing.assert_close(mapping.tdi_mm[1:], torch.tensor([.5, 1., .5], device=device), atol=.012, rtol=0)
    assert mapping.tdi_mm[0] == 0


def test_fmls_count_sentinel_discards_unassigned_bin():
    """The FMLS LUT's count sentinel must produce no SIFT2 contribution."""
    shape = (1, 1, 1)
    ids, first, count, lut, dirs = _one_fixel_lookup("cpu", shape)
    lut[:, 0] = 1
    path = torch.tensor([[-.25, 0., 0.], [.25, 0., 0.]])
    mapping = map_streamlines_to_fixels(
        [path], torch.eye(4), shape, ids, first, count, lut, dirs,
        step_size_mm=.1, n_fixels=2,
    )
    assert len(mapping.track_index) == 0
    assert mapping.tdi_mm.sum() == 0


def test_world_translation_does_not_change_fixel_lengths():
    """MRtrix Hermite interpolation must be translation invariant in world mm."""
    shape = (3, 1, 1)
    ids, first, count, lut, dirs = _one_fixel_lookup("cpu", shape)
    base = torch.tensor([[0., 0., 0.], [2., 0., 0.]])
    base_map = map_streamlines_to_fixels(
        [base], torch.eye(4), shape, ids, first, count, lut, dirs,
        step_size_mm=1., n_fixels=4,
    )
    shifted_affine = torch.eye(4)
    shifted_affine[0, 3] = 100
    moved_map = map_streamlines_to_fixels(
        [base + torch.tensor([100., 0., 0.])], shifted_affine,
        shape, ids, first, count, lut, dirs, step_size_mm=1., n_fixels=4,
    )
    torch.testing.assert_close(moved_map.tdi_mm, base_map.tdi_mm, atol=.012, rtol=0)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA absent"))])
def test_mrtrix_first_fit_byte_packing(device: str):
    """Multiple dixels combine until 255 bytes, then open another record."""
    from fnit.connectome.sift2_mapping import _pack_fixel_bytes

    track = torch.zeros(8, device=device, dtype=torch.long)
    fixel = torch.tensor([1, 1, 1, 2, 2, 2, 3, 3], device=device)
    encoded = torch.tensor([100, 100, 20, 170, 170, 170, 250, 10], device=device)
    out_track, out_fixel, out_bytes = _pack_fixel_bytes(track, fixel, encoded, 4)
    assert out_track.tolist() == [0] * 6
    assert out_fixel.tolist() == [1, 2, 2, 2, 3, 3]
    assert out_bytes.tolist() == [220, 170, 170, 170, 250, 10]


def test_single_step_can_cross_an_intermediate_voxel():
    """MRtrix precise mapping revisits one segment until all voxels are exited."""
    shape = (3, 1, 1)
    ids, first, count, lut, dirs = _one_fixel_lookup("cpu", shape)
    path = torch.tensor([[0., 0., 0.], [2., 0., 0.]])
    mapped = map_streamlines_to_fixels(
        [path], torch.eye(4), shape, ids, first, count, lut, dirs,
        step_size_mm=.1, n_fixels=4,
    )
    assert mapped.fixel_index.tolist() == [1, 2, 3]
    torch.testing.assert_close(mapped.tdi_mm[1:], torch.tensor([.5, 1., .5]), atol=.012, rtol=0)
