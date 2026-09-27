"""Analytic checks for the MRtrix FMLS fixel segmentation port."""

import math

import torch

from fnit.connectome.sift2_fixels import segment_fod_fixels


def test_anisotropic_fod_single_lobe_and_mrtrix_voxel_order() -> None:
    sh = torch.zeros((2, 1, 2, 6), dtype=torch.float32)
    sh[..., 0] = torch.tensor([1.0, 1.2, 1.4, 1.6]).reshape(2, 1, 2)
    sh[..., 3] = 0.5 * sh[..., 0]
    mask = torch.ones(sh.shape[:3])
    result = segment_fod_fixels(sh, mask)

    assert result.voxel_ids.tolist() == [0, 1, 2, 3]
    lobes = int(result.count[0])
    assert lobes == 1
    assert result.count.tolist() == [lobes] * 4
    assert result.first_fixel_index.tolist() == [1, 1 + 2 * lobes, 1 + lobes, 1 + 3 * lobes]
    assert torch.all(result.lookup_table < lobes)
    assert torch.all(result.count_image == lobes)
    expected = sh[..., 0].to(torch.float64) * math.sqrt(4 * math.pi)
    torch.testing.assert_close(result.target_image, expected, atol=1e-7, rtol=1e-7)
    observed = result.fixel_integrals[1:].reshape(4, lobes).sum(dim=1)
    torch.testing.assert_close(observed, expected.permute(2, 1, 0).reshape(-1), atol=1e-7, rtol=1e-7)


def test_empty_mask_has_no_fixels() -> None:
    result = segment_fod_fixels(torch.ones((1, 1, 1, 1), dtype=torch.float32), torch.zeros((1, 1, 1)))
    assert result.voxel_ids.numel() == 0
    assert result.fixel_integrals.tolist() == [0.0]
    assert torch.isnan(result.target_image).all()


def test_many_equal_height_lobes_do_not_overflow_lookup() -> None:
    # A flat FOD seeds hundreds of lobes before FMLS can join them; no uint8 wrap.
    sh = torch.ones((1, 1, 1, 1), dtype=torch.float32)
    result = segment_fod_fixels(sh, torch.ones(sh.shape[:3]))
    assert int(result.count[0]) > 255
    assert result.count.dtype == torch.int16
    assert int(result.count_image[0, 0, 0]) == int(result.count[0])
    assert torch.all(result.lookup_table < result.count[0])
