"""Small analytic checks for ACT-derived SIFT2 processing-mask geometry."""

import torch

from fnit.connectome.sift2_proc_mask import processing_mask_from_5tt


def test_matching_grid_squares_wm_fraction() -> None:
    sh = torch.ones((2, 1, 1, 1), dtype=torch.float32)
    five_tt = torch.zeros((2, 1, 1, 5), dtype=torch.float32)
    five_tt[0, 0, 0, 2] = 0.25
    five_tt[1, 0, 0, 2] = float("nan")
    result = processing_mask_from_5tt(sh, torch.eye(4), five_tt, torch.eye(4))
    torch.testing.assert_close(result.flatten(), torch.tensor([0.0625, 0.0]))


def test_different_grid_classifies_1000_subvoxels() -> None:
    sh = torch.ones((1, 1, 1, 1), dtype=torch.float32)
    five_tt = torch.zeros((5, 5, 5, 5), dtype=torch.float32)
    five_tt[..., 2] = 1.0
    fod_affine = torch.eye(4)
    fod_affine[:3, 3] = 2.0
    result = processing_mask_from_5tt(sh, fod_affine, five_tt, torch.eye(4))
    torch.testing.assert_close(result, torch.ones((1, 1, 1)))
    sh.zero_()
    assert processing_mask_from_5tt(sh, fod_affine, five_tt, torch.eye(4)).item() == 0


def test_equal_shape_different_affine_still_resamples() -> None:
    sh = torch.zeros((5, 5, 5, 1), dtype=torch.float32)
    sh[2, 2, 2, 0] = 1.0
    five_tt = torch.zeros((5, 5, 5, 5), dtype=torch.float32)
    five_tt[..., 0] = 1.0
    five_tt[:2, ..., 0] = 0.0
    five_tt[:2, ..., 2] = 1.0
    shifted_affine = torch.eye(4)
    shifted_affine[0, 3] = 1.0
    result = processing_mask_from_5tt(sh, torch.eye(4), five_tt, shifted_affine)
    assert result[2, 2, 2].item() == 1.0
    assert torch.count_nonzero(result).item() == 1
