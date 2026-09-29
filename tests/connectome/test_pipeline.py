"""Input geometry and gradient contracts for the official connectome path."""

import nibabel as nib
import numpy as np
import pytest
import torch

from fnit.connectome.pipeline import _scalar_on_grid


def test_fsl_gradient_x_flip_matches_mrconvert(tmp_path):
    from fnit.connectome.pipeline import _gradients

    bval = tmp_path / "grad.bval"
    bvec = tmp_path / "grad.bvec"
    np.savetxt(bval, np.array([[0., 1000.]]))
    np.savetxt(bvec, np.array([[0., .6], [0., .8], [0., 0.]]))
    for affine in (torch.eye(4), torch.diag(torch.tensor([-1., 1., 1., 1.]))):
        _, direction = _gradients(bval, bvec, 2, affine, torch.device("cpu"))
        torch.testing.assert_close(direction[0], torch.zeros(3))
        torch.testing.assert_close(direction[1], torch.tensor([-.6, .8, 0.]))


def test_scalar_mask_accepts_axis_flip_and_rejects_shift(tmp_path):
    reference = nib.Nifti1Image(np.zeros((3, 4, 2), dtype=np.float32), np.eye(4))
    mask = np.zeros((3, 4, 2), dtype=np.uint8)
    mask[0, 2, 1] = 1
    flipped_affine = np.diag([-1., 1., 1., 1.])
    flipped_affine[0, 3] = 2
    path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(mask[::-1], flipped_affine), path)
    loaded = _scalar_on_grid(path, reference, torch.device("cpu"), binary=True)
    assert torch.equal(loaded, torch.as_tensor(mask.astype(bool)))
    rounded = flipped_affine.copy()
    rounded[0, 3] += 8e-5
    nib.save(nib.Nifti1Image(mask[::-1], rounded), path)
    assert torch.equal(_scalar_on_grid(path, reference, torch.device("cpu"), binary=True),
                       torch.as_tensor(mask.astype(bool)))
    shifted = flipped_affine.copy()
    shifted[1, 3] = 1
    nib.save(nib.Nifti1Image(mask[::-1], shifted), path)
    with pytest.raises(ValueError, match="does not share DWI voxel centers"):
        _scalar_on_grid(path, reference, torch.device("cpu"), binary=True)


def test_automatic_bet_maps_ras_bids_mask_back_to_original_grid(monkeypatch):
    import fnit.connectome.pipeline as module

    mean_b0 = torch.zeros((3, 4, 2), dtype=torch.float32)
    mean_b0[0, 2, 1] = 1
    reference = nib.Nifti1Image(np.zeros(mean_b0.shape, np.float32), np.eye(4))
    observed = []

    def fake_bet(*, mean_b0, **kwargs):
        observed.append(torch.nonzero(mean_b0).tolist())
        return mean_b0 > 0

    monkeypatch.setattr(module, "bet_mask", fake_bet)
    mask = module._bet_on_dwi_grid(mean_b0, reference, torch.device("cpu"))
    assert observed == [[[2, 2, 1]]]
    assert torch.equal(mask, mean_b0 > 0)
