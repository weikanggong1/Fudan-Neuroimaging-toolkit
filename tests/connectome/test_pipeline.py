"""A small paired T1/DWI fixture exercises all GPU compute stages."""

import math

import nibabel as nib
import numpy as np
import pytest
import torch

from fnit.connectome.pipeline import UKBConnectome


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_corrected_dwi_to_region_matrices(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    shape = (22, 11, 11)
    segmentation = np.zeros(shape, dtype=np.int16)
    segmentation[1, 1:10, 1:10] = 3
    segmentation[2:20, 1:10, 1:10] = 2
    segmentation[20, 1:10, 1:10] = 42
    segmentation[0, 1:10, 1:10] = 4
    segmentation[21, 1:10, 1:10] = 4

    index = np.arange(32)
    z = 1 - 2 * (index + 0.5) / len(index)
    angle = index * math.pi * (3 - math.sqrt(5))
    directions = np.stack((
        np.sqrt(1 - z * z) * np.cos(angle),
        np.sqrt(1 - z * z) * np.sin(angle),
        z,
    ), axis=-1)
    bvecs = np.vstack((np.zeros((1, 3)), directions, directions))
    bvals = np.r_[0., np.full(32, 1000.), np.full(32, 2000.)]
    wm = np.exp(-bvals * (0.0003 + 0.0012 * bvecs[:, 0] ** 2))
    gm = np.exp(-bvals * 0.0008)
    csf = np.exp(-bvals * 0.0025)
    dwi = np.zeros((*shape, len(bvals)), dtype=np.float32)
    dwi[segmentation == 2] = 1000 * wm
    dwi[(segmentation == 3) | (segmentation == 42)] = 1000 * gm
    dwi[segmentation == 4] = 1000 * csf
    t1 = np.where(segmentation == 2, 100., np.where(
        (segmentation == 3) | (segmentation == 42), 70., 20.
    )).astype(np.float32)
    files = {
        "dwi": tmp_path / "sub-01_dwi.nii.gz",
        "t1": tmp_path / "sub-01_T1w.nii.gz",
        "seg": tmp_path / "sub-01_seg.nii.gz",
        "bval": tmp_path / "sub-01_dwi.bval",
        "bvec": tmp_path / "sub-01_dwi.bvec",
    }
    for key, data in (("dwi", dwi), ("t1", t1), ("seg", segmentation)):
        nib.save(nib.Nifti1Image(data, np.eye(4)), files[key])
    np.savetxt(files["bval"], bvals[None], fmt="%.1f")
    np.savetxt(files["bvec"], bvecs.T, fmt="%.8f")

    result = UKBConnectome(device=device)(
        files["dwi"], files["bval"], files["bvec"], files["t1"],
        t1_segmentation=files["seg"], dwi_to_t1_world=np.eye(4),
        n_seeds=200, seed=11,
    )
    assert result.region_labels == (3, 42)
    assert result.atlas.shape == result.tissues.shape == shape
    assert result.wm_sh.shape == (*shape, 15)
    assert result.fa.shape == shape
    assert len(result.tractogram.paths) > 20
    assert set(result.matrices) == {"count", "sift2_fbc", "mean_length", "mean_fa"}
    for matrix in result.matrices.values():
        assert matrix.shape == (2, 2)
        torch.testing.assert_close(matrix, matrix.T)
        assert torch.isfinite(matrix).all()
    assert result.matrices["count"][0, 1] > 20
    assert (result.sift2_weights >= 0).all()
    if device == "cpu":
        # MRtrix can preserve an atlas grid while updating its DWI-world affine.
        padded = np.pad(result.atlas.cpu().numpy(), ((1, 0), (0, 0), (0, 0)))
        atlas_affine = np.eye(4)
        atlas_affine[0, 3] = -1
        atlas_path = tmp_path / "sub-01_space-dwi_atlas.nii.gz"
        nib.save(nib.Nifti1Image(padded, atlas_affine), atlas_path)
        atlas_result = UKBConnectome(device="cpu")(
            files["dwi"], files["bval"], files["bvec"], files["t1"],
            atlas_dwi=atlas_path, t1_segmentation=files["seg"],
            dwi_to_t1_world=np.eye(4), n_seeds=200, seed=11,
        )
        assert atlas_result.atlas.shape == padded.shape
        torch.testing.assert_close(atlas_result.atlas_affine, torch.as_tensor(atlas_affine, dtype=torch.float32))
        torch.testing.assert_close(atlas_result.matrices["count"], result.matrices["count"])


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
