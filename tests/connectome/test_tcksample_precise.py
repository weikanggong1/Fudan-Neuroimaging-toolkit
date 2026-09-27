"""Analytic tests for MRtrix precise per-track image sampling."""

import pytest
import torch

from fnit.connectome.tcksample_precise import sample_streamline_mean_precise


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA absent"))])
def test_length_weighted_mean_crosses_intermediate_voxel(device: str):
    """One 2 mm step crosses 0.5/1/0.5 mm in voxels valued 0.2/0.4/0.6."""
    image = torch.tensor([.2, .4, .6], device=device).reshape(3, 1, 1)
    path = torch.tensor([[0., 0., 0.], [2., 0., 0.]], device=device)
    result = sample_streamline_mean_precise([path], image, torch.eye(4, device=device))
    assert result.shape == (1,) and result.dtype == torch.float32
    torch.testing.assert_close(result, torch.tensor([.4], device=device), atol=.003, rtol=0)


def test_world_translation_and_out_of_image_track():
    """Image affine translation preserves the mean; absent voxels give NaN."""
    image = torch.tensor([.2, .4, .6]).reshape(3, 1, 1)
    affine = torch.eye(4)
    affine[0, 3] = 100
    inside = torch.tensor([[100., 0., 0.], [102., 0., 0.]])
    outside = torch.tensor([[200., 0., 0.], [202., 0., 0.]])
    result = sample_streamline_mean_precise([inside, outside], image, affine)
    torch.testing.assert_close(result[:1], torch.tensor([.4]), atol=.003, rtol=0)
    assert torch.isnan(result[1])
