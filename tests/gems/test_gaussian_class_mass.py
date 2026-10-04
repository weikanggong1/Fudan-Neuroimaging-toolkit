"""Each Gaussian class divides its weighted sum by its own mass."""

import pytest
import torch

from fnit.gems.gaussian import update_gaussians


@pytest.mark.parametrize('modalities', [1, 2])
def test_no_hyperprior_mean_preserves_class_and_modality_axes(modalities):
    image = torch.tensor([[[[2., 4., 8.]]], [[[20., 40., 80.]]]])[:modalities]
    responsibilities = torch.tensor([[[[3., 1., 0.]]], [[[0., 0., 2.]]], [[[1., 1., 1.]]]])
    result = update_gaussians(image[0] if modalities == 1 else image, responsibilities)
    expected = torch.tensor([[2.5, 25.], [8., 80.], [14. / 3., 140. / 3.]])[:, :modalities]
    assert result.means.shape == (3, modalities)
    assert torch.equal(result.means, expected)
    assert result.covariances.shape == (3, modalities, modalities)
    assert torch.isfinite(result.covariances).all()
