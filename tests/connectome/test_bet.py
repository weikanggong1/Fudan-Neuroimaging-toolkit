"""BET mean-b0 interface and MRtrix image-header parity checks."""

import pytest
import torch

from fnit.connectome.bet import mean_bzero, mrtrix_roundtrip_voxel_size


def test_mean_bzero_uses_bvalue_selection_and_double_accumulation() -> None:
    dwi = torch.tensor([1e8, 1, -1e8, 1, 1, 1000], dtype=torch.float32).reshape(1, 1, 1, 6)
    bvalues = torch.tensor([0, 49, 0, 49, 0, 50], dtype=torch.float64)
    mean = mean_bzero(dwi=dwi, bvalues=bvalues, bzero_threshold=50.0)
    assert mean.shape == (1, 1, 1)
    assert mean.dtype == torch.float32
    assert mean.device == dwi.device
    assert mean.item() == torch.tensor(3 / 5, dtype=torch.float32).item()


def test_mean_bzero_rejects_mismatched_gradients_and_empty_bzero() -> None:
    dwi = torch.ones((2, 2, 2, 2), dtype=torch.float32)
    with pytest.raises(ValueError, match='one value per DWI volume'):
        mean_bzero(dwi=dwi, bvalues=torch.tensor([0.0]))
    with pytest.raises(ValueError, match='no b=0 volumes'):
        mean_bzero(dwi=dwi, bvalues=torch.tensor([1000.0, 1000.0]))


def test_mrtrix_roundtrip_voxel_size_follows_six_significant_digits() -> None:
    original = (2.019230842590332, 2.019230842590332, 2.0)
    assert mrtrix_roundtrip_voxel_size(original) == (
        torch.tensor(2.01923, dtype=torch.float32).item(),
        torch.tensor(2.01923, dtype=torch.float32).item(),
        2.0,
    )
    with pytest.raises(ValueError, match='three finite positive'):
        mrtrix_roundtrip_voxel_size((2.0, 0.0, 2.0))
