"""Focused tests of the original UKB surface-to-ribbon assignment rule."""

import pytest
import torch

from fnit.connectome.atlas_surface import surface_annotation_to_volume


def test_surface_annotation_pial_white_and_exhaustive_fallback():
    ribbon = torch.zeros((3, 1, 1), dtype=torch.int32)
    ribbon[0, 0, 0] = 3
    ribbon[2, 0, 0] = 42
    result = surface_annotation_to_volume(
        ribbon, torch.eye(4),
        torch.tensor([[10.0, 0, 0], [20.0, 0, 0]]),
        torch.tensor([[11.0, 0, 0], [21.0, 0, 0]]),
        torch.tensor([7, 8]),
        torch.tensor([[8.0, 0, 0], [20.0, 0, 0]]),
        torch.tensor([[9.0, 0, 0], [21.0, 0, 0]]),
        torch.tensor([9, 10]),
        bin_width_mm=2.0, query_batch=1,
    )
    assert result.tolist() == [[[7]], [[0]], [[9]]]
    assert result.dtype == torch.int32


def test_surface_annotation_requires_matched_vertices():
    with pytest.raises(ValueError, match="matching"):
        surface_annotation_to_volume(
            torch.zeros((1, 1, 1)), torch.eye(4),
            torch.zeros((2, 3)), torch.zeros((1, 3)), torch.zeros(2),
            torch.zeros((1, 3)), torch.zeros((1, 3)), torch.zeros(1),
        )
