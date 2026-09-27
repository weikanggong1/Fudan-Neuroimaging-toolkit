"""Focused checks for MRtrix-style FreeSurfer anatomy operators."""

import torch

from fnit.connectome.anatomy import (
    combine_cortical_subcortical,
    freesurfer_five_tissue,
    gmwmi_from_five_tissue,
    resample_labels_nearest,
)


def test_freesurfer_5tt_label_convention():
    labels = torch.tensor([[[0, 2, 3, 4, 10, 1000]]])
    five = freesurfer_five_tissue(labels)
    assert five.dtype == torch.float32
    assert five.shape == (1, 1, 6, 5)
    assert five.argmax(-1).flatten().tolist() == [0, 2, 0, 3, 1, 0]
    assert five.sum(-1).flatten().tolist() == [0, 1, 1, 1, 1, 1]


def test_gmwmi_boundary_gradient():
    five = torch.zeros((3, 2, 2, 5))
    five[0, ..., 0] = 1
    five[1:, ..., 2] = 1
    seed = gmwmi_from_five_tissue(five)
    assert torch.equal(seed[:, 0, 0], torch.tensor([1.0, 0.5, 0.0]))


def test_atlas_resample_and_cortical_precedence():
    cortical = torch.zeros((3, 2, 2), dtype=torch.int32)
    cortical[1, 0, 0] = 7
    transform = torch.eye(4)
    transform[0, 3] = 1
    resampled = resample_labels_nearest(
        cortical, torch.eye(4), (3, 2, 2), torch.eye(4), transform
    )
    assert resampled[0, 0, 0] == 7
    assert resampled.sum() == 7
    sub = torch.zeros_like(resampled)
    sub[0, 0, 0] = 2
    sub[1, 0, 0] = 3
    combined = combine_cortical_subcortical(resampled, sub, 7)
    assert combined[0, 0, 0] == 7
    assert combined[1, 0, 0] == 10
