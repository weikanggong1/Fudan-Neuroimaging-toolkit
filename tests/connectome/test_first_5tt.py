"""Focused tests for the two MRtrix eeab681 FIRST tissue-blending rules."""

import pytest
import torch

from fnit.connectome.first_5tt import FIRST_LABELS, freesurfer_first_five_tissue


def test_directory_first_pve_blends_old_sgm_into_white_matter():
    base = torch.zeros((4, 1, 1, 5), dtype=torch.float32)
    base[0, 0, 0, 0] = 1  # cortical GM
    base[1, 0, 0, 1] = 1  # old FreeSurfer subcortical GM
    base[2, 0, 0, 2] = 1  # white matter
    pve = torch.zeros((4, 1, 1, len(FIRST_LABELS)))
    pve[0, 0, 0, 0] = 0.25
    pve[1, 0, 0, :2] = torch.tensor([0.6, 0.5])
    pve[2, 0, 0, 0] = 0.4
    pve[3, 0, 0, 0] = 0.7  # clipped by FreeSurfer brain support
    result = freesurfer_first_five_tissue(base, first_pve=pve)
    expected = torch.tensor([[0.75, 0.25, 0, 0, 0],
                             [0, 1, 0, 0, 0],
                             [0, 0.4, 0.6, 0, 0],
                             [0, 0, 0, 0, 0]])
    torch.testing.assert_close(result[:, 0, 0], expected)


def test_image_firstseg_replaces_and_reclassifies_old_sgm():
    base = torch.zeros((3, 1, 1, 5), dtype=torch.float32)
    base[0, 0, 0, 1] = 1  # old SGM absent in FIRST -> WM
    base[1, 0, 0, 0] = 1  # new FIRST SGM -> SGM
    first = torch.tensor([[[0]], [[FIRST_LABELS[0]]], [[FIRST_LABELS[-1]]]])
    result = freesurfer_first_five_tissue(base, first_labels=first)
    expected = torch.tensor([[0, 0, 1, 0, 0],
                             [0, 1, 0, 0, 0],
                             [0, 1, 0, 0, 0]], dtype=torch.float32)
    torch.testing.assert_close(result[:, 0, 0], expected)


def test_requires_one_matching_first_input():
    base = torch.zeros((2, 2, 2, 5))
    with pytest.raises(ValueError, match="exactly one"):
        freesurfer_first_five_tissue(base)
    with pytest.raises(ValueError, match="14 structure"):
        freesurfer_first_five_tissue(base, first_pve=torch.zeros((2, 2, 2, 13)))
