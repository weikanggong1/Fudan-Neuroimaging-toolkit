"""FreeSurfer atlas node order and noncontiguous label IDs."""

import nibabel as nib
import numpy as np
import pytest
import torch

from fnit.connectome.freesurfer_subject import (
    fs_aparc_a2009s_atlas, fs_aparc_atlas, fs_aparc_nodes,
)


def test_fs_aparc_uses_fixed_node_order_and_background():
    nodes = fs_aparc_nodes()
    assert len(nodes) == 84
    segmentation = torch.tensor([[[0, 1001, 8, 49, nodes[-1].original_label, 9999]]], dtype=torch.int32)
    atlas, returned_nodes = fs_aparc_atlas(segmentation=segmentation)
    assert returned_nodes is nodes
    assert atlas.dtype == torch.int32
    assert atlas.flatten().tolist() == [0, 1, 35, 43, 84, 0]


def test_fs_aparc_rejects_fractional_segmentation():
    with pytest.raises(ValueError, match="integer"):
        fs_aparc_atlas(segmentation=torch.tensor([[[1001.5]]]))


def test_fs_a2009s_uses_subject_annotations_and_contiguous_nodes(monkeypatch, tmp_path):
    def read_annot(path):
        return np.array([1, 3, -1]), None, [b"Unknown", b"Region-1", b"Region-2", b"Region-3"]

    monkeypatch.setattr(nib.freesurfer, "read_annot", read_annot)
    segmentation = torch.tensor([[[11101, 11103, 12101, 12103, 10, 1000, 0]]])
    atlas, nodes = fs_aparc_a2009s_atlas(segmentation, tmp_path)
    assert atlas.flatten().tolist() == [1, 2, 3, 4, 6, 0, 0]
    assert len(nodes) == 20
    assert [node.index for node in nodes] == list(range(1, 21))
    assert nodes[0].name == "ctx-lh-Region-1"
    assert nodes[2].name == "ctx-rh-Region-1"
    with pytest.raises(ValueError, match="absent"):
        fs_aparc_a2009s_atlas(torch.tensor([[[11102]]]), tmp_path)
