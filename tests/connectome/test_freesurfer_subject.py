"""FreeSurfer atlas node order and noncontiguous label IDs."""

import pytest
import torch

from fnit.connectome.freesurfer_subject import fs_aparc_atlas, fs_aparc_nodes


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
