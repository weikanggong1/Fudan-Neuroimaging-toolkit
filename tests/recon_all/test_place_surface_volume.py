"""固定源码整数邻居边界与255恢复规则；真实体积回归另外记录。"""
import numpy as np
import pytest

from fnit.recon_all.place_surface_volume import prepare_placement_volume


@pytest.mark.parametrize("surface", ["white", "pial"])
@pytest.mark.parametrize("white_neighbors", [12, 13])
def test_bright_seed_uses_integer_source_threshold(surface, white_neighbors):
    brain = np.zeros((7, 7, 7), np.uint8)
    wm = np.zeros_like(brain)
    center = (3, 3, 3)
    neighbors = [(x, y, z) for x in (2, 3, 4) for y in (2, 3, 4) for z in (2, 3, 4)
                 if (x, y, z) != center]
    for position in neighbors[:white_neighbors]:
        wm[position] = 110
    brain[center] = 140
    volume, labels = prepare_placement_volume(brain=brain, wm=wm, surface=surface, mid_gray=50.)
    assert volume.dtype == labels.dtype == np.uint8
    assert labels[center] == (130 if white_neighbors == 12 else 0)
    assert volume[center] == ((0 if surface == "white" else 255) if white_neighbors == 12 else 140)


@pytest.mark.parametrize("restore", [False, True])
def test_white_original_255_restore_is_independent_of_bright_mask(restore):
    brain = np.zeros((7, 7, 7), np.uint8)
    brain[3, 3, 3] = 255
    result, labels = prepare_placement_volume(brain=brain, wm=np.zeros_like(brain),
        surface="white", mid_gray=50., restore_255=restore)
    assert labels[3, 3, 3] == 130
    assert result[3, 3, 3] == (110 if restore else 0)
