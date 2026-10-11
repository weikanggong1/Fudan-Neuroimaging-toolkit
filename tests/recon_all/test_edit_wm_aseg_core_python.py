import numpy as np
import pytest

from fnit.recon_all.edit_wm_aseg_core_python import (
    edit_segmentation_no_fill,
    edit_until_propagation,
    remove_paths_to_cortex,
    spackle_wm_superior_to_mtl,
)


def test_erase_cerebellum_fill_subcortex_and_propagate_to_aseg_wm():
    wm = np.zeros((16, 16, 16), dtype=np.uint8)
    brain = np.zeros_like(wm)
    aseg = np.zeros(wm.shape, dtype=np.int32)
    wm[5, 5, 5] = 110
    aseg[5, 5, 5] = 7
    aseg[10, 10, 10] = 11
    aseg[11, 10, 10] = 2
    result = edit_until_propagation(wm, aseg)
    assert result[5, 5, 5] == 0
    assert result[10, 10, 10] == result[11, 10, 10] == 250
    assert wm[5, 5, 5] == 110
    assert np.array_equal(edit_segmentation_no_fill(wm, brain, aseg), result)


def test_spackle_superior_cortex_at_hippocampal_edge():
    wm = np.zeros((24, 24, 24), dtype=np.uint8)
    aseg = np.zeros(wm.shape, dtype=np.int32)
    aseg[12, 10, 10:14] = 17
    aseg[12, 9, 10] = 3
    result = spackle_wm_superior_to_mtl(wm, aseg)
    assert result[12, 9, 10] == 250
    assert wm[12, 9, 10] == 0


def test_core_rejects_mismatched_shapes():
    wm = np.zeros((8, 8, 8), dtype=np.uint8)
    with pytest.raises(ValueError):
        edit_segmentation_no_fill(wm, wm, np.zeros((7, 8, 8), dtype=np.int32))


def test_int32_path_proof_and_complete_active_roi_without_seeds():
    wm = np.zeros((24, 24, 24), dtype=np.uint8)
    aseg = np.zeros(wm.shape, dtype=np.int32)
    aseg[15, 12, 12] = 17
    result, changed = remove_paths_to_cortex(wm, wm, aseg)
    assert changed == 0 and np.array_equal(result, wm)
    aseg[15, 12, 12] = 0
    aseg[7, 12, 12] = 17
    result, changed = remove_paths_to_cortex(wm, wm, aseg)
    assert changed == 0 and np.array_equal(result, wm)
