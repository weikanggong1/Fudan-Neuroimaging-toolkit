"""HIP component selection preserves label identity across a one-voxel gap."""
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
from scipy import ndimage
import torch

from fnit.gems.context import SubregionContext
from fnit.gems.recipes import hippo_amygdala as production


candidate = production


def outcome(raw, side="left"):
    affine = np.diag([.33333, .33333, .33333, 1.])
    image = nib.Nifti1Image(np.ones(raw.shape, np.float32), affine)
    coarse = np.full(raw.shape, 17 if side == "left" else 53, np.int32)
    context = SubregionContext(image, np.asarray(image.dataobj), coarse, None, None)
    ids = np.asarray([0, 201, 203, 7006, 7010], np.int32)
    posterior = torch.full((len(ids), *raw.shape), .2)
    fit = SimpleNamespace(labels=torch.as_tensor(raw.copy()), posterior=posterior, affine=affine)
    atlas = SimpleNamespace(label_ids=ids)
    return candidate.HippoAmygdalaRecipe(side, Path("unused")).postprocess(fit, context, atlas)


def one_gap_labels():
    raw = np.zeros((18, 18, 18), np.int32)
    raw[2:7, 2:7, 2:7] = 203
    raw[8:11, 3:6, 3:6] = 7010  # One empty working voxel between thick structures.
    raw[3:5, 3:5, 3:5] = 7006
    raw[14:16, 14:16, 14:16] = 7010  # Remote fragment must remain excluded.
    raw[7, 4, 4] = 201  # Closing's bridge cannot become a foreground label.
    return raw


@pytest.mark.parametrize("side,offset", [("left", 0), ("right", 10000)])
def test_one_gap_is_selected_without_adding_bridges_or_changing_label_ids(side, offset):
    raw = one_gap_labels()
    before = raw.copy()
    old_components, _ = ndimage.label(np.isin(raw, [203, 7006, 7010]))
    assert old_components[4, 4, 4] != old_components[9, 4, 4]
    result = outcome(raw, side)
    highres = np.asarray(result.highres_labels.dataobj)
    assert highres[9, 4, 4] == 7010 + offset
    assert highres[3, 3, 3] == 7006 + offset
    assert highres[14, 14, 14] == highres[7, 4, 4] == 0
    nonzero = highres != 0
    np.testing.assert_array_equal(highres[nonzero], raw[nonzero] + offset)
    assert not np.any(nonzero & ~np.isin(raw, [203, 7006, 7010]))
    np.testing.assert_array_equal(raw, before)
    np.testing.assert_array_equal(result.native_labels, highres)
    assert result.report["component_selection"]["new_labeled_voxels"] == 0
    assert result.report["component_selection"]["changed_label_ids"] == 0
    assert result.soft_volumes_mm3[7010 + offset] == pytest.approx(
        .2 * raw.size * abs(np.linalg.det(result.fit.affine[:3, :3])), rel=1e-6)


def test_diagonal_one_gap_between_thick_regions_retains_original_aaa():
    raw = np.zeros((16, 16, 16), np.int32)
    raw[2:7, 2:7, 2:7] = 203
    raw[7:10, 7:10, 3:6] = 7010
    components, _ = ndimage.label(raw != 0)
    assert components[6, 6, 4] != components[7, 7, 4]
    output = np.asarray(outcome(raw).highres_labels.dataobj)
    np.testing.assert_array_equal(output, raw)


def test_boundary_foreground_survives_closing_border_erosion():
    raw = np.zeros((12, 12, 12), np.int32)
    raw[:5, :5, :5] = 203
    raw[9:11, 9:11, 9:11] = 7010
    output = np.asarray(outcome(raw).highres_labels.dataobj)
    np.testing.assert_array_equal(output[:5, :5, :5], raw[:5, :5, :5])
    assert not output[9:11, 9:11, 9:11].any()


def test_empty_foreground_stays_empty_and_reports_zero_counts():
    raw = np.full((8, 8, 8), 201, np.int32)
    result = outcome(raw)
    assert not np.asarray(result.highres_labels.dataobj).any()
    assert result.report["component_selection"]["input_foreground_voxels"] == 0
    assert result.report["component_selection"]["retained_original_foreground_voxels"] == 0


def test_component_selection_and_counts_are_cpu_deterministic():
    raw = one_gap_labels()
    first, second = outcome(raw), outcome(raw)
    np.testing.assert_array_equal(first.highres_labels.dataobj, second.highres_labels.dataobj)
    assert first.report == second.report
    report = first.report["component_selection"]
    assert report["radius_working_voxels"] == 1
    assert report["working_voxel_sizes_mm"] == pytest.approx([.33333] * 3)
    assert report["input_label_counts"]["7010"] == 35
    assert report["retained_label_counts"]["7010"] == 27
