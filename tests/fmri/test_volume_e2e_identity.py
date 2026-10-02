"""Independent pipeline comparisons must be tied to completed real runs."""

import importlib.util
from pathlib import Path
import sys

import pytest
import nibabel as nib
import numpy as np


root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "validation/fmri"))
spec = importlib.util.spec_from_file_location(
    "volume_e2e_comparison", root / "validation/fmri/compare_volume_e2e.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def records(monkeypatch):
    hashes = {name: name + "-hash" for name in ("bold", "t1w", "sbref")}
    monkeypatch.setattr(comparison, "sha256", lambda path: hashes[path])
    candidate = {"input_sha256": hashes, "configuration": {"slice_timing": False},
                 "source_unchanged_during_run": True}
    reference = {"raw_bold_sha256": hashes["bold"], "raw_t1_sha256": hashes["t1w"],
                 "raw_sbref_sha256": hashes["sbref"], "exit_code": 0,
                 "validation_complete": True, "STC": False, "SDC": False}
    return candidate, reference, {name: name for name in hashes}


def test_raw_hashes_match_both_completed_workflows(monkeypatch):
    candidate, reference, raw = records(monkeypatch)
    assert comparison.require_raw_identity(candidate, reference, raw) == candidate["input_sha256"]


@pytest.mark.parametrize("field,value", [
    ("raw_bold_sha256", "old-run"), ("raw_t1_sha256", "different-subject"),
    ("raw_sbref_sha256", None), ("exit_code", 1), ("validation_complete", False),
    ("STC", True), ("SDC", True),
])
def test_reject_different_or_incomplete_reference(monkeypatch, field, value):
    candidate, reference, raw = records(monkeypatch)
    reference[field] = value
    with pytest.raises(ValueError):
        comparison.require_raw_identity(candidate, reference, raw)


def test_reject_unchecked_candidate_configuration(monkeypatch):
    candidate, reference, raw = records(monkeypatch)
    candidate["configuration"] = {}
    with pytest.raises(ValueError, match="STC"):
        comparison.require_raw_identity(candidate, reference, raw)


def test_reject_changed_candidate_source(monkeypatch):
    candidate, reference, raw = records(monkeypatch)
    candidate["source_unchanged_during_run"] = False
    with pytest.raises(ValueError, match="source identity"):
        comparison.require_raw_identity(candidate, reference, raw)


def test_lossless_axis_flip_keeps_world_voxel_correspondence():
    # 两份存盘图具有相反的x索引，但每个世界格点上的四帧值相同。
    values = np.arange(2 * 3 * 5 * 4, dtype=np.float32).reshape(2, 3, 5, 4)
    candidate = nib.Nifti1Image(values, np.diag([-2., 2., 2., 1.]))
    reference_affine = np.diag([2., 2., 2., 1.])
    reference_affine[0, 3] = -2.
    reference = nib.Nifti1Image(values[::-1], reference_affine)
    images, report = comparison.common_mni_views(candidate, reference)
    assert np.array_equal(images[0].dataobj, images[1].dataobj)
    assert np.array_equal(images[0].affine, images[1].affine)
    assert report["reference_reindexed"]
    assert report["interpolation_performed"] is False


def test_lossless_axis_reindex_rejects_displaced_physical_lattice():
    values = np.zeros((2, 3, 5, 4), dtype=np.float32)
    candidate = nib.Nifti1Image(values, np.eye(4))
    shifted = np.eye(4)
    shifted[0, 3] = .25
    with pytest.raises(ValueError, match="grids differ"):
        comparison.common_mni_views(candidate, nib.Nifti1Image(values, shifted))
