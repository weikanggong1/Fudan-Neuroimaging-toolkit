"""MNI reference content/geometry participate in paired template cache identity."""
from dataclasses import replace

import nibabel as nib
import numpy as np
import pytest
import torch

from fnit._transforms import DenseWarp
import fnit.connectome.paired_pipeline as paired
from fnit.connectome.template_inputs import TemplatePair
from test_paired_e2e import setup, run


def context(setup, tmp_path):
    result = run(setup, [TemplatePair("base", setup["volume_a"], setup["volume_b"])])
    t1 = tmp_path / "reference.nii.gz"
    reference = nib.Nifti1Image(np.ones(setup["shape"], np.float32), np.eye(4))
    nib.save(reference, t1)
    warp = DenseWarp(np.zeros((*setup["shape"], 3), np.float32), source=reference, target=reference)
    spec = replace(setup["volume_a"], space="mni")
    options = dict(subject_dir=setup["subject"], t1_reference_path=t1,
                   mni_to_t1_transform=warp, checkpoint_dir=tmp_path / "mni_cache",
                   assignment_radius=.1, device="cpu")
    return result, [TemplatePair("MNI", spec, spec)], options


def test_reference_content_change_invalidates_mapping_and_A_B_A_restores(setup, tmp_path, monkeypatch):
    result, pairs, options = context(setup, tmp_path)
    original = paired.prepare_template
    calls = []
    def prepare(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(paired, "prepare_template", prepare)
    path = options["t1_reference_path"]
    before = path.read_bytes()
    first = paired.build_template_pairs(result, pairs, **options)["MNI"]
    nib.save(nib.Nifti1Image(np.full(setup["shape"], 2, np.float32), np.eye(4)), path)
    changed = paired.build_template_pairs(result, pairs, **options)["MNI"]
    assert len(calls) == 2 and first.cache_status == changed.cache_status == "completed"
    path.write_bytes(before)
    restored = paired.build_template_pairs(result, pairs, **options)["MNI"]
    assert len(calls) == 2 and restored.cache_status == "skipped"
    assert all(torch.equal(first.matrices[key], restored.matrices[key]) for key in first.matrices)


@pytest.mark.parametrize("saved_warp", [False, True])
def test_changed_reference_grid_rejected_before_cached_template_return(setup, tmp_path, saved_warp):
    result, pairs, options = context(setup, tmp_path)
    if saved_warp:
        path = tmp_path / "warp.nii.gz"
        options["mni_to_t1_transform"].save(path)
        options["mni_to_t1_transform"] = path
    paired.build_template_pairs(result, pairs, **options)
    affine = np.eye(4); affine[0, 3] = 1
    nib.save(nib.Nifti1Image(np.ones(setup["shape"], np.float32), affine), options["t1_reference_path"])
    with pytest.raises(ValueError, match="transform target must match"):
        paired.build_template_pairs(result, pairs, **options)


def test_reference_changed_during_mapping_not_published(setup, tmp_path, monkeypatch):
    result, pairs, options = context(setup, tmp_path)
    original = paired.prepare_template
    def change(*args, **kwargs):
        template = original(*args, **kwargs)
        nib.save(nib.Nifti1Image(np.full(setup["shape"], 3, np.float32), np.eye(4)), options["t1_reference_path"])
        return template
    monkeypatch.setattr(paired, "prepare_template", change)
    with pytest.raises(RuntimeError, match="template inputs changed"):
        paired.build_template_pairs(result, pairs, **options)
    assert not list(options["checkpoint_dir"].rglob("complete.json"))


def test_reference_changed_during_cache_hit_not_returned(setup, tmp_path, monkeypatch):
    result, pairs, options = context(setup, tmp_path)
    paired.build_template_pairs(result, pairs, **options)
    original = paired.CheckpointStore.load
    def change(self, stage, key, **kwargs):
        loaded = original(self, stage, key, **kwargs)
        if stage == "template" and loaded is not None:
            nib.save(nib.Nifti1Image(np.full(setup["shape"], 3, np.float32), np.eye(4)), options["t1_reference_path"])
        return loaded
    monkeypatch.setattr(paired.CheckpointStore, "load", change)
    with pytest.raises(RuntimeError, match="template inputs changed"):
        paired.build_template_pairs(result, pairs, **options)


def test_reference_changed_during_matrix_not_published(setup, tmp_path, monkeypatch):
    result, pairs, options = context(setup, tmp_path)
    original = paired.build_pair_connectomes
    def change(*args, **kwargs):
        matrices = original(*args, **kwargs)
        nib.save(nib.Nifti1Image(np.full(setup["shape"], 3, np.float32), np.eye(4)), options["t1_reference_path"])
        return matrices
    monkeypatch.setattr(paired, "build_pair_connectomes", change)
    with pytest.raises(RuntimeError, match="template inputs changed"):
        paired.build_template_pairs(result, pairs, **options)
    assert not list((options["checkpoint_dir"] / "pairs/matrix").rglob("complete.json"))


def test_reference_changed_during_matrix_hit_not_returned(setup, tmp_path, monkeypatch):
    result, pairs, options = context(setup, tmp_path)
    paired.build_template_pairs(result, pairs, **options)
    original = paired.CheckpointStore.load
    def change(self, stage, key, **kwargs):
        loaded = original(self, stage, key, **kwargs)
        if stage == "matrix" and loaded is not None:
            nib.save(nib.Nifti1Image(np.full(setup["shape"], 3, np.float32), np.eye(4)), options["t1_reference_path"])
        return loaded
    monkeypatch.setattr(paired.CheckpointStore, "load", change)
    with pytest.raises(RuntimeError, match="template inputs changed"):
        paired.build_template_pairs(result, pairs, **options)
