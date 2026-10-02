"""Input geometry and gradient contracts for the official connectome path."""

import nibabel as nib
import numpy as np
import pytest
import torch
from collections import Counter
from types import SimpleNamespace

from fnit.connectome.pipeline import _scalar_on_grid


def test_fsl_gradient_x_flip_matches_mrconvert(tmp_path):
    from fnit.connectome.pipeline import _gradients

    bval = tmp_path / "grad.bval"
    bvec = tmp_path / "grad.bvec"
    np.savetxt(bval, np.array([[0., 1000.]]))
    np.savetxt(bvec, np.array([[0., .6], [0., .8], [0., 0.]]))
    for affine in (torch.eye(4), torch.diag(torch.tensor([-1., 1., 1., 1.]))):
        _, direction = _gradients(bval, bvec, 2, affine, torch.device("cpu"))
        torch.testing.assert_close(direction[0], torch.zeros(3))
        torch.testing.assert_close(direction[1], torch.tensor([-.6, .8, 0.]))


def test_scalar_mask_accepts_axis_flip_and_rejects_shift(tmp_path):
    reference = nib.Nifti1Image(np.zeros((3, 4, 2), dtype=np.float32), np.eye(4))
    mask = np.zeros((3, 4, 2), dtype=np.uint8)
    mask[0, 2, 1] = 1
    flipped_affine = np.diag([-1., 1., 1., 1.])
    flipped_affine[0, 3] = 2
    path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(mask[::-1], flipped_affine), path)
    loaded = _scalar_on_grid(path, reference, torch.device("cpu"), binary=True)
    assert torch.equal(loaded, torch.as_tensor(mask.astype(bool)))
    rounded = flipped_affine.copy()
    rounded[0, 3] += 8e-5
    nib.save(nib.Nifti1Image(mask[::-1], rounded), path)
    assert torch.equal(_scalar_on_grid(path, reference, torch.device("cpu"), binary=True),
                       torch.as_tensor(mask.astype(bool)))
    shifted = flipped_affine.copy()
    shifted[1, 3] = 1
    nib.save(nib.Nifti1Image(mask[::-1], shifted), path)
    with pytest.raises(ValueError, match="does not share DWI voxel centers"):
        _scalar_on_grid(path, reference, torch.device("cpu"), binary=True)


def test_automatic_bet_maps_ras_bids_mask_back_to_original_grid(monkeypatch):
    import fnit.connectome.pipeline as module

    mean_b0 = torch.zeros((3, 4, 2), dtype=torch.float32)
    mean_b0[0, 2, 1] = 1
    reference = nib.Nifti1Image(np.zeros(mean_b0.shape, np.float32), np.eye(4))
    observed = []

    def fake_bet(*, mean_b0, **kwargs):
        observed.append(torch.nonzero(mean_b0).tolist())
        return mean_b0 > 0

    monkeypatch.setattr(module, "bet_mask", fake_bet)
    mask = module._bet_on_dwi_grid(mean_b0, reference, torch.device("cpu"))
    assert observed == [[[2, 2, 1]]]
    assert torch.equal(mask, mean_b0 > 0)


@pytest.mark.parametrize("registration", ["synthmorph", "fnirt"])
def test_multiple_atlases_reuse_anatomy_with_identical_outputs(tmp_path, monkeypatch, registration):
    """Shared atlas construction must match independent calls, including nodes.

    Expensive reconstruction is stubbed here to isolate orchestration. Real
    labels, resampling, radial endpoint assignment and all four matrices run.
    """
    import fnit.connectome.pipeline as module

    shape = (5, 4, 3)
    dwi_path = tmp_path / "dwi.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((*shape, 2), np.float32), np.eye(4)), dwi_path)
    subject_dir = tmp_path / "subject"
    (subject_dir / "mri").mkdir(parents=True)
    for name in ("brain.mgz", "aparc+aseg.mgz"):
        nib.save(nib.MGHImage(np.full(shape, 2, np.float32), np.eye(4)), subject_dir / "mri" / name)
    templates = tmp_path / "templates"
    templates.mkdir()
    for scale, names in ((1, ("one-lh", "two-rh")),
                         (4, ("one-lh", "two-rh", "three-lh"))):
        (templates / f"Tian_Subcortex_S{scale}_3T_label.txt").write_text("\n".join(names))

    monkeypatch.setattr(module, "_gradients", lambda *args: (
        torch.tensor([0., 1000.]), torch.tensor([[0., 0., 0.], [1., 0., 0.]])))
    monkeypatch.setattr(module, "_scalar_on_grid", lambda *args, binary=False: (
        torch.ones(shape, dtype=torch.bool) if binary else torch.full(shape, .5)))
    monkeypatch.setattr(module, "estimate_mrtrix_dhollander", lambda *args: (
        torch.tensor([0., 1000.]), None, None, None, None))
    fod = torch.ones((*shape, 45), dtype=torch.float32)
    scalar = torch.ones(shape)
    monkeypatch.setattr(module, "fit_mrtrix_msmt_csd", lambda *args: (fod, scalar, scalar))
    monkeypatch.setattr(module, "normalise_mrtrix_three_tissue", lambda *args: SimpleNamespace(wm=fod))
    paths = (torch.tensor([[0., 1., 1.], [4., 1., 1.]]),
             torch.tensor([[1., 2., 1.], [3., 2., 1.]]))
    monkeypatch.setattr(module, "probabilistic_tractography", lambda *args, **kwargs: module.Tractogram(
        paths=paths, endpoints=torch.stack([path[[0, -1]] for path in paths]),
        lengths_mm=torch.tensor([4., 2.]), mean_fa=None, seeds_attempted=2,
        accepted_seeds=torch.stack([path[0] for path in paths]),
    ))
    monkeypatch.setattr(module, "estimate_sift2_weights", lambda *args, **kwargs: torch.tensor([.75, 1.25], dtype=torch.float64))
    monkeypatch.setattr(module, "sample_streamline_mean_precise", lambda *args: torch.tensor([.45, .55]))

    calls = Counter()
    seen_transforms = []
    shared_transform = object()

    def cortex(name):
        calls[("cortex", name)] += 1
        labels = np.zeros(shape, np.int32)
        labels[0:2] = 1
        labels[2] = 2
        nodes = tuple(module.ConnectomeNode(i, 100 + i, "L" if i == 1 else "R", f"{name}-{i}")
                      for i in (1, 2))
        return nib.Nifti1Image(labels, np.eye(4)), nodes

    monkeypatch.setattr(module, "native_annotation_to_t1", lambda **kwargs: cortex(kwargs["annotation"]))
    monkeypatch.setattr(module, "schaefer_to_t1", lambda **kwargs: cortex(kwargs["left_annot"].name))
    monkeypatch.setattr(module, "glasser_to_t1", lambda **kwargs: cortex("glasser"))

    def tian(**kwargs):
        scale = 1 if "_S1_" in str(kwargs["tian_mni"]) else 4
        calls[("tian", scale)] += 1
        labels = np.zeros(shape, np.int32)
        labels[3] = 1
        labels[4] = 2
        if scale == 4:
            labels[4, 2:] = 3
        image = nib.Nifti1Image(labels, np.eye(4))
        if registration == "synthmorph":
            seen_transforms.append(kwargs["transform"])
            return image, shared_transform
        return image

    monkeypatch.setattr(module, f"{registration}_tian_to_t1", tian)
    atlas_names = (
        "aparc+tian-s1", "aparc.a2009s+tian-s1", "glasser+tian-s1",
        "glasser+tian-s4", "schaefer200+tian-s1", "schaefer500+tian-s4",
        "schaefer1000+tian-s4",
    )
    options = dict(
        dwi=dwi_path, bvals="unused", bvecs="unused", freesurfer_subject_dir=subject_dir,
        atlas_templates_dir=templates, fsaverage_dir=tmp_path / "fsaverage",
        n_seeds=2, shell_bvals=(0., 1000.), brain_mask="unused", response_mask="unused",
        fod_mask="unused", normalise_mask="unused", fa_map="unused",
        dwi_to_t1_world=torch.eye(4),
        **({"mni_template": "unused"} if registration == "synthmorph" else {"tian_fnirt_coeff": "unused"}),
    )
    runner = module.UKBConnectome_pipeline(device="cpu")
    combined = runner(atlas=atlas_names, **options)
    assert list(combined.atlas_results) == list(atlas_names)
    assert calls[("cortex", "glasser")] == 1
    assert calls[("tian", 1)] == calls[("tian", 4)] == 1
    assert sum(count for (kind, _), count in calls.items() if kind == "cortex") == 6
    if registration == "synthmorph":
        assert seen_transforms == [None, shared_transform]

    # Every standalone call starts a new cache and provides the parity oracle.
    for atlas_name in atlas_names:
        independent = runner(atlas=atlas_name, **options).atlas_results[atlas_name]
        reused = combined.atlas_results[atlas_name]
        assert reused.nodes == independent.nodes
        assert reused.region_labels == independent.region_labels
        assert torch.equal(reused.atlas, independent.atlas)
        assert torch.equal(reused.atlas_affine, independent.atlas_affine)
        assert reused.matrices.keys() == independent.matrices.keys()
        for name in reused.matrices:
            assert torch.equal(reused.matrices[name], independent.matrices[name])
    assert calls[("cortex", "glasser")] == 3
    assert calls[("tian", 1)] == 5
    assert calls[("tian", 4)] == 4
