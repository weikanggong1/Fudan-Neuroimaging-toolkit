"""Stage dependency and complete result restoration, with a mocked core.

The real atlas resampler and endpoint/matrix builder run on CPU. The expensive
numerical core is deliberately stubbed: these tests are not imaging benchmarks.
"""
from collections import Counter
import json
import os
import sys
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
import torch

import fnit.connectome.pipeline as module


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    # This cache contract stubs the entire numerical core. Resolving/building
    # the real native runtime would be unrelated to its assertions.
    monkeypatch.setattr(module, "_native_tracking_fingerprint", lambda: {
        "backend": "unit-test-core", "binary_sha256": "0" * 64})
    shape = (4, 3, 2)
    dwi = tmp_path / "dwi.nii"
    nib.save(nib.Nifti1Image(np.ones((*shape, 2), np.float32), np.eye(4)), dwi)
    bvals, bvecs = tmp_path / "dwi.bval", tmp_path / "dwi.bvec"
    np.savetxt(bvals, [[0., 1000.]])
    np.savetxt(bvecs, [[0., 1.], [0., 0.], [0., 0.]])
    brain = tmp_path / "brain.nii"
    seg = tmp_path / "seg.nii"
    atlas = tmp_path / "atlas.nii"
    nib.save(nib.Nifti1Image(np.ones(shape, np.float32), np.eye(4)), brain)
    nib.save(nib.Nifti1Image(np.full(shape, 2, np.float32), np.eye(4)), seg)
    labels = np.ones(shape, np.int32)
    labels[2:] = 2
    nib.save(nib.Nifti1Image(labels, np.eye(4)), atlas)
    calls = Counter()
    paths = (torch.tensor([[0., 1., 1.], [1., 1., 1.], [3., 1., 1.]]),
             torch.tensor([[0., 2., 1.], [3., 2., 1.]]))
    fa = torch.full(shape, .5)
    fa[0, 0, 0] = float("nan")

    def core(**options):
        calls["core"] += 1
        eye = torch.eye(4, dtype=torch.float64)
        return dict(seg=torch.full(shape, 2.), seg_affine=eye,
                    five=torch.ones((*shape, 5)), gmwmi=torch.ones(shape),
                    transform=eye, five_affine=eye, wm_sh=torch.ones((*shape, 45)),
                    fa=fa.clone(), mask=torch.ones(shape, dtype=torch.bool),
                    tracks=module.Tractogram(paths, torch.stack([p[[0, -1]] for p in paths]),
                                             torch.tensor([3., 3.]), torch.tensor([.45, .55]),
                                             options["n_seeds"], None,
                                             {"backend": "unit-test-core", "tck_header": {"step_size": "0.5"}}),
                    weights=torch.tensor([.75, 1.25], dtype=torch.float64),
                    dwi_affine=eye, dwi_shape=shape)

    monkeypatch.setattr(module, "_compute_shared_core", core)
    options = dict(dwi=dwi, bvals=bvals, bvecs=bvecs, t1_brain=brain,
                   t1_segmentation=seg, atlas_dwi=atlas, n_seeds=3,
                   dwi_to_t1_world=torch.eye(4, dtype=torch.float64),
                   checkpoint_dir=tmp_path / "checkpoints")
    return module.UKBConnectome_pipeline(device="cpu"), options, calls


def assert_result_same(a, b):
    for name in ("atlas", "five_tissue", "five_tissue_affine", "gmwmi", "wm_sh",
                 "fa", "brain_mask", "sift2_weights", "dwi_affine", "atlas_affine",
                 "dwi_to_t1_world"):
        torch.testing.assert_close(getattr(a, name), getattr(b, name), rtol=0, atol=0,
                                   equal_nan=True)
        assert getattr(a, name).dtype == getattr(b, name).dtype
    assert a.region_labels == b.region_labels and a.nodes == b.nodes
    for name in a.matrices:
        torch.testing.assert_close(a.matrices[name], b.matrices[name], rtol=0, atol=0, equal_nan=True)
    for first, second in zip(a.tractogram.paths, b.tractogram.paths, strict=True):
        assert torch.equal(first, second)
    for name in ("endpoints", "lengths_mm", "mean_fa"):
        assert torch.equal(getattr(a.tractogram, name), getattr(b.tractogram, name))
    assert a.tractogram.accepted_seeds is b.tractogram.accepted_seeds is None
    assert a.tractogram.native_provenance == b.tractogram.native_provenance
    assert a.tractogram.seeds_attempted == b.tractogram.seeds_attempted


def test_full_result_restored_template_and_radius_do_not_recompute_core(fixture):
    runner, options, calls = fixture
    first = runner(**options)
    second = runner(**options)
    assert first.cache_status["core"] == "completed"
    assert second.cache_status["core"] == "skipped"
    assert first.cache_status["core_key"] == second.cache_status["core_key"]
    assert calls["core"] == 1
    assert_result_same(first, second)
    atlas = nib.load(str(options["atlas_dwi"]))
    nib.save(nib.Nifti1Image(np.full(atlas.shape, 3, np.int32), atlas.affine), options["atlas_dwi"])
    changed = runner(**options, assignment_radius=2.)
    assert calls["core"] == 1 and changed.cache_status["core"] == "skipped"
    assert changed.cache_status["core_key"] == first.cache_status["core_key"]
    assert changed.region_labels == (1, 2, 3)
    assert changed.matrices["count"].shape == (3, 3)


def test_template_pair_hook_receives_only_completed_shared_fields(fixture, monkeypatch):
    runner, options, calls = fixture
    observed = []

    def pairs(result, templates, **kwargs):
        observed.append((result, templates, kwargs))
        return {"pair": "computed"}

    monkeypatch.setitem(sys.modules, "fnit.connectome.paired_pipeline",
                        SimpleNamespace(build_template_pairs=pairs))
    from fnit.connectome.template_inputs import TemplatePair, TemplateSpec
    a = TemplateSpec("A", "volume", "dwi", volume_path=options["atlas_dwi"])
    b = TemplateSpec("B", "volume", "dwi", volume_path=options["atlas_dwi"])
    first = runner(**options, template_pairs=[TemplatePair("AA", a, a)])
    second = runner(**options, template_pairs=[TemplatePair("BB", b, b)], assignment_radius=3.)
    assert calls["core"] == 1
    assert first.pair_results == second.pair_results == {"pair": "computed"}
    assert observed[1][2]["assignment_radius"] == 3.
    assert observed[1][2]["checkpoint_dir"] == options["checkpoint_dir"]
    assert observed[1][0].cache_status["core"] == "skipped"


@pytest.mark.parametrize("change", ["input", "seed", "transform", "shells", "revision", "policy", "native"])
def test_numerical_inputs_parameters_revision_policy_invalidate_core(fixture, monkeypatch, change):
    runner, options, calls = fixture
    first = runner(**options)
    if change == "input":
        path = options["bvals"]
        stat = path.stat()
        data = path.read_bytes()
        path.write_bytes(data.replace(b"1.000", b"2.000", 1))
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    elif change == "seed":
        options["seed"] = 7
    elif change == "transform":
        options["dwi_to_t1_world"][0, 3] = .1
    elif change == "shells":
        options["shell_bvals"] = (0., 1000.)
    elif change == "revision":
        monkeypatch.setattr(module, "CONNECTOME_NUMERICAL_REVISION", "test-changed")
    elif change == "native":
        monkeypatch.setattr(module, "_native_tracking_fingerprint", lambda: {
            "backend": "unit-test-core", "binary_sha256": "1" * 64})
    else:
        original = module._checkpoint_policy
        monkeypatch.setattr(module, "_checkpoint_policy", lambda device: {**original(device), "test_policy": 2})
    second = runner(**options)
    assert calls["core"] == 2
    assert second.cache_status["core"] == "completed"
    assert second.cache_status["core_key"] != first.cache_status["core_key"]


def test_semantically_invalid_complete_payload_is_not_restored(fixture):
    runner, options, calls = fixture
    first = runner(**options)
    root = options["checkpoint_dir"] / "shared" / "core" / first.cache_status["core_key"]
    marker = json.loads((root / "complete.json").read_bytes())
    (root / marker["generation"] / "track_offsets.npy").unlink()
    result = runner(**options)
    assert calls["core"] == 2 and result.cache_status["core"] == "completed"
    assert len(list(root.glob("generation-*"))) == 2


def test_direct_call_opt_in_and_overwrite_keeps_old_generations(fixture):
    runner, options, calls = fixture
    first = runner(**options)
    result = runner(**options, overwrite=True)
    assert calls["core"] == 2 and result.cache_status["core"] == "completed"
    root = options["checkpoint_dir"] / "shared" / "core" / first.cache_status["core_key"]
    assert len(list(root.glob("generation-*"))) == 2
    options["checkpoint_dir"] = None
    result = runner(**options)
    assert result.cache_status["enabled"] is False and result.cache_status["core_key"] is None


def test_changed_source_or_input_during_core_is_not_published(fixture, monkeypatch):
    runner, options, calls = fixture
    original = module._compute_shared_core

    def changed(**kwargs):
        result = original(**kwargs)
        options["bvecs"].write_text("changed during processing")
        return result

    monkeypatch.setattr(module, "_compute_shared_core", changed)
    with pytest.raises(RuntimeError, match="not publishing"):
        runner(**options)
    assert not list(options["checkpoint_dir"].rglob("complete.json"))


def test_bids_default_checkpoint_backend_and_stages_are_forwarded(monkeypatch, tmp_path):
    import fnit.connectome.bids as bids
    preparation, compute = {}, {}

    def prepare(*args, **kwargs):
        preparation.update(kwargs)
        return SimpleNamespace(dwi="corrected", bvals="bval", bvecs="bvec",
                               freesurfer_subject_dir="fs", stages={"eddy": "skipped"})

    def run(self, *args, **kwargs):
        compute.update(kwargs)
        return SimpleNamespace(preparation_stages=None)

    monkeypatch.setattr(bids, "prepare_bids_connectome", prepare)
    monkeypatch.setattr(module.UKBConnectome_pipeline, "__call__", run)
    result = module.UKBConnectome_pipeline(device="cpu").run_bids(
        "raw", tmp_path, subject="01", n_seeds=3, recon_backend="provided",
        recon_options={"threads": 4}, seed=8, overwrite=True)
    assert preparation["recon_backend"] == "provided"
    assert preparation["recon_options"] == {"threads": 4}
    assert compute["checkpoint_dir"] == tmp_path / "checkpoints"
    assert compute["overwrite"] is True and compute["seed"] == 8
    assert result.preparation_stages == {"eddy": "skipped"}


def test_invalid_radius_rejected_before_core(fixture):
    runner, options, calls = fixture
    for radius in (float("nan"), float("inf"), -1., 0.):
        with pytest.raises(ValueError, match="assignment_radius"):
            runner(**options, assignment_radius=radius)
    assert not calls
