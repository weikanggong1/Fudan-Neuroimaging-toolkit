"""Prevent stale anatomy/transforms from being reused across BIDS runs."""

from dataclasses import replace
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from fnit.fmri import _anatomical as cache
from fnit.fmri.normalization import T1MNIResult
from fnit.fnirt import T1FNIRTConfig


@pytest.fixture
def anatomy(tmp_path, monkeypatch):
    source = tmp_path / "T1.nii.gz"
    template = tmp_path / "MNI.nii.gz"
    mask = tmp_path / "mask.nii.gz"
    weights = tmp_path / "strip.pt"
    weights.write_bytes(b"checkpoint identity")
    # Stored pixdim deliberately differs from the sheared sform's column norm.
    affine = np.eye(4)
    affine[0, 1] = .15
    for path, dtype in ((source, np.int16), (template, np.int16), (mask, np.uint8)):
        image = nib.Nifti1Image(np.ones((6, 7, 8), dtype=dtype), affine)
        image.header.set_zooms((1, 1, 1))
        nib.save(image, path)
    calls = {"strip": 0, "fast": 0, "registration": 0}

    class Volume:
        def __init__(self, data, image):
            self.data = data
            self.image = image

        def save(self, path):
            cache._save(self.data, self.image, path)

    class Strip:
        model_path = weights

        def __call__(self, path):
            calls["strip"] += 1
            image = nib.load(path)
            return SimpleNamespace(image=Volume(np.asarray(image.dataobj, dtype=np.float32) + .25, image),
                                   mask=Volume(np.ones(image.shape, dtype=np.uint8), image))

    class FAST:
        def __init__(self, *, device):
            assert device == "cpu"

        def __call__(self, path, *, mask):
            calls["fast"] += 1
            image = nib.load(path)
            return SimpleNamespace(pve_wm=Volume(np.ones(image.shape, dtype=np.float32) * .75, image),
                                   pve_csf=Volume(np.ones(image.shape, dtype=np.float32) * .1, image))

    def registration(source, fixed, output, *, backend, **kwargs):
        calls["registration"] += 1
        affine = output / "T1_to_MNI152_2mm_affine.mat"
        pull = output / "MNI152_2mm_to_T1_pull_ras.nii.gz"
        np.savetxt(affine, np.eye(4))
        reference = nib.load(fixed)
        nib.save(nib.Nifti1Image(np.zeros((*reference.shape, 3), dtype=np.float32), reference.affine), pull)
        return T1MNIResult(affine, pull, backend, np.eye(4), {"valid": True},
                           {"t1_to_mni_affine": 1., "t1_to_mni_nonlinear": 2., "warp_conversion": .1})

    monkeypatch.setattr(cache, "TorchFAST", FAST)
    monkeypatch.setattr(cache, "register_t1_to_mni", registration)
    options = dict(template_mask=mask, strip=Strip(), backend="fnirt", morph_weights=None,
                   fnirt_config=T1FNIRTConfig(), device="cpu", work_dir=tmp_path / "work",
                   cache_dir=tmp_path / "derivatives" / "anat" / ".fnit_anatomical")
    return source, template, options, calls


def test_second_run_reuses_verified_anatomy_but_changed_inputs_do_not(anatomy):
    source, template, options, calls = anatomy
    first = cache.prepare_anatomical(source, template, **options)
    second = cache.prepare_anatomical(source, template, **options)
    assert not first.reused and second.reused
    assert first.fingerprint == second.fingerprint
    assert calls == {"strip": 1, "fast": 1, "registration": 1}
    assert second.timing_seconds["t1_to_mni_nonlinear"] == 0
    np.testing.assert_array_equal(second.registration.moving_to_fixed_world, np.eye(4))
    image = nib.load(first.path("T1_brain.nii.gz"))
    assert image.get_data_dtype() == np.dtype("float32")
    np.testing.assert_array_equal(image.get_fdata(), 1.25)
    np.testing.assert_array_equal(image.header.get_zooms(), (1, 1, 1))
    assert nib.load(first.path("T1_mask.nii.gz")).get_data_dtype() == np.dtype("uint8")
    options["fnirt_config"] = replace(T1FNIRTConfig(), bias_regularization=9999)
    third = cache.prepare_anatomical(source, template, **options)
    assert not third.reused and third.fingerprint != first.fingerprint
    assert calls["registration"] == 2
    options["strip"].model_path.write_bytes(b"different weights")
    fourth = cache.prepare_anatomical(source, template, **options)
    assert not fourth.reused and fourth.fingerprint != third.fingerprint
    assert calls["registration"] == 3


def test_corrupt_or_interrupted_cache_is_rebuilt(anatomy):
    source, template, options, calls = anatomy
    first = cache.prepare_anatomical(source, template, **options)
    first.path("T1_pve_wm.nii.gz").write_bytes(b"broken")
    rebuilt = cache.prepare_anatomical(source, template, **options)
    assert not rebuilt.reused and calls["registration"] == 2
    rebuilt.path("manifest.json").unlink()
    interrupted = cache.prepare_anatomical(source, template, **options)
    assert not interrupted.reused and calls["registration"] == 3
    assert cache.prepare_anatomical(source, template, **options).reused


def test_disabled_cache_runs_again_without_reusing_persistent_results(anatomy):
    source, template, options, calls = anatomy
    cache.prepare_anatomical(source, template, **options)
    uncached = cache.prepare_anatomical(source, template, **options, reuse=False)
    assert not uncached.reused and uncached.fingerprint is None
    assert uncached.directory == options["work_dir"] / "anatomical"
    assert calls["registration"] == 2


def test_cache_hit_retains_cold_fast_cuda_policy_without_running_a_gpu_fit(anatomy, monkeypatch):
    import torch
    source, template, options, calls = anatomy
    first = cache.prepare_anatomical(source, template, **options)
    # Exercise the CUDA cache-hit policy using the verified fixture; a hit
    # must preserve backend settings without another fit or any GPU work.
    monkeypatch.setattr(cache, "_fingerprint", lambda *args: first.fingerprint)
    monkeypatch.setitem(options, "device", "cuda:0")
    with monkeypatch.context() as state:
        state.setattr(torch.backends.cudnn, "benchmark", True)  # SynthStrip policy
        state.setattr(torch.backends.cudnn, "deterministic", False)
        state.setattr(torch.backends.cudnn, "allow_tf32", False)
        state.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
        hit = cache.prepare_anatomical(source, template, **options)
        assert hit.reused and calls["registration"] == 1
        assert not torch.backends.cudnn.benchmark
        assert torch.backends.cudnn.deterministic
        assert torch.backends.cudnn.allow_tf32 and torch.backends.cuda.matmul.allow_tf32
