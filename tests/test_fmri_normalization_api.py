"""Current FNIT FLIRT/SynthMorph interfaces used by the fMRI pipeline."""

from types import SimpleNamespace

import nibabel as nib
import numpy as np

from fnit.fmri import normalization
import fnit.synthmorph


def test_t1_to_mni_uses_nibabel_images_and_world_pull(tmp_path, monkeypatch):
    moving = tmp_path / "t1.nii.gz"
    fixed = tmp_path / "mni.nii.gz"
    for path in (moving, fixed):
        nib.save(nib.Nifti1Image(np.ones((4, 5, 6), dtype=np.float32), np.eye(4)), path)

    class FakeFLIRT:
        def __init__(self, *, device):
            assert device == "cpu"

        def __call__(self, source, target):
            assert isinstance(source, nib.spatialimages.SpatialImage)
            assert isinstance(target, nib.spatialimages.SpatialImage)
            return SimpleNamespace(matrix=np.eye(4), moving_to_fixed_world=np.eye(4))

    class FakeSynthMorph:
        def __init__(self, *, weights, device, model):
            assert model == "deform"

        def __call__(self, source, target, *, init):
            assert isinstance(source, nib.spatialimages.SpatialImage)
            assert isinstance(target, nib.spatialimages.SpatialImage)
            np.testing.assert_array_equal(init, np.eye(4))
            return SimpleNamespace(transform=nib.Nifti1Image(
                np.zeros((*target.shape, 3), dtype=np.float32), target.affine
            ))

    monkeypatch.setattr(normalization, "TorchFLIRT", FakeFLIRT)
    monkeypatch.setattr(fnit.synthmorph, "SynthMorph", FakeSynthMorph)
    result = normalization.register_t1_to_mni(
        moving, fixed, tmp_path / "registration", device="cpu"
    )
    assert nib.load(str(result.pull_ras)).shape == (4, 5, 6, 3)
    np.testing.assert_array_equal(result.moving_to_fixed_world, np.eye(4))
