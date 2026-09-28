"""Connectome registration uses the current NiBabel TorchFLIRT API."""

from types import SimpleNamespace

import nibabel as nib
import numpy as np
import torch

from fnit.connectome.pipeline import _registration


def test_registration_passes_nibabel_image(monkeypatch):
    import fnit.flirt

    affine = torch.eye(4, dtype=torch.float64)
    affine[0, 3] = 12.5

    class FakeFLIRT:
        def __init__(self, *, device, dof, cost):
            assert (device, dof, cost) == ("cpu", 6, "normmi")

        def __call__(self, moving, fixed):
            assert isinstance(moving, nib.spatialimages.SpatialImage)
            np.testing.assert_allclose(moving.affine, affine.numpy())
            assert moving.shape == (3, 4, 5)
            assert fixed == "brain.mgz"
            return SimpleNamespace(moving_to_fixed_world=np.eye(4))

    monkeypatch.setattr(fnit.flirt, "TorchFLIRT", FakeFLIRT)
    result = _registration(torch.ones((3, 4, 5)), affine, "brain.mgz", torch.device("cpu"))
    torch.testing.assert_close(result, torch.eye(4, dtype=torch.float64))
