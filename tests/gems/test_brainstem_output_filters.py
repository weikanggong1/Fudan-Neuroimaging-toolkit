import json
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
import torch

from fnit.gems.context import SubregionContext
from fnit.gems.recipes import brainstem


@pytest.mark.parametrize("resolution", (None, 1.0))
def test_brainstem_highres_labels_obey_component_support_and_positive_confidence(
        tmp_path, monkeypatch, resolution):
    shape = (8, 8, 8)
    data = np.ones(shape, np.float32)
    image = nib.Nifti1Image(data, np.eye(4))
    coarse = np.full(shape, 16, np.int32)
    coarse[2, 1, 1] = 0  # Largest component, outside coarse support.
    context = SubregionContext(image, data, coarse, None, None)
    labels = np.zeros(shape, np.int32)
    labels[1:3, 1:3, 1:3] = 173
    labels[6, 6, 6] = 173  # Separate smaller component.
    posterior = torch.stack((torch.ones(shape), torch.zeros(shape)))
    foreground = torch.as_tensor(labels != 0)
    posterior[0, foreground] = .1
    posterior[1, foreground] = .9
    posterior[:, 1, 2, 1] = 0  # Largest component, no supported posterior.
    config = {"label_classes": [0, 1], "support_coarse_label_ids": [16]}
    if resolution is not None:
        config["working_resolution_mm"] = resolution
    (tmp_path / "config.json").write_text(json.dumps(config))
    np.save(tmp_path / "atlas_to_native_voxel.npy", np.eye(4))

    class Atlas:
        vertices = np.asarray([[0., 0., 0.], [7., 7., 7.]])
        label_ids = np.asarray([0, 173])
        def transformed(self, matrix, **kwargs):
            return self

    atlas = Atlas()
    monkeypatch.setattr(brainstem.GEMSAtlas, "from_freesurfer", lambda *args, **kwargs: atlas)
    monkeypatch.setattr(brainstem, "brainstem_gaussian_hyperparameters", lambda *args:
                        (np.asarray([1., 2.], np.float32), np.asarray([10., 10.], np.float32)))
    monkeypatch.setattr(brainstem, "fit_brainstem_segmentation", lambda current, *args, **kwargs:
                        (current, {"mock": True}))
    monkeypatch.setattr(brainstem, "make_brainstem_working_image", lambda *args, **kwargs:
                        (image, coarse, {"mock": True}))

    class Engine:
        def __init__(self, current, **kwargs):
            assert current is atlas and kwargs["device"] == torch.device("cpu")
        def __call__(self, target, **kwargs):
            assert tuple(target.shape) == shape
            return SimpleNamespace(labels=torch.as_tensor(labels), posterior=posterior)

    monkeypatch.setattr(brainstem, "TorchGEMS", Engine)
    outcome = brainstem._fit_brainstem(context, tmp_path, torch.device("cpu"))
    expected = labels.copy()
    expected[6, 6, 6] = expected[2, 1, 1] = expected[1, 2, 1] = 0
    np.testing.assert_array_equal(outcome.highres_labels.dataobj, expected)
    np.testing.assert_array_equal(outcome.native_labels, expected)
    np.testing.assert_array_equal(outcome.native_support, expected != 0)
    assert np.count_nonzero(expected) == 6
    assert outcome.native_confidence[6, 6, 6] == outcome.native_confidence[2, 1, 1] == 0
    assert outcome.native_confidence[1, 2, 1] == 0
    # Filtering winner labels preserves posterior-derived soft volumes.
    assert outcome.soft_volumes_mm3[173] == pytest.approx(float(posterior[1].sum()))
