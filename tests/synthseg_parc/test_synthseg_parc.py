"""Synthetic checks for the PyTorch SynthSeg parcellation head."""

import os
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from fnit.synthseg_parc import SynthSegParc
from fnit.synthseg_parc.model import ParcUNet


def test_unet_has_69_normalized_channels():
    torch.set_num_threads(2)
    model = ParcUNet().eval()
    with torch.inference_mode():
        posterior = model(torch.zeros(1, 3, 32, 32, 32))
    assert posterior.shape == (1, 69, 32, 32, 32)
    assert torch.allclose(posterior.sum(dim=1), torch.ones_like(posterior[:, 0]), atol=1e-6)
    with pytest.raises(ValueError, match="divisible by 32"):
        model(torch.zeros(1, 3, 31, 32, 32))


def test_official_h5_load_and_cortex_only_output():
    weights = os.environ.get("FREESURFER_SYNTHSEG_PARC_WEIGHTS")
    labels = os.environ.get("FREESURFER_SYNTHSEG_PARC_LABELS")
    if not weights or not labels:
        pytest.skip("Set official SynthSeg parcellation weights and labels to run integration check")
    torch.set_num_threads(2)
    model = SynthSegParc(Path(weights), Path(labels))
    with h5py.File(weights, "r") as h5:
        kernel = h5["unet_parc_conv_downarm_0_0"]["unet_parc_conv_downarm_0_0"]["kernel:0"][:]
    expected = torch.from_numpy(np.asarray(kernel).transpose(4, 3, 0, 1, 2).copy())
    assert torch.equal(model.model.down[0].conv0.weight.cpu(), expected)

    image = torch.zeros(32, 32, 32)
    segmentation = torch.zeros_like(image, dtype=torch.long)
    segmentation[8:24, 8:24, 8:24] = 3
    parcels = model(image, segmentation)
    assert parcels.shape == segmentation.shape
    assert torch.all(parcels[segmentation == 0] == 0)
    assert torch.all(parcels[segmentation == 3] != 0)
    assert set(torch.unique(parcels).tolist()) <= set(np.load(labels).tolist())
    parcels_with_volumes, soft_volumes = model(
        image, segmentation, soft_volumes=True,
        content_slices=(slice(None), slice(None), slice(None)))
    assert torch.equal(parcels_with_volumes, parcels)
    assert soft_volumes.shape == (68,)
    assert np.all(np.isfinite(soft_volumes)) and np.all(soft_volumes > 0)
    with pytest.raises(ValueError, match="aligned 3-D"):
        model(image, segmentation[1:])
