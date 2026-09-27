"""Check official SynthSeg 2.0 segmentation weights and parcel chaining."""

import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from fnit.synthseg_parc import SynthSegParc
from fnit.synthseg_parc.segment import SynthSegSegmenter, run_synthseg_parc_t1


def test_segmenter_keeps_default_tf32_enabled(monkeypatch):
    seen = []

    class RecordingModel(torch.nn.Module):
        def forward(self, image):
            seen.append(torch.backends.cudnn.allow_tf32)
            posterior = torch.zeros((1, 33, *image.shape[2:]), device=image.device)
            posterior[:, 0] = 1
            return posterior

    segmenter = object.__new__(SynthSegSegmenter)
    segmenter.device = torch.device("cpu")
    segmenter.model = RecordingModel()
    segmenter.flip_indices = torch.arange(33)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    segmenter.posterior(torch.zeros((32, 32, 32)))
    assert seen == [True, True]
    assert torch.backends.cudnn.allow_tf32 is True


def test_segmentation_and_parcellation_chain():
    root = os.environ.get("FREESURFER_SYNTHSEG_WEIGHTS")
    labels = os.environ.get("FREESURFER_SYNTHSEG_LABELS")
    parc_weights = os.environ.get("FREESURFER_SYNTHSEG_PARC_WEIGHTS")
    parc_labels = os.environ.get("FREESURFER_SYNTHSEG_PARC_LABELS")
    if not all((root, labels, parc_weights, parc_labels)):
        pytest.skip("Set official SynthSeg segmentation and parcellation paths")

    torch.set_num_threads(2)
    segmenter = SynthSegSegmenter(Path(root), Path(labels))
    assert torch.equal(segmenter.flip_indices[segmenter.flip_indices], torch.arange(33))
    image = torch.linspace(0, 1, 32 ** 3).reshape(32, 32, 32)
    posterior = segmenter.posterior(image)
    assert posterior.shape == (33, 32, 32, 32)
    assert torch.allclose(posterior[:, 16, 16, 16].sum(), torch.tensor(1.0), atol=1e-6)
    segmentation = segmenter(image)
    assert set(torch.unique(segmentation).tolist()) <= set(np.unique(np.load(labels)).tolist())

    parcellator = SynthSegParc(Path(parc_weights), Path(parc_labels))
    parcels = parcellator(image, segmentation)
    cortex = (segmentation == 3) | (segmentation == 42)
    assert torch.all(parcels[~cortex] == 0)
    assert torch.all(parcels[cortex] != 0)
    assert set(torch.unique(parcels).tolist()) <= set(np.load(parc_labels).tolist())


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
@pytest.mark.parametrize("fast", [False, True])
def test_t1_to_segmentation_and_parc_output_grid(tmp_path, device, fast):
    paths = [os.environ.get(name) for name in (
        "FREESURFER_SYNTHSEG_WEIGHTS", "FREESURFER_SYNTHSEG_LABELS",
        "FREESURFER_SYNTHSEG_PARC_WEIGHTS", "FREESURFER_SYNTHSEG_PARC_LABELS")]
    if not all(paths):
        pytest.skip("Set official SynthSeg segmentation and parcellation paths")

    data = np.arange(16 * 18 * 20, dtype=np.float32).reshape(16, 18, 20)
    source_affine = np.diag([-1.0, 1.0, 1.0, 1.0])
    t1 = tmp_path / "t1.nii.gz"
    nib.save(nib.Nifti1Image(data, source_affine), t1)
    result = run_synthseg_parc_t1(t1, *paths, device=device, min_pad=32, fast=fast)

    assert result.segmentation.shape == result.parcellation.shape == result.combined.shape == data.shape
    assert result.segmentation.device.type == device
    assert result.source_shape == data.shape
    np.testing.assert_array_equal(result.source_affine, source_affine)
    np.testing.assert_array_equal(result.affine[:3, :3], np.eye(3))
    np.testing.assert_array_equal(result.affine[:3, 3], [-15, 0, 0])
    assert torch.equal(result.combined, torch.where(result.parcellation != 0,
                                                    result.parcellation, result.segmentation))
    assert set(torch.unique(result.segmentation).tolist()) <= set(np.unique(np.load(paths[1])).tolist())
    assert set(torch.unique(result.parcellation).tolist()) <= set(np.load(paths[3]).tolist())
