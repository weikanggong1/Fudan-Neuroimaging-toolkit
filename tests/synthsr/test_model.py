from pathlib import Path
import os

import h5py
import nibabel as nib
import numpy as np
import pytest
import torch

from fnit.synthsr.model import SynthSRUNet, load_h5_weights
from fnit.synthsr.pipeline import SynthSRImage, _load_image


def test_unet_output_shape():
    model = SynthSRUNet().eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 1, 32, 32, 32))
    assert output.shape == (1, 1, 32, 32, 32)
    assert torch.isfinite(output).all()


def test_missing_h5_layer_is_rejected(tmp_path):
    path = tmp_path / "incomplete.h5"
    with h5py.File(path, "w"):
        pass
    with pytest.raises(ValueError, match="unet_conv_downarm_0_0"):
        load_h5_weights(SynthSRUNet(), path)


def test_npz_input_keeps_source_dtype(tmp_path):
    path = tmp_path / "image.npz"
    np.savez_compressed(path, vol_data=np.ones((32, 32, 32), dtype=np.uint16))
    data, affine, _ = _load_image(path)
    assert data.dtype == np.uint16
    np.testing.assert_array_equal(affine, np.eye(4))


def test_nibabel_memory_input_preserves_affine_and_header():
    affine = np.diag([1.2, 1.3, 1.4, 1.0])
    source = nib.Nifti1Image(np.ones((4, 5, 6), dtype=np.float32), affine)
    data, loaded_affine, header = _load_image(source)
    np.testing.assert_array_equal(data, np.ones((4, 5, 6)))
    np.testing.assert_array_equal(loaded_affine, affine)
    assert isinstance(header, nib.Nifti1Header)


def test_npz_output_uses_unquantized_data(tmp_path):
    image = SynthSRImage(np.zeros((2, 2, 2), dtype=np.uint8), np.eye(4),
                         None, np.full((2, 2, 2), 3.25))
    path = tmp_path / "output.npz"
    image.save(path)
    np.testing.assert_array_equal(np.load(path)["vol_data"], image.float_data)


@pytest.mark.parametrize("filename", (
    "synthsr_v20_230130.h5",
    "synthsr_lowfield_v20_230130.h5",
    "synthsr_v10_210712.h5",
))
def test_official_checkpoints_load_when_available(filename):
    home = os.environ.get("FREESURFER_HOME")
    path = Path(home) / "models" / filename if home else None
    if path is None or not path.is_file():
        pytest.skip("FreeSurfer checkpoints are optional for package tests")
    model = load_h5_weights(SynthSRUNet(), path)
    with h5py.File(path) as weights:
        expected = weights["unet_conv_downarm_0_0"]["unet_conv_downarm_0_0"]["kernel:0"][:]
    np.testing.assert_array_equal(model.down[0][0].weight.detach().numpy(),
                                  expected.transpose(4, 3, 0, 1, 2))
