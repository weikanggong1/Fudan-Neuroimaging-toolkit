"""SynthSR inference adapted from FreeSurfer's mri_synthsr script."""

from dataclasses import dataclass
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter
import torch

from ..weights import resolve_weights
from .model import SynthSRUNet, load_h5_weights
from .spatial import align_volume_to_ref, crop_volume_with_idx, pad_volume, resample_volume


_WEIGHTS = {
    "general": "synthsr_v20_230130.h5",
    "lowfield": "synthsr_lowfield_v20_230130.h5",
    "v1": "synthsr_v10_210712.h5",
}


@dataclass
class SynthSRImage:
    """SynthSR output with the original file-format conversion rules."""

    data: np.ndarray
    affine: np.ndarray
    header: nib.Nifti1Header
    float_data: np.ndarray

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".npz":
            np.savez_compressed(path, vol_data=self.float_data)
            return
        if not str(path).endswith((".nii", ".nii.gz", ".mgz")):
            raise ValueError("SynthSR output must be .nii, .nii.gz, .mgz, or .npz")
        image = nib.Nifti1Image(self.data, self.affine, self.header)
        image.set_data_dtype(np.uint8)
        nib.save(image, str(path))


@dataclass
class SynthSRResult:
    image: SynthSRImage


class SynthSR:
    """Load one official checkpoint and synthesize 1 mm T1-weighted images."""

    def __init__(self, weights=None, device="cpu", lowfield=False, v1=False, threads=None):
        self.device = torch.device(device)
        if threads is not None:
            torch.set_num_threads(os.cpu_count() if threads < 0 else threads)
        variant = "v1" if v1 else "lowfield" if lowfield else "general"
        checkpoint = resolve_weights(_WEIGHTS[variant], explicit=weights)
        self.model = SynthSRUNet().to(self.device)
        load_h5_weights(self.model, checkpoint)
        self.model.eval()
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    @torch.inference_mode()
    def __call__(self, image, ct=False, disable_flipping=False, disable_sharpening=False):
        data, affine, header = _load_image(image)
        data, affine = resample_volume(data, affine)
        data = align_volume_to_ref(data, affine, aff_ref=np.eye(4), n_dims=3)
        target = (np.ceil(np.asarray(data.shape) / 32) * 32).astype(int)
        data, pad_idx = pad_volume(data, target, return_pad_idx=True)
        if ct:
            data = np.clip(data, 0, 80)
        data = data - np.min(data)
        data = data / np.max(data)
        tensor = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32))[None, None].to(self.device)
        pred1 = self.model(tensor)[0, 0].cpu().numpy()
        if disable_flipping:
            prediction = np.clip(255 * pred1, 0, 128)
        else:
            pred2 = self.model(torch.flip(tensor, dims=(2,)))[0, 0]
            pred2 = torch.flip(pred2, dims=(0,)).cpu().numpy()
            prediction = (0.5 * np.clip(255 * pred1, 0, 128)
                          + 0.5 * np.clip(255 * pred2, 0, 128))
        prediction = crop_volume_with_idx(prediction.clip(0, 128), pad_idx)
        if not disable_sharpening:
            prediction = prediction + (prediction - gaussian_filter(prediction, 1.5))
        prediction = align_volume_to_ref(prediction, np.eye(4), aff_ref=affine, n_dims=3)
        quantized = np.clip(prediction * 2, 0, 255).astype(np.uint8)
        return SynthSRResult(SynthSRImage(quantized, affine, header, prediction))



def _load_image(image):
    preserve_dtype = False
    if isinstance(image, (str, Path)):
        path = Path(image)
        if path.suffix == ".npz":
            data = np.load(path)["vol_data"]
            affine, header = np.eye(4), nib.Nifti1Header()
            preserve_dtype = True
        else:
            source = nib.load(str(path))
            data, affine, header = source.get_fdata(), source.affine, source.header
    elif isinstance(image, nib.spatialimages.SpatialImage):
        data, affine = np.asanyarray(image.dataobj), image.affine
        header = nib.Nifti1Header.from_header(image.header)
    else:
        raise TypeError("image must be a file path or nibabel spatial image")
    data = np.squeeze(data)
    if data.ndim == 4 and data.shape[-1] <= 10:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError("SynthSR accepts one 3D image (or the first of up to 10 channels)")
    if not preserve_dtype:
        data = np.asarray(data, dtype=np.float64)
    return np.asarray(data), np.asarray(affine, dtype=np.float64), header
