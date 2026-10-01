"""Importable PyTorch implementation of FreeSurfer SynthStrip.

The official mri_synthstrip implementation already uses PyTorch. This module
preserves its network parameter names and image geometry operations with
nibabel and SciPy. Reference: FreeSurfer 8.2.0, build 20260314-d932c45,
python/scripts/mri_synthstrip.
Source: https://github.com/freesurfer/freesurfer/blob/d932c45/mri_synthstrip/mri_synthstrip
SynthStrip: Hoopes et al., NeuroImage (2022), doi:10.1016/j.neuroimage.2022.119474.
"""

from dataclasses import dataclass
from pathlib import Path

from nibabel.processing import conform, resample_from_to
import numpy as np
from scipy.ndimage import (
    binary_fill_holes,
    distance_transform_edt,
    generate_binary_structure,
    label,
)
import torch

from .._nib import FNITNifti1Image, load_image, new_image
from ..weights import resolve_weights
from .model import StripModel


def extend_sdt(sdt, border=1):
    """Preserve the official outer-distance extension for large mask borders."""
    data = np.asanyarray(sdt.dataobj)
    if border < int(data.max()):
        return sdt
    mask = data < 1
    keep = np.nonzero(mask)
    if not keep[0].size:
        return sdt
    low = np.min(keep, axis=-1)
    upp = np.max(keep, axis=-1)
    gap = int(border + 0.5)
    low = (max(i - gap, 0) for i in low)
    upp = (min(i + gap, d - 1) for i, d in zip(upp, mask.shape))
    ind = tuple(slice(a, b + 1) for a, b in zip(low, upp))
    out = np.full_like(data, fill_value=100)
    local = mask[ind]
    out[ind] = distance_transform_edt(~local) - distance_transform_edt(local)
    out[keep] = data[keep]
    return new_image(out, sdt)


def _crop_nonzero(image):
    data = np.asanyarray(image.dataobj)
    indices = np.nonzero(data)
    if not indices[0].size:
        return image
    low = np.asarray([axis.min() for axis in indices])
    high = np.asarray([axis.max() + 1 for axis in indices])
    slices = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
    affine = image.affine.copy()
    affine[:3, 3] += affine[:3, :3] @ low
    return new_image(data[slices], image, affine=affine)


def _reshape_center(image, target_shape):
    data = np.asanyarray(image.dataobj)
    target_shape = np.asarray(target_shape, dtype=int)
    source_shape = np.asarray(data.shape, dtype=int)
    source_start = np.maximum((source_shape - target_shape) // 2, 0)
    output_start = np.maximum((target_shape - source_shape) // 2, 0)
    length = np.minimum(source_shape, target_shape)
    source_slices = tuple(slice(int(a), int(a + n))
                          for a, n in zip(source_start, length))
    output_slices = tuple(slice(int(a), int(a + n))
                          for a, n in zip(output_start, length))
    output = np.zeros(tuple(target_shape), dtype=data.dtype)
    output[output_slices] = data[source_slices]
    affine = image.affine.copy()
    affine[:3, 3] += affine[:3, :3] @ (source_start - output_start)
    return new_image(output, image, affine=affine)


def _largest_filled_component(mask):
    components, count = label(mask, structure=generate_binary_structure(3, 1))
    if count == 0:
        return np.zeros(mask.shape, dtype=np.uint8)
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    largest = components == int(np.argmax(sizes))
    return binary_fill_holes(largest).astype(np.uint8)


@dataclass
class StripResult:
    image: FNITNifti1Image
    mask: FNITNifti1Image
    distance: FNITNifti1Image


class SynthStrip:
    """Load an official checkpoint once and process one or more image volumes.

    Images retain their original voxel grid and geometry. A 4D input is processed
    one frame at a time. ``distance`` contains millimetre signed distances;
    the binary mask uses ``distance < border`` followed by connected components.
    """

    # configure_precision=False仅保留调用方TF32策略；默认True保持公共行为。
    def __init__(self, weights=None, device="cpu", no_csf=False, threads=None, *, configure_precision=True):
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if threads is not None:
            torch.set_num_threads(threads)
        # Match the executable's convolution backend settings.
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        if configure_precision:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        name = "synthstrip.nocsf.1.pt" if no_csf else "synthstrip.1.pt"
        self.model_path = Path(resolve_weights(name, explicit=weights))
        self.model = StripModel().to(self.device).eval()
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    @torch.no_grad()
    def __call__(self, image, border=1, fill=None, *, precision_report=None):
        image = load_image(image)
        source = np.asanyarray(image.dataobj)
        if source.ndim not in (3, 4):
            raise ValueError("SynthStrip accepts a 3D image or a 4D frame series")
        frames = source[..., None] if source.ndim == 3 else source
        distances, masks = [], []
        for frame_index in range(frames.shape[-1]):
            frame = new_image(frames[..., frame_index], image)
            shape_1mm = tuple(np.maximum(1, np.ceil(
                np.asarray(frame.shape) * np.asarray(frame.header.get_zooms()[:3])
            ).astype(int)))
            conformed = conform(frame, out_shape=shape_1mm, voxel_size=(1.0,) * 3,
                                order=0, orientation="LIA", cval=0.0)
            conformed = new_image(np.asarray(conformed.dataobj, dtype=np.float32),
                                  frame, affine=conformed.affine)
            conformed = _crop_nonzero(conformed)
            shape = np.clip(np.ceil(np.array(conformed.shape[:3]) / 64).astype(int) * 64,
                            192, 320)
            conformed = _reshape_center(conformed, shape)
            conformed_data = np.asanyarray(conformed.dataobj)
            conformed_data = conformed_data - conformed_data.min()
            percentile = np.percentile(conformed_data, 99)
            if percentile > 0:
                conformed_data = np.clip(conformed_data / percentile, 0, 1)
            tensor = torch.from_numpy(
                np.ascontiguousarray(conformed_data[np.newaxis, np.newaxis])
            ).to(self.device)
            if precision_report is not None:
                from fnit.recon_all.profiling import record_network_forward
                record_network_forward(self.model, tensor, precision_report, model="SynthStrip")
            prediction = self.model(tensor).squeeze().cpu().numpy()
            distance = extend_sdt(new_image(prediction, conformed), border=border)
            resampled = resample_from_to(distance, (frame.shape, frame.affine),
                                         order=1, cval=100.0)
            distance_data = np.asarray(resampled.dataobj, dtype=np.float32)
            distances.append(distance_data)
            masks.append(_largest_filled_component(distance_data < border))
        distance_data = np.stack(distances, axis=-1)
        mask_data = np.stack(masks, axis=-1)
        if source.ndim == 3:
            distance_data = distance_data[..., 0]
            mask_data = mask_data[..., 0]
        output = source.copy()
        background = min(float(np.min(source)), 0.0) if fill is None else fill
        output[mask_data == 0] = background
        return StripResult(new_image(output, image), new_image(mask_data, image),
                           new_image(distance_data, image))
