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

from nibabel.orientations import apply_orientation, inv_ornt_aff, io_orientation, ornt_transform, axcodes2ornt
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


def _geometry_image(data, reference, affine):
    """Keep intermediate geometry in float64 until the public NIfTI is saved."""
    header = reference.header.copy()
    header.set_data_dtype(np.asarray(data).dtype)
    return FNITNifti1Image(np.asarray(data), np.asarray(affine, dtype=np.float64), header)


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
    return _geometry_image(out, sdt, sdt.affine)


def _crop_nonzero(image):
    data = np.asanyarray(image.dataobj)
    # SynthStrip crops the positive-data bounding box before normalization.
    indices = np.nonzero(data > 0)
    if not indices[0].size:
        return image
    low = np.asarray([axis.min() for axis in indices])
    high = np.asarray([axis.max() + 1 for axis in indices])
    slices = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
    affine = image.affine.copy()
    affine[:3, 3] += affine[:3, :3] @ low
    return _geometry_image(data[slices], image, affine)


def _reshape_center(image, target_shape):
    data = np.asanyarray(image.dataobj)
    target_shape = np.asarray(target_shape, dtype=int)
    source_shape = np.asarray(data.shape, dtype=int)
    # For odd cropping differences, the extra removed voxel is at the high end.
    source_start = np.maximum(-np.ceil((target_shape - source_shape) / 2).astype(int), 0)
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
    return _geometry_image(output, image, affine)


def _resample_affine(data, source_affine, target_shape, target_affine, *,
                     device="cpu", nearest=False, fill=0.0):
    """Sample an affine grid with SynthStrip's float32 coordinate convention.

    Nearest sampling accepts coordinates in ``[0, shape)`` and rounds half
    voxels upwards. Linear sampling clamps the high neighbour in the last
    voxel, and uses ``fill`` when the low neighbour lies outside the image.
    Coordinate products use scalar operations, independent of TF32 settings.
    """
    data = np.asarray(data, dtype=np.float32)
    target_shape = tuple(int(value) for value in target_shape)
    pull = np.linalg.inv(np.asarray(source_affine, dtype=np.float64)) @ np.asarray(target_affine, dtype=np.float64)
    # Cropping/padding on an unchanged grid needs no interpolation.
    if np.allclose(pull[:3, :3], np.eye(3), atol=1e-5, rtol=0) and np.allclose(
        pull[:3, 3], np.round(pull[:3, 3]), atol=1e-5, rtol=0
    ):
        offset = np.round(pull[:3, 3]).astype(int)
        output = np.full(target_shape, fill, dtype=np.float32)
        target_start = np.maximum(-offset, 0)
        source_start = np.maximum(offset, 0)
        length = np.minimum(np.asarray(target_shape) - target_start, np.asarray(data.shape) - source_start)
        if np.all(length > 0):
            source_slices = tuple(slice(int(a), int(a + n)) for a, n in zip(source_start, length))
            target_slices = tuple(slice(int(a), int(a + n)) for a, n in zip(target_start, length))
            output[target_slices] = data[source_slices]
        return output
    source = torch.as_tensor(np.ascontiguousarray(data), device=device)
    matrix = torch.as_tensor(pull[:3], dtype=torch.float32, device=device)
    limits = torch.as_tensor(data.shape, device=device).reshape(3, 1)
    output = torch.empty(int(np.prod(target_shape)), dtype=torch.float32, device=device)
    yz = target_shape[1] * target_shape[2]
    for start in range(0, output.numel(), 1 << 20):
        linear = torch.arange(start, min(start + (1 << 20), output.numel()), device=device)
        index = ((linear // yz).float(), ((linear // target_shape[2]) % target_shape[1]).float(), (linear % target_shape[2]).float())
        coordinates = torch.stack([((matrix[row, 0] * index[0] + matrix[row, 1] * index[1]) + matrix[row, 2] * index[2]) + matrix[row, 3] for row in range(3)])
        valid = ((coordinates >= 0) & (coordinates < limits)).all(dim=0)
        if nearest:
            rounded = torch.floor(coordinates + 0.5).long()
            rounded = torch.minimum(torch.maximum(rounded, torch.zeros_like(rounded)), limits - 1)
            values = source[rounded[0], rounded[1], rounded[2]]
        else:
            low = torch.floor(coordinates).long()
            low_safe = torch.minimum(torch.maximum(low, torch.zeros_like(low)), limits - 1)
            high = torch.minimum(low_safe + 1, limits - 1)
            fraction = coordinates - low.float()
            values = torch.zeros_like(coordinates[0])
            # Eight neighbours, in the original interpolator's summation order.
            for neighbour in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
                              (1, 0, 1), (0, 1, 1), (1, 1, 0), (1, 1, 1)):
                weights = [(fraction[axis] if neighbour[axis] else 1 - fraction[axis]) for axis in range(3)]
                positions = [(high[axis] if neighbour[axis] else low_safe[axis]) for axis in range(3)]
                values = values + (weights[0] * weights[1] * weights[2]) * source[positions[0], positions[1], positions[2]]
        output[start:start + len(linear)] = torch.where(valid, values, float(fill))
    return output.reshape(target_shape).cpu().numpy()


def _conform_lia_1mm(image, *, device="cpu"):
    """Reorient discretely, then resize around the original ``shape / 2`` centre.

    SynthStrip's geometry uses the centre of the full field of view. Nibabel's
    general ``conform`` instead centres the first and last voxel centres; the
    two definitions differ by half a voxel when changing voxel size.
    """
    orientation = ornt_transform(io_orientation(image.affine), axcodes2ornt("LIA"))
    source = apply_orientation(np.asanyarray(image.dataobj), orientation)
    affine = np.asarray(image.affine, dtype=np.float64) @ inv_ornt_aff(orientation, image.shape)
    q, r = np.linalg.qr(affine[:3, :3])
    # The original NIfTI header's pixdim supplies voxel sizes. Deriving them
    # again from the rounded sform can change ceil(shape * voxel_size) by one.
    reordered_axes = np.argsort(orientation[:, 0].astype(int))
    voxel_size = np.asarray(image.header.get_zooms()[:3], dtype=np.float64)[reordered_axes]
    rotation = q @ np.diag(np.sign(np.diag(r)))
    shape = np.asarray(source.shape)
    if np.allclose(voxel_size, 1.0, atol=1e-5, rtol=0):
        return _geometry_image(np.asarray(source, dtype=np.float32), image, affine)
    target_shape = np.ceil(shape * voxel_size).astype(int)
    center = affine[:3, :3] @ (shape / 2) + affine[:3, 3]
    target_affine = np.eye(4)
    target_affine[:3, :3] = rotation
    target_affine[:3, 3] = center - rotation @ (target_shape / 2)
    resized = _resample_affine(source, affine, target_shape, target_affine,
                               device=device, nearest=True)
    return _geometry_image(resized, image, target_affine)


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

    def __init__(self, weights=None, device="cpu", no_csf=False, threads=None):
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if threads is not None:
            torch.set_num_threads(threads)
        # Fixed algorithm selection avoids cross-process boundary changes
        # from timing-based autotuning on shared GPUs; retain TF32 below.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        name = "synthstrip.nocsf.1.pt" if no_csf else "synthstrip.1.pt"
        self.model_path = Path(resolve_weights(name, explicit=weights))
        self.model = StripModel().to(self.device).eval()
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    @torch.no_grad()
    def __call__(self, image, border=1, fill=None):
        image = load_image(image)
        source = np.asanyarray(image.dataobj)
        if source.ndim not in (3, 4):
            raise ValueError("SynthStrip accepts a 3D image or a 4D frame series")
        frames = source[..., None] if source.ndim == 3 else source
        distances, masks = [], []
        for frame_index in range(frames.shape[-1]):
            frame = new_image(frames[..., frame_index], image)
            conformed = _conform_lia_1mm(frame, device=self.device)
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
            prediction = self.model(tensor).squeeze().cpu().numpy()
            distance = extend_sdt(_geometry_image(prediction, conformed, conformed.affine), border=border)
            distance_data = _resample_affine(np.asarray(distance.dataobj), distance.affine,
                                             frame.shape, frame.affine, device=self.device,
                                             fill=100.0)
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
