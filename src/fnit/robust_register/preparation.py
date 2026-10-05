"""Prepare the binary SAMSEG alignment target without an installed FreeSurfer.

Modified PyTorch/nibabel adaptation of FreeSurfer SAMSEG subregions and
Surfa 0.6.3 image geometry/interpolation. The FreeSurfer Software License
applies to the adapted workflow (licenses/FreeSurfer.txt). Geometry and
sampling adaptations retain the Surfa developers' MIT notice
(licenses/surfa-MIT.txt). Original sources and version boundaries are listed
in docs/robust_register/PREPARATION.md. This is not an official release.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from time import monotonic

import nibabel as nib
from nibabel.freesurfer.mghformat import MGHHeader, MGHImage
import numpy as np
import torch

from .._nib import load_image


@dataclass
class AlignmentTargetPreparation:
    """Float32 0/255 MGH image and preparation measurements.

    ``image.affine`` is the computational RAS-mm geometry. Its MGH header
    stores the source-defined float32 voxel sizes/directions/center; saving
    and loading therefore uses the MGH storage boundary. ``report`` contains
    grid, crop, operation, dtype and wall-time measurements, not image arrays.
    """

    image: MGHImage
    report: dict


@dataclass
class _Geometry:
    shape: np.ndarray
    voxel_sizes: np.ndarray
    rotation: np.ndarray
    center: np.ndarray
    affine: np.ndarray


def _compose(shape, sizes, rotation, center):
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = rotation @ np.diag(sizes)
    affine[:3, 3] = center - (affine @ np.append(shape / 2, 1))[:3]
    return affine


def _geometry(image) -> _Geometry:
    shape = np.asarray(image.shape[:3], dtype=np.int64)
    if isinstance(image.header, MGHHeader):
        # MGH fields are decoded into Double before composition, as in the
        # installed Surfa reader. nibabel's MGH get_affine multiplies its F32
        # fields first; that rounded product is a different computation.
        if int(image.header["goodRASFlag"]):
            sizes = np.asarray(image.header["delta"], dtype=np.float64)
            rotation = np.asarray(image.header["Mdc"], dtype=np.float64).T
            center = np.asarray(image.header["Pxyz_c"], dtype=np.float64)
        else:
            sizes = np.ones(3)
            rotation = np.array([[-1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)
            center = np.zeros(3)
        affine = _compose(shape, sizes, rotation, center)
    elif isinstance(image.header, (nib.Nifti1Header, nib.Nifti2Header)):
        unit = image.header.get_xyzt_units()[0]
        if unit not in ("mm", "unknown"):
            raise ValueError("alignment preparation requires NIfTI coordinates in mm")
        affine = np.array(image.affine, dtype=np.float64, copy=True)
        sizes = np.asarray(image.header.get_zooms()[:3], dtype=np.float64)
        q, r = np.linalg.qr(affine[:3, :3])
        scale = np.abs(np.diag(r))
        if np.any(scale == 0):
            raise ValueError("image affine must be invertible")
        rotation = q @ np.diag(np.diag(r) / scale)
        center = (affine @ np.append(shape / 2, 1))[:3]
    else:
        raise TypeError("alignment preparation supports MGH/MGZ or NIfTI images")
    if (not np.isfinite(affine).all() or not np.isfinite(sizes).all()
            or np.any(sizes <= 0) or np.linalg.det(affine[:3, :3]) == 0):
        raise ValueError("image geometry must be finite with positive voxel sizes")
    return _Geometry(shape, sizes, rotation, center, affine)


def _mgh_image(data, geometry: _Geometry, reference):
    header = reference.header.copy() if isinstance(reference.header, MGHHeader) else MGHHeader()
    header.set_data_shape(data.shape)
    header.set_data_dtype(data.dtype)
    header["dof"] = 1  # installed Surfa writer uses 1 for prepared images
    header["goodRASFlag"] = 1
    header["delta"] = geometry.voxel_sizes
    header["Mdc"] = geometry.rotation.T
    header["Pxyz_c"] = geometry.center
    header["fov"] = max(geometry.voxel_sizes * geometry.shape)
    if not isinstance(reference.header, MGHHeader):
        time_unit = reference.header.get_xyzt_units()[1]
        factor = {"sec": 1000, "msec": 1, "usec": .001}.get(time_unit, 0)
        header["tr"] = float(reference.header["pixdim"][4]) * factor
    # Filled fields are retained by nibabel when their affine is allclose to
    # the Double geometry. Do not derive header directions from column norms
    # again; the upstream geometry can contain stored non-unit directions.
    image = MGHImage(data, geometry.affine, header)
    for name in ("delta", "Mdc", "Pxyz_c"):
        if not np.array_equal(image.header[name], header[name]):
            raise ValueError("nibabel changed the source-defined MGH geometry")
    return image


@torch.inference_mode()
def _nearest_mask(mask, shape, pull, chunk_size):
    """Surfa C-float coordinates, pre-round FOV test, round-away ties.

    The generic world resampler uses grid_sample nearest, whose ties and
    outer half-voxel support differ. Reuse nibabel I/O, but keep this explicit
    sampler local to the source-defined preparation profile.
    """
    shape = tuple(int(v) for v in shape)
    output = torch.empty(prod(shape), dtype=torch.bool, device=mask.device)
    matrix = torch.as_tensor(np.asarray(pull, dtype=np.float32), device=mask.device)
    for start in range(0, output.numel(), chunk_size):
        stop = min(start + chunk_size, output.numel())
        index = torch.arange(start, stop, dtype=torch.int64, device=mask.device)
        x = torch.div(index, shape[1] * shape[2], rounding_mode="floor").float()
        y = torch.remainder(torch.div(index, shape[2], rounding_mode="floor"), shape[1]).float()
        z = torch.remainder(index, shape[2]).float()
        coordinates = []
        valid = torch.ones(stop - start, dtype=torch.bool, device=mask.device)
        for axis in range(3):
            # Explicit operation order follows the pinned Cython expression.
            c = ((matrix[axis, 0] * x + matrix[axis, 1] * y)
                 + matrix[axis, 2] * z) + matrix[axis, 3]
            valid &= (c >= 0) & (c < mask.shape[axis])
            # Valid coordinates are nonnegative. Test the fractional part
            # rather than add .5 in F32, which can itself round a point just
            # below the half to the next integer. No Double GPU volume is used.
            floor = torch.floor(c)
            selected = (floor + (c - floor >= .5)).long()
            coordinates.append(selected.clamp(0, mask.shape[axis] - 1))
        output[start:stop] = mask[tuple(coordinates)] & valid
    return output.reshape(shape)


def _crop(mask, geometry, margin):
    lower, upper = [], []
    if bool(mask.any()):
        for axis in range(3):
            occupied = mask.any(dim=tuple(v for v in range(3) if v != axis))
            positions = occupied.nonzero().flatten()
            lower.append(max(int(positions[0]) - margin[axis], 0))
            upper.append(min(int(positions[-1]) + margin[axis] + 1, mask.shape[axis]))
    else:
        lower, upper = [0] * 3, list(mask.shape)
    cropped = mask[tuple(slice(a, b) for a, b in zip(lower, upper))]
    shape = np.asarray(cropped.shape)
    corner_matrix = geometry.affine.copy()
    corner_matrix[:3, 3] = np.dot(geometry.affine, np.append(lower, 1))[:3]
    center = (corner_matrix @ np.append(shape / 2, 1))[:3]
    affine = _compose(shape, geometry.voxel_sizes, geometry.rotation, center)
    return cropped, _Geometry(shape, geometry.voxel_sizes, geometry.rotation, center, affine), lower, upper


@torch.inference_mode()
def prepare_subregion_alignment_target(
    coarse_segmentation,
    target_label_ids,
    *,
    target_voxel_mm: float = 1.0,
    bbox_margin_voxels: int | tuple[int, int, int] = 6,
    smoothing: str | None = "backward",
    device: str | torch.device = "cpu",
    spatial_chunk_size: int = 262144,
    memory_budget_gb: float = 20.0,
) -> AlignmentTargetPreparation:
    """Select labels, optionally resize, crop and smooth a 0/255 target.

    Labels are selected in the input image's voxel grid. As in SAMSEG, resize
    runs only if the input mean voxel size is below .99 mm. The requested
    isotropic grid keeps its shape/2 world center and uses ceil dimensions.
    The radius-one voxel sphere includes the center and six face neighbours.
    ``backward`` is erosion with exterior=True then dilation with exterior=False;
    ``forward`` reverses them; None skips them. Output is a nibabel MGHImage
    with float32 values and source-defined RAS-mm geometry. No registration,
    label inference, model download, GEMS fit or external program is called.
    """
    started = monotonic()
    if (not np.isfinite(target_voxel_mm) or target_voxel_mm <= 0
            or not isinstance(spatial_chunk_size, int) or spatial_chunk_size < 1
            or not np.isfinite(memory_budget_gb) or memory_budget_gb <= 0):
        raise ValueError("voxel size, chunk size and memory budget must be positive")
    if smoothing not in ("backward", "forward", None):
        raise ValueError("smoothing must be backward, forward, or None")
    labels = np.asarray(target_label_ids)
    if labels.ndim != 1 or labels.size == 0 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("target_label_ids must be a nonempty integer sequence")
    margin = np.repeat(bbox_margin_voxels, 3) if np.isscalar(bbox_margin_voxels) else np.asarray(bbox_margin_voxels)
    if margin.shape != (3,) or not np.issubdtype(margin.dtype, np.integer) or np.any(margin < 0):
        raise ValueError("bbox_margin_voxels must contain nonnegative integers")
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("device must be CPU or CUDA")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable")
    image = load_image(coarse_segmentation, "coarse_segmentation")
    if image.ndim != 3:
        raise ValueError("coarse_segmentation must be one 3D label image")
    geometry = _geometry(image)
    original_geometry = geometry
    resize = float(geometry.voxel_sizes.mean()) < .99
    sizes = np.repeat(float(target_voxel_mm), 3)
    resize_applied = resize and not np.allclose(geometry.voxel_sizes, sizes, atol=1e-5, rtol=0)
    raw_shape = (np.ceil(geometry.voxel_sizes * geometry.shape / target_voxel_mm)
                 if resize_applied else geometry.shape)
    if not np.isfinite(raw_shape).all() or np.any(raw_shape < 1) or np.any(raw_shape > 2**31 - 1):
        raise MemoryError("requested grid dimensions exceed the preparation bounds")
    target_shape = raw_shape.astype(np.int64)
    # Bound full label input, bool masks, cropped float32 output and chunk
    # coordinates before reading data or allocating a device tensor.
    estimated_bytes = (prod(int(v) for v in geometry.shape) * (2 * image.header.get_data_dtype().itemsize + 2)
                       + prod(int(v) for v in target_shape) * 12 + spatial_chunk_size * 96)
    if estimated_bytes > memory_budget_gb * 1e9:
        raise MemoryError("preparation estimate exceeds memory_budget_gb")
    array = np.asarray(image.dataobj)
    if not np.isfinite(array).all():
        raise ValueError("coarse_segmentation must be finite")
    array = np.array(array, dtype=array.dtype.newbyteorder("="), copy=True)
    values = torch.as_tensor(array, device=device)
    ids = torch.as_tensor(labels.astype(np.int64), device=device)
    mask = torch.isin(values, ids)
    del values
    input_foreground = int(mask.count_nonzero())
    read_select_seconds = monotonic() - started
    resize_started = monotonic()
    if resize_applied:
        affine = _compose(target_shape, sizes, geometry.rotation, geometry.center)
        mask = _nearest_mask(mask, target_shape, np.linalg.inv(geometry.affine) @ affine, spatial_chunk_size)
        geometry = _Geometry(target_shape, sizes, geometry.rotation, geometry.center, affine)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    resize_seconds = monotonic() - resize_started
    crop_started = monotonic()
    mask, geometry, lower, upper = _crop(mask, geometry, margin)
    before_smoothing = int(mask.count_nonzero())
    crop_seconds = monotonic() - crop_started
    morphology_started = monotonic()
    if smoothing is not None:
        # Reuse the mature bool six-connected FNIT primitive. Complementing
        # zero-exterior dilation implements the required one-exterior erosion.
        from ..connectome.masks import maskfilter_six_connected
        erode = lambda v: ~maskfilter_six_connected(~v, "dilate", 1)
        dilate = lambda v: maskfilter_six_connected(v, "dilate", 1)
        mask = dilate(erode(mask)) if smoothing == "backward" else erode(dilate(mask))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    morphology_seconds = monotonic() - morphology_started
    output = mask.cpu().numpy().astype(np.float32) * np.float32(255)
    result = _mgh_image(output, geometry, image)
    report = {
        "scope": "target_preparation_only_no_registration", "device": str(device),
        "input_shape": original_geometry.shape.tolist(),
        "input_voxel_sizes_mm": original_geometry.voxel_sizes.tolist(),
        "target_label_ids": [int(v) for v in labels], "resize_triggered": resize,
        "resize_applied": bool(resize_applied),
        "resize_threshold_mean_mm": .99, "resize_target_voxel_mm": float(target_voxel_mm),
        "resized_shape": [int(v) for v in target_shape],
        "bbox_lower": lower, "bbox_upper_exclusive": upper,
        "bbox_margin_voxels": margin.tolist(), "smoothing": smoothing,
        "input_selected_voxels": input_foreground, "cropped_selected_voxels": before_smoothing,
        "output_selected_voxels": int(np.count_nonzero(output)),
        "output_shape": list(output.shape), "output_affine_RAS_mm": geometry.affine.tolist(),
        "output_dtype": "float32", "mask_tensor_dtype": "bool",
        "interpolation_coordinate_dtype": "float32", "small_geometry_dtype": "float64",
        "estimated_workspace_bytes": estimated_bytes,
        "seconds": {"read_select": read_select_seconds, "resize": resize_seconds,
                    "crop": crop_seconds, "morphology": morphology_seconds,
                    "api_total": monotonic() - started},
    }
    return AlignmentTargetPreparation(result, report)


def reflect_atlas_header(atlas_image) -> MGHImage:
    """Reflect RAS row 0, retaining voxel values and their order.

    This reproduces the right-hemisphere SAMSEG preparation before the
    separate rigid and affine registrations. It does not resample the atlas.
    """
    image = load_image(atlas_image, "atlas_image")
    if image.ndim != 3:
        raise ValueError("atlas_image must be one 3D image")
    geometry = _geometry(image)
    reflected = geometry.affine.copy()
    reflected[0, :] *= -1
    # The installed setter decomposes its reflected affine with Double QR
    # while retaining the previously stored voxel sizes.
    q, r = np.linalg.qr(reflected[:3, :3])
    rotation = q @ np.diag(np.diag(r) / np.abs(np.diag(r)))
    center = (reflected @ np.append(geometry.shape / 2, 1))[:3]
    updated = _Geometry(geometry.shape, geometry.voxel_sizes, rotation, center, reflected)
    array = np.array(image.dataobj, copy=True)
    array = array.astype(array.dtype.newbyteorder("="), copy=False)
    return _mgh_image(array, updated, image)
