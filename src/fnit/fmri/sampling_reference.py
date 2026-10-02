"""T1w sampling grid at the native BOLD resolution.

The grid follows NiWorkflows 1.14.4 GenerateSamplingReference: RAS axes,
rounded native voxel sizes, and a brain bounding box padded by two voxels.

Modified in FNIT (2026) for direct nibabel/NumPy I/O and validation. Original
grid semantics: NiWorkflows (Copyright The NiPreps Developers, Apache 2.0)
and Nilearn 0.11.1 (Copyright The nilearn developers, BSD 3-Clause).
Full licenses and attribution are preserved in THIRD_PARTY_NOTICES.md.
"""

from itertools import product
from pathlib import Path

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np


def native_bold_sampling_reference(fixed_image, moving_image, fov_mask, output):
    """Write a 3D T1w reference at BOLD resolution without a workflow runtime."""
    fixed = nib.load(str(fixed_image))
    moving = nib.as_closest_canonical(nib.load(str(moving_image)))
    mask = nib.load(str(fov_mask))
    if fixed.ndim != 3 or mask.shape != fixed.shape or not np.allclose(
        mask.affine, fixed.affine, atol=1e-5, rtol=0
    ):
        raise ValueError("fov_mask must match the 3D T1w image")
    zooms = np.round(moving.header.get_zooms()[:3], 3)
    if not np.isfinite(zooms).all() or np.any(zooms <= 0):
        raise ValueError("BOLD voxel sizes must be finite and positive")
    affine = np.diag([*zooms, 1.0])
    corners = np.array(list(product(*[(0, n - 1) for n in fixed.shape])))
    world = nib.affines.apply_affine(fixed.affine, corners)
    coordinates = world / zooms
    lower = coordinates.min(axis=0)
    affine[:3, 3] = lower * zooms
    shape = tuple((np.ceil(coordinates.max(axis=0) - lower) + 1).astype(int))
    sampled_mask = resample_from_to(mask, (shape, affine), order=0,
                                   mode="constant", cval=0)
    values = np.asarray(sampled_mask.dataobj)
    if not np.isfinite(values).all():
        raise ValueError("fov_mask contains nonfinite values")
    bounds = np.argwhere(values > 0)
    if not len(bounds):
        raise ValueError("fov_mask is empty on the native BOLD sampling grid")
    origin = np.maximum(bounds.min(axis=0) - 2, 0)
    end = np.minimum(bounds.max(axis=0) + 2, np.asarray(shape) - 1)
    affine[:3, 3] += affine[:3, :3] @ origin
    shape = tuple(end - origin + 1)
    sampled = resample_from_to(fixed, (shape, affine), order=0,
                              mode="constant", cval=0)
    # Locked Nilearn resampling creates a fresh NIfTI header (sform code 2).
    code = 2
    sampled.set_qform(affine, code)
    sampled.set_sform(affine, code)
    sampled.header["descrip"] = "FNIT native BOLD resolution T1w sampling reference"
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    nib.save(sampled, str(destination))
    return destination
