"""Convert a SynthMorph RAS pull field to an FSL relative displacement field."""

from __future__ import annotations

import nibabel as nib
import numpy as np

from .._nib import FNITNifti1Image, load_image
from .._transforms import DenseWarp, same_geometry
from ..flirt.coordinates import voxel_to_fsl_scaled_mm


def convert_warp_to_fsl(warp, *, moving, fixed):
    """Return a fixed-grid FSL intent-2006 warp for moving-to-fixed resampling.

    ``warp`` is SynthMorph's fixed-grid world-RAS pull displacement. Both
    images are required because FSL scaled-mm coordinates depend on each
    image's voxel size, storage handedness and shape.
    """
    moving = load_image(moving, "moving")
    fixed = load_image(fixed, "fixed")
    if len(moving.shape) != 3 or len(fixed.shape) != 3:
        raise ValueError("moving and fixed must be single-frame 3D images")
    if not isinstance(warp, DenseWarp):
        warp = load_image(warp, "warp")
        if isinstance(warp, (nib.Nifti1Image, nib.Nifti2Image)) and int(
            warp.header["intent_code"]
        ) == 2006:
            raise ValueError("warp is already an FSL intent-2006 field")
    if tuple(warp.shape) != (*fixed.shape, 3):
        raise ValueError("warp must have fixed shape (X, Y, Z, 3)")
    if isinstance(warp, DenseWarp) and (
        not same_geometry(warp.source, moving)
        or not same_geometry(warp.target, fixed)
    ):
        raise ValueError("warp source/target geometry does not match moving/fixed")
    if not same_geometry(warp, fixed):
        raise ValueError("warp grid does not match fixed image")
    displacement = np.asarray(warp.dataobj, dtype=np.float32)
    if not np.isfinite(displacement).all():
        raise ValueError("warp must contain only finite values")

    moving_fsl = voxel_to_fsl_scaled_mm(
        moving.affine, moving.shape, moving.header.get_zooms()[:3]
    )
    fixed_fsl = voxel_to_fsl_scaled_mm(
        fixed.affine, fixed.shape, fixed.header.get_zooms()[:3]
    )
    ras_to_moving_fsl = moving_fsl @ np.linalg.inv(moving.affine)
    base = ras_to_moving_fsl @ fixed.affine - fixed_fsl
    result = np.empty(displacement.shape, dtype=np.float32)
    for start in range(0, fixed.shape[2], 16):
        end = min(start + 16, fixed.shape[2])
        slab = np.einsum(
            "ab,...b->...a",
            ras_to_moving_fsl[:3, :3],
            displacement[:, :, start:end],
            optimize=True,
        )
        slab += base[:3, 3]
        for axis, (low, high) in enumerate(
            ((0, fixed.shape[0]), (0, fixed.shape[1]), (start, end))
        ):
            coordinate = np.arange(low, high, dtype=np.float64)
            view = [1, 1, 1, 1]
            view[axis] = high - low
            slab += coordinate.reshape(view) * base[:3, axis]
        result[:, :, start:end] = slab

    header = fixed.header.copy()
    header.set_data_dtype(np.float32)
    header["intent_code"] = 2006  # FSL_FNIRT_DISPLACEMENT_FIELD
    output = FNITNifti1Image(result, fixed.affine, header)
    qform, qcode = fixed.get_qform(coded=True)
    sform, scode = fixed.get_sform(coded=True)
    output.set_qform(qform, int(qcode))
    output.set_sform(sform, int(scode))
    return output


__all__ = ["convert_warp_to_fsl"]
