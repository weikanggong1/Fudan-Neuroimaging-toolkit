"""Read and write FSL FNIRT cubic coefficient NIfTI files."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np

from .spline import fsl_control_shape


FSL_CUBIC_SPLINE_COEFFICIENTS = 2007


@dataclass(frozen=True)
class FSLFNIRTCoefficients:
    coefficients: np.ndarray
    field_shape: tuple[int, int, int]
    field_voxel_sizes: tuple[float, float, float]
    knot_spacing: tuple[int, int, int]
    affine_forward: np.ndarray
    image: nib.Nifti1Image | nib.Nifti2Image


def _triplet(values, name, cast):
    if len(values) != 3:
        raise ValueError(f"{name} must contain three values")
    result = tuple(cast(value) for value in values)
    if any(not np.isfinite(value) or value <= 0 for value in result):
        raise ValueError(f"{name} must contain positive finite values")
    return result


def _affine(value):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("affine_forward must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], (0, 0, 0, 1), atol=1e-8, rtol=0):
        raise ValueError("affine_forward must be homogeneous")
    if abs(float(np.linalg.det(matrix[:3, :3]))) < 1e-10:
        raise ValueError("affine_forward must be invertible")
    return matrix


def make_fsl_coefficient_image(
    coefficients,
    field_shape: Sequence[int],
    field_voxel_sizes: Sequence[float],
    knot_spacing: Sequence[int],
    affine_forward,
):
    """Create the coefficient-file representation written by FNIRT 2203.0.

    The unusual header contract is intentional: the sform stores the input to
    reference FLIRT affine, qform offsets store the dense field matrix size,
    pixdims store knot spacing, and intent parameters store dense field voxel
    sizes.  Coefficients are ordered ``[Cx, Cy, Cz, 3]``.
    """
    field_shape = _triplet(field_shape, "field_shape", int)
    field_voxel_sizes = _triplet(
        field_voxel_sizes, "field_voxel_sizes", float
    )
    knot_spacing = _triplet(knot_spacing, "knot_spacing", int)
    affine_forward = _affine(affine_forward)
    values = np.asarray(coefficients, dtype=np.float32)
    expected = (*fsl_control_shape(field_shape, knot_spacing), 3)
    if values.shape != expected:
        raise ValueError(
            f"coefficients must have shape {expected}, received {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("coefficients must contain only finite values")

    qform = np.diag([*knot_spacing, 1.0]).astype(np.float64)
    qform[:3, 3] = field_shape
    image = nib.Nifti1Image(values, affine_forward)
    image.set_qform(qform, code=1)
    image.set_sform(affine_forward, code=1)
    # Nibabel registers FSL intent 2007 as having no standard NIfTI
    # parameters, while FSL deliberately stores the dense voxel sizes there.
    # Assign the raw header fields to preserve that FSL extension.
    image.header["intent_code"] = FSL_CUBIC_SPLINE_COEFFICIENTS
    image.header["intent_p1"] = field_voxel_sizes[0]
    image.header["intent_p2"] = field_voxel_sizes[1]
    image.header["intent_p3"] = field_voxel_sizes[2]
    image.header["intent_name"] = b""
    image.header.set_data_dtype(np.float32)
    image.header.set_slope_inter(1.0, 0.0)
    # NEWIMAGE's NIfTI writer declares millimetres and seconds even for
    # coefficient volumes; nibabel otherwise leaves both units unknown.
    image.header.set_xyzt_units("mm", "sec")
    image.header["cal_min"] = 0
    image.header["cal_max"] = 0
    return image


def save_fsl_coefficients(
    filename,
    coefficients,
    field_shape: Sequence[int],
    field_voxel_sizes: Sequence[float],
    knot_spacing: Sequence[int],
    affine_forward,
):
    """Write a cubic coefficient file accepted by FSL FNIRT tools."""
    image = make_fsl_coefficient_image(
        coefficients,
        field_shape,
        field_voxel_sizes,
        knot_spacing,
        affine_forward,
    )
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(filename))
    return filename


def load_fsl_coefficients(value):
    """Load an FSL cubic coefficient file without spatial reorientation."""
    if isinstance(value, (str, os.PathLike)):
        image = nib.load(str(value))
    elif isinstance(value, (nib.Nifti1Image, nib.Nifti2Image)):
        image = value
    else:
        raise TypeError("value must be a NIfTI path or image")
    if int(image.header["intent_code"]) != FSL_CUBIC_SPLINE_COEFFICIENTS:
        raise ValueError("image is not an FSL cubic coefficient file")
    if len(image.shape) != 4 or image.shape[3] != 3:
        raise ValueError("coefficient image must have shape [Cx, Cy, Cz, 3]")
    field_shape = tuple(
        int(round(float(image.header[name])))
        for name in ("qoffset_x", "qoffset_y", "qoffset_z")
    )
    field_voxel_sizes = tuple(
        float(image.header[name])
        for name in ("intent_p1", "intent_p2", "intent_p3")
    )
    knot_spacing = tuple(
        int(np.floor(abs(float(value)) + 0.5))
        for value in image.header["pixdim"][1:4]
    )
    _triplet(field_shape, "field_shape", int)
    _triplet(field_voxel_sizes, "field_voxel_sizes", float)
    _triplet(knot_spacing, "knot_spacing", int)
    expected = (*fsl_control_shape(field_shape, knot_spacing), 3)
    if image.shape != expected:
        raise ValueError(
            "coefficient dimensions are inconsistent with FNIRT header metadata"
        )
    coefficients = np.asarray(image.dataobj, dtype=np.float32)
    if not np.isfinite(coefficients).all():
        raise ValueError("coefficients must contain only finite values")
    affine_forward = _affine(image.get_sform())
    return FSLFNIRTCoefficients(
        coefficients=coefficients,
        field_shape=field_shape,
        field_voxel_sizes=field_voxel_sizes,
        knot_spacing=knot_spacing,
        affine_forward=affine_forward,
        image=image,
    )


__all__ = [
    "FSL_CUBIC_SPLINE_COEFFICIENTS",
    "FSLFNIRTCoefficients",
    "load_fsl_coefficients",
    "make_fsl_coefficient_image",
    "save_fsl_coefficients",
]
