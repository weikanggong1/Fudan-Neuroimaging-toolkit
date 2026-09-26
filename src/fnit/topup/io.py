"""FSL TOPUP coefficient and image output contracts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np

from ..fnirt.spline import fsl_control_shape


FSL_TOPUP_CUBIC_SPLINE_COEFFICIENTS = 2016
FSL_TOPUP_FIELD = 2018


def image_path(value):
    """Resolve an FSL image basename using ``FSLOUTPUTTYPE``."""
    path = Path(value).expanduser()
    if path.name.endswith((".nii", ".nii.gz")):
        return path
    if path.suffix:
        raise ValueError("image output must be a basename, .nii, or .nii.gz")
    output_type = os.environ.get("FSLOUTPUTTYPE", "NIFTI_GZ").upper()
    extension = {"NIFTI": ".nii", "NIFTI_GZ": ".nii.gz"}.get(output_type)
    if extension is None:
        raise NotImplementedError(
            "extensionless output requires FSLOUTPUTTYPE=NIFTI or NIFTI_GZ"
        )
    return path.with_name(path.name + extension)


def make_topup_coefficient_image(
    coefficients,
    field_shape: Sequence[int],
    field_voxel_sizes: Sequence[float],
    knot_spacing: Sequence[int],
):
    """Create the intent-2016 coefficient file consumed by FSL applytopup/eddy."""
    field_shape = tuple(int(value) for value in field_shape)
    voxel_sizes = tuple(float(value) for value in field_voxel_sizes)
    knot_spacing = tuple(int(value) for value in knot_spacing)
    if not (len(field_shape) == len(voxel_sizes) == len(knot_spacing) == 3):
        raise ValueError("TOPUP field geometry must contain three values")
    if any(value <= 0 for value in (*field_shape, *voxel_sizes, *knot_spacing)):
        raise ValueError("TOPUP field geometry must be positive")
    values = np.asarray(coefficients, dtype=np.float32)
    expected = fsl_control_shape(field_shape, knot_spacing)
    if values.shape != expected:
        raise ValueError(f"coefficients must have shape {expected}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("coefficients must be finite")

    # FSL deliberately stores two incompatible geometries in this header:
    # knot spacing in pixdim and field size in qform offsets.  Calling
    # nibabel set_qform/set_sform would reconcile them and destroy the TOPUP
    # contract, so assign the raw NIfTI fields and construct with affine=None.
    header = nib.Nifti1Header()
    header.set_data_shape(values.shape)
    header.set_data_dtype(np.float32)
    header.set_slope_inter(1.0, 0.0)
    header.set_xyzt_units("mm", "sec")
    header["pixdim"][0] = 1
    header["pixdim"][1:4] = knot_spacing
    header["qform_code"] = 1
    header["quatern_b"] = 0
    header["quatern_c"] = 0
    header["quatern_d"] = 0
    header["qoffset_x"] = field_shape[0]
    header["qoffset_y"] = field_shape[1]
    header["qoffset_z"] = field_shape[2]
    header["sform_code"] = 0
    header["srow_x"] = (1, 0, 0, 0)
    header["srow_y"] = (0, 1, 0, 0)
    header["srow_z"] = (0, 0, 1, 0)
    header["intent_code"] = FSL_TOPUP_CUBIC_SPLINE_COEFFICIENTS
    header["intent_p1"] = voxel_sizes[0]
    header["intent_p2"] = voxel_sizes[1]
    header["intent_p3"] = voxel_sizes[2]
    header["intent_name"] = b""
    return nib.Nifti1Image(values, None, header=header)


def make_topup_jacobian_image(data, reference):
    """Match TOPUP ``--jacout`` Analyze-style NIfTI geometry."""
    values = np.asarray(data, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("a TOPUP Jacobian must be 3D")
    header = nib.Nifti1Header()
    header.set_data_shape(values.shape)
    header.set_data_dtype(np.float32)
    header.set_slope_inter(1.0, 0.0)
    header.set_xyzt_units("mm", "sec")
    header["pixdim"][0] = 1
    header["pixdim"][1:4] = reference.header.get_zooms()[:3]
    header["pixdim"][4:8] = 1
    header["qform_code"] = 0
    header["qoffset_x"] = 0
    header["qoffset_y"] = 0
    header["qoffset_z"] = 0
    header["sform_code"] = 0
    header["srow_x"] = (1, 0, 0, 0)
    header["srow_y"] = (0, 1, 0, 0)
    header["srow_z"] = (0, 0, 1, 0)
    return nib.Nifti1Image(values, None, header=header)


def make_output_image(data, reference, *, intent=None):
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    header.set_slope_inter(1.0, 0.0)
    image = nib.Nifti1Image(
        np.asarray(data, dtype=np.float32), reference.affine, header=header
    )
    image.set_qform(reference.get_qform(), code=int(reference.header["qform_code"]))
    image.set_sform(reference.get_sform(), code=int(reference.header["sform_code"]))
    if intent is not None:
        image.header["intent_code"] = int(intent)
        image.header["intent_p1"] = 0
        image.header["intent_p2"] = 0
        image.header["intent_p3"] = 0
        image.header["intent_name"] = b""
    return image


__all__ = [
    "FSL_TOPUP_CUBIC_SPLINE_COEFFICIENTS",
    "FSL_TOPUP_FIELD",
    "image_path",
    "make_output_image",
    "make_topup_coefficient_image",
    "make_topup_jacobian_image",
]
