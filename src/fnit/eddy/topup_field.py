"""Decode TOPUP spline coefficients on the EDDY input grid."""

from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from ..fnirt.spline import expand_coefficients, spline_bases


def _load_topup_field(prefix, shape, device, pe_axis):
    if prefix is None:
        zero = torch.zeros(tuple(shape), dtype=torch.float32, device=device)
        return zero, zero.clone(), None
    prefix = Path(prefix)
    coefficient_path = prefix.with_name(prefix.name + "_fieldcoef.nii.gz")
    if not coefficient_path.exists():
        coefficient_path = prefix.with_name(prefix.name + "_fieldcoef.nii")
    image = nib.load(str(coefficient_path))
    coefficients = torch.as_tensor(
        np.asarray(image.dataobj, dtype=np.float32), device=device
    )
    header = image.header
    field_shape = tuple(
        int(round(float(header[key])))
        for key in ("qoffset_x", "qoffset_y", "qoffset_z")
    )
    if field_shape != tuple(shape):
        raise ValueError(
            f"TOPUP coefficient field shape {field_shape} does not match DWI {shape}"
        )
    spacing = tuple(int(round(float(v))) for v in header["pixdim"][1:4])
    voxel_sizes = tuple(
        float(v)
        for v in (header["intent_p1"], header["intent_p2"], header["intent_p3"])
    )
    bases = spline_bases(shape, spacing, (1, 1, 1), device=device, dtype=torch.float32)
    field = expand_coefficients(coefficients[None], bases)[0]
    derivatives = [0, 0, 0]
    derivatives[pe_axis] = 1
    derivative_bases = spline_bases(
        shape,
        spacing,
        (1, 1, 1),
        device=device,
        dtype=torch.float32,
        derivatives=tuple(derivatives),
    )
    derivative = expand_coefficients(coefficients[None], derivative_bases)[0]
    return field, derivative, voxel_sizes
