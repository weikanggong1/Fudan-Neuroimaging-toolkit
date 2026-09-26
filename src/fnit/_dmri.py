"""Shared diffusion-MRI file and geometry helpers."""

from __future__ import annotations

from pathlib import Path
import os

import nibabel as nib
import numpy as np
import torch


def load_bvals(path):
    values = np.loadtxt(os.fspath(path), dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("b-values must be finite and non-negative")
    return values


def load_bvecs(path, count=None):
    values = np.loadtxt(os.fspath(path), dtype=np.float64)
    if values.ndim == 1 and values.size == 3:
        values = values.reshape(3, 1)
    if values.ndim != 2:
        raise ValueError("b-vectors must be a 3xN or Nx3 matrix")
    if values.shape[0] != 3 and values.shape[1] == 3:
        values = values.T
    if values.shape[0] != 3 or (count is not None and values.shape[1] != count):
        raise ValueError("b-vectors must be a 3xN matrix matching the DWI")
    if not np.isfinite(values).all():
        raise ValueError("b-vectors must be finite")
    norms = np.linalg.norm(values, axis=0)
    nonzero = norms > 0
    values[:, nonzero] /= norms[nonzero]
    return values


def output_path(value):
    path = Path(value).expanduser()
    if path.name.endswith((".nii", ".nii.gz")):
        return path
    if path.suffix:
        raise ValueError("image output must be a basename, .nii, or .nii.gz")
    return path.with_name(
        path.name
        + (
            ".nii"
            if os.environ.get("FSLOUTPUTTYPE", "NIFTI_GZ").upper() == "NIFTI"
            else ".nii.gz"
        )
    )


def image_like(data, reference, *, intent=None):
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    header.set_slope_inter(1.0, 0.0)
    image = nib.Nifti1Image(
        np.asarray(data, dtype=np.float32), reference.affine, header
    )
    image.set_qform(reference.get_qform(), int(reference.header["qform_code"]))
    image.set_sform(reference.get_sform(), int(reference.header["sform_code"]))
    if intent is not None:
        image.header.set_intent(intent)
    return image


def configure_device(device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    result = torch.device(device)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if result.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return result
