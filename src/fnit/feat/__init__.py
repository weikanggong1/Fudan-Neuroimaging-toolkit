"""FEAT-compatible temporal preprocessing subfunctions."""

from .temporal import (
    gaussian_highpass,
    gaussian_highpass_matrix,
    grand_mean_scale,
    highpass_nifti,
    scale_nifti,
)

__all__ = [
    "gaussian_highpass",
    "gaussian_highpass_matrix",
    "grand_mean_scale",
    "highpass_nifti",
    "scale_nifti",
]
