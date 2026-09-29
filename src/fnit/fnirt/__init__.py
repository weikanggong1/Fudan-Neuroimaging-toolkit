"""GPU FNIRT registration components."""

from .io import (
    FSL_CUBIC_SPLINE_COEFFICIENTS,
    FSLFNIRTCoefficients,
    load_fsl_coefficients,
    make_fsl_coefficient_image,
    save_fsl_coefficients,
)


def __getattr__(name):
    if name in {
        "FSL_SOURCE_VERSIONS",
        "GMFNIRTConfig",
        "T1FNIRTConfig",
        "TorchFNIRT",
        "TorchFNIRTResult",
        "spm_like_mean",
    }:
        from importlib import import_module
        module = import_module(".registration", __name__)
        return getattr(module, name)
    raise AttributeError(name)

__all__ = [
    "FSL_SOURCE_VERSIONS",
    "FSL_CUBIC_SPLINE_COEFFICIENTS",
    "FSLFNIRTCoefficients",
    "GMFNIRTConfig",
    "T1FNIRTConfig",
    "TorchFNIRT",
    "TorchFNIRTResult",
    "load_fsl_coefficients",
    "make_fsl_coefficient_image",
    "save_fsl_coefficients",
    "spm_like_mean",
]
