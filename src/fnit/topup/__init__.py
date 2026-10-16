"""PyTorch TOPUP and UK Biobank AP/PA preparation."""

from .core import (
    FSL_TOPUP_COMMIT,
    FSL_TOPUP_VERSION,
    TOPUPConfig,
    TOPUPResult,
    TorchTOPUP,
)
from .io import (
    FSL_TOPUP_CUBIC_SPLINE_COEFFICIENTS,
    FSL_TOPUP_FIELD,
    make_topup_coefficient_image,
    make_topup_jacobian_image,
)
from .ukb import run_ukb_topup

__all__ = [
    "FSL_TOPUP_COMMIT",
    "FSL_TOPUP_CUBIC_SPLINE_COEFFICIENTS",
    "FSL_TOPUP_VERSION",
    "FSL_TOPUP_FIELD",
    "TOPUPConfig",
    "TOPUPResult",
    "TorchTOPUP",
    "make_topup_coefficient_image",
    "make_topup_jacobian_image",
    "run_ukb_topup",
]

# 统一公开名称；保留成熟实现对象及其构造、调用和运行方法。
FNIT_topup = TorchTOPUP
__all__.append("FNIT_topup")
