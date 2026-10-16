"""FSL 2111 EDDY volume correction on PyTorch GPU."""

from .fsl2111_strict import FSL2111Config as EDDYConfig
from .fsl2111_strict import StrictEDDYResult as EDDYResult
from .fsl2111_strict import TorchEDDYFSL2111 as TorchEDDY
from .ukb import run_ukb_eddy

__all__ = ["EDDYConfig", "EDDYResult", "TorchEDDY", "run_ukb_eddy"]

# 统一公开名称；保留成熟实现对象及其构造、调用和运行方法。
FNIT_eddy = TorchEDDY
__all__.append("FNIT_eddy")
