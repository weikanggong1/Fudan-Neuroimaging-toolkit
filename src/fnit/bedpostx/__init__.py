"""Bayesian crossing-fibre estimation for diffusion MRI."""

from .core import BedpostXConfig, BedpostXResult, TorchBEDPOSTX

__all__ = ["BedpostXConfig", "BedpostXResult", "TorchBEDPOSTX"]

# 统一公开名称；保留成熟实现对象及其构造、调用和运行方法。
FNIT_bedpostx = TorchBEDPOSTX
__all__.append("FNIT_bedpostx")
