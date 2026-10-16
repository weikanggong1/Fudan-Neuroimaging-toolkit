"""FSL-compatible diffusion tensor fitting with PyTorch."""

from .core import DTIFITResult, TorchDTIFIT, select_shell

__all__ = ["DTIFITResult", "TorchDTIFIT", "select_shell"]

# 统一公开名称；保留成熟实现对象及其构造、调用和运行方法。
FNIT_dtifit = TorchDTIFIT
__all__.append("FNIT_dtifit")
