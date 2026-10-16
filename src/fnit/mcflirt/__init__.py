"""原生 PyTorch MCFLIRT 刚体运动估计。"""
from .core import MCFLIRTResult, TorchMCFLIRT

__all__ = ["MCFLIRTResult", "TorchMCFLIRT"]

# Unified public name; the mature implementation and defaults are unchanged.
FNIT_mcflirt = TorchMCFLIRT
__all__.append("FNIT_mcflirt")
