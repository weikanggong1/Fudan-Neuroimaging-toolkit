"""FSL-compatible EDDY path on PyTorch GPU."""

from .core import EDDYConfig, EDDYResult, TorchEDDY
from .ukb import run_ukb_eddy

__all__ = ["EDDYConfig", "EDDYResult", "TorchEDDY", "run_ukb_eddy"]
