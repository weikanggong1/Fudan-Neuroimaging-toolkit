"""FSL-compatible EDDY path on PyTorch GPU."""

from .core import EDDYConfig, EDDYResult, TorchEDDY
from .ukb import prepare_ukb_eddy, run_ukb_eddy

__all__ = ["EDDYConfig", "EDDYResult", "TorchEDDY", "prepare_ukb_eddy", "run_ukb_eddy"]
