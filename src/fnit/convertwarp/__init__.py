"""Compose FSL affine and nonlinear transforms in PyTorch."""

from .core import TorchConvertWarp, WarpFieldResult, convertwarp

__all__ = ["TorchConvertWarp", "WarpFieldResult", "convertwarp"]

# Unified public name; the mature implementation and defaults are unchanged.
FNIT_convertwarp = convertwarp
__all__.append("FNIT_convertwarp")
