"""Invert FSL deformation fields in PyTorch."""

from .core import TorchInvWarp, invwarp
from ..convertwarp.core import WarpFieldResult

__all__ = ["TorchInvWarp", "WarpFieldResult", "invwarp"]

# Unified public name; the mature implementation and defaults are unchanged.
FNIT_invwarp = invwarp
__all__.append("FNIT_invwarp")
