"""PyTorch implementation of the common FSL fslmaths operations."""

from .core import run_fslmaths

__all__ = ["run_fslmaths"]

# Unified public name; the mature implementation and defaults are unchanged.
FNIT_fslmaths = run_fslmaths
__all__.append("FNIT_fslmaths")
