"""GPU FAST VBM Python API."""

from .pipeline import FastVBM, FastVBMResult, OUTPUT_FILENAMES
from .registration import VBMRegistrationResult

__all__ = [
    "FastVBMResult",
    "FastVBM",
    "OUTPUT_FILENAMES",
    "VBMRegistrationResult",
]

# Unified public name; the mature implementation and defaults are unchanged.
FNIT_fslvbm = FastVBM
__all__.append("FNIT_fslvbm")
