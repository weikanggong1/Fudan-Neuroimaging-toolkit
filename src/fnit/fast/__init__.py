"""Single-channel T1 tissue segmentation and bias correction."""

from .algorithm import FASTConfig, FASTTensorResult, segment_t1
from .pipeline import FASTResult, TorchFAST

__all__ = [
    "FASTConfig", "FASTResult", "FASTTensorResult", "TorchFAST", "segment_t1",
]

# Unified public name; the mature implementation and defaults are unchanged.
FNIT_fslfast = TorchFAST
__all__.append("FNIT_fslfast")
