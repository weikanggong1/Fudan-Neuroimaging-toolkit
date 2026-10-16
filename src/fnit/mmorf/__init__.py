"""PyTorch scalar and diffusion-tensor nonlinear registration."""

from .core import MMORFConfig, MMORFResult, TorchMMORF, MMORFWarpPlan, apply_mmorf_warp, prepare_mmorf_warp
from .standalone import run_mmorf

__all__ = [
    "MMORFConfig",
    "MMORFResult",
    "MMORFWarpPlan",
    "TorchMMORF",
    "apply_mmorf_warp",
    "prepare_mmorf_warp",
    "run_mmorf",
]

# Unified public name; the mature implementation and defaults are unchanged.
FNIT_mmorf = run_mmorf
__all__.append("FNIT_mmorf")
