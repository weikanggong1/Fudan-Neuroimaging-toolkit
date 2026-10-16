"""SynthSR image synthesis."""

from .pipeline import SynthSR, SynthSRImage, SynthSRResult

__all__ = ["SynthSR", "SynthSRImage", "SynthSRResult"]

# Unified public name; the mature implementation and defaults are unchanged.
FNIT_SynthSR = SynthSR
__all__.append("FNIT_SynthSR")
