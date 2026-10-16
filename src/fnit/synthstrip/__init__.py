"""Brain extraction using the official SynthStrip checkpoints."""
from .model import ConvBlock, StripModel
from .pipeline import StripResult, SynthStrip, extend_sdt

__all__ = ["ConvBlock", "StripModel", "StripResult", "SynthStrip", "extend_sdt"]

# Unified public name; the mature implementation and defaults are unchanged.
FNIT_SynthStrip = SynthStrip
__all__.append("FNIT_SynthStrip")
