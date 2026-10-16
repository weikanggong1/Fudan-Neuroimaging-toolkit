"""WMH-SynthSeg anatomy and lesion segmentation."""

from .pipeline import LABEL_IDS, LABEL_NAMES, WMHResult, WMHSynthSeg

__all__ = ["LABEL_IDS", "LABEL_NAMES", "WMHResult", "WMHSynthSeg"]

# Unified public name; the mature implementation and defaults are unchanged.
FNIT_WMHSynthSeg = WMHSynthSeg
__all__.append("FNIT_WMHSynthSeg")
