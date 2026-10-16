"""Pure PyTorch SynthSeg 2.0 segmentation and volumetric parcellation."""

from .pipeline import SynthSegParc
from .segment import SynthSegParcResult, SynthSegSegmenter, run_synthseg_parc_t1
from .synthseg import SynthSeg, SynthSegResult
from .synthseg_plus import SynthSegPlus, SynthSegPlusResult

__all__ = ["SynthSegParc", "SynthSegParcResult", "SynthSegSegmenter", "run_synthseg_parc_t1",
           "SynthSeg", "SynthSegResult", "SynthSegPlus", "SynthSegPlusResult"]

# Preserve reusable model instances under the unified public names.
FNIT_SynthSeg = SynthSeg
FNIT_SynthSegPlus = SynthSegPlus
__all__.extend(["FNIT_SynthSeg", "FNIT_SynthSegPlus"])
