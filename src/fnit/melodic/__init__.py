"""Standalone PyTorch single-subject spatial PICA (MELODIC-style)."""

from .ica import ICAResult, decompose_spatial_ica
from .bids import MelodicBIDSResult, run_melodic_bids

__all__ = ["ICAResult", "decompose_spatial_ica", "MelodicBIDSResult", "run_melodic_bids"]

FNIT_melodic = decompose_spatial_ica
FNIT_melodic_bids = run_melodic_bids
__all__ += ["FNIT_melodic", "FNIT_melodic_bids"]
