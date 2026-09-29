"""Standalone PyTorch single-subject spatial PICA (MELODIC-style)."""

from .ica import ICAResult, decompose_spatial_ica
from .bids import MelodicBIDSResult, run_melodic_bids

__all__ = ["ICAResult", "decompose_spatial_ica", "MelodicBIDSResult", "run_melodic_bids"]
