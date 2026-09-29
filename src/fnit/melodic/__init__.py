"""Standalone PyTorch single-subject spatial PICA (MELODIC-style)."""

from .ica import ICAResult, decompose_spatial_ica

__all__ = ["ICAResult", "decompose_spatial_ica"]
