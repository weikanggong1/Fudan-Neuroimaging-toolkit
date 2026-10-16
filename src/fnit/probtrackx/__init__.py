"""Probabilistic tractography from FSL-compatible bedpostX samples."""

from .pipeline import ProbTrackXResult, TorchProbtrackX

__all__ = ["ProbTrackXResult", "TorchProbtrackX"]

# 统一公开名称；保留成熟实现对象及其构造、调用和运行方法。
FNIT_probtrackx2 = TorchProbtrackX
__all__.append("FNIT_probtrackx2")
