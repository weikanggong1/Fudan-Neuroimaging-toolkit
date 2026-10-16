"""Single-subject diffusion preprocessing and standard-space pipelines."""

from .pipeline import DMRIPipeline, DMRIPipelineResult, STANDARD_MAP_NAMES

__all__ = [
    "DMRIPipeline",
    "DMRIPipelineResult",
    "STANDARD_MAP_NAMES",
]

# 统一公开名称；保留成熟实现对象及其构造、调用和运行方法。
FNIT_dMRIPipeline = DMRIPipeline
__all__.append("FNIT_dMRIPipeline")
