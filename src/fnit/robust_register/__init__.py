"""Independent preparation for the source-defined robust registration workflow.

This module does not register images or alter the GEMS default pipeline.
"""

from .preparation import (
    AlignmentTargetPreparation,
    prepare_subregion_alignment_target,
    reflect_atlas_header,
)

__all__ = [
    "AlignmentTargetPreparation",
    "prepare_subregion_alignment_target",
    "reflect_atlas_header",
]
