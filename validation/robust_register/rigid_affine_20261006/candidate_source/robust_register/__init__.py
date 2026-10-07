"""Independent preparation for the source-defined robust registration workflow.

Experimental registration is explicit and does not alter the GEMS default.
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
    "RobustRegistrationResult",
    "robust_register",
    "robust_rigid_affine",
]


def __getattr__(name):
    # A preparation users do not import the optional registration/QR helpers.
    if name in ("RobustRegistrationResult", "robust_register", "robust_rigid_affine"):
        from . import registration
        return getattr(registration, name)
    raise AttributeError(name)
