"""PyTorch FLIRT implementation with FSL image and matrix contracts."""

from .coordinates import (
    flirt_to_world_affine,
    flirt_to_world_pull,
    voxel_to_fsl_scaled_mm,
    world_to_flirt_affine,
)


def __getattr__(name):
    if name in {
        "FSLCorrelationRatio",
        "FSL_FLIRT_COMMIT",
        "FSL_FLIRT_VERSION",
        "TorchFLIRT",
        "fsl_affine_from_parameters",
        "fsl_coordinate_optimize",
        "fsl_parameters_from_affine",
    }:
        from importlib import import_module
        module = import_module(".core", __name__)
        return getattr(module, name)
    if name == "FLIRTResult":
        from .types import FLIRTResult
        return FLIRTResult
    if name == "run_flirt":
        from .standalone import run_flirt
        return run_flirt
    raise AttributeError(name)

__all__ = [
    "FLIRTResult",
    "FSLCorrelationRatio",
    "FSL_FLIRT_COMMIT",
    "FSL_FLIRT_VERSION",
    "TorchFLIRT",
    "flirt_to_world_affine",
    "flirt_to_world_pull",
    "fsl_affine_from_parameters",
    "fsl_coordinate_optimize",
    "fsl_parameters_from_affine",
    "voxel_to_fsl_scaled_mm",
    "world_to_flirt_affine",
    "run_flirt",
]
