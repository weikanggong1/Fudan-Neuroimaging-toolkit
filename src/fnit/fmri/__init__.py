"""BIDS volumetric and fsLR32k fMRI preprocessing."""

from .bids import BIDSInputs, locate_bids_inputs
from .bbr import BBRResult, register_bbr
from .pipeline import FeatCoreResult, run_feat_core
from .ica import ICAResult, decompose_spatial_ica
from .aroma_pipeline import AromaResult, run_aroma_pipeline
from .aroma import classify_aroma, denoise_aroma
from .confounds import clean_confounds, motion_regressors
from .surface import SurfaceHemisphere, SurfaceProjectionResult, run_surface_projection
from .surface_qc import SurfaceQCResult, make_ribbon_goodvoxels
from .surface_msmsulc import run_msmsulc
from .surface_registration import MSMSulcInputs, prepare_msmsulc_inputs
from .surface_pipeline import SurfacePipelineInputs, SurfacePipelineResult, run_surface_from_mni, run_surface_from_volume
from .surface_prepare import (MNISurfacePair, MNISurfaceResult, SurfacePreparationResult,
                              prepare_mni_surface_geometry, prepare_fs_sphere_projection_inputs)


def __getattr__(name):
    if name in ("FMRIPipelineResult", "run_fmri_pipeline"):
        from . import end_to_end
        return getattr(end_to_end, name)
    if name in ("T1MNIResult", "register_t1_to_mni", "resample_world"):
        from . import normalization
        return getattr(normalization, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BIDSInputs", "locate_bids_inputs", "BBRResult", "register_bbr",
    "FeatCoreResult", "run_feat_core",
    "FMRIPipelineResult", "run_fmri_pipeline", "T1MNIResult",
    "register_t1_to_mni", "resample_world",
    "ICAResult", "decompose_spatial_ica", "AromaResult", "run_aroma_pipeline",
    "classify_aroma", "denoise_aroma", "clean_confounds", "motion_regressors",
    "SurfaceHemisphere", "SurfaceProjectionResult", "run_surface_projection",
    "SurfaceQCResult", "make_ribbon_goodvoxels",
    "MSMSulcInputs", "prepare_msmsulc_inputs", "run_msmsulc",
    "SurfacePipelineInputs", "SurfacePipelineResult", "run_surface_from_mni", "run_surface_from_volume",
    "MNISurfacePair", "MNISurfaceResult", "SurfacePreparationResult",
    "prepare_mni_surface_geometry", "prepare_fs_sphere_projection_inputs",
]
