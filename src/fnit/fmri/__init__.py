"""BIDS volumetric and fsLR32k fMRI preprocessing."""

from .bids import BIDSInputs, locate_bids_inputs
from .bbr import BBRResult, register_bbr
from .pipeline import FeatCoreResult, run_feat_core
from ..melodic import ICAResult, decompose_spatial_ica
from .aroma_pipeline import AromaResult, run_aroma_pipeline
from .aroma import classify_aroma, denoise_aroma
from .confounds import clean_confounds, motion_regressors
from .surface import SurfaceHemisphere, SurfaceProjectionResult
from .surface_fmriprep import create_fmriprep_cifti, run_fmriprep_surface_projection
from ..msm import MSMSulcInputs, prepare_msmsulc_inputs, run_msmsulc
from ..msm.config import MSMSulcConfig
from .surface_pipeline import FMRISurfaceResult, fMRISurface_pipeline
from .surface_volume import SurfaceVolumeStatus, inspect_surface_volume
from .surface_reconstruction import ReconstructionResult, prepare_surface_reconstruction
from .surface_prepare import (T1SurfacePair, T1SurfaceGeometry, T1SurfacePreparation,
                              prepare_fmriprep_surface_inputs, prepare_t1w_surface_geometry)


def __getattr__(name):
    if name in ("FMRIVolumeResult", "fMRIVolume_pipeline"):
        from . import end_to_end
        return getattr(end_to_end, name)
    if name in ("T1MNIResult", "register_t1_to_mni", "resample_world"):
        from . import normalization
        return getattr(normalization, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BIDSInputs", "locate_bids_inputs", "BBRResult", "register_bbr",
    "FeatCoreResult", "run_feat_core",
    "FMRIVolumeResult", "fMRIVolume_pipeline", "T1MNIResult",
    "register_t1_to_mni", "resample_world",
    "ICAResult", "decompose_spatial_ica", "AromaResult", "run_aroma_pipeline",
    "classify_aroma", "denoise_aroma", "clean_confounds", "motion_regressors",
    "SurfaceHemisphere", "SurfaceProjectionResult",
    "create_fmriprep_cifti", "run_fmriprep_surface_projection",
    "MSMSulcConfig", "MSMSulcInputs", "prepare_msmsulc_inputs", "run_msmsulc",
    "FMRISurfaceResult", "fMRISurface_pipeline",
    "SurfaceVolumeStatus", "inspect_surface_volume",
    "ReconstructionResult", "prepare_surface_reconstruction",
    "T1SurfacePair", "T1SurfaceGeometry", "T1SurfacePreparation",
    "prepare_fmriprep_surface_inputs", "prepare_t1w_surface_geometry",
]
