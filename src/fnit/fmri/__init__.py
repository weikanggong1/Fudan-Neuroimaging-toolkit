"""BIDS volumetric fMRI preprocessing, ICA denoising, and confound regression."""

from .bids import BIDSInputs, locate_bids_inputs
from .pipeline import FeatCoreResult, run_feat_core
from .ica import ICAResult, decompose_spatial_ica
from .aroma_pipeline import AromaResult, run_aroma_pipeline
from .aroma import classify_aroma, denoise_aroma
from .confounds import clean_confounds, motion_regressors

__all__ = [
    "BIDSInputs", "locate_bids_inputs", "FeatCoreResult", "run_feat_core",
    "ICAResult", "decompose_spatial_ica", "AromaResult", "run_aroma_pipeline",
    "classify_aroma", "denoise_aroma", "clean_confounds", "motion_regressors",
]
