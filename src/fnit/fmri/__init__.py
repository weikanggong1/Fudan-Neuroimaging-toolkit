"""BIDS volumetric fMRI preprocessing, ICA denoising, and confound regression."""

from .bids import BIDSInputs, locate_bids_inputs
from .bbr import BBRResult, register_bbr
from .end_to_end import FMRIPipelineResult, run_fmri_pipeline
from .normalization import T1MNIResult, register_t1_to_mni, resample_world
from .pipeline import FeatCoreResult, run_feat_core
from .ica import ICAResult, decompose_spatial_ica
from .aroma_pipeline import AromaResult, run_aroma_pipeline
from .aroma import classify_aroma, denoise_aroma
from .confounds import clean_confounds, motion_regressors

__all__ = [
    "BIDSInputs", "locate_bids_inputs", "BBRResult", "register_bbr",
    "FeatCoreResult", "run_feat_core",
    "FMRIPipelineResult", "run_fmri_pipeline", "T1MNIResult",
    "register_t1_to_mni", "resample_world",
    "ICAResult", "decompose_spatial_ica", "AromaResult", "run_aroma_pipeline",
    "classify_aroma", "denoise_aroma", "clean_confounds", "motion_regressors",
]
