"""Fudan Neuroimaging Toolkit."""
__version__ = "0.16.0"


def __getattr__(name):
    if name in ('SynthStrip', 'StripResult'):
        from . import synthstrip
        return getattr(synthstrip, name)
    if name in ('SynthMorph', 'RegistrationResult', 'apply_transform',
                'convert_warp_to_fsl'):
        from . import synthmorph
        return getattr(synthmorph, name)
    if name in ('WMHSynthSeg', 'WMHResult'):
        from . import wmh_synthseg
        return getattr(wmh_synthseg, name)
    if name in ('SynthSeg', 'SynthSegResult', 'SynthSegPlus', 'SynthSegPlusResult'):
        from . import synthseg_parc
        return getattr(synthseg_parc, name)
    if name in ('GEMSAtlas', 'TorchGEMS', 'TorchGEMSResult',
                'SubregionResult', 'segment_subregions'):
        from . import gems
        return getattr(gems, name)
    if name in ('SynthSR', 'SynthSRResult', 'SynthSRImage'):
        from . import synthsr
        return getattr(synthsr, name)
    if name in ('TorchFAST', 'FASTResult', 'FASTConfig', 'FASTTensorResult',
                'segment_t1'):
        from . import fast
        return getattr(fast, name)
    if name in ('TorchApplyWarp', 'ApplyWarpResult'):
        from importlib import import_module
        module = import_module('.applywarp', __name__)
        return getattr(module, name)
    if name in ('TorchConvertWarp', 'WarpFieldResult'):
        from importlib import import_module
        module = import_module('.convertwarp', __name__)
        return getattr(module, name)
    if name == 'TorchInvWarp':
        from importlib import import_module
        module = import_module('.invwarp', __name__)
        return getattr(module, name)
    if name in ('TorchMCFLIRT', 'MCFLIRTResult'):
        from importlib import import_module
        return getattr(import_module('.mcflirt', __name__), name)
    if name in ('FLIRTResult', 'TorchFLIRT',
                'flirt_to_world_affine', 'flirt_to_world_pull',
                'voxel_to_fsl_scaled_mm', 'world_to_flirt_affine'):
        from importlib import import_module
        module = import_module('.flirt', __name__)
        return getattr(module, name)
    if name in ('TorchFNIRT', 'TorchFNIRTResult', 'FNIRTConfig',
                'GMFNIRTConfig', 'T1FNIRTConfig', 'TBSSFNIRTConfig',
                'resolve_fnirt_config'):
        from importlib import import_module
        module = import_module('.fnirt', __name__)
        return getattr(module, name)
    if name in ('TorchTOPUP', 'TOPUPResult', 'TOPUPConfig', 'run_ukb_topup'):
        from importlib import import_module
        module = import_module('.topup', __name__)
        return getattr(module, name)
    if name in ('TorchEDDY', 'EDDYResult', 'EDDYConfig', 'run_ukb_eddy'):
        from importlib import import_module
        module = import_module('.eddy', __name__)
        return getattr(module, name)
    if name in ('TorchDTIFIT', 'DTIFITResult', 'select_shell'):
        from importlib import import_module
        module = import_module('.dtifit', __name__)
        return getattr(module, name)
    if name in ('TorchAMICONODDI', 'AMICONODDIResult', 'AMICONODDIConfig'):
        from importlib import import_module
        module = import_module('.amico_noddi', __name__)
        return getattr(module, name)
    if name in ('TorchMMORF', 'MMORFResult', 'MMORFConfig',
                'apply_mmorf_warp', 'run_mmorf'):
        from importlib import import_module
        module = import_module('.mmorf', __name__)
        return getattr(module, name)
    if name in ('DMRIPipeline', 'DMRIPipelineResult', 'STANDARD_MAP_NAMES'):
        from importlib import import_module
        module = import_module('.dmri_pipeline', __name__)
        return getattr(module, name)
    if name in ('TorchBEDPOSTX', 'BedpostXResult'):
        from importlib import import_module
        module = import_module('.bedpostx', __name__)
        return getattr(module, name)
    if name in ('TorchProbtrackX', 'ProbTrackXResult'):
        from importlib import import_module
        module = import_module('.probtrackx', __name__)
        return getattr(module, name)
    if name == 'run_fslmaths':
        from .fslmaths import run_fslmaths
        return run_fslmaths
    if name == 'convert_space':
        from .space_conversion import convert_space
        return convert_space
    if name == 'run_superbigflica':
        from .superbigflica import run_superbigflica
        return run_superbigflica
    if name in ('run_bwas', 'BWASResult'):
        from . import bwas
        return getattr(bwas, name)
    if name in ('FastVBM', 'FastVBMResult',
                'VBMRegistrationResult'):
        from . import fast_vbm
        return getattr(fast_vbm, name)
    if name in ('FeatCoreResult', 'run_feat_core', 'BIDSInputs', 'locate_bids_inputs',
                'BBRResult', 'register_bbr',
                'FMRIVolumeResult', 'fMRIVolume_pipeline', 'T1MNIResult',
                'register_t1_to_mni', 'resample_world',
                'ICAResult', 'decompose_spatial_ica', 'AromaResult',
                'run_aroma_pipeline', 'classify_aroma', 'denoise_aroma',
                'clean_confounds', 'motion_regressors',
                'SurfaceHemisphere', 'SurfaceProjectionResult',
                'create_fmriprep_cifti', 'run_fmriprep_surface_projection',
                'MSMSulcInputs', 'prepare_msmsulc_inputs', 'run_msmsulc',
                'FMRISurfaceResult', 'fMRISurface_pipeline',
                'T1SurfacePair', 'T1SurfaceGeometry', 'T1SurfacePreparation',
                'prepare_fmriprep_surface_inputs', 'prepare_t1w_surface_geometry'):
        from . import fmri
        return getattr(fmri, name)
    if name in ('UKBConnectome_pipeline', 'UKBConnectome', 'ConnectomeResult'):
        from . import connectome
        return getattr(connectome, name)
    raise AttributeError(name)
