"""Standalone PyTorch brain MRI inference tools."""
__version__ = '0.10.0'


def __getattr__(name):
    if name in ('SynthStrip', 'StripResult'):
        from . import synthstrip
        return getattr(synthstrip, name)
    if name in ('SynthMorph', 'RegistrationResult', 'apply_transform'):
        from . import synthmorph
        return getattr(synthmorph, name)
    if name in ('WMHSynthSeg', 'WMHResult'):
        from . import wmh_synthseg
        return getattr(wmh_synthseg, name)
    if name in ('SynthSeg', 'SynthSegResult'):
        from . import synthseg_parc
        return getattr(synthseg_parc, name)
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
    if name in ('FLIRTResult', 'TorchFLIRT',
                'flirt_to_world_affine', 'flirt_to_world_pull',
                'voxel_to_fsl_scaled_mm', 'world_to_flirt_affine'):
        from importlib import import_module
        module = import_module('.flirt', __name__)
        return getattr(module, name)
    if name in ('TorchFNIRT', 'TorchFNIRTResult', 'GMFNIRTConfig'):
        from importlib import import_module
        module = import_module('.fnirt', __name__)
        return getattr(module, name)
    if name in ('TorchTOPUP', 'TOPUPResult', 'TOPUPConfig',
                'prepare_ukb_topup', 'run_ukb_topup'):
        from importlib import import_module
        module = import_module('.topup', __name__)
        return getattr(module, name)
    if name in ('FastVBM', 'FastVBMResult',
                'VBMRegistrationResult'):
        from . import fast_vbm
        return getattr(fast_vbm, name)
    raise AttributeError(name)
