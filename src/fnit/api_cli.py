"""Unified console names forwarding to mature parsers without subprocesses."""

from importlib import import_module
import sys

COMMANDS = {'FNIT_fslmaths': ('fnit.fslmaths.cli', ()),
 'FNIT_flirt': ('fnit.flirt.cli', ()),
 'FNIT_fnirt': ('fnit.fnirt.cli', ()),
 'FNIT_applywarp': ('fnit.applywarp.cli', ()),
 'FNIT_convertwarp': ('fnit.convertwarp.cli', ()),
 'FNIT_invwarp': ('fnit.invwarp.cli', ()),
 'FNIT_convert_space': ('fnit.space_conversion', ()),
 'FNIT_mmorf': ('fnit.mmorf.cli', ()),
 'FNIT_SynthStrip': ('fnit.cli', ('synthstrip',)),
 'FNIT_SynthMorph': ('fnit.cli', ('synthmorph',)),
 'FNIT_SynthSeg': ('fnit.cli', ('synthseg',)),
 'FNIT_SynthSegPlus': ('fnit.cli', ('synthseg', '--parc')),
 'FNIT_SynthSR': ('fnit.cli', ('synthsr',)),
 'FNIT_WMHSynthSeg': ('fnit.cli', ('wmh-synthseg',)),
 'FNIT_fslfast': ('fnit.cli', ('fast',)),
 'FNIT_fs_segment_subregions': ('fnit.cli', ('segment-4-subregions',)),
 'FNIT_mri_robust_register': ('fnit.robust_register.cli', ()),
 'FNIT_recon_all': ('fnit.recon_all.native_free', ()),
 'FNIT_fslvbm': ('fnit.cli', ('fast-vbm',)),
 'FNIT_mcflirt': ('fnit.mcflirt.cli', ()),
 'FNIT_fMRIVolume_pipeline': ('fnit.fmri.cli', ('volume',)),
 'FNIT_fMRISurface_pipeline': ('fnit.fmri.cli', ('surface',)),
 'FNIT_newmsm': ('fnit.msm.cli', ()),
 'FNIT_melodic': ('fnit.api_cli', ()),
 'FNIT_mshbm': ('fnit.mshbm.cli', ()),
 'FNIT_topup': ('fnit.topup.cli', ()),
 'FNIT_eddy': ('fnit.eddy.cli', ()),
 'FNIT_dtifit': ('fnit.dtifit.cli', ()),
 'FNIT_noddi': ('fnit.amico_noddi.cli', ()),
 'FNIT_bedpostx': ('fnit.bedpostx.cli', ()),
 'FNIT_probtrackx2': ('fnit.probtrackx.cli', ()),
 'FNIT_dMRIPipeline': ('fnit.dmri_pipeline.cli', ()),
 'FNIT_UKBconnectomePipeline': ('fnit.cli', ('UKBConnectome_pipeline',))}


def _melodic_main(argv):
    import argparse
    parser = argparse.ArgumentParser(prog='FNIT_melodic', description='单次4D BOLD空间PICA；复用FNIT已有MELODIC实现。')
    parser.add_argument('--input-bold', required=True, help='4D NIfTI [X,Y,Z,T] BOLD')
    parser.add_argument('--brain-mask', required=True, help='同空间3D脑mask')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--n-components', type=int, default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--voxel-batch-size', type=int, default=8192)
    parser.add_argument('--max-iter', type=int, default=500)
    parser.add_argument('--tolerance', type=float, default=1e-3)
    parser.add_argument('--random-state', type=int, default=0)
    parser.add_argument('--mm-threshold', type=float, default=0.5)
    args = parser.parse_args(argv)
    from .melodic import FNIT_melodic
    result = FNIT_melodic(**vars(args))
    for key in ('component_maps', 'thresholded_maps', 'posterior_maps', 'mixing', 'frequency_power'):
        print(getattr(result, key))
    return None


def _msm_positional_method(arguments):
    """Locate the legacy position without mistaking option values for a method.

    The existing MSM CLI constructs its parser inside main rather than exposing
    a parser factory. This small parser consumes only its value/flag arities;
    choices, required arguments, file inputs and execution stay with that CLI.
    """
    import argparse
    parser = argparse.ArgumentParser(prog='FNIT_newmsm', add_help=False)
    parser.add_argument('method', nargs='?')
    for option in ('--inputs-json', '--output-dir', '--device', '--config',
                   '--execution', '--qc-policy', '--cpu-threads'):
        parser.add_argument(option)
    parser.add_argument('--no-parallel', action='store_true')
    parsed, _ = parser.parse_known_args(arguments)
    return parsed.method


def run_cli(name, argv=None):
    """Run one canonical command; algorithm parameters remain with its parser."""
    if name not in COMMANDS:
        raise ValueError(f'unknown FNIT command: {name}')
    arguments = list(sys.argv[1:] if argv is None else argv)
    if name == 'FNIT_newmsm':
        methods = []
        remaining = []
        iterator = iter(arguments)
        for argument in iterator:
            if argument == '--method':
                try:
                    methods.append(next(iterator))
                except StopIteration:
                    raise ValueError('--method requires msmsulc or msmall') from None
            elif argument.startswith('--method='):
                methods.append(argument.split('=', 1)[1])
            else:
                remaining.append(argument)
        positional = _msm_positional_method(remaining) is not None
        if len(methods) > 1 or (methods and positional):
            raise ValueError('select MSM method exactly once')
        if methods and methods[0] not in ('msmsulc', 'msmall'):
            raise ValueError('MSM method must be msmsulc or msmall')
        arguments = remaining if positional else [methods[0] if methods else 'msmsulc', *remaining]
    if name == 'FNIT_noddi':
        if any(a == '--method' or a.startswith('--method=') for a in arguments):
            if any(a == '--fit-method' or a.startswith('--fit-method=') for a in arguments):
                raise ValueError('use either --method or --fit-method, not both')
            arguments = ['--fit-method' + a[len('--method'):] if a == '--method' or a.startswith('--method=') else a for a in arguments]
    if name == 'FNIT_melodic':
        return _melodic_main(arguments)
    module_name, prefix = COMMANDS[name]
    status = import_module(module_name).main([*prefix, *arguments])
    return status if isinstance(status, (int, type(None))) else 0


def _entrypoint(name):
    def entry(argv=None):
        return run_cli(name, argv)
    entry.__name__ = name
    entry.__doc__ = f'Canonical command for {name}; see its --help.'
    return entry


for _name in COMMANDS:
    globals()[_name] = _entrypoint(_name)

__all__ = tuple(COMMANDS)
