"""Public naming contracts; the scientific implementations are unchanged."""
from importlib import import_module
from pathlib import Path
import inspect

import pytest


BINDINGS = [
    ("fslmaths", "FNIT_fslmaths", "fnit.fslmaths", "run_fslmaths"),
    ("flirt", "FNIT_flirt", "fnit.flirt.standalone", "run_flirt"),
    ("fnirt", "FNIT_fnirt", "fnit.fnirt.standalone", "run_fnirt"),
    ("applywarp", "FNIT_applywarp", "fnit.applywarp", "applywarp"),
    ("convertwarp", "FNIT_convertwarp", "fnit.convertwarp", "convertwarp"),
    ("invwarp", "FNIT_invwarp", "fnit.invwarp", "invwarp"),
    ("space_conversion", "FNIT_convert_space", "fnit.space_conversion", "convert_space"),
    ("mmorf", "FNIT_mmorf", "fnit.mmorf", "run_mmorf"),
    ("synthstrip", "FNIT_SynthStrip", "fnit.synthstrip", "SynthStrip"),
    ("synthmorph", "FNIT_SynthMorph", "fnit.synthmorph", "SynthMorph"),
    ("synthseg_parc", "FNIT_SynthSeg", "fnit.synthseg_parc", "SynthSeg"),
    ("synthseg_parc", "FNIT_SynthSegPlus", "fnit.synthseg_parc", "SynthSegPlus"),
    ("synthsr", "FNIT_SynthSR", "fnit.synthsr", "SynthSR"),
    ("wmh_synthseg", "FNIT_WMHSynthSeg", "fnit.wmh_synthseg", "WMHSynthSeg"),
    ("fast", "FNIT_fslfast", "fnit.fast", "TorchFAST"),
    ("gems", "FNIT_fs_segment_subregions", "fnit.gems", "segment_4_subregions"),
    ("recon_all", "FNIT_recon_all", "fnit.recon_all.native_free", "run_recon_all_python"),
    ("fast_vbm", "FNIT_fslvbm", "fnit.fast_vbm", "FastVBM"),
    ("mcflirt", "FNIT_mcflirt", "fnit.mcflirt", "TorchMCFLIRT"),
]


@pytest.mark.parametrize("module,name,original_module,original", BINDINGS)
def test_public_name_is_original_callable(module, name, original_module, original):
    package = import_module("fnit." + module)
    canonical = getattr(package, name)
    mature = getattr(import_module(original_module), original)
    assert canonical is mature
    assert inspect.signature(canonical) == inspect.signature(mature)
    if hasattr(package, "__all__"):
        assert name in package.__all__


def test_removed_preparation_helpers_are_not_production_exports():
    module = import_module("fnit.robust_register")
    for name in ("prepare_subregion_alignment_target", "reflect_atlas_header",
                 "AlignmentTargetPreparation"):
        assert not hasattr(module, name)
        assert name not in module.__all__
    assert not Path(module.__file__).with_name("preparation.py").exists()


def test_recon_all_public_name_is_single_subject_not_batch():
    package = import_module("fnit.recon_all")
    signature = inspect.signature(package.FNIT_recon_all)
    assert list(signature.parameters)[:4] == ["t1", "subject_dir", "weights_dir", "assets_dir"]
    assert "jobs" not in signature.parameters
