"""Standalone surface imports do not require the volumetric Surfa backend."""

import subprocess
import sys


def test_surface_api_imports_without_surfa():
    code = """
import builtins
import importlib
import sys

original_import = builtins.__import__
def reject_surfa(name, *args, **kwargs):
    if name == "surfa" or name.startswith("surfa."):
        raise AssertionError("surface import reached Surfa")
    return original_import(name, *args, **kwargs)

builtins.__import__ = reject_surfa
surface = importlib.import_module("fnit.fmri.surface")
from fnit import run_surface_projection
from fnit.fmri import (
    SurfaceHemisphere, SurfacePipelineInputs, run_surface_from_mni,
    prepare_fs_sphere_projection_inputs,
)
assert run_surface_projection is surface.run_surface_projection
assert SurfaceHemisphere is surface.SurfaceHemisphere
assert callable(run_surface_from_mni)
assert callable(prepare_fs_sphere_projection_inputs)
assert "surfa" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_volumetric_public_api_still_resolves():
    code = """
import fnit.fmri as fmri
from fnit.fmri import end_to_end, normalization
assert fmri.FMRIPipelineResult is end_to_end.FMRIPipelineResult
assert fmri.run_fmri_pipeline is end_to_end.run_fmri_pipeline
assert fmri.T1MNIResult is normalization.T1MNIResult
assert fmri.register_t1_to_mni is normalization.register_t1_to_mni
assert fmri.resample_world is normalization.resample_world
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
