"""Benchmark validation must use the same native reference as the public API."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest


_spec = importlib.util.spec_from_file_location(
    "fmri_benchmark_bids", Path(__file__).parents[1] / "validation/fmri/benchmark_bids.py"
)
benchmark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(benchmark)


def _save(path, shape, affine):
    image = nib.Nifti1Image(np.ones(shape, dtype=np.float32), affine)
    image.header.set_xyzt_units("mm", "sec")
    if len(shape) == 4:
        image.header.set_zooms((*image.header.get_zooms()[:3], .8))
    nib.save(image, path)
    return path


def test_native_benchmark_accepts_selected_sbref_grid(tmp_path):
    bold = _save(tmp_path / "bold.nii.gz", (4, 5, 6, 8), np.eye(4))
    sbref_affine = np.diag([2., 2., 2., 1.])
    sbref_affine[:3, 3] = [-4., 5., 8.]
    sbref = _save(tmp_path / "sbref.nii.gz", (5, 6, 7), sbref_affine)
    native = _save(tmp_path / "clean_native.nii.gz", (5, 6, 7, 8), sbref_affine)
    inputs = SimpleNamespace(bold=bold, sbref=sbref)

    checks = benchmark.check_native_volume(native, inputs)
    assert checks["shape"] == [5, 6, 7, 8]
    assert checks["nonfinite_values"] == 0
    with pytest.raises(ValueError, match="reference grid"):
        benchmark.check_volume(native, nib.load(bold))


def test_native_benchmark_without_sbref_uses_bold_grid(tmp_path):
    bold = _save(tmp_path / "bold.nii.gz", (4, 5, 6, 8), np.eye(4))
    native = _save(tmp_path / "clean_native.nii.gz", (4, 5, 6, 8), np.eye(4))
    checks = benchmark.check_native_volume(native, SimpleNamespace(bold=bold, sbref=None))
    assert checks["shape"] == [4, 5, 6, 8]
