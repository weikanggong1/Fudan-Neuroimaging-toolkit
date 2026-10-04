"""Acceptance diagnostics must not hide differing NaN/inf support."""
import importlib.util
from pathlib import Path

import nibabel as nib
import numpy as np


def test_nonfinite_mismatch_is_reported_even_when_finite_values_match(tmp_path):
    path = Path(__file__).resolve().parents[1] / "tools/benchmark_multimodal_cpu.py"
    spec = importlib.util.spec_from_file_location("multimodal_cpu_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference = np.array([1, np.nan, 2, np.inf, -np.inf], np.float32).reshape(5, 1, 1)
    candidate = np.array([1, 2, np.nan, -np.inf, -np.inf], np.float32).reshape(5, 1, 1)
    paths = [tmp_path / name for name in ["candidate.nii", "reference.nii"]]
    for values, destination in zip([candidate, reference], paths):
        nib.save(nib.Nifti1Image(values, np.eye(4)), destination)
    result = module.compare_outputs({"volume":paths[0]}, {"volume":paths[1]})["volume"]
    assert result["max_absolute_error"] == 0
    assert result["element_count"] == 5 and result["finite_pair_count"] == 1
    assert result["candidate_finite_count"] == result["reference_finite_count"] == 2
    assert result["nonfinite_pattern_equal"] is False
    assert result["nonfinite_pattern_mismatch_count"] == 3
