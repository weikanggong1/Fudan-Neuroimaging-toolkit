"""Reporting regressions; fixtures here are not performance benchmarks."""
import importlib.util
from pathlib import Path

import numpy as np


def _metrics():
    path = Path(__file__).resolve().parents[2] / "validation/topup/benchmark_matched.py"
    spec = importlib.util.spec_from_file_location("topup_benchmark_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.metrics


def test_two_scan_statistics_flatten_values_before_correlation():
    reference = np.arange(24, dtype=np.float32).reshape(2, 2, 3, 2)
    candidate = reference.copy()
    candidate[..., 1] *= -2
    mask = np.ones(reference.shape[:3], dtype=bool)
    result = _metrics()(candidate, reference, mask)
    expected = np.corrcoef(candidate.ravel(), reference.ravel())[0, 1]
    assert np.isclose(result["pearson_r"], expected)
    assert result["spatial_voxels"] == 12
    assert result["scalar_values"] == 24
    assert np.isclose(result["rmse"], np.sqrt(np.mean((candidate-reference)**2)))


def test_empty_roi_reports_no_statistics():
    image = np.zeros((2, 2, 3, 2), np.float32)
    result = _metrics()(image, image, np.zeros(image.shape[:3], dtype=bool))
    assert result["spatial_voxels"] == result["scalar_values"] == 0
    assert result["pearson_r"] is result["rmse"] is None
