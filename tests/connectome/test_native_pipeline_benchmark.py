"""Evaluator contracts only; generated arrays are not neuroimaging benchmarks."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch


TOOL = Path(__file__).resolve().parents[2] / "tools/benchmark_connectome_native_pipeline.py"
spec = importlib.util.spec_from_file_location("native_pipeline_benchmark", TOOL)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_metrics_equal_zero_support_is_defined():
    matrix = np.zeros((3, 3))
    metrics = benchmark.matrix_metrics(matrix, matrix)
    assert metrics["support_dice"] == 1
    assert metrics["relative_l1"] is None
    assert metrics["pearson"] is None
    assert metrics["full_matrix"]["unequal_values"] == 0


def test_full_matrix_reports_diagonal_changes():
    left = np.eye(3)
    right = np.zeros((3, 3))
    metrics = benchmark.matrix_metrics(left, right)
    assert metrics["unequal_values"] == 0
    assert metrics["full_matrix"]["unequal_values"] == 3


def test_vector_nonfinite_diagnostics():
    metrics = benchmark.vector_metrics([1, np.nan, np.inf], [1, 2, np.inf])
    assert metrics["finite_pairs"] == 1
    assert metrics["nonfinite_mismatch"] == 1
    assert metrics["unequal_values"] == 1


def test_matrix_shape_is_not_padded():
    with pytest.raises(ValueError, match="shapes differ"):
        benchmark.matrix_metrics(np.eye(3), np.eye(2))


def test_track_digest_preserves_boundaries_and_order():
    points = np.arange(18, dtype=np.float32).reshape(6, 3)
    original = benchmark.path_digest([points[:2], points[2:]])
    moved_boundary = benchmark.path_digest([points[:3], points[3:]])
    reverse_order = benchmark.path_digest([points[2:], points[:2]])
    assert original["points_sha256"] == moved_boundary["points_sha256"]
    assert original["offsets_sha256"] != moved_boundary["offsets_sha256"]
    assert original["points_sha256"] != reverse_order["points_sha256"]
    assert original == benchmark.path_digest([torch.from_numpy(points[:2]), torch.from_numpy(points[2:])])


def test_track_digest_preserves_signed_zero_bits():
    original = np.zeros((2, 3), dtype=np.float32)
    changed = original.copy()
    changed[0, 0] = -0.
    assert benchmark.path_digest([original])["points_sha256"] != benchmark.path_digest([changed])["points_sha256"]


def test_track_digest_checks_shape():
    with pytest.raises(ValueError, match="streamline must"):
        benchmark.path_digest([np.zeros((3, 2))])


def test_stage_wrapper_preserves_return_and_arguments():
    rows, marker = [], object()
    def original(value, *, option):
        assert value == 4 and option == 7
        return marker
    wrapped = benchmark.measured_call(original, device=torch.device("cpu"), stage_rows=rows, name="operator")
    assert wrapped(4, option=7) is marker
    assert rows[0]["stage"] == "operator" and rows[0]["seconds"] >= 0


def test_stage_failure_keeps_timing():
    rows = []
    def original():
        raise RuntimeError("actual failure")
    wrapped = benchmark.measured_call(original, device=torch.device("cpu"), stage_rows=rows, name="operator")
    with pytest.raises(RuntimeError, match="actual failure"):
        wrapped()
    assert len(rows) == 1


def test_json_report_is_strict_and_keeps_unavailable(tmp_path):
    path = tmp_path / "report.json"
    benchmark.write_report(path, dict(a=np.float64(np.nan), b=np.int64(8), c=np.inf, path=tmp_path))
    assert json.loads(path.read_text()) == dict(a=None, b=8, c=None, path=str(tmp_path))
    assert "NaN" not in path.read_text()


def test_native_reference_defaults_are_not_replaced():
    images = {name: dict(path=name + ".nii") for name in ("wm_fod", "five_tissue", "gmwmi")}
    command = benchmark._tracking_command(Path("/explicit/build/bin"), images, "oracle.tck", 100, 1)
    assert command[0] == Path("/explicit/build/bin/tckgen")
    assert "-step" not in command and "-minlength" not in command
    assert command[command.index("-angle") + 1] == "45"
    assert command[command.index("-seeds") + 1] == "100"
    assert command[command.index("-nthreads") + 1] == "1"
