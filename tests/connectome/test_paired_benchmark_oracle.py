"""Independent paired reduction checks, including preserved undefined FA."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import torch


_path = (Path(__file__).resolve().parents[2] / "validation/connectome/"
         "paired_pipeline_20261003/run_ten_benchmark.py")
_spec = spec_from_file_location("fnit_paired_benchmark_test", _path)
_benchmark = module_from_spec(_spec)
_spec.loader.exec_module(_benchmark)


def _snapshot():
    return dict(
        first=torch.tensor([1, 2, 1, 0]).reshape(4, 1, 1),
        second=torch.tensor([1, 3, 2, 0]).reshape(4, 1, 1),
        first_affine=torch.eye(4), second_affine=torch.eye(4),
        first_count=2, second_count=3,
        endpoints=torch.tensor([
            [[0., 0, 0], [1., 0, 0]],
            [[0., 0, 0], [0., 0, 0]],
            [[2., 0, 0], [1., 0, 0]],
            [[3., 0, 0], [3., 0, 0]],
        ]),
        weights=torch.tensor([2., 3., 5., 7.]),
        lengths=torch.tensor([10., 20., 30., 40.]),
        fa=torch.tensor([.125, .25, .375, .5]),
        matrices=dict(
            count=np.array([[1, 0, 2], [1, 1, 0]], np.int64),
            sift2_fbc=np.array([[3., 0, 7], [2., 5., 0]], np.float32),
            mean_length=np.array([[20., 0, 170 / 7], [10., 30., 0]], np.float32),
            mean_fa=np.array([[.25, 0, 2.125 / 7], [.125, .375, 0]], np.float32),
        ),
    )


def test_oracle_undirected_pairing_deduplicates_and_keeps_two_distinct_cells():
    report = _benchmark.cpu_pair_oracle(_snapshot(), .4)
    assert report["passed"]
    assert report["matched_unique_streamlines"] == 3
    assert report["matrix_count_sum"] == 5
    assert all(row["different_elements"] == 0 for row in report["comparisons"].values())


def test_oracle_matching_nan_statistics_are_equal_and_have_finite_error_report():
    snapshot = _snapshot()
    snapshot["fa"][1] = float("nan")
    snapshot["matrices"]["mean_fa"][0, 0] = np.nan
    report = _benchmark.cpu_pair_oracle(snapshot, .4)
    assert report["passed"]
    comparison = report["comparisons"]["mean_fa"]
    assert comparison["nan_mask_equal"]
    assert comparison["different_elements"] == 0
    assert comparison["max_absolute_error"] == 0
    assert comparison["finite_comparison_elements"] == 5


def test_oracle_missing_nan_reports_exact_mismatch():
    snapshot = _snapshot()
    snapshot["fa"][1] = float("nan")
    report = _benchmark.cpu_pair_oracle(snapshot, .4)
    assert not report["passed"]
    comparison = report["comparisons"]["mean_fa"]
    assert not comparison["nan_mask_equal"]
    assert comparison["different_elements"] == 1
    assert np.isfinite(comparison["max_absolute_error"])
