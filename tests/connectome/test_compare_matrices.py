"""Edge policy, undefined metrics, and cross-seed report checks."""

import json

import numpy as np
import pytest

from tools.compare_connectome_matrices import NAMES, compare_matrices, main


def _write(directory, matrices):
    directory.mkdir()
    for name in NAMES:
        np.savetxt(directory / f"connectome_{name}.csv", matrices[name], delimiter=",")


def test_diagonal_separate_and_length_fa_only_common_count_edges(tmp_path):
    reference = {
        "count": np.array([[1, 2, 0], [2, 0, 3], [0, 3, 0]]),
        "sift2_fbc": np.array([[1, 2, 0], [2, 0, 6], [0, 6, 0]]),
        "mean_length": np.array([[10, 10, 0], [10, 0, 20], [0, 20, 0]]),
        "mean_fa": np.array([[.1, .3, 0], [.3, 0, .4], [0, .4, 0]]),
    }
    candidate = {
        "count": np.array([[9, 2, 4], [2, 0, 0], [4, 0, 0]]),
        "sift2_fbc": np.array([[9, 2, 8], [2, 0, 0], [8, 0, 0]]),
        "mean_length": np.array([[90, 12, 100], [12, 0, 0], [100, 0, 0]]),
        "mean_fa": np.array([[.9, .5, .8], [.5, 0, 0], [.8, 0, 0]]),
    }
    ref_dir, cand_dir = tmp_path / "ref", tmp_path / "candidate"
    _write(ref_dir, reference)
    _write(cand_dir, candidate)
    metrics = compare_matrices(ref_dir, [cand_dir])["candidates"][0]["metrics"]
    count = metrics["count"]
    assert count["upper_triangle_excludes_diagonal"]
    assert count["nonzero_support"]["dice"] == .5
    assert count["diagonal"]["mae"] == pytest.approx(8 / 3)
    assert count["values"]["evaluated_edges"] == 3
    length = metrics["mean_length"]
    assert length["nonzero_support"]["dice"] == .5
    assert length["values"]["evaluated_edges"] == 1
    assert length["values"]["mae"] == 2
    assert length["values"]["rmse"] == 2
    assert length["values"]["normalized_mae"] == .2
    assert length["values"]["pearson"] is None
    assert metrics["mean_fa"]["values"]["mae"] == pytest.approx(.2)


def test_all_zero_and_no_common_edges_are_explicit(tmp_path):
    zeros = {name: np.zeros((3, 3)) for name in NAMES}
    ref_dir, cand_dir = tmp_path / "ref", tmp_path / "candidate"
    _write(ref_dir, zeros)
    _write(cand_dir, zeros)
    report = compare_matrices(ref_dir, [cand_dir])
    metrics = report["candidates"][0]["metrics"]
    assert metrics["count"]["nonzero_support"]["dice"] == 1
    assert metrics["count"]["values"]["mae"] == 0
    assert metrics["count"]["values"]["normalized_mae"] is None
    assert metrics["count"]["values"]["pearson"] is None
    assert metrics["mean_length"]["values"]["mae"] is None
    assert metrics["mean_length"]["values"]["undefined_reason"] == "no common count edges"
    assert report["stability"]["pair_count"] == 0
    assert report["stability"]["summary"]["count"]["mean_support_dice"] is None


def test_correlations_seed_stability_and_cli_json(tmp_path, capsys):
    upper = np.array([1., 2., 3., 4., 5., 6.])
    indices = np.triu_indices(4, k=1)

    def matrices(factor):
        matrix = np.zeros((4, 4))
        matrix[indices] = factor * upper
        matrix[(indices[1], indices[0])] = factor * upper
        return {name: matrix.copy() for name in NAMES}

    ref_dir, first, second = tmp_path / "ref", tmp_path / "seed1", tmp_path / "seed2"
    _write(ref_dir, matrices(1))
    _write(first, matrices(2))
    _write(second, matrices(3))
    output = tmp_path / "report.json"
    main(["--reference-dir", str(ref_dir), "--candidate-dir", str(first),
          "--candidate-dir", str(second), "--output", str(output)])
    report = json.loads(output.read_text())
    assert report == json.loads(capsys.readouterr().out)
    assert report["stability"]["pair_count"] == 1
    first_count = report["candidates"][0]["metrics"]["count"]
    assert first_count["values"]["pearson"] == pytest.approx(1)
    assert first_count["values"]["spearman"] == pytest.approx(1)
    assert first_count["values"]["normalized_mae"] == pytest.approx(1)
    assert first_count["nonzero_support"]["dice"] == 1
    assert report["stability"]["summary"]["count"]["mean_pairwise_mae"] == pytest.approx(3.5)
    assert len(report["reference"]["sha256"]["count"]) == 64


def test_zero_fa_on_shared_count_edge_is_still_evaluated(tmp_path):
    count = np.array([[0, 1], [1, 0]])
    bundle = {name: np.zeros((2, 2)) for name in NAMES}
    bundle["count"] = count
    ref_dir, cand_dir = tmp_path / "ref", tmp_path / "candidate"
    _write(ref_dir, bundle)
    _write(cand_dir, bundle)
    fa = compare_matrices(ref_dir, [cand_dir])["candidates"][0]["metrics"]["mean_fa"]
    assert fa["values"]["evaluated_edges"] == 1
    assert fa["values"]["mae"] == 0
    assert fa["nonzero_support"]["dice"] == 1
