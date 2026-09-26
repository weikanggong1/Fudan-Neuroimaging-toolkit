"""Compare four fixed-name connectome CSVs against a reference and across seeds.

Primary value metrics use the strict upper triangle (self edges are reported
separately). Mean length and FA are evaluated only where both count matrices
have an edge; their nonzero-value support Dice still uses their own matrices.
Normalized errors divide by the mean absolute nonzero reference value on the
evaluated edges. Undefined correlations and zero-reference normalization are
reported as JSON null with an explanation.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


NAMES = ("count", "sift2_fbc", "mean_length", "mean_fa")


def _load(directory: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    matrices = {}
    hashes = {}
    shape = None
    for name in NAMES:
        path = directory / f"connectome_{name}.csv"
        data = np.loadtxt(path, delimiter=",", ndmin=2)
        if data.ndim != 2 or data.shape[0] != data.shape[1] or not np.isfinite(data).all():
            raise ValueError(f"{path}: expected a finite square matrix")
        if shape is not None and data.shape != shape:
            raise ValueError(f"{path}: shape {data.shape} differs from {shape}")
        shape = data.shape
        matrices[name] = data
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return matrices, hashes


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2:
        return None
    x = x - x.mean()
    y = y - y.mean()
    denominator = np.sqrt(np.dot(x, x) * np.dot(y, y))
    if denominator == 0:
        return None
    return float(np.clip(np.dot(x, y) / denominator, -1.0, 1.0))


def _one_matrix(
    reference: np.ndarray,
    candidate: np.ndarray,
    reference_count: np.ndarray,
    candidate_count: np.ndarray,
    name: str,
) -> dict:
    if reference.shape != candidate.shape:
        raise ValueError(f"{name}: reference {reference.shape} != candidate {candidate.shape}")
    upper = np.triu_indices(reference.shape[0], k=1)
    ref = reference[upper]
    cand = candidate[upper]
    ref_support = ref != 0
    cand_support = cand != 0
    shared = int(np.count_nonzero(ref_support & cand_support))
    n_ref = int(np.count_nonzero(ref_support))
    n_cand = int(np.count_nonzero(cand_support))
    dice = 1.0 if n_ref + n_cand == 0 else 2.0 * shared / (n_ref + n_cand)
    if name in ("mean_length", "mean_fa"):
        evaluate = (reference_count[upper] != 0) & (candidate_count[upper] != 0)
        policy = "common nonzero count edges"
    else:
        evaluate = np.ones(ref.shape, dtype=bool)
        policy = "all off-diagonal edges"
    x = ref[evaluate]
    y = cand[evaluate]
    values = {
        "policy": policy,
        "evaluated_edges": int(x.size),
        "pearson": None,
        "spearman": None,
        "mae": None,
        "rmse": None,
        "normalized_mae": None,
        "normalized_rmse": None,
        "normalization_scale": None,
    }
    if x.size:
        error = y - x
        values["mae"] = float(np.mean(np.abs(error)))
        values["rmse"] = float(np.sqrt(np.mean(error * error)))
        values["pearson"] = _pearson(x, y)
        values["spearman"] = _pearson(rankdata(x), rankdata(y))
        if values["pearson"] is None or values["spearman"] is None:
            values["correlation_reason"] = "fewer than two values or a constant vector"
        nonzero_reference = np.abs(x[x != 0])
        if nonzero_reference.size:
            scale = float(nonzero_reference.mean())
            values["normalization_scale"] = scale
            values["normalized_mae"] = values["mae"] / scale
            values["normalized_rmse"] = values["rmse"] / scale
        else:
            values["normalization_reason"] = "no nonzero reference values on evaluated edges"
    else:
        values["undefined_reason"] = "no common count edges" if name in ("mean_length", "mean_fa") else "no off-diagonal edges"

    ref_diagonal = np.diag(reference)
    cand_diagonal = np.diag(candidate)
    diagonal_error = cand_diagonal - ref_diagonal
    return {
        "shape": list(reference.shape),
        "upper_triangle_excludes_diagonal": True,
        "nonzero_support": {
            "reference_edges": n_ref,
            "candidate_edges": n_cand,
            "shared_edges": shared,
            "dice": dice,
            "empty_empty_dice_convention": 1.0,
        },
        "values": values,
        "diagonal": {
            "reference_nonzero_nodes": int(np.count_nonzero(ref_diagonal)),
            "candidate_nonzero_nodes": int(np.count_nonzero(cand_diagonal)),
            "mae": float(np.mean(np.abs(diagonal_error))),
            "rmse": float(np.sqrt(np.mean(diagonal_error * diagonal_error))),
        },
    }


def _compare(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray]) -> dict:
    return {
        name: _one_matrix(reference[name], candidate[name],
                          reference["count"], candidate["count"], name)
        for name in NAMES
    }


def compare_matrices(reference_dir: Path, candidate_dirs: list[Path]) -> dict:
    """Return a reproducible report for one reference and one or more candidates."""
    if not candidate_dirs:
        raise ValueError("at least one candidate directory is required")
    reference_dir = Path(reference_dir).resolve()
    candidate_dirs = [Path(path).resolve() for path in candidate_dirs]
    reference, reference_hashes = _load(reference_dir)
    candidates = []
    candidate_matrices = []
    for directory in candidate_dirs:
        matrices, hashes = _load(directory)
        candidates.append({"directory": str(directory), "sha256": hashes,
                           "metrics": _compare(reference, matrices)})
        candidate_matrices.append(matrices)
    pairs = []
    for i, j in itertools.combinations(range(len(candidates)), 2):
        pairs.append({"candidate_a": str(candidate_dirs[i]),
                      "candidate_b": str(candidate_dirs[j]),
                      "metrics": _compare(candidate_matrices[i], candidate_matrices[j])})
    stability_summary = {}
    for name in NAMES:
        dices = [pair["metrics"][name]["nonzero_support"]["dice"] for pair in pairs]

        def pairwise_mean(key: str) -> float | None:
            values = [pair["metrics"][name]["values"][key] for pair in pairs]
            valid = [value for value in values if value is not None]
            return float(np.mean(valid)) if valid else None

        stability_summary[name] = {
            "mean_support_dice": float(np.mean(dices)) if dices else None,
            "minimum_support_dice": float(np.min(dices)) if dices else None,
            "mean_pairwise_pearson": pairwise_mean("pearson"),
            "mean_pairwise_spearman": pairwise_mean("spearman"),
            "mean_pairwise_mae": pairwise_mean("mae"),
            "mean_pairwise_rmse": pairwise_mean("rmse"),
            "mean_pairwise_normalized_mae": pairwise_mean("normalized_mae"),
            "mean_pairwise_normalized_rmse": pairwise_mean("normalized_rmse"),
        }
    return {
        "metric_policy": {
            "files": [f"connectome_{name}.csv" for name in NAMES],
            "edge_subset": "strict upper triangle; diagonal reported separately",
            "length_fa_values": "only edges with nonzero counts in both matrices",
            "support": "nonzero values of each metric; empty/empty Dice is 1",
            "normalized_error": "raw error / mean absolute nonzero reference value on evaluated edges",
            "stability": "all unordered candidate-directory pairs with the same comparison policy",
        },
        "reference": {"directory": str(reference_dir), "sha256": reference_hashes},
        "candidates": candidates,
        "stability": {"pair_count": len(pairs), "pairwise": pairs,
                      "summary": stability_summary},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, action="append", type=Path)
    parser.add_argument("--output", type=Path, help="Write JSON report to this path")
    args = parser.parse_args(argv)
    report = compare_matrices(args.reference_dir, args.candidate_dir)
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
