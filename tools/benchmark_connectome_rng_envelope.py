"""Compare three fixed-seed MRtrix and three FNIT connectomes on one dataset.

Run with six directories containing four square CSVs each. The program accepts
MRtrix's count.csv naming, published connectome_count.csv naming, and FNIT's
candidate_count.csv naming. It writes Pearson r, normalized MAE, support Dice,
and evaluated-edge counts for the three within-arm pairs and all nine cross-arm
pairs. The metric definitions are shared with compare_connectome_matrices.py.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from compare_connectome_matrices import NAMES, _compare


def load_matrices(directory: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Read a four-matrix directory; return arrays and file SHA-256 hashes."""
    for prefix in ("connectome_", "candidate_", ""):
        paths = {name: directory / f"{prefix}{name}.csv" for name in NAMES}
        if all(path.is_file() for path in paths.values()):
            break
    else:
        raise FileNotFoundError(f"{directory}: four connectome CSVs not found")
    arrays = {name: np.loadtxt(path, delimiter=",", ndmin=2)
              for name, path in paths.items()}
    shape = arrays["count"].shape
    if len(shape) != 2 or shape[0] != shape[1] or any(
        array.shape != shape or not np.isfinite(array).all()
        for array in arrays.values()
    ):
        raise ValueError(f"{directory}: expected four finite square matrices")
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
              for name, path in paths.items()}
    return arrays, hashes


def compact_metrics(first: dict[str, np.ndarray], second: dict[str, np.ndarray]) -> dict:
    """Return strict-upper-triangle correlations, support, and both error scales."""
    full = _compare(first, second)
    upper = np.triu_indices(first["count"].shape[0], k=1)
    result = {}
    for name in NAMES:
        reference = first[name][upper]
        candidate = second[name][upper]
        common = (reference != 0) & (candidate != 0)
        error = np.abs(candidate - reference)
        reference_mass = np.abs(reference).sum()
        common_mass = np.abs(reference[common]).sum()
        result[name] = {
            "pearson": full[name]["values"]["pearson"],
            "nonzero_mean_scaled_mae": full[name]["values"]["normalized_mae"],
            "relative_l1_full_upper": (float(error.sum() / reference_mass)
                                        if reference_mass else None),
            "relative_l1_common_nonzero": (float(error[common].sum() / common_mass)
                                           if common_mass else None),
            "support_dice": full[name]["nonzero_support"]["dice"],
            "evaluated_edges": full[name]["values"]["evaluated_edges"],
            "shared_support_edges": full[name]["nonzero_support"]["shared_edges"],
            "reference_support_edges": full[name]["nonzero_support"]["reference_edges"],
            "candidate_support_edges": full[name]["nonzero_support"]["candidate_edges"],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", nargs=3, type=Path, required=True,
                        metavar=("SEED0", "SEED1", "SEED2"))
    parser.add_argument("--fnit", nargs=3, type=Path, required=True,
                        metavar=("SEED0", "SEED1", "SEED2"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, help="Save a four-panel normalized-error figure")
    args = parser.parse_args()
    loaded = {arm: [load_matrices(directory) for directory in dirs]
              for arm, dirs in (("official", args.official), ("fnit", args.fnit))}
    pairs = {"official": [], "fnit": [], "cross": []}
    for arm in ("official", "fnit"):
        for i, j in itertools.combinations(range(3), 2):
            pairs[arm].append({"first_seed": i, "second_seed": j,
                               "metrics": compact_metrics(loaded[arm][i][0],
                                                          loaded[arm][j][0])})
    for i in range(3):
        for j in range(3):
            pairs["cross"].append({"official_seed": i, "fnit_seed": j,
                                   "metrics": compact_metrics(loaded["official"][i][0],
                                                              loaded["fnit"][j][0])})
    ranges = {}
    for group, comparisons in pairs.items():
        ranges[group] = {}
        for name in NAMES:
            ranges[group][name] = {}
            for field in ("pearson", "nonzero_mean_scaled_mae", "relative_l1_full_upper",
                          "relative_l1_common_nonzero", "support_dice", "evaluated_edges"):
                values = [item["metrics"][name][field] for item in comparisons]
                ranges[group][name][field] = [float(min(values)), float(max(values))]
    report = {
        "dataset": "OpenNeuro ds004666 sub-01 ses-2mm, corrected paired T1/DWI",
        "metric_policy": {
            "edges": "strict upper triangle of 20x20 matrices, excluding diagonal",
            "count_and_fbc": "all 190 edges",
            "length_and_fa": "only edges with nonzero count in both compared matrices",
            "nonzero_mean_scaled_mae": "MAE on evaluated edges divided by mean absolute nonzero first-matrix value on those edges; matches compare_connectome_matrices.py",
            "relative_l1_full_upper": "sum absolute error / sum absolute first-matrix value across all 190 upper-triangle edges; matches integrated benchmark relative_l1_upper",
            "relative_l1_common_nonzero": "sum absolute error / sum absolute first-matrix value where both metric matrices are nonzero",
            "support_dice": "2*shared nonzero edges/(nonzero edges in first and second matrices)",
        },
        "input_sha256": {arm: [{"seed": i, "matrices": item[1]}
                                for i, item in enumerate(items)]
                         for arm, items in loaded.items()},
        "pairwise": pairs,
        "ranges": ranges,
        "cross_exceeds_official_random_range": {
            name: {
                "all_cross_relative_l1_full_upper_above_official_max":
                    ranges["cross"][name]["relative_l1_full_upper"][0] >
                    ranges["official"][name]["relative_l1_full_upper"][1],
                "all_cross_support_dice_below_official_min":
                    ranges["cross"][name]["support_dice"][1] <
                    ranges["official"][name]["support_dice"][0],
            }
            for name in NAMES
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        labels = {"count": "Streamline count", "sift2_fbc": "SIFT2 FBC",
                  "mean_length": "Mean length", "mean_fa": "Mean FA"}
        groups = (("official", "MRtrix/MRtrix", "#2374ab"),
                  ("fnit", "FNIT/FNIT", "#499c64"),
                  ("cross", "FNIT/MRtrix", "#d55e00"))
        for ax, name in zip(axes.flat, NAMES):
            for x, (group, label, color) in enumerate(groups):
                values = [pair["metrics"][name]["relative_l1_full_upper"]
                          for pair in pairs[group]]
                jitter = np.linspace(-0.13, 0.13, len(values))
                ax.scatter(x + jitter, values, color=color, s=34, zorder=3)
                ax.plot([x - 0.2, x + 0.2], [np.median(values)] * 2,
                        color=color, linewidth=2)
            ax.axhline(ranges["official"][name]["relative_l1_full_upper"][1],
                       color="#2374ab", linestyle="--", linewidth=1)
            ax.set_xticks(range(3), [group[1] for group in groups], rotation=14)
            ax.set_ylabel("Relative L1, all 190 edges")
            ax.set_title(labels[name])
            ax.grid(axis="y", alpha=0.2)
        fig.suptitle("ds004666: 3 fixed seeds per implementation; 3 / 3 / 9 comparisons")
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure, dpi=180)
        plt.close(fig)
    print(json.dumps({"ranges": ranges,
                      "cross_exceeds_official_random_range":
                      report["cross_exceeds_official_random_range"]},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
