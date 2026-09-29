"""七套真实 atlas：三次 MRtrix 互比与一份 FNIT 独立 100k 追踪矩阵比较。"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from connectome_benchmark_common import NAMES


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metrics(first: dict[str, np.ndarray], second: dict[str, np.ndarray]) -> dict:
    upper = np.triu_indices(first["count"].shape[0], k=1)
    count_a, count_b = first["count"][upper], second["count"][upper]
    support_a, support_b = count_a > 0, count_b > 0
    common = support_a & support_b
    result = {
        "count_support_dice": float(2 * common.sum() / (support_a.sum() + support_b.sum())),
        "count_pearson": float(np.corrcoef(count_a, count_b)[0, 1]),
        "common_edges": int(common.sum()),
    }
    for name in ("count", "sift2_fbc"):
        a, b = first[name][upper], second[name][upper]
        result[f"{name}_relative_l1"] = float(np.abs(a - b).sum() / np.abs(a).sum())
    for name in ("mean_length", "mean_fa"):
        a, b = first[name][upper][common], second[name][upper][common]
        result[f"{name}_common_normalized_mae"] = (
            float(np.abs(a - b).mean() / np.abs(a).mean()) if len(a) and np.abs(a).sum()
            else None
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", type=Path, nargs=3, required=True)
    parser.add_argument("--fnit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    native_report = json.loads((args.fnit / "report.json").read_text())
    report = {
        "dataset": "OpenNeuro ds004666 sub-01/ses-2mm, same FOD/5TT/FA and seven atlas volumes",
        "design": "three independent MRtrix 100k seed runs, one FNIT 100k run",
        "profiles": {},
    }
    fields = ("count_relative_l1", "sift2_fbc_relative_l1",
              "count_support_dice", "mean_length_common_normalized_mae",
              "mean_fa_common_normalized_mae")
    for profile, info in native_report["profiles"].items():
        official = []
        official_hashes = []
        for root in args.official:
            paths = {name: root / profile / f"{name}.csv" for name in NAMES}
            official.append({name: np.loadtxt(path, delimiter=",") for name, path in paths.items()})
            official_hashes.append({name: _sha(path) for name, path in paths.items()})
        with np.load(args.fnit / f"{profile}.npz") as archive:
            fnit = {name: archive[name] for name in NAMES}
        expected_shape = (info["nodes"], info["nodes"])
        for matrices in (*official, fnit):
            for matrix in matrices.values():
                if matrix.shape != expected_shape or not np.isfinite(matrix).all():
                    raise ValueError(f"{profile}: invalid matrix shape or values")
        within = [_metrics(official[i], official[j])
                  for i, j in itertools.combinations(range(3), 2)]
        cross = [_metrics(matrices, fnit) for matrices in official]
        ranges = {
            field: {
                "official_min_max": [float(min(item[field] for item in within)),
                                     float(max(item[field] for item in within))],
                "fnit_vs_official": [float(item[field]) for item in cross],
                "inside_count": sum(min(item[field] for item in within) <= value[field] <=
                                    max(item[field] for item in within) for value in cross),
            } for field in fields
        }
        report["profiles"][profile] = {
            "nodes": info["nodes"], "atlas_sha256": info["atlas_sha256"],
            "official_matrix_sha256": official_hashes,
            "fnit_matrix_npz_sha256": _sha(args.fnit / f"{profile}.npz"),
            "ranges": ranges,
        }
        print(f"{profile}: count L1 {ranges['count_relative_l1']}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        profiles = list(report["profiles"])
        shown = ("count_relative_l1", "sift2_fbc_relative_l1",
                 "mean_fa_common_normalized_mae")
        fig, axes = plt.subplots(1, 3, figsize=(13, 5), constrained_layout=True)
        for ax, field in zip(axes, shown):
            for row, profile in enumerate(profiles):
                values = report["profiles"][profile]["ranges"][field]
                low, high = values["official_min_max"]
                ax.plot((low, high), (row, row), color="#2374ab", linewidth=4)
                ax.scatter(values["fnit_vs_official"], [row - .12, row, row + .12],
                           color="#d55e00", s=25)
            ax.set(title=field.replace("_", " "), yticks=range(len(profiles)),
                   yticklabels=profiles, xlabel="error")
            ax.grid(axis="x", alpha=.2)
        fig.suptitle("ds004666, 100k attempts · MRtrix self range (blue) vs FNIT (orange)")
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure, dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
