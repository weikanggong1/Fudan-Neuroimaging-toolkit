"""对同一 DWI 网格的 84 节点 atlas，比较 FNIT 和 MRtrix 四张真实矩阵。"""

import argparse
import json
from pathlib import Path

import numpy as np


NAMES = ("count", "sift2_fbc", "mean_length", "mean_fa")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fnit-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"dataset": args.dataset, "node_count": 84, "matrix_metrics": {}}
    arrays = []
    upper = np.triu_indices(84, 1)
    for name in NAMES:
        fnit = np.loadtxt(args.fnit_dir / f"connectome_{name}.csv", delimiter=",")
        reference = np.loadtxt(args.reference_dir / f"{name}.csv", delimiter=",")
        if fnit.shape != reference.shape or fnit.shape != (84, 84):
            raise ValueError(f"{name}: expected matching 84x84 matrices")
        if not np.isfinite(fnit).all() or not np.isfinite(reference).all():
            raise ValueError(f"{name}: non-finite values")
        candidate, official = fnit[upper], reference[upper]
        common = (candidate != 0) & (official != 0)
        corr = float(np.corrcoef(candidate, official)[0, 1])
        metrics = {
            "upper_triangle_pearson_r": corr,
            "upper_triangle_relative_l1": float(np.abs(candidate - official).sum() /
                                                max(np.abs(official).sum(), 1e-12)),
            "upper_triangle_max_abs_error": float(np.abs(candidate - official).max()),
            "common_nonzero_edges": int(common.sum()),
            "common_edge_mean_abs_error": (float(np.abs(candidate[common] - official[common]).mean())
                                           if common.any() else None),
        }
        if name == "count":
            a, b = candidate > 0, official > 0
            metrics.update({
                "fnit_upper_nonzero_edges": int(a.sum()),
                "reference_upper_nonzero_edges": int(b.sum()),
                "nonzero_edge_dice": float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1)),
                "fnit_sum_all_matrix": int(fnit.sum()),
                "reference_sum_all_matrix": int(reference.sum()),
            })
        report["matrix_metrics"][name] = metrics
        arrays.append((fnit, reference))
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    for column, (name, (fnit, reference)) in enumerate(zip(NAMES, arrays)):
        if name in ("count", "sift2_fbc"):
            fnit, reference = np.log1p(fnit), np.log1p(reference)
        maximum = max(float(fnit.max()), float(reference.max()))
        for row, (matrix, label) in enumerate(((reference, "MRtrix"), (fnit, "FNIT"))):
            axes[row, column].imshow(matrix, cmap="magma", vmin=0, vmax=maximum)
            axes[row, column].set_title(f"{label}: {name}")
            axes[row, column].set_xlabel("node index")
            axes[row, column].set_ylabel("node index")
    fig.savefig(args.output_dir / "connectome_comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
