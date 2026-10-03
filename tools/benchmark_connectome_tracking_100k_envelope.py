"""单 atlas：至少三次官方追踪与一或多次 FNIT 的四矩阵随机重复性对照。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
try:
    from .connectome_repeat_common import (
        METRIC_POLICY, NAMES, check_metadata, compact_metrics, envelope,
        fnit_repeat_envelope, load_matrices, overall_status, pairwise, seed_labels,
    )
except ImportError:
    from connectome_repeat_common import (
        METRIC_POLICY, NAMES, check_metadata, compact_metrics, envelope,
        fnit_repeat_envelope, load_matrices, overall_status, pairwise, seed_labels,
    )


def compare(official_dirs, fnit_dirs, *, dataset="unspecified", n_seeds=100000,
            official_seeds=None, fnit_seeds=None) -> dict:
    official = [load_matrices(path) for path in official_dirs]
    fnit = [load_matrices(path) for path in fnit_dirs]
    identity = check_metadata([*official, *fnit], "single atlas")
    pairs = pairwise([item[0] for item in official], [item[0] for item in fnit], compact_metrics)
    fields = ("relative_l1_full_upper", "nonzero_mean_scaled_mae", "support_dice", "pearson")
    ranges = {name: {
        field: envelope([item["metrics"][name][field] for item in pairs["official"]],
                        [item["metrics"][name][field] for item in pairs["cross"]],
                        similarity=field in ("support_dice", "pearson"))
        for field in fields} for name in NAMES}
    fnit_ranges = {name: {
        field: fnit_repeat_envelope([item["metrics"][name][field] for item in pairs["official"]],
                                    [item["metrics"][name][field] for item in pairs["fnit"]],
                                    similarity=field in ("support_dice", "pearson"))
        for field in fields} for name in NAMES}
    return {
        "dataset": dataset, "n_seed_attempts": n_seeds,
        "design": f"{len(official)} supplied official repeats and {len(fnit)} FNIT runs; supplied existing matrices",
        "official_seeds": seed_labels(official_seeds, len(official), "official"),
        "fnit_seeds": seed_labels(fnit_seeds, len(fnit), "fnit"),
        "metric_policy": {**METRIC_POLICY,
                          "relative_l1_full_upper": "all upper-triangle values, also recorded for FA/length as a support-sensitive secondary measure",
                          "nonzero_mean_scaled_mae": "evaluated-edge MAE / mean absolute nonzero reference value on evaluated edges"},
        "official_input_sha256": [item[1]["matrix_sha256"] for item in official],
        "fnit_input_sha256": (fnit[0][1]["matrix_sha256"] if len(fnit) == 1 else
                              [item[1]["matrix_sha256"] for item in fnit]),
        "provenance": {"official": [item[1] for item in official], "fnit": [item[1] for item in fnit]},
        "input_identity": identity, "pairwise": pairs, "ranges": ranges,
        "comparison_counts": {name: len(values) for name, values in pairs.items()},
        "matrix_envelope_status": overall_status([item for values in ranges.values() for item in values.values()]),
        "fnit_reproducibility_ranges": fnit_ranges,
        "fnit_reproducibility_status": overall_status([item for values in fnit_ranges.values() for item in values.values()]),
    }


def figure(path: Path, official_dir: Path, fnit_dir: Path, report: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    reference = load_matrices(official_dir)[0]
    candidate = load_matrices(fnit_dir)[0]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for row, name in enumerate(("count", "mean_fa")):
        difference = candidate[name] - reference[name]
        images = ((np.log1p(reference[name]), np.log1p(candidate[name]),
                   np.sign(difference) * np.log1p(np.abs(difference))) if name == "count" else
                  (reference[name], candidate[name], difference))
        maximum = max(float(images[0].max()), float(images[1].max()), 1e-12)
        limit = max(float(np.abs(images[2]).max()), 1e-12)
        for column, (values, title) in enumerate(zip(images, ("MRtrix run 0", "FNIT run 0", "FNIT - MRtrix"))):
            axes[row, column].imshow(values, origin="lower", interpolation="nearest",
                                    cmap="coolwarm" if column == 2 else "viridis",
                                    vmin=-limit if column == 2 else 0,
                                    vmax=limit if column == 2 else maximum)
            axes[row, column].set(title=f"{name}: {title}", xlabel="node", ylabel="node")
    fig.suptitle(f"{report['dataset']} · {report['n_seed_attempts']:,} attempts · {reference['count'].shape[0]} nodes")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", type=Path, nargs="+", required=True)
    parser.add_argument("--fnit", type=Path, nargs="+", required=True)
    parser.add_argument("--dataset", default="unspecified (provide --dataset)")
    parser.add_argument("--n-seeds", type=int, default=100000, help="attempted seeds, not accepted streamline count")
    parser.add_argument("--official-seeds", type=int, nargs="+")
    parser.add_argument("--fnit-seeds", type=int, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args(argv)
    if len(args.official) < 3 or args.n_seeds < 1:
        parser.error("at least three official repeats and positive --n-seeds are required")
    report = compare(args.official, args.fnit, dataset=args.dataset, n_seeds=args.n_seeds,
                     official_seeds=args.official_seeds, fnit_seeds=args.fnit_seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.figure:
        figure(args.figure, args.official[0], args.fnit[0], report)
    print(json.dumps({"ranges": report["ranges"], "matrix_envelope_status": report["matrix_envelope_status"]}, allow_nan=False))


if __name__ == "__main__":
    main()
