"""任意多 atlas：至少三次 MRtrix 互比与 FNIT 已有四矩阵的随机重复性比较。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
try:
    from .connectome_repeat_common import (
        METRIC_POLICY, check_metadata, envelope, load_profiles, overall_status,
        fnit_repeat_envelope, pairwise, profile_metrics, seed_labels,
    )
except ImportError:
    from connectome_repeat_common import (
        METRIC_POLICY, check_metadata, envelope, load_profiles, overall_status,
        fnit_repeat_envelope, pairwise, profile_metrics, seed_labels,
    )

_metrics = profile_metrics  # retain the existing tool's metric definitions


def compare(official_dirs, fnit_dirs, *, dataset="unspecified", n_seeds=100000,
            official_seeds=None, fnit_seeds=None, atlases=None) -> dict:
    official = [load_profiles(path) for path in official_dirs]
    fnit = [load_profiles(path) for path in fnit_dirs]
    if len(official) < 3 or not fnit:
        raise ValueError("at least three official repeats and one FNIT run are required")
    names = list(atlases) if atlases else list(fnit[0])
    if not names or len(set(names)) != len(names):
        raise ValueError("atlas names must be nonempty and unique")
    for item in (*official, *fnit):
        if (set(item) != set(names) if not atlases else not set(names).issubset(item)):
            raise ValueError("atlas profiles differ between runs; use --atlas only for an explicit subset")
    report = {
        "dataset": dataset, "n_seed_attempts": n_seeds,
        "design": f"{len(official)} supplied official repeats and {len(fnit)} FNIT runs; no matrix recomputation",
        "official_seeds": seed_labels(official_seeds, len(official), "official"),
        "fnit_seeds": seed_labels(fnit_seeds, len(fnit), "fnit"),
        "metric_policy": {**METRIC_POLICY,
                          "relative_l1": "sum absolute error / sum absolute first matrix upper-triangle values",
                          "common_normalized_mae": "MAE on common count edges / mean absolute first matrix value on those edges"},
        "profiles": {},
    }
    fields = ("count_relative_l1", "sift2_fbc_relative_l1", "count_support_dice", "count_pearson",
              "mean_length_common_normalized_mae", "mean_fa_common_normalized_mae")
    for profile in names:
        loaded = [item[profile] for item in (*official, *fnit)]
        identity = check_metadata(loaded, profile)
        pairs = pairwise([item[profile][0] for item in official],
                         [item[profile][0] for item in fnit], profile_metrics)
        ranges = {
            field: envelope([item["metrics"][field] for item in pairs["official"]],
                            [item["metrics"][field] for item in pairs["cross"]],
                            similarity=field in ("count_support_dice", "count_pearson"))
            for field in fields}
        fnit_ranges = {
            field: fnit_repeat_envelope([item["metrics"][field] for item in pairs["official"]],
                                        [item["metrics"][field] for item in pairs["fnit"]],
                                        similarity=field in ("count_support_dice", "count_pearson"))
            for field in fields}
        fnit_metadata = [item[profile][1] for item in fnit]
        report["profiles"][profile] = {
            "nodes": loaded[0][1]["nodes"], "atlas_sha256": fnit_metadata[0]["atlas_sha256"],
            "official_matrix_sha256": [item[profile][1].get("matrix_sha256") for item in official],
            "fnit_matrix_npz_sha256": fnit_metadata[0].get("matrix_npz_sha256"),
            "provenance": {"official": [item[profile][1] for item in official], "fnit": fnit_metadata},
            "input_identity": identity, "pairwise": pairs, "ranges": ranges,
            "comparison_counts": {name: len(values) for name, values in pairs.items()},
            "matrix_envelope_status": overall_status(list(ranges.values())),
            "fnit_reproducibility_ranges": fnit_ranges,
            "fnit_reproducibility_status": overall_status(list(fnit_ranges.values())),
        }
    statuses = [item["matrix_envelope_status"] for item in report["profiles"].values()]
    report["matrix_envelope_status"] = ("failed" if "failed" in statuses else
                                         "passed" if all(item == "passed" for item in statuses) else "not_assessed")
    self_statuses = [item["fnit_reproducibility_status"] for item in report["profiles"].values()]
    report["fnit_reproducibility_status"] = ("failed" if "failed" in self_statuses else
                                             "passed" if all(item == "passed" for item in self_statuses) else "not_assessed")
    return report


def figure(path: Path, report: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import textwrap

    profiles = list(report["profiles"])
    shown = (("count_relative_l1", "Count relative L1"),
             ("sift2_fbc_relative_l1", "SIFT2 FBC relative L1"),
             ("count_pearson", "Count Pearson correlation"),
             ("count_support_dice", "Count support Dice"),
             ("mean_length_common_normalized_mae", "Common count edges:\nmean length normalized MAE"),
             ("mean_fa_common_normalized_mae", "Common count edges:\nmean FA normalized MAE"))
    fig, axes = plt.subplots(2, 3, figsize=(15, max(9, len(profiles) * .85)),
                             sharey=True, constrained_layout=True)
    for ax, (field, title) in zip(axes.ravel(), shown):
        for row, profile in enumerate(profiles):
            values = report["profiles"][profile]["ranges"][field]
            if values["official_min_max"] is not None:
                low, high = values["official_min_max"]
                ax.plot((low, high), (row, row), color="#2374ab", linewidth=4)
            offsets = np.linspace(-.12, .12, len(values["fnit_vs_official"]))
            for value, accepted, offset in zip(values["fnit_vs_official"],
                                               values["comparison_accepted"], offsets):
                if value is not None:
                    ax.scatter(value, row + offset,
                               color="#d62728" if accepted is False else "#d55e00", s=25)
        ax.set(title=title, yticks=range(len(profiles)), yticklabels=profiles,
               xlabel="similarity" if field in ("count_support_dice", "count_pearson") else "normalized error")
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="x", alpha=.2)
    fig.suptitle(textwrap.fill(f"{report['dataset']} · {report['n_seed_attempts']:,} attempts", 110) +
                 "\nBlue: official observed range; orange: accepted cross comparison; red: outside predeclared gate", fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", type=Path, nargs="+", required=True)
    parser.add_argument("--fnit", type=Path, nargs="+", required=True)
    parser.add_argument("--dataset", default="unspecified (provide --dataset)")
    parser.add_argument("--n-seeds", type=int, default=100000)
    parser.add_argument("--official-seeds", type=int, nargs="+")
    parser.add_argument("--fnit-seeds", type=int, nargs="+")
    parser.add_argument("--atlas", nargs="+", help="explicit profile subset; default checks every output atlas")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args(argv)
    if len(args.official) < 3 or args.n_seeds < 1:
        parser.error("at least three official repeats and positive --n-seeds are required")
    report = compare(args.official, args.fnit, dataset=args.dataset, n_seeds=args.n_seeds,
                     official_seeds=args.official_seeds, fnit_seeds=args.fnit_seeds, atlases=args.atlas)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.figure:
        figure(args.figure, report)
    print(json.dumps({"matrix_envelope_status": report["matrix_envelope_status"],
                      "profiles": {name: item["matrix_envelope_status"] for name, item in report["profiles"].items()}}, allow_nan=False))


if __name__ == "__main__":
    main()
