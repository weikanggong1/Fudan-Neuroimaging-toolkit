"""绘制真实 FOD 的 iFOD2 初始方向五次重复及冻结输入矩阵 A/B。"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _ecdf(values):
    ordered = np.sort(values)
    return ordered, np.arange(1, len(ordered) + 1) / len(ordered)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction-report", type=Path, required=True, help="五次初始方向比较 JSON")
    parser.add_argument("--candidate-npz", type=Path, required=True, help="对应 PyTorch 逐种子结果")
    parser.add_argument("--official", type=Path, required=True, help="官方 seed 0 五列结果")
    parser.add_argument("--matrix-report", type=Path, action="append", required=True, help="矩阵 A/B JSON；重复五次")
    parser.add_argument("--output", type=Path, required=True, help="输出 PNG")
    args = parser.parse_args()
    direction = json.loads(args.direction_report.read_text())
    reports = [json.loads(path.read_text()) for path in args.matrix_report]
    official = np.loadtxt(args.official)
    with np.load(args.candidate_npz, allow_pickle=False) as candidate:
        amplitude = candidate["amplitudes"][0][candidate["valid"][0]]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    ids = np.arange(len(direction["official_valid_counts"]))
    axes[0, 0].plot(ids, direction["official_valid_counts"], "o-", label="MRtrix")
    axes[0, 0].plot(ids, direction["fnit_valid_counts"], "s-", label="PyTorch")
    axes[0, 0].set(xlabel="RNG seed", ylabel="Valid starts / 10,000")
    for values, label in ((official[official[:, 0] == 1, 4], "MRtrix"), (amplitude, "PyTorch")):
        axes[0, 1].plot(*_ecdf(values), label=label)
    axes[0, 1].set(xlabel="Initial FOD amplitude", ylabel="Cumulative fraction")
    for axis, metric, field, label in (
        (axes[1, 0], "count", "relative_l1_upper", "Count relative L1"),
        (axes[1, 1], "mean_fa", "normalized_mae_common_support", "Mean FA common-edge nMAE"),
    ):
        old, new = [], []
        for report in reports:
            for reference in report["arms"]["discrete128"]["metrics"]:
                old.append(report["arms"]["discrete128"]["metrics"][reference][metric][field])
                new.append(report["arms"]["continuous1000"]["metrics"][reference][metric][field])
        axis.scatter(old, new, s=20)
        limit = max(old + new) * 1.05
        axis.plot([0, limit], [0, limit], color="gray", linewidth=1)
        axis.set(xlabel="Discrete 128", ylabel="Continuous 1000", title=label,
                 xlim=(0, limit), ylim=(0, limit))
    for axis in axes[0]:
        axis.legend(frameon=False)
    for axis in axes.flat:
        axis.grid(alpha=.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=170)
    plt.close(figure)


if __name__ == "__main__":
    main()
