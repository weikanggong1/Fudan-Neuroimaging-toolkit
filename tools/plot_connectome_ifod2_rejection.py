"""绘制真实 FOD 拒绝概率和同一 20 节点连接矩阵的对照图。"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rejection-csv", type=Path, required=True)
    parser.add_argument("--official-count", type=Path, required=True)
    parser.add_argument("--fnit-count", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arc = np.loadtxt(args.rejection_csv, delimiter=",", skiprows=1)
    official = np.loadtxt(args.official_count, delimiter=",")
    fnit = np.loadtxt(args.fnit_count, delimiter=",")
    if arc.shape != (80, 6) or official.shape != (20, 20) or fnit.shape != (20, 20):
        raise ValueError("expected 80 real arcs and paired 20×20 count matrices")
    official = official.copy()
    fnit = fnit.copy()
    np.fill_diagonal(official, 0)
    np.fill_diagonal(fnit, 0)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for axis, column, title in ((axes[0, 0], 1, "Calibrated maximum"),
                                (axes[0, 1], 2, "Proposal probability")):
        reference, candidate = arc[:, column], arc[:, column + 3]
        axis.scatter(reference, candidate, s=15, alpha=.75)
        limit = max(reference.max(), candidate.max()) * 1.05
        axis.plot([0, limit], [0, limit], color="black", linewidth=1)
        axis.set(xlim=(0, limit), ylim=(0, limit), xlabel="MRtrix", ylabel="FNIT", title=title)
    error = np.abs(arc[:, 1:3] - arc[:, 4:6])
    axes[0, 2].hist(error[:, 0], bins=16, alpha=.7, label="Maximum")
    axes[0, 2].hist(error[:, 1], bins=16, alpha=.7, label="Proposal")
    axes[0, 2].set(xlabel="Absolute error", ylabel="Arc count", title="80 fixed real FOD arcs")
    axes[0, 2].legend(frameon=False)
    vmax = max(official.max(), fnit.max())
    for axis, matrix, title in ((axes[1, 0], official, "MRtrix count"),
                                (axes[1, 1], fnit, "FNIT count")):
        axis.imshow(matrix, vmin=0, vmax=vmax, cmap="viridis", interpolation="nearest")
        axis.set(title=title, xlabel="Node", ylabel="Node")
    difference = fnit - official
    scale = np.abs(difference).max()
    axes[1, 2].imshow(difference, vmin=-scale, vmax=scale, cmap="coolwarm",
                      interpolation="nearest")
    axes[1, 2].set(title="FNIT minus MRtrix", xlabel="Node", ylabel="Node")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
