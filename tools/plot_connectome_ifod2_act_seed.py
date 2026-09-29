"""绘制真实 5TT 的 ACT 种子分类和同输入连接矩阵。"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("five-tissue", "seeds", "act-csv", "official-count", "candidate-count", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--ab-report", type=Path, action="append", required=True)
    args = parser.parse_args()

    image = nib.load(str(args.five_tissue))
    tissue = np.asarray(image.dataobj, dtype=np.float32)
    points = np.loadtxt(args.seeds, dtype=np.float32)
    act = np.loadtxt(args.act_csv, delimiter=",")
    voxels = nib.affines.apply_affine(np.linalg.inv(image.affine), points)
    slice_index = int(np.median(voxels[act[:, 0] == 1, 2]))
    selected = (act[:, 0] == 1) & (np.abs(voxels[:, 2] - slice_index) < 1.5)
    one_way = selected & (act[:, 1] == 1)
    other = selected & (act[:, 1] == 0)
    official = np.loadtxt(args.official_count, delimiter=",")
    candidate = np.loadtxt(args.candidate_count, delimiter=",")
    official = official.copy()
    candidate = candidate.copy()
    np.fill_diagonal(official, 0)
    np.fill_diagonal(candidate, 0)
    reports = [json.loads(path.read_text()) for path in args.ab_report]
    values = {key: [] for key in ("bidirectional", "act_one_way")}
    for report in reports:
        for arm in values:
            for reference in report["arms"][arm]["metrics"].values():
                values[arm].append(reference["count"]["relative_l1_upper"])

    figure, axes = plt.subplots(1, 4, figsize=(16, 4.2), layout="constrained")
    contrast = tissue[:, :, slice_index, 0] - tissue[:, :, slice_index, 2]
    axes[0].imshow(contrast.T, origin="lower", cmap="coolwarm", vmin=-1, vmax=1)
    axes[0].scatter(voxels[one_way, 0], voxels[one_way, 1], s=2, c="black", alpha=.6,
                    label=f"cGM one-way: {one_way.sum()}")
    axes[0].scatter(voxels[other, 0], voxels[other, 1], s=8, c="lime", alpha=.8,
                    label=f"other: {other.sum()}")
    axes[0].set_title(f"Real 5TT, axial voxel {slice_index}")
    axes[0].legend(loc="upper right", fontsize=7)
    vmax = max(float(official.max()), float(candidate.max()))
    for axis, matrix, title in zip(axes[1:3], (official, candidate),
                                   ("MRtrix count, off diagonal", "FNIT count, off diagonal")):
        axis.imshow(matrix, cmap="magma", vmin=0, vmax=vmax)
        axis.set_title(title)
        axis.set_xlabel("Node")
        axis.set_ylabel("Node")
    x = np.arange(len(values["bidirectional"]))
    axes[3].plot(x, values["bidirectional"], "o", ms=3, label="bidirectional")
    axes[3].plot(x, values["act_one_way"], "o", ms=3, label="ACT one-way")
    for index in x:
        axes[3].plot([index, index], [values["bidirectional"][index],
                                       values["act_one_way"][index]], color=".7", lw=.6)
    axes[3].set_title("Paired count error, 3 seeds x 3 MRtrix")
    axes[3].set_xlabel("Comparison")
    axes[3].set_ylabel("Upper-triangle relative L1")
    axes[3].legend(fontsize=7)
    figure.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
