"""绘制同一真实 5TT 上两次 MRtrix 与一次 FNIT 的 GMWMI 播种投影。"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--official-repeat", type=Path, required=True)
    parser.add_argument("--fnit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seeds = [np.loadtxt(path, delimiter=",", comments="#", usecols=(2, 3, 4))
             for path in (args.official, args.official_repeat)]
    seeds.append(np.load(args.fnit))
    all_points = np.concatenate(seeds)
    bins = [np.arange(np.floor(all_points[:, axis].min() / 8) * 8,
                      np.ceil(all_points[:, axis].max() / 8) * 8 + 8, 8)
            for axis in (0, 1)]
    hist = [np.histogram2d(points[:, 0], points[:, 1], bins=bins)[0]
            for points in seeds]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), constrained_layout=True)
    extent = (bins[0][0], bins[0][-1], bins[1][0], bins[1][-1])
    maximum = max(float(values.max()) for values in hist)
    for axis, values, title in zip(axes[:3], hist, ("MRtrix seed 0", "MRtrix seed 1", "FNIT seed 0")):
        view = axis.imshow(values.T, origin="lower", extent=extent, cmap="magma",
                           vmin=0, vmax=maximum, interpolation="nearest")
        axis.set(title=title, xlabel="RAS x (mm)")
    axes[0].set_ylabel("RAS y (mm)")
    difference = hist[2] - (hist[0] + hist[1]) / 2
    limit = max(float(np.abs(difference).max()), 1.)
    diff_view = axes[3].imshow(difference.T, origin="lower", extent=extent,
                                cmap="RdBu_r", vmin=-limit, vmax=limit,
                                interpolation="nearest")
    axes[3].set(title="FNIT - MRtrix mean", xlabel="RAS x (mm)")
    fig.colorbar(view, ax=axes[:3], label="seeds / 8 mm bin")
    fig.colorbar(diff_view, ax=axes[3], label="seed difference")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
