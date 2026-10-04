"""Plot actual same-grid hard labels; never resample quantitative comparisons."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def plot(reference, candidate, output):
    first, second = nib.load(reference), nib.load(candidate)
    if first.shape != second.shape or not np.array_equal(first.affine, second.affine):
        raise ValueError("The figure requires exactly the same saved output grid")
    x, y = np.asanyarray(first.dataobj), np.asanyarray(second.dataobj)
    indices = np.argwhere(x > 0)
    middle = np.rint(np.median(indices, axis=0)).astype(int)
    difference = x != y
    changed = np.argwhere(difference)
    figure, axes = plt.subplots(3, 3, figsize=(10, 10), constrained_layout=True)
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_under("black")
    for axis in range(3):
        planes = [np.take(volume, middle[axis], axis=axis).T for volume in (x, y)]
        for row, plane in enumerate(planes):
            axes[row, axis].imshow(plane, origin="lower", interpolation="nearest",
                                   cmap=cmap, vmin=1, vmax=60)
            axes[row, axis].axis("off")
        plane_index = int(changed[0, axis]) if len(changed) else int(middle[axis])
        axes[2, axis].imshow(np.take(x > 0, plane_index, axis=axis).T,
                            origin="lower", cmap="gray", interpolation="nearest")
        overlay = np.ma.masked_where(~np.take(difference, plane_index, axis=axis).T,
                                    np.ones(np.take(difference, plane_index, axis=axis).T.shape))
        axes[2, axis].imshow(overlay, origin="lower", cmap="autumn", vmin=0, vmax=1,
                            interpolation="nearest")
        if len(changed):
            remaining = [v for v in range(3) if v != axis]
            axes[2, axis].scatter(changed[0, remaining[0]], changed[0, remaining[1]],
                                  s=120, facecolors="none", edgecolors="red", linewidths=1)
        axes[2, axis].axis("off")
        axes[0, axis].set_title(["Sagittal", "Coronal", "Axial"][axis])
    for axis, title in zip(axes[:, 0], ["Official", "FNIT CPU", "Changed voxel location"]):
        axis.text(-0.04, 0.5, title, rotation=90, transform=axis.transAxes,
                  va="center", ha="right")
    figure.suptitle(f"Real T1 case01: {len(changed)} differing label voxel(s); identical saved grid")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plot(args.reference, args.candidate, args.output)
