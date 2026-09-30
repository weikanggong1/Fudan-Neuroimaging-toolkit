"""Plot temporal agreement on standard fsLR spheres; no subject geometry is exported."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--left-sphere", type=Path, required=True)
    parser.add_argument("--right-sphere", type=Path, required=True)
    parser.add_argument("--figure-out", type=Path, required=True)
    args = parser.parse_args()
    candidate = nib.load(str(args.candidate))
    reference = nib.load(str(args.reference))
    if candidate.shape != reference.shape or not all(
            candidate.header.get_axis(i) == reference.header.get_axis(i) for i in (0, 1)):
        raise ValueError("CIFTI axes differ")
    first = np.asarray(candidate.dataobj, dtype=np.float64)
    second = np.asarray(reference.dataobj, dtype=np.float64)
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("nonfinite CIFTI")
    first -= first.mean(axis=0)
    second -= second.mean(axis=0)
    denominator = np.sqrt((first * first).sum(axis=0) * (second * second).sum(axis=0))
    correlation = np.full(candidate.shape[1], np.nan)
    valid = denominator > 1e-8
    correlation[valid] = (first[:, valid] * second[:, valid]).sum(axis=0) / denominator[valid]
    models = candidate.header.get_axis(1)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for index, (hemi, sphere_path) in enumerate((("LEFT", args.left_sphere), ("RIGHT", args.right_sphere))):
        name = "CIFTI_STRUCTURE_CORTEX_" + hemi
        points = nib.load(str(sphere_path)).agg_data("pointset")
        if points.shape != (32492, 3):
            raise ValueError("expected standard fsLR32k sphere")
        values = np.full(32492, np.nan)
        selected = models.name == name
        values[models.vertex[selected]] = correlation[selected]
        visible = points[:, 0] >= 0
        scale = np.linalg.norm(points, axis=1).mean()
        axes[index].add_patch(plt.Circle((0, 0), scale, color="0.88"))
        scatter = axes[index].scatter(points[visible, 1], points[visible, 2],
                                      c=values[visible], cmap="viridis", vmin=0.5, vmax=1,
                                      s=3, linewidths=0, rasterized=True)
        axes[index].set_aspect("equal")
        axes[index].set_xlim(-scale * 1.04, scale * 1.04)
        axes[index].set_ylim(-scale * 1.04, scale * 1.04)
        axes[index].axis("off")
        axes[index].set_title(f"fsLR32k standard sphere: {hemi.title()} (+x)")
        finite = values[np.isfinite(values)]
        axes[2].hist(finite, bins=np.linspace(-1, 1, 101), histtype="step",
                     label=f"{hemi.title()}: mean r={finite.mean():.4f}", linewidth=1.5)
    figure.colorbar(scatter, ax=axes[:2], shrink=0.7, label="Temporal Pearson r (490 frames)")
    axes[2].set_xlabel("Temporal Pearson r")
    axes[2].set_ylabel("Cortical grayordinates")
    axes[2].legend(loc="upper left", fontsize=8)
    axes[2].set_title("All valid cortical grayordinates")
    figure.suptitle("FNIT vs official newMSM spheres on the same cleaned BOLD\n"
                     "Same projection and assembly; gray = medial wall / constant signal", fontsize=12)
    args.figure_out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure_out, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
