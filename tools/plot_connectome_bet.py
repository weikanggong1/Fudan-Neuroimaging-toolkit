"""Plot FSL and native BET masks on a paired public mean-b0 image."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mean-b0", required=True, type=Path)
    parser.add_argument("--reference-mask", required=True, type=Path)
    parser.add_argument("--candidate-mask", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    image = np.asarray(nib.load(str(args.mean_b0)).dataobj)
    reference = np.asarray(nib.load(str(args.reference_mask)).dataobj).astype(bool)
    candidate = np.asarray(nib.load(str(args.candidate_mask)).dataobj).astype(bool)
    if image.shape != reference.shape or image.shape != candidate.shape:
        raise ValueError("all inputs must use the same 3D voxel grid")
    center = np.rint(np.argwhere(reference).mean(axis=0)).astype(int)
    views = (("Axial", 2, int(center[2])), ("Coronal", 1, int(center[1])))
    low, high = np.percentile(image[reference | candidate], [2, 99])
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)
    for row, (name, axis, index) in enumerate(views):
        base = np.rot90(np.take(image, index, axis=axis))
        masks = [np.rot90(np.take(m, index, axis=axis)) for m in
                 (reference, candidate, reference ^ candidate)]
        for col, mask in enumerate(masks):
            ax = axes[row, col]
            ax.imshow(base, cmap="gray", vmin=low, vmax=high, interpolation="nearest")
            if col < 2:
                ax.contour(mask.astype(float), levels=[.5], colors=["#ff7f0e"], linewidths=.8)
            else:
                ax.imshow(np.ma.masked_where(~mask, mask), cmap="autumn", vmin=0, vmax=1,
                          alpha=.9, interpolation="nearest")
            ax.set_title(f"{('FSL BET', 'FNIT PyTorch', 'XOR')[col]} · {name}")
            ax.axis("off")
    fig.suptitle(f"OpenNeuro ds004666 · mask XOR {int(np.count_nonzero(reference ^ candidate)):,} voxels")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
