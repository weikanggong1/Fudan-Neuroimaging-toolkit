"""Plot public real-DWI original and native brain-mask slices."""

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
    parser.add_argument("--dwi", required=True, type=Path)
    parser.add_argument("--reference-mask", required=True, type=Path)
    parser.add_argument("--candidate-mask", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = np.asanyarray(nib.load(str(args.dwi)).dataobj)[..., 0]
    reference = np.asanyarray(nib.load(str(args.reference_mask)).dataobj).astype(bool)
    candidate = np.asanyarray(nib.load(str(args.candidate_mask)).dataobj).astype(bool)
    if reference.shape != candidate.shape or reference.shape != data.shape:
        raise ValueError("DWI and masks must share a voxel grid")
    center = np.rint(np.argwhere(reference).mean(axis=0)).astype(int)
    views = (("axial", 2, center[2]), ("coronal", 1, center[1]))
    figure, axes = plt.subplots(2, 3, figsize=(10.5, 6.5), constrained_layout=True)
    for row, (name, axis, index) in enumerate(views):
        image = np.rot90(np.take(data, index, axis=axis))
        original = np.rot90(np.take(reference, index, axis=axis))
        native = np.rot90(np.take(candidate, index, axis=axis))
        difference = original ^ native
        positive = image[original | native]
        low, high = np.percentile(positive, [2, 99])
        for column, mask in enumerate((original, native, difference)):
            panel = axes[row, column]
            panel.imshow(image, cmap="gray", vmin=low, vmax=high, interpolation="nearest")
            if column < 2:
                panel.contour(mask.astype(float), levels=[.5], colors=["#f97316"], linewidths=.8)
            else:
                panel.imshow(np.ma.masked_where(~mask, mask), cmap="autumn", vmin=0, vmax=1,
                             interpolation="nearest", alpha=.8)
            panel.set_title(("MRtrix legacy", "FNIT PyTorch", "XOR")[column] +
                            f" · {name} {index}", fontsize=10)
            panel.axis("off")
    figure.suptitle(f"OpenNeuro ds004666 · 3D mask XOR {np.count_nonzero(reference ^ candidate):,} voxels",
                    fontsize=12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
