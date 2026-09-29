"""Plot three high-disagreement conform slices from two real recon-all subjects."""

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
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = np.asarray(nib.load(str(args.candidate / "mri/orig.mgz")).dataobj)
    got = np.asarray(nib.load(str(args.candidate / "mri/filled.mgz")).dataobj)
    want = np.asarray(nib.load(str(args.reference / "mri/filled.mgz")).dataobj)
    if not (base.shape == got.shape == want.shape):
        raise ValueError("reference and candidate must have the same conform grid")
    changed = got != want
    slices = [int(np.argmax(changed.sum(axis=tuple(i for i in range(3)
                                                   if i != axis))))
              for axis in range(3)]
    fig, axes = plt.subplots(3, 3, figsize=(11, 10), constrained_layout=True)
    for col, (axis, index) in enumerate(zip(range(3), slices)):
        image = np.rot90(np.take(base, index, axis=axis))
        reference = np.rot90(np.take(want, index, axis=axis))
        candidate = np.rot90(np.take(got, index, axis=axis))
        diff = reference != candidate
        for row, label in enumerate(("Reference", "FNIT", "Disagreement")):
            panel = axes[row, col]
            panel.imshow(image, cmap="gray", vmin=0, vmax=180,
                         interpolation="nearest")
            overlay = np.zeros((*image.shape, 4), dtype=np.float32)
            if row < 2:
                values = reference if row == 0 else candidate
                overlay[values == 127] = (0., .8, .9, .45)
                overlay[values == 255] = (.9, .2, .6, .45)
            else:
                overlay[diff] = (1., .2, 0., .9)
            panel.imshow(overlay, interpolation="nearest")
            title = f"{label}: axis {axis}, slice {index}"
            if row == 2:
                title += f" ({np.count_nonzero(diff)} voxels)"
            panel.set_title(title, fontsize=10)
            panel.axis("off")
    fig.suptitle("filled segmentation: reference vs FNIT", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)


if __name__ == "__main__":
    main()
