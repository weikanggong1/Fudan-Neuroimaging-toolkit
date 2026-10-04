"""Render brain-only real FAST PVE maps after benchmark timing completes."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-prefix", required=True)
    parser.add_argument("--second-prefix", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mask_image = nib.as_closest_canonical(nib.load(args.mask))
    mask = np.asarray(mask_image.dataobj) > 0
    positions = np.argwhere(mask)
    slice_z = int(np.median(positions[:, 2]))
    lower = positions.min(axis=0)
    upper = positions.max(axis=0) + 1
    crop = (slice(int(lower[0]), int(upper[0])),
            slice(int(lower[1]), int(upper[1])), slice_z)
    figure, axes = plt.subplots(3, 3, figsize=(9, 9), constrained_layout=True)
    for row, tissue in enumerate(("CSF", "GM", "WM")):
        maps = []
        for prefix in (args.first_prefix, args.second_prefix):
            image = nib.as_closest_canonical(nib.load(f"{prefix}_pve_{row}.nii.gz"))
            if image.shape != mask.shape or not np.array_equal(image.affine, mask_image.affine):
                raise ValueError("maps and mask must share the canonical geometry")
            maps.append(np.where(mask, np.asarray(image.dataobj), 0)[crop].T)
        for column in range(3):
            values = maps[column] if column < 2 else np.abs(maps[0] - maps[1])
            view = axes[row, column].imshow(values, origin="lower", cmap="gray" if column < 2 else "magma",
                                            vmin=0, vmax=1 if column < 2 else 0.01)
            axes[row, column].axis("off")
            axes[row, column].set_title(f"{tissue}: " + ("FSL CPU", "FNIT CPU fsl", "absolute difference")[column])
            if column == 2:
                figure.colorbar(view, ax=axes[row, column], shrink=0.6)
    figure.suptitle("Same real brain input; central axial PVE slice")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
