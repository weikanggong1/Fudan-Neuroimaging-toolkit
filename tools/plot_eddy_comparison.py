#!/usr/bin/env python3
"""Plot one real AP DWI slice from matched EDDY runs."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--fsl", type=Path, required=True)
    parser.add_argument("--fnit", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mask = np.asarray(nib.load(str(args.mask)).dataobj) > 0
    images = []
    for path in (args.raw, args.fsl, args.fnit):
        data = np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)
        if data.shape[:3] != mask.shape or data.ndim != 4:
            raise ValueError(f"invalid 4D image shape: {path}")
        images.append(data.mean(axis=3))
    z = int(np.argmax(mask.sum(axis=(0, 1))))
    raw, fsl, fnit = images
    vmax = float(np.percentile(np.concatenate((raw[mask], fsl[mask], fnit[mask])), 99))
    diff = np.abs(fnit - fsl)
    diff_max = max(float(np.percentile(diff[mask], 99)), 1e-6)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), constrained_layout=True)
    for ax, data, title, limit, cmap in zip(
        axes,
        (raw, fsl, fnit, diff),
        ("Raw AP mean", "FSL eddy_cuda10.2", "FNIT TorchEDDY", "Absolute difference"),
        (vmax, vmax, vmax, diff_max),
        ("gray", "gray", "gray", "magma"),
    ):
        artist = ax.imshow(np.rot90(data[:, :, z]), cmap=cmap, vmin=0, vmax=limit)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.colorbar(artist, ax=axes[3], fraction=0.046, pad=0.02)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
