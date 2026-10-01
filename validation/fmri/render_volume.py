"""Render deidentified template-space BOLD examples from a completed real run."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bold", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--figure-out", type=Path, required=True)
    parser.add_argument("--signal", choices=("clean", "preproc"), default="clean")
    parser.add_argument("--third-map", choices=("frame", "mean"), default="frame")
    parser.add_argument("--title", help="Explicit snapshot/scope label for a new figure")
    args = parser.parse_args()
    bold, mask, template = (nib.load(str(p)) for p in (args.bold, args.mask, args.template))
    if bold.shape[:3] != template.shape or mask.shape != template.shape or not all(
            np.allclose(image.affine, template.affine, atol=1e-4, rtol=0) for image in (bold, mask)):
        raise ValueError("BOLD, mask and template grids differ")
    values = np.asarray(bold.dataobj, dtype=np.float32)
    brain = np.asarray(mask.dataobj) > 0
    frame = bold.shape[3] // 2
    third = values.mean(axis=3, dtype=np.float64) if args.third_map == "mean" else values[..., frame]
    maps = (np.asarray(template.dataobj, dtype=np.float32), values.std(axis=3, dtype=np.float64), third)
    labels = ("MNI template (anatomical reference)", f"FNIT {args.signal} BOLD: temporal SD",
              f"FNIT {args.signal} BOLD: " + ("temporal mean\n" if args.third_map == "mean" else f"frame {frame}\n") +
              ("after confound regression" if args.signal == "clean" else "original intensity scale"))
    planes = ("Sagittal", "Coronal", "Axial")
    fig, axes = plt.subplots(3, 3, figsize=(10, 10), facecolor="white")
    for row, (data, label) in enumerate(zip(maps, labels)):
        display = np.where(brain, data, 0)
        scale = max(float(np.percentile(np.abs(display[brain]), 99)), 1e-6)
        for axis in range(3):
            plane = np.rot90(np.take(display, display.shape[axis] // 2, axis=axis))
            signed = row == 2 and args.signal == "clean"
            rendered = axes[row, axis].imshow(plane, cmap="coolwarm" if signed else "gray",
                                             vmin=-scale if signed else 0, vmax=scale)
            axes[row, axis].axis("off")
            axes[row, axis].set_title(f"{label}\n{planes[axis]}", fontsize=10)
        fig.colorbar(rendered, ax=axes[row, -1], shrink=0.7)
    fig.suptitle(args.title or f"Real BOLD: {bold.shape[3]} frames; output MNI grid {bold.shape[:3]}", fontsize=12)
    fig.tight_layout()
    args.figure_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure_out, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
