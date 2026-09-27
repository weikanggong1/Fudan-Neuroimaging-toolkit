"""Plot public DWI NODDI maps from DIPY-backed and standalone FNIT runs."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--brain", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Public DWI: old and standalone FNIT NODDI")
    args = parser.parse_args()
    brain = np.asarray(nib.load(str(args.brain)).dataobj, dtype=np.float32)
    mask = np.asarray(nib.load(str(args.mask)).dataobj, dtype=np.uint8) == 1
    if brain.shape != mask.shape:
        raise ValueError("brain and mask shapes must match")
    z = int(np.argmax(mask.sum(axis=(0, 1))))
    top = float(np.percentile(brain[brain > 0], 99))
    fig, axes = plt.subplots(3, 3, figsize=(13, 12), constrained_layout=True)
    for row, name in enumerate(("NDI", "ODI", "FWF")):
        ref = nib.load(str(args.reference_dir / f"fit_{name}.nii.gz"))
        candidate = nib.load(str(args.candidate_dir / f"fit_{name}.nii.gz"))
        if ref.shape != mask.shape or candidate.shape != mask.shape:
            raise ValueError(f"{name} atlas and mask shapes must match")
        first = np.asarray(ref.dataobj, dtype=np.float32)
        second = np.asarray(candidate.dataobj, dtype=np.float32)
        delta = np.abs(first - second)
        for column, (values, label) in enumerate((
            (first, "DIPY-backed FNIT"),
            (second, "Standalone FNIT"),
            (delta, "Absolute difference"),
        )):
            ax = axes[row, column]
            ax.imshow(brain[:, :, z].T, cmap="gray", origin="lower", vmin=0, vmax=top)
            visible = mask[:, :, z] & ((values[:, :, z] > 0) if column == 2 else True)
            overlay = np.ma.masked_where(~visible, values[:, :, z])
            ax.imshow(overlay.T, cmap="magma" if column == 2 else "viridis",
                      origin="lower", alpha=0.85, vmin=0,
                      vmax=max(1e-7, float(delta.max())) if column == 2 else 1)
            ax.set_title(f"{name}: {label}")
            ax.axis("off")
    fig.suptitle(args.title)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
