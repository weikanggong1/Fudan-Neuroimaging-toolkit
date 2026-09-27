"""Render a fixed-grid brain-image comparison of original and PyTorch cortical atlases."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--brain", type=Path, required=True)
    parser.add_argument("--ribbon", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Real FreeSurfer aparc; identical ribbon, vertices and labels")
    args = parser.parse_args()
    reference, candidate, brain, ribbon = [nib.load(str(path)) for path in
                                            (args.reference, args.candidate, args.brain, args.ribbon)]
    labels = np.asarray(reference.dataobj)
    predicted = np.asarray(candidate.dataobj)
    intensity = np.asarray(brain.dataobj)
    mask = np.isin(np.asarray(ribbon.dataobj), (3, 42))
    assert labels.shape == predicted.shape == intensity.shape == mask.shape
    assert np.allclose(reference.affine, candidate.affine)
    z = int(np.argmax(mask.sum(axis=(0, 1))))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    top = float(np.percentile(intensity[intensity > 0], 99))
    for ax, overlay, title in zip(
        axes, (labels, predicted, (labels != predicted).astype(int)),
        ("Original UKB Python", "FNIT PyTorch", "Mismatched voxels"),
    ):
        ax.imshow(intensity[:, :, z].T, cmap="gray", origin="lower", vmin=0, vmax=top)
        visible = mask[:, :, z] & (overlay[:, :, z] != 0)
        ax.imshow(np.ma.masked_where(~visible, overlay[:, :, z]).T,
                  cmap="turbo" if title != "Mismatched voxels" else "Reds",
                  origin="lower", alpha=0.82,
                  vmin=0, vmax=(1 if title == "Mismatched voxels"
                                else max(int(labels.max()), int(predicted.max()))))
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(args.title)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
