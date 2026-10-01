"""Plot six axial slices of a real T1 and two subregion segmentations."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np


FAMILIES = (
    ("Brainstem", (1., .65, .1)), ("Thalamus", (.1, .8, .9)),
    ("Left hippocampus", (.25, .45, 1.)), ("Left amygdala", (.75, .3, .95)),
    ("Right hippocampus", (.35, .9, .35)), ("Right amygdala", (1., .3, .35)),
)


def colors(labels):
    masks = (np.isin(labels, (173, 174, 175, 178)),
             (labels >= 8100) & (labels < 8300),
             (labels >= 200) & (labels <= 246), (labels >= 7000) & (labels < 8000),
             (labels >= 10200) & (labels <= 10246), (labels >= 17000) & (labels < 18000))
    rgba = np.zeros((*labels.shape, 4), np.float32)
    for mask, (_, rgb) in zip(masks, FAMILIES):
        rgba[mask, :3] = rgb
        rgba[mask, 3] = .75
    return rgba


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t1", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--reference", required=True, nargs="+", type=Path,
                        help="one unified FNIT label image, or four official structure images")
    parser.add_argument("--reference-name", default="FreeSurfer 8.2")
    parser.add_argument("--candidate-name", default="FNIT TorchGEMS")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if len(args.reference) not in (1, 4):
        parser.error("--reference requires one unified label image or four official structure images")
    image = nib.as_closest_canonical(nib.load(args.t1))
    grid = (image.shape, image.affine)
    data = np.asarray(image.dataobj, dtype=np.float32)
    candidate = np.asarray(resample_from_to(nib.load(args.candidate), grid, order=0).dataobj, np.int32)
    reference = np.zeros(image.shape, np.int32)
    for index, path in enumerate(args.reference):
        labels = np.asarray(resample_from_to(nib.load(path), grid, order=0).dataobj, np.int32)
        if len(args.reference) == 4 and index == 3:
            labels = np.where(labels != 0, labels + 10000, 0)
        reference[labels != 0] = labels[labels != 0]
    points = np.argwhere((reference != 0) | (candidate != 0))
    lo = np.maximum(points.min(0) - 8, 0)
    hi = np.minimum(points.max(0) + 9, image.shape)
    crop = (slice(lo[0], hi[0]), slice(lo[1], hi[1]))
    indices = np.linspace(points[:, 2].min(), points[:, 2].max(), 8).round().astype(int)[1:-1]
    fig, axes = plt.subplots(3, 6, figsize=(15, 7.2), facecolor="black")
    vmax = np.percentile(data[data > 0], 99)
    for col, index in enumerate(indices):
        t1 = data[crop + (index,)].T
        for row, labels in enumerate((reference, candidate)):
            axes[row, col].imshow(t1, cmap="gray", vmin=0, vmax=vmax, origin="lower")
            overlay = colors(labels[crop + (index,)])
            axes[row, col].imshow(overlay.transpose(1, 0, 2), origin="lower")
        axes[2, col].imshow(t1, cmap="gray", vmin=0, vmax=vmax, origin="lower")
        difference = reference[crop + (index,)] != candidate[crop + (index,)]
        overlay = np.zeros((*difference.shape, 4), np.float32)
        overlay[difference] = (1., .2, .15, .85)
        axes[2, col].imshow(overlay.transpose(1, 0, 2), origin="lower")
        z = (image.affine @ np.array([0, 0, index, 1]))[2]
        axes[0, col].set_title(f"z = {z:.1f} mm", color="white", fontsize=11)
    for row, name in enumerate((args.reference_name, args.candidate_name, "Label differences")):
        axes[row, 0].set_ylabel(name, color="white", fontsize=11)
    for axis in axes.flat:
        axis.set_xticks([]); axis.set_yticks([])
    fig.legend(handles=[Patch(color=rgb, label=name) for name, rgb in FAMILIES],
               loc="lower center", ncol=3, labelcolor="white", facecolor="black", edgecolor="none")
    fig.subplots_adjust(left=.06, right=.995, bottom=.11, top=.96, wspace=.02, hspace=.04)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, facecolor="black")
    plt.close(fig)


if __name__ == "__main__":
    main()
