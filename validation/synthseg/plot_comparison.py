#!/usr/bin/env python3
"""Plot matched slices from an input T1 and two SynthSeg label maps."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np


def _plane(data, axis, index):
    return np.rot90(np.take(data, index, axis=axis))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="input T1w image")
    parser.add_argument("--reference", type=Path, required=True,
                        help="FreeSurfer mri_synthseg label image")
    parser.add_argument("--candidate", type=Path, required=True,
                        help="FNIT SynthSeg label image")
    parser.add_argument("--output", type=Path, required=True, help="output PNG")
    args = parser.parse_args()

    reference_image = nib.load(args.reference)
    candidate_image = nib.load(args.candidate)
    reference = np.asarray(reference_image.dataobj)
    candidate = np.asarray(candidate_image.dataobj)
    if reference.shape != candidate.shape or not np.allclose(
            reference_image.affine, candidate_image.affine, atol=1e-5, rtol=0):
        raise ValueError("reference and candidate label grids differ")

    t1 = resample_from_to(nib.load(args.image), reference_image, order=1).get_fdata(
        dtype=np.float32)
    brain = reference != 0
    coordinates = np.argwhere(brain)
    if not len(coordinates):
        raise ValueError("reference segmentation has no foreground voxels")
    center = np.round(np.median(coordinates, axis=0)).astype(int)
    slices = ((2, int(center[2]), "Axial"), (1, int(center[1]), "Coronal"))
    foreground = t1[brain]
    low, high = np.percentile(foreground, (1, 99))
    difference = reference != candidate

    figure, axes = plt.subplots(2, 4, figsize=(13.5, 7), constrained_layout=True)
    titles = ("Public T1w", "FreeSurfer mri_synthseg", "FNIT SynthSeg", "Different labels")
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, fontsize=12)
    for row, (axis, index, label) in enumerate(slices):
        background = _plane(t1, axis, index)
        axes[row, 0].imshow(background, cmap="gray", vmin=low, vmax=high)
        for column, segmentation in ((1, reference), (2, candidate)):
            axes[row, column].imshow(background, cmap="gray", vmin=low, vmax=high)
            overlay = np.ma.masked_where(
                _plane(segmentation, axis, index) == 0,
                _plane(segmentation, axis, index),
            )
            axes[row, column].imshow(overlay, cmap="nipy_spectral", alpha=0.62,
                                     interpolation="nearest")
        axes[row, 3].imshow(background, cmap="gray", vmin=low, vmax=high)
        diff_slice = np.ma.masked_where(
            ~_plane(difference, axis, index), _plane(difference, axis, index))
        axes[row, 3].imshow(diff_slice, cmap="autumn", vmin=0, vmax=1, alpha=0.9,
                            interpolation="nearest")
        axes[row, 0].set_ylabel(label, fontsize=11)
        for panel in axes[row]:
            panel.set_xticks([])
            panel.set_yticks([])

    agreement = float(np.mean(reference == candidate))
    figure.suptitle(
        f"33-class labels: voxel agreement {agreement:.8f}; "
        f"{np.count_nonzero(difference):,} different voxels",
        fontsize=13,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
