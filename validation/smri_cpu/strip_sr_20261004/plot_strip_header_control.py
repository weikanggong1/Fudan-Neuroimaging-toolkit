"""Render the verified real T1 masks; no resampling or inference is performed."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.orientations import apply_orientation, axcodes2ornt, io_orientation, ornt_transform
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference-mask", type=Path, required=True)
    parser.add_argument("--old-mask", type=Path, required=True)
    parser.add_argument("--corrected-mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve previous figures and choose a fresh output")
    images = [nib.load(str(path)) for path in
              (args.input, args.reference_mask, args.old_mask, args.corrected_mask)]
    if any(image.shape != images[0].shape or not np.allclose(image.affine, images[0].affine, rtol=0, atol=1e-6)
           for image in images[1:]):
        raise ValueError("plot needs already matched voxel grids")
    orientation = ornt_transform(io_orientation(images[0].affine), axcodes2ornt("RAS"))
    intensity, reference, old, corrected = [apply_orientation(np.asanyarray(image.dataobj), orientation)
                                           for image in images]
    reference, old, corrected = reference != 0, old != 0, corrected != 0
    full_grid_old_difference = int(np.count_nonzero(old != reference))
    full_grid_corrected_difference = int(np.count_nonzero(corrected != reference))
    locations = np.where(reference)
    bounds = [slice(max(int(axis.min()) - 5, 0), min(int(axis.max()) + 6, reference.shape[i]))
              for i, axis in enumerate(locations)]
    intensity, reference, old, corrected = [array[tuple(bounds)]
                                           for array in (intensity, reference, old, corrected)]
    old_difference = old != reference
    maximum = float(np.percentile(intensity[reference], 99))
    figure, axes = plt.subplots(3, 3, figsize=(11, 10))
    columns = ("Official mask", "Old header: disagreement", "Corrected CPU: disagreement")
    for axis, view_name in enumerate(("Sagittal", "Coronal", "Axial")):
        counts = old_difference.sum(axis=tuple(i for i in range(3) if i != axis))
        index = int(np.argmax(counts))
        image_slice = np.take(intensity, index, axis=axis).T
        reference_slice = np.take(reference, index, axis=axis).T
        for column in range(3):
            panel = axes[axis, column]
            panel.imshow(image_slice, cmap="gray", vmin=0, vmax=maximum, origin="lower")
            if column == 0:
                panel.contour(reference_slice, levels=[.5], colors=["#68bcff"], linewidths=.7)
            else:
                mask = old if column == 1 else corrected
                mask_slice = np.take(mask, index, axis=axis).T
                colors = np.zeros(image_slice.shape + (4,))
                colors[reference_slice & ~mask_slice] = (1, .2, .1, .9)
                colors[~reference_slice & mask_slice] = (0, .85, 1, .9)
                panel.imshow(colors, origin="lower")
            panel.axis("off")
            if axis == 0:
                panel.set_title(columns[column], fontsize=11)
            if column == 0:
                panel.text(0, .5, view_name, transform=panel.transAxes, rotation=90,
                           ha="right", va="center", fontsize=11)
    figure.suptitle("Public raw T1 | same input and original output grid", fontsize=13)
    figure.text(.5, .022, "Red: missing; cyan: extra. Slices maximize old disagreement in each plane.\n"
                + "All-grid mask differences: old %d; corrected %d." %
                (full_grid_old_difference, full_grid_corrected_difference), ha="center", fontsize=10)
    figure.tight_layout(rect=(.02, .065, 1, .96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
