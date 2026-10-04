"""Plot same-grid real CPU SynthSR outputs and their quantization differences."""

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
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a fresh output")
    first, second = nib.load(args.reference), nib.load(args.candidate)
    if first.shape != second.shape or not np.allclose(first.affine, second.affine, rtol=0, atol=1e-6):
        raise ValueError("plots require matched voxel grids")
    orientation = ornt_transform(io_orientation(first.affine), axcodes2ornt("RAS"))
    a, b = [apply_orientation(np.asanyarray(image.dataobj), orientation).astype(np.float32)
            for image in (first, second)]
    difference = np.abs(b - a)
    count = int(np.count_nonzero(difference))
    figure, axes = plt.subplots(3, 3, figsize=(11, 10))
    for axis, name in enumerate(("Sagittal", "Coronal", "Axial")):
        scores = np.count_nonzero(difference, axis=tuple(index for index in range(3) if index != axis))
        index = int(np.argmax(scores))
        for column, image in enumerate((a, b, a)):
            panel = axes[axis, column]
            panel.imshow(np.take(image, index, axis=axis).T, origin="lower", cmap="gray", vmin=0, vmax=255)
            if column == 2:
                mask = np.take(difference != 0, index, axis=axis).T
                overlay = np.zeros(mask.shape + (4,))
                overlay[mask] = (1, .1, 0, 1)
                panel.imshow(overlay, origin="lower")
            panel.axis("off")
            if axis == 0:
                panel.set_title(("Official TensorFlow CPU", "FNIT PyTorch CPU", "Red: different quantized voxel")[column])
            if column == 0:
                panel.text(0, .5, name, transform=panel.transAxes, rotation=90, ha="right", va="center")
    figure.suptitle("Public raw T1 | 1 mm SynthSR outputs | no display resampling", fontsize=13)
    figure.text(.5, .025, f"All-grid differences: {count}/{a.size}; max intensity difference: {difference.max():g}.\n"
                "Slices maximize the quantization disagreement in each plane.", ha="center")
    figure.tight_layout(rect=(0, .075, 1, .96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
