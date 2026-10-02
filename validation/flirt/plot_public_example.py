#!/usr/bin/env python3
"""Plot paired real T1w outputs on their reference grid, without registration."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moving", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--fsl-moved", required=True)
    parser.add_argument("--fnit-moved", required=True)
    parser.add_argument("--figure", required=True)
    args = parser.parse_args(argv)

    paths = {"input": args.moving, "reference": args.reference,
             "fsl": args.fsl_moved, "fnit": args.fnit_moved}
    images = {name: nib.as_closest_canonical(nib.load(path))
              for name, path in paths.items()}
    reference = images["reference"]
    for name in ("fsl", "fnit"):
        if (images[name].shape != reference.shape
                or not np.allclose(images[name].affine, reference.affine,
                                   atol=1e-5, rtol=0)):
            raise ValueError(f"{name} output must match the reference grid")
    display = {name: np.asarray(image.dataobj, dtype=np.float32)
               for name, image in images.items()}
    display["difference"] = np.abs(display["fnit"] - display["fsl"])
    mask = display["reference"] > 0
    centre = np.rint(np.argwhere(mask).mean(axis=0)).astype(int)
    input_centre = np.rint(np.argwhere(display["input"] > 0).mean(axis=0)).astype(int)
    output_limits = np.percentile(np.concatenate(
        [display[name][mask] for name in ("fsl", "fnit")]), [1, 99])
    difference_max = max(float(np.percentile(display["difference"][mask], 99.5)), 1e-6)
    names = ("input", "reference", "fsl", "fnit", "difference")
    titles = ("Input", "Reference", "FSL FLIRT", "FNIT batched GPU", "Absolute difference")
    figure, axes = plt.subplots(3, 5, figsize=(13.5, 8.2), constrained_layout=True)
    for row, (axis, view) in enumerate(((2, "Axial"), (1, "Coronal"), (0, "Sagittal"))):
        for column, name in enumerate(names):
            data = display[name]
            index = int(input_centre[axis] if name == "input" else centre[axis])
            if name == "difference":
                vmin, vmax, cmap = 0.0, difference_max, "magma"
            elif name in ("fsl", "fnit"):
                vmin, vmax = output_limits
                cmap = "gray"
            else:
                vmin, vmax = np.percentile(data[data > 0], [1, 99])
                cmap = "gray"
            axes[row, column].imshow(np.rot90(np.take(data, index, axis=axis)),
                                    cmap=cmap, vmin=vmin, vmax=vmax)
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(titles[column], fontsize=11)
            if column == 0:
                axes[row, column].text(-0.04, 0.5, view, rotation=90, va="center",
                                       ha="right", transform=axes[row, column].transAxes)
    figure.suptitle("OpenNeuro ds000114: 12-DOF correlation-ratio registration")
    output = Path(args.figure)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
