#!/usr/bin/env python3
"""Plot one de-identified FSL/FNIT TOPUP comparison."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def _load(path):
    return np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)


def _slice(volume, index):
    return np.rot90(volume[:, :, index])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="two-volume b0 NIfTI")
    parser.add_argument("--fsl-dir", required=True)
    parser.add_argument("--fnit-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    raw = _load(args.input)
    fsl_dir = Path(args.fsl_dir)
    fnit_dir = Path(args.fnit_dir)
    fsl_iout = _load(fsl_dir / "fieldmap_iout.nii.gz").mean(axis=3)
    fnit_iout = _load(fnit_dir / "fieldmap_iout.nii.gz").mean(axis=3)
    fsl_field = _load(fsl_dir / "fieldmap_fout.nii.gz")
    fnit_field = _load(fnit_dir / "fieldmap_fout.nii.gz")
    z = raw.shape[2] // 2

    anatomical = [raw[..., 0], raw[..., 1], fsl_iout, fnit_iout]
    vmax = np.percentile(np.concatenate([value[value > 0] for value in anatomical]), 99.5)
    field_limit = np.percentile(np.abs(np.concatenate((fsl_field.ravel(), fnit_field.ravel()))), 99)
    corrected_difference = np.abs(fsl_iout - fnit_iout)
    field_difference = np.abs(fsl_field - fnit_field)

    fig, axes = plt.subplots(2, 4, figsize=(13.2, 7.0), constrained_layout=True)
    for axis, image, title in zip(
        axes[0], anatomical,
        ("Raw AP b0", "Raw PA b0", "FSL corrected mean", "FNIT corrected mean"),
    ):
        axis.imshow(_slice(image, z), cmap="gray", vmin=0, vmax=vmax)
        axis.set_title(title)
        axis.axis("off")

    panels = (
        (fsl_field, "FSL field (Hz)", "coolwarm", -field_limit, field_limit),
        (fnit_field, "FNIT field (Hz)", "coolwarm", -field_limit, field_limit),
        (field_difference, "Absolute field difference (Hz)", "magma", 0, np.percentile(field_difference, 99)),
        (corrected_difference, "Absolute corrected-image difference", "magma", 0, np.percentile(corrected_difference, 99)),
    )
    for axis, (image, title, cmap, vmin, vmax_panel) in zip(axes[1], panels):
        shown = axis.imshow(_slice(image, z), cmap=cmap, vmin=vmin, vmax=vmax_panel)
        axis.set_title(title)
        axis.axis("off")
        fig.colorbar(shown, ax=axis, fraction=0.046, pad=0.02)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
