#!/usr/bin/env python3
"""Render the public EDDY, DTIFIT, and NODDI comparison figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def load(path):
    return np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)


def plane(volume, axis, index):
    return np.rot90(np.take(volume, index, axis=axis))


def best_plane(mask):
    scores = [
        mask.sum(axis=tuple(i for i in range(3) if i != axis)) for axis in range(3)
    ]
    axis = int(np.argmax([score.max() for score in scores]))
    return axis, int(np.argmax(scores[axis]))


def show(ax, volume, axis, index, title, *, vmin=0, vmax=None, cmap="gray"):
    artist = ax.imshow(plane(volume, axis, index), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    return artist


def save_eddy(args, mask, axis, index):
    raw = load(args.raw).mean(axis=3)
    official = load(args.eddy_official).mean(axis=3)
    fnit = load(args.eddy_fnit).mean(axis=3)
    values = np.concatenate((raw[mask], official[mask], fnit[mask]))
    vmax = float(np.percentile(values, 99))
    difference = np.abs(fnit - official)
    diff_max = max(float(np.percentile(difference[mask], 99)), 1e-6)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), constrained_layout=True)
    show(axes[0], raw, axis, index, "Raw AP mean", vmax=vmax)
    show(axes[1], official, axis, index, "FSL EDDY", vmax=vmax)
    show(axes[2], fnit, axis, index, "FNIT TorchEDDY", vmax=vmax)
    artist = show(
        axes[3],
        difference,
        axis,
        index,
        "Absolute difference",
        vmax=diff_max,
        cmap="magma",
    )
    fig.colorbar(artist, ax=axes[3], fraction=0.046, pad=0.02)
    fig.savefig(args.output_dir / "eddy_fsl_comparison.png", dpi=180)
    plt.close(fig)


def save_dtifit(args, mask, axis, index):
    official = load(str(args.dtifit_official) + "_FA.nii.gz")
    fnit = load(str(args.dtifit_fnit) + "_FA.nii.gz")
    difference = np.abs(fnit - official)
    diff_max = max(float(np.percentile(difference[mask], 99.9)), 1e-8)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), constrained_layout=True)
    show(axes[0], official, axis, index, "FSL DTIFIT FA", vmax=1)
    show(axes[1], fnit, axis, index, "FNIT TorchDTIFIT FA", vmax=1)
    artist = show(
        axes[2],
        difference,
        axis,
        index,
        "Absolute difference",
        vmax=diff_max,
        cmap="magma",
    )
    fig.colorbar(artist, ax=axes[2], fraction=0.046, pad=0.02)
    fig.savefig(args.output_dir / "dtifit_fsl_comparison.png", dpi=180)
    plt.close(fig)


def save_noddi(args, mask, axis, index):
    names = (
        ("NDI", "Neurite density"),
        ("ODI", "Orientation dispersion"),
        ("FWF", "Free-water fraction"),
    )
    fig, axes = plt.subplots(3, 3, figsize=(8.2, 8.2), constrained_layout=True)
    for row, (name, label) in enumerate(names):
        official = load(args.noddi_official / f"fit_{name}.nii.gz")
        fnit = load(args.noddi_fnit / f"fit_{name}.nii.gz")
        difference = np.abs(fnit - official)
        diff_max = max(float(np.percentile(difference[mask], 99)), 1e-6)
        show(axes[row, 0], official, axis, index, f"AMICO {label}", vmax=1)
        show(axes[row, 1], fnit, axis, index, f"FNIT {label}", vmax=1)
        artist = show(
            axes[row, 2],
            difference,
            axis,
            index,
            "Absolute difference",
            vmax=diff_max,
            cmap="magma",
        )
        fig.colorbar(artist, ax=axes[row, 2], fraction=0.046, pad=0.02)
    fig.savefig(args.output_dir / "amico_noddi_comparison.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--eddy-official", type=Path, required=True)
    parser.add_argument("--eddy-fnit", type=Path, required=True)
    parser.add_argument("--dtifit-official", type=Path, required=True)
    parser.add_argument("--dtifit-fnit", type=Path, required=True)
    parser.add_argument("--noddi-official", type=Path, required=True)
    parser.add_argument("--noddi-fnit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask = load(args.mask) > 0
    axis, index = best_plane(mask)
    save_eddy(args, mask, axis, index)
    save_dtifit(args, mask, axis, index)
    save_noddi(args, mask, axis, index)


if __name__ == "__main__":
    main()
