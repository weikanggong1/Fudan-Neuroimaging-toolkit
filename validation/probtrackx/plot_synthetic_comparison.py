"""Render FSL and FNIT tract densities using only generated synthetic data."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--synthetic-dir", type=Path, required=True)
args = parser.parse_args()
root = args.synthetic_dir.resolve()

def image(name):
    return nib.load(str(root / name / "fdt_paths.nii.gz")).get_fdata(dtype=np.float32)


fsl_seed = image("fsl_seed")
fnit_seed = image("fnit_seed")
fsl_network = image("fsl_network")
fnit_network = image("fnit_network")
seed_roi = nib.load(str(root / "roi_01.nii.gz")).get_fdata()[:, :, 20].T
target_roi = nib.load(str(root / "roi_02.nii.gz")).get_fdata()[:, :, 20].T

fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.5), sharex=True, sharey=True,
                         constrained_layout=True)
for row, (fsl, fnit, label) in enumerate(((fsl_seed, fnit_seed, "Seed-to-voxel"),
                                          (fsl_network, fnit_network, "ROI network"))):
    vmax = np.log1p(max(float(fsl.max()), float(fnit.max())))
    for col, (data, title) in enumerate(((fsl, "FSL probtrackx2"),
                                         (fnit, "FNIT probtrackx"))):
        ax = axes[row, col]
        im = ax.imshow(np.log1p(data[:, :, 20].T), origin="lower", cmap="magma",
                       vmin=0, vmax=vmax, interpolation="nearest")
        ax.contour(seed_roi, levels=[0.5], colors=["cyan"], linewidths=1.0)
        ax.contour(target_roi, levels=[0.5], colors=["lime"], linewidths=1.0)
        ax.set_xlim(3, 36)
        ax.set_ylim(14, 26)
        ax.set_title(title if row == 0 else label + " — " + title)
        ax.set_xlabel("voxel x")
        if col == 0:
            ax.set_ylabel("voxel y")
    fig.colorbar(im, ax=axes[row, :], shrink=0.78, label="log(1 + visits)")

fsl_matrix = np.loadtxt(root / "fsl_network/fdt_network_matrix", dtype=int)
fnit_matrix = np.loadtxt(root / "fnit_network/fdt_network_matrix", dtype=int)
fig.suptitle("Synthetic one-fibre posterior only · P=200, S=240, random seed=20260927")
fig.text(0.5, 0.005,
         f"Directed ROI counts (1→2, 2→1): FSL {fsl_matrix[0,1]}, {fsl_matrix[1,0]}  |  "
         f"FNIT {fnit_matrix[0,1]}, {fnit_matrix[1,0]}.  Cyan=seed ROI, green=target ROI.",
         ha="center", fontsize=8.5)
fig.savefig(root / "fsl_fnit_synthetic_comparison.png", dpi=220)
fig.savefig(root / "fsl_fnit_synthetic_comparison.svg")
print(root / "fsl_fnit_synthetic_comparison.png")
