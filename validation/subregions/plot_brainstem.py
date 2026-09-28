"""Real-T1 brainstem comparison in the source T1 voxel grid."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t1", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--fnit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    t1 = nib.load(args.t1)
    official = resample_from_to(nib.load(args.official), (t1.shape[:3], t1.affine), order=0)
    fnit = nib.load(args.fnit)
    if fnit.shape != t1.shape or not np.allclose(fnit.affine, t1.affine):
        raise ValueError("FNIT output must be on the source T1 grid")
    raw = np.asanyarray(t1.dataobj)
    x = np.asanyarray(official.dataobj).round().astype(np.int32)
    y = np.asanyarray(fnit.dataobj).astype(np.int32)
    mask = np.isin(x, (173, 174, 175, 178)) | np.isin(y, (173, 174, 175, 178))
    xyz = np.argwhere(mask)
    if not len(xyz):
        raise ValueError("No brainstem subregions to plot")
    sx = int(np.median(xyz[:, 0]))
    lo = np.maximum(xyz[:, 1:].min(0) - 8, 0)
    hi = np.minimum(xyz[:, 1:].max(0) + 9, np.asarray(raw.shape[1:]))
    sl = (sx, slice(lo[0], hi[0]), slice(lo[1], hi[1]))
    background = np.rot90(raw[sl])
    upper = np.percentile(background[np.isfinite(background)], 99)
    labels = (173, 174, 175, 178)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for axis in axes:
        axis.imshow(background, cmap="gray", vmin=0, vmax=upper)
        axis.axis("off")
    for axis, data, title in zip(axes[:2], (x, y), ("FreeSurfer", "FNIT TorchGEMS")):
        section = np.rot90(data[sl])
        for i, label in enumerate(labels):
            overlay = np.ma.masked_where(section != label, np.ones_like(section))
            axis.imshow(overlay, cmap=ListedColormap([plt.cm.tab10(i)]), alpha=0.6)
        axis.set_title(title)
    difference = np.rot90((x[sl] != y[sl]) & (mask[sl]))
    axes[2].imshow(np.ma.masked_where(~difference, difference),
                   cmap=ListedColormap(["red"]), alpha=0.75)
    axes[2].set_title("Label disagreement")
    figure.savefig(args.output, dpi=170)


if __name__ == "__main__":
    main()
