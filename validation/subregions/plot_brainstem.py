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
    scp_xyz = np.argwhere((x == 178) | (y == 178))
    lo = np.maximum(xyz[:, 1:].min(0) - 8, 0)
    hi = np.minimum(xyz[:, 1:].max(0) + 9, np.asarray(raw.shape[1:]))
    scp_lo = np.maximum(scp_xyz[:, :2].min(0) - 8, 0)
    scp_hi = np.minimum(scp_xyz[:, :2].max(0) + 9, np.asarray(raw.shape[:2]))
    sections = [
        (int(np.median(xyz[:, 0])), slice(lo[0], hi[0]), slice(lo[1], hi[1])),
        (slice(scp_lo[0], scp_hi[0]), slice(scp_lo[1], scp_hi[1]),
         int(np.median(scp_xyz[:, 2]))),
    ]
    labels = (173, 174, 175, 178)
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    for row, sl in enumerate(sections):
        background = np.rot90(raw[sl])
        upper = np.percentile(background[np.isfinite(background)], 99)
        for axis in axes[row]:
            axis.imshow(background, cmap="gray", vmin=0, vmax=upper)
            axis.axis("off")
        for axis, data in zip(axes[row, :2], (x, y)):
            section = np.rot90(data[sl])
            for i, label in enumerate(labels):
                overlay = np.ma.masked_where(section != label, np.ones_like(section))
                axis.imshow(overlay, cmap=ListedColormap([plt.cm.tab10(i)]), alpha=0.6)
        difference = np.rot90((x[sl] != y[sl]) & (mask[sl]))
        axes[row, 2].imshow(np.ma.masked_where(~difference, difference),
                            cmap=ListedColormap(["red"]), alpha=0.75)
        axes[row, 0].text(0.01, 0.98, "brainstem sagittal" if row == 0 else "SCP axial",
                          transform=axes[row, 0].transAxes, color="white", va="top")
    for axis, title in zip(axes[0], ("FreeSurfer", "FNIT TorchGEMS", "Label disagreement")):
        axis.set_title(title)
    figure.savefig(args.output, dpi=170)


if __name__ == "__main__":
    main()
