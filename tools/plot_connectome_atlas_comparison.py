"""绘制真实 T1 上同网格 atlas 的原版、FNIT 和差异切片。"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("background", "reference", "candidate", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--space-label", default="T1", help="grid name shown in the figure title")
    args = parser.parse_args()
    background, reference, candidate = (
        nib.as_closest_canonical(nib.load(str(path)))
        for path in (args.background, args.reference, args.candidate)
    )
    if reference.shape != candidate.shape or not np.allclose(reference.affine, candidate.affine):
        raise ValueError("reference and candidate must share one voxel grid")
    if background.shape != reference.shape or not np.allclose(background.affine, reference.affine):
        raise ValueError("background must share the atlas voxel grid")
    anatomy = np.asarray(background.dataobj)
    expected = np.asarray(reference.dataobj)
    actual = np.asarray(candidate.dataobj)
    voxels = np.argwhere((expected > 0) | (actual > 0))
    if not len(voxels):
        raise ValueError("both atlases are empty")
    mismatch_per_slice = np.count_nonzero(expected != actual, axis=(0, 1))
    z = (int(mismatch_per_slice.argmax()) if mismatch_per_slice.max() else
         int(np.median(voxels[:, 2])))
    x0, y0 = np.maximum(voxels[:, :2].min(axis=0) - 8, 0)
    x1, y1 = np.minimum(voxels[:, :2].max(axis=0) + 9, anatomy.shape[:2])
    background_slice = np.rot90(anatomy[x0:x1, y0:y1, z])
    scale = np.percentile(background_slice, 99)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    for axis, labels, title in zip(
        axes, (expected, actual, expected != actual),
        ("Reference", "FNIT", f"Disagreement: {np.count_nonzero(expected != actual):,} voxels"),
    ):
        axis.imshow(background_slice, cmap="gray", vmin=0, vmax=scale)
        overlay = np.rot90(labels[x0:x1, y0:y1, z])
        if title.startswith("Disagreement"):
            axis.imshow(np.ma.masked_where(~overlay, overlay), cmap="Reds",
                        vmin=0, vmax=1, alpha=1)
        else:
            axis.imshow(np.ma.masked_where(overlay == 0, overlay), cmap="nipy_spectral", alpha=0.55)
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(f"{args.space_label} axial voxel z={z}; reference and FNIT on the same grid")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
