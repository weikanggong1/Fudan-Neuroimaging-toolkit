"""将新旧 FNIT 与官方 BOLD 的时间标准差画在同一模板网格、同一色阶。"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("current", "previous", "official", "mask", "template", "figure-out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    template = nib.load(args.template)
    mask_image = nib.load(args.mask)
    if mask_image.shape != template.shape or not np.allclose(
        mask_image.affine, template.affine, atol=1e-4, rtol=0
    ):
        raise ValueError("Mask and template grids differ")
    brain = np.asarray(mask_image.dataobj) > 0
    maps = [np.asarray(template.dataobj, dtype=np.float32)]
    frames = []
    for path in (args.current, args.previous, args.official):
        image = nib.load(path)
        if len(image.shape) != 4 or image.shape[:3] != template.shape or not np.allclose(
            image.affine, template.affine, atol=1e-4, rtol=0
        ):
            raise ValueError("BOLD and template grids differ")
        values = np.asarray(image.dataobj, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite BOLD")
        maps.append(values.std(axis=3, dtype=np.float64).astype(np.float32))
        frames.append(image.shape[3])
        del values
    if len(set(frames)) != 1:
        raise ValueError("Frame counts differ")
    scale = float(np.percentile(np.concatenate([m[brain] for m in maps[1:]]), 99))
    labels = ("MNI anatomical template", "Current FNIT: ICA-AROMA + confounds",
              "Previous FNIT: ICA-AROMA + confounds", "Official UKB: FIX + official spatial maps")
    fig, axes = plt.subplots(4, 3, figsize=(10, 12), facecolor="white")
    for row, (data, label) in enumerate(zip(maps, labels)):
        display = np.where(brain, data, 0)
        vmax = float(np.percentile(display[brain], 99)) if row == 0 else scale
        for axis, plane_name in enumerate(("Sagittal", "Coronal", "Axial")):
            plane = np.rot90(np.take(display, display.shape[axis] // 2, axis=axis))
            rendered = axes[row, axis].imshow(plane, cmap="gray", vmin=0, vmax=vmax)
            axes[row, axis].axis("off")
            axes[row, axis].set_title(label + "\n" + plane_name, fontsize=10)
        fig.colorbar(rendered, ax=axes[row, -1], shrink=0.7)
    fig.suptitle(f"Real {frames[0]}-frame BOLD; temporal SD uses a shared scale; no spatial smoothing", fontsize=12)
    fig.tight_layout()
    args.figure_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure_out, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
