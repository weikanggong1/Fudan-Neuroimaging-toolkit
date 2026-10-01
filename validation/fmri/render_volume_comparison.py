"""将两份或三份真实 BOLD 的时间标准差画在同一模板网格、同一色阶。"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def temporal_correlation(first_path, second_path, brain):
    first = np.asarray(nib.load(first_path).dataobj, dtype=np.float32)[brain]
    second = np.asarray(nib.load(second_path).dataobj, dtype=np.float32)[brain]
    correlation = np.full(len(first), np.nan, dtype=np.float32)
    for start in range(0, len(first), 4096):
        left = first[start:start + 4096].astype(np.float64)
        right = second[start:start + 4096].astype(np.float64)
        left -= left.mean(axis=1, keepdims=True)
        right -= right.mean(axis=1, keepdims=True)
        left_norm = np.square(left).sum(axis=1)
        right_norm = np.square(right).sum(axis=1)
        valid = (left_norm > left.shape[1] * 1e-12) & (right_norm > right.shape[1] * 1e-12)
        block = correlation[start:start + 4096]
        block[valid] = np.clip((left[valid] * right[valid]).sum(axis=1) /
                              np.sqrt(left_norm[valid] * right_norm[valid]), -1, 1)
    result = np.full(brain.shape, np.nan, dtype=np.float32)
    result[brain] = correlation
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("current", "previous", "mask", "template", "figure-out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--official", type=Path)
    parser.add_argument("--current-label", default="Current FNIT: ICA-AROMA + confounds")
    parser.add_argument("--previous-label", default="Previous FNIT: ICA-AROMA + confounds")
    parser.add_argument("--official-label", default="Official UKB: FIX + official spatial maps")
    parser.add_argument("--show-correlation", action="store_true",
                        help="增加 current 与 previous 的逐体素时间相关图，常数时序留空")
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
    paths = [args.current, args.previous]
    labels = ["MNI anatomical template", args.current_label, args.previous_label]
    if args.official is not None:
        paths.append(args.official)
        labels.append(args.official_label)
    for path in paths:
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
    correlation_row = None
    if args.show_correlation:
        correlation_row = len(maps)
        maps.append(temporal_correlation(args.current, args.previous, brain))
        labels.append("FNIT / matched native: voxel temporal Pearson r")
    fig, axes = plt.subplots(len(maps), 3, figsize=(10, 3 * len(maps)), facecolor="white")
    for row, (data, label) in enumerate(zip(maps, labels)):
        display = np.where(brain, data, 0)
        vmax = float(np.percentile(display[brain], 99)) if row == 0 else scale
        cmap, vmin = "gray", 0
        if row == correlation_row:
            display = np.where(brain, data, np.nan)
            cmap, vmin, vmax = "coolwarm", -1, 1
        for axis, plane_name in enumerate(("Sagittal", "Coronal", "Axial")):
            plane = np.rot90(np.take(display, display.shape[axis] // 2, axis=axis))
            rendered = axes[row, axis].imshow(plane, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[row, axis].axis("off")
            axes[row, axis].set_title((label + "\n" if axis == 1 else "") + plane_name,
                                      fontsize=10)
        fig.colorbar(rendered, ax=axes[row, -1], shrink=0.7)
    fig.suptitle(f"Real {frames[0]}-frame BOLD; temporal SD uses a shared scale; no spatial smoothing", fontsize=12)
    fig.tight_layout()
    args.figure_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure_out, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
