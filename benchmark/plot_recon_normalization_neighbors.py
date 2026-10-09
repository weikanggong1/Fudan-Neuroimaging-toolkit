"""真实归一化输出与差异图；读取比较文件，不修改或修补候选。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    """两个同网格三维 MGZ→PNG与哈希JSON；灰度为归一化强度单位。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="同输入旧FNIT输出，仅作图比较")
    parser.add_argument("--candidate", type=Path, required=True, help="完整候选输出")
    parser.add_argument("--output", type=Path, required=True, help="新PNG路径")
    parser.add_argument("--label", required=True, help="公开被试和阶段名称，不含私有路径")
    parser.add_argument("--candidate-title", default="GPU neighbor candidate",
                        help="候选面板标题；默认邻域候选，初始偏场可显式命名")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    images = [nib.load(str(path)) for path in (args.reference, args.candidate)]
    if images[0].shape != images[1].shape or not np.array_equal(images[0].affine, images[1].affine):
        raise ValueError("plots require the same shape and millimeter affine")
    baseline, candidate = [np.asarray(image.dataobj, dtype=np.float32) for image in images]
    if baseline.ndim != 3:
        raise ValueError("plots require 3D inputs")
    difference = np.abs(baseline - candidate)
    slice_index = baseline.shape[2] // 2
    figure, axes = plt.subplots(1, 3, figsize=(10, 3.7), constrained_layout=True)
    for axis, data, title in zip(axes[:2], (baseline, candidate), ("Existing FNIT", args.candidate_title)):
        axis.imshow(data[:, :, slice_index].T, origin="lower", cmap="gray", vmin=0, vmax=255)
        axis.set_title(title)
        axis.axis("off")
    panel = axes[2].imshow(difference[:, :, slice_index].T, origin="lower", cmap="magma",
                           vmin=0, vmax=max(1., float(difference.max())))
    axes[2].set_title(f"Absolute error ({np.count_nonzero(difference):,} voxels differ)")
    axes[2].axis("off")
    figure.colorbar(panel, ax=axes[2], label="Normalized intensity units", shrink=.7)
    figure.suptitle(args.label)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)
    args.output.with_suffix(".json").write_text(json.dumps({
        "scope": "mid-index source grid slice; no RAS resampling or clinical claim",
        "reference_sha256": sha(args.reference), "candidate_sha256": sha(args.candidate),
        "script_sha256": sha(__file__), "figure_sha256": sha(args.output),
        "candidate_title": args.candidate_title,
        "shape": list(baseline.shape), "slice_axis": 2, "slice_index": slice_index,
        "different_voxels": int(np.count_nonzero(difference)), "max_abs": float(difference.max()),
        "p99_abs": float(np.percentile(difference, 99)), "affine_equal": True,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
