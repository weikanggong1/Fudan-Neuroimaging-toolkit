"""公开真实T1同网格脑图：强度、原生/GPU标签与差异，限独立benchmark。"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="三维uint8强度输入")
    parser.add_argument("--native", type=Path, required=True, help="同网格原生histogram参考")
    parser.add_argument("--candidate", type=Path, required=True, help="同网格GPU输出")
    parser.add_argument("--output", type=Path, required=True, help="新PNG文件")
    parser.add_argument("--case-label", required=True, help="公开数据集和被试的图标题")
    parser.add_argument("--histogram-pass", type=int, choices=(1, 2), default=1, help="遍次，默认1")
    parser.add_argument("--slice-z", type=int, default=128, help="z体素索引，默认128")
    args = parser.parse_args()
    images = [nib.load(path) for path in (args.source, args.native, args.candidate)]
    if any(image.shape != images[0].shape or not np.array_equal(image.affine, images[0].affine)
           or image.get_data_dtype() != np.dtype(np.uint8) for image in images):
        raise ValueError("expected uint8 images with exactly matching voxel grids")
    if not 0 <= args.slice_z < images[0].shape[2] or args.output.exists():
        raise ValueError("slice outside voxel grid or output already exists")
    arrays = [np.asarray(image.dataobj) for image in images]
    if any(not np.isin(array, (1, 128, 255)).all() for array in arrays[1:]):
        raise ValueError("unexpected trinary labels")
    figure, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
    for axis in axes:
        axis.set_xlabel("x voxel")
        axis.set_ylabel("y voxel")
    axes[0].imshow(arrays[0][:, :, args.slice_z].T, origin="lower", cmap="gray", vmin=0, vmax=140)
    axes[0].set_title("FNIT frozen intensity")
    for index, title in ((1, "Native histogram labels"), (2, "Torch GPU histogram labels")):
        axes[index].imshow(arrays[0][:, :, args.slice_z].T, origin="lower", cmap="gray", vmin=0, vmax=140)
        labels = arrays[index][:, :, args.slice_z].T
        overlay = np.ma.masked_where(labels == 1, labels)
        axes[index].imshow(overlay, origin="lower", cmap="viridis", vmin=128, vmax=255, alpha=.65)
        axes[index].set_title(title)
    difference = arrays[1] != arrays[2]
    axes[3].imshow(difference[:, :, args.slice_z].T, origin="lower", cmap="Reds", vmin=0, vmax=1)
    axes[3].set_title(f"Different voxels: {np.count_nonzero(difference)}")
    figure.suptitle(f"{args.case_label} / histogram pass {args.histogram_pass} / z={args.slice_z} voxel; identical grid")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=140)
    plt.close(figure)
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    args.output.with_suffix(".json").write_text(json.dumps({
        "scope": "same-input histogram public brain visualization; not whole reconstruction",
        "source_sha256": sha(args.source), "native_sha256": sha(args.native),
        "candidate_sha256": sha(args.candidate), "png_sha256": sha(args.output),
        "script_sha256": sha(__file__), "shape": [int(axis) for axis in images[0].shape],
        "affine_mm": images[0].affine.tolist(), "slice_z_voxel": args.slice_z,
        "case_label": args.case_label, "histogram_pass": args.histogram_pass,
        "different_voxels": int(np.count_nonzero(difference)),
        "data_license_evidence": "validation/recon_all/accuracy_20261003/cohort: ds000114 CC0-1.0 fixed snapshot"
    }, indent=2))


if __name__ == "__main__":
    main()
