"""公开真实T1的filled同网格标签与差异脑图，输入只用于benchmark。"""
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
    parser.add_argument("--source", type=Path, required=True, help="同网格uint8强度")
    parser.add_argument("--reference", type=Path, required=True, help="旧FNIT冻结filled，仅比较")
    parser.add_argument("--candidate", type=Path, required=True, help="完整新filled")
    parser.add_argument("--output", type=Path, required=True, help="新PNG文件")
    parser.add_argument("--case-label", required=True)
    parser.add_argument("--slice-z", type=int, default=128, help="z体素索引，默认128")
    parser.add_argument("--label-kind", choices=("filled", "wm"), default="filled",
                        help="默认filled的0/127/255；wm以>=5显示WM掩膜，原始uint8逐体素比较不变")
    parser.add_argument("--runtime-label", default="warm CPU API",
                        help="图标题中的实际运行范围，默认warm CPU API；不参与数值计算")
    args = parser.parse_args()
    paths = (args.source, args.reference, args.candidate)
    images = [nib.load(path) for path in paths]
    if any(image.shape != images[0].shape or not np.array_equal(image.affine, images[0].affine)
           or image.get_data_dtype() != np.dtype(np.uint8) for image in images):
        raise ValueError("require uint8 volumes with exactly matching voxel grids")
    if args.output.exists() or not 0 <= args.slice_z < images[0].shape[2]:
        raise ValueError("output already exists or slice outside grid")
    arrays = [np.asarray(image.dataobj) for image in images]
    if args.label_kind == "filled" and any(not np.isin(array, (0, 127, 255)).all() for array in arrays[1:]):
        raise ValueError("unexpected filled labels")
    figure, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
    for axis in axes:
        axis.set_xlabel("x voxel"); axis.set_ylabel("y voxel")
    axes[0].imshow(arrays[0][:, :, args.slice_z].T, origin="lower", cmap="gray", vmin=0, vmax=140)
    axes[0].set_title("Frozen FNIT intensity")
    titles = ((1, "Original FNIT filled"), (2, "Torch + ordered Numba filled")) if args.label_kind == "filled" else (
        (1, "Existing FNIT WM"), (2, "Cached geometry + ordered WM"))
    for index, title in titles:
        axes[index].imshow(arrays[0][:, :, args.slice_z].T, origin="lower", cmap="gray", vmin=0, vmax=140)
        labels = arrays[index][:, :, args.slice_z].T
        if args.label_kind == "wm":
            labels = (labels >= 5).astype(np.uint8)
        axes[index].imshow(np.ma.masked_where(labels == 0, labels), origin="lower",
                           cmap="coolwarm" if args.label_kind == "filled" else "Greens",
                           vmin=127 if args.label_kind == "filled" else 0,
                           vmax=255 if args.label_kind == "filled" else 1, alpha=.6)
        axes[index].set_title(title)
    delta = arrays[1] != arrays[2]
    axes[3].imshow(delta[:, :, args.slice_z].T, origin="lower", cmap="Reds", vmin=0, vmax=1)
    axes[3].set_title(f"Different voxels: {np.count_nonzero(delta)}")
    figure.suptitle(f"{args.case_label} / same voxel grid / z={args.slice_z}; {args.runtime_label}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=140); plt.close(figure)
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    args.output.with_suffix(".json").write_text(json.dumps({
        "scope": f"frozen same-input complete {args.label_kind}; existing FNIT reference; not whole reconstruction",
        "source_sha256": sha(args.source), "reference_sha256": sha(args.reference),
        "candidate_sha256": sha(args.candidate), "png_sha256": sha(args.output),
        "script_sha256": sha(__file__), "different_voxels": int(np.count_nonzero(delta)),
        "slice_z_voxel": args.slice_z, "case_label": args.case_label,
        "label_kind": args.label_kind, "runtime_label": args.runtime_label,
        "shape": [int(axis) for axis in images[0].shape], "affine_mm": images[0].affine.tolist(),
        "public_data_license": "ds000114 CC0-1.0; frozen cohort source/input hashes in this run manifest"}, indent=2))


if __name__ == "__main__":
    main()
