"""核对真实 DWI 的 Schaefer200+Tian S1 四矩阵及节点输出。"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--time-file", type=Path)
    args = parser.parse_args()
    root = args.output_dir
    with (root / "nodes.tsv").open(newline="") as stream:
        nodes = list(csv.DictReader(stream, delimiter="\t"))
    indices = np.array([int(row["index"]) for row in nodes])
    labels = np.loadtxt(root / "region_labels.csv", delimiter=",").astype(int)
    if len(nodes) != 216 or not np.array_equal(indices, np.arange(1, 217)) or not np.array_equal(
        labels, indices
    ):
        raise ValueError("nodes.tsv and region_labels.csv must define 1..216")
    atlas = np.asarray(nib.load(str(root / "atlas_dwi.nii.gz")).dataobj)
    if np.min(atlas) < 0 or np.max(atlas) > 216:
        raise ValueError("DWI atlas labels lie outside 0..216")
    matrices = {}
    stats = {}
    for name in ("count", "sift2_fbc", "mean_length", "mean_fa"):
        matrix = np.loadtxt(root / f"connectome_{name}.csv", delimiter=",")
        if matrix.shape != (216, 216) or not np.isfinite(matrix).all() or not np.allclose(
            matrix, matrix.T, atol=1e-5
        ):
            raise ValueError(f"{name} matrix is not finite symmetric 216x216")
        matrices[name] = matrix
        stats[name] = {
            "shape": [216, 216],
            "upper_nonzero_edges": int(np.count_nonzero(np.triu(matrix, 1))),
            "upper_sum": float(np.triu(matrix, 1).sum()),
        }
    report = {
        "nodes": len(nodes),
        "atlas_shape": list(atlas.shape),
        "atlas_present_nodes": int(np.count_nonzero(np.unique(atlas) > 0)),
        "matrices": stats,
    }
    if args.time_file:
        seconds, rss_kib = args.time_file.read_text().split()
        report["cli_wall_seconds"] = float(seconds)
        report["cli_peak_rss_kib"] = int(rss_kib)
    (root / "output_qc.json").write_text(json.dumps(report, indent=2) + "\n")
    fig, axes = plt.subplots(2, 2, figsize=(8, 7.5), constrained_layout=True)
    for axis, name in zip(axes.flat, matrices):
        matrix = matrices[name]
        display = np.log1p(matrix) if name in ("count", "sift2_fbc") else matrix
        axis.imshow(display, cmap="viridis", interpolation="nearest")
        axis.set_title(name)
        axis.set_xlabel("node index")
        axis.set_ylabel("node index")
    fig.savefig(root / "four_matrices.png", dpi=150)
    plt.close(fig)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
