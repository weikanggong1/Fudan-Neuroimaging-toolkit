"""绘制完整球面的径向负面；显示抽样与质量计数明确分开。"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel.freesurfer.io as fsio
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal-sphere", type=Path, required=True)
    parser.add_argument("--candidate-sphere", type=Path, required=True)
    parser.add_argument("--registered-sphere", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    cases = [("Normal79", args.normal_sphere),
             ("v4 sphere", args.candidate_sphere),
             ("new2 sphere.reg", args.registered_sphere)]
    figure, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    receipt = []
    for column, (label, path) in enumerate(cases):
        vertices, faces = fsio.read_geometry(path)
        vertices = vertices.astype(np.float32)
        points = vertices.astype(np.float64)[faces]
        cross = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
        negative = np.sum(points.sum(axis=1) * cross, axis=1) < 0
        centers = points.mean(axis=1)
        for row, (dimensions, view) in enumerate([((0, 1), "dorsal"), ((1, 2), "lateral")]):
            axis = axes[row, column]
            axis.scatter(vertices[::20, dimensions[0]], vertices[::20, dimensions[1]],
                         s=.35, c="#c9cdd1", alpha=.5, rasterized=True)
            axis.scatter(centers[negative, dimensions[0]], centers[negative, dimensions[1]],
                         s=1.2, c="#d42c34", alpha=.65, rasterized=True)
            axis.set_aspect("equal")
            axis.set_xlim(-105, 105)
            axis.set_ylim(-105, 105)
            axis.set_xlabel(["X", "Y", "Z"][dimensions[0]] + " / mm")
            axis.set_ylabel(["X", "Y", "Z"][dimensions[1]] + " / mm")
            axis.set_title(f"{label}: {view}\nnegative faces={int(negative.sum())}")
        receipt.append({"label": label,
                        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "negative_faces_double": int(negative.sum()),
                        "visual_only": "gray vertices display every twentieth row; red centers display all negative faces; counts from full mesh"})
    figure.suptitle("Actual sphere QC: red = inward radial faces; two views include front and back points")
    figure.savefig(args.output, dpi=160)
    plt.close(figure)
    args.report.write_text(json.dumps({
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inputs": receipt}, indent=2) + "\n")


if __name__ == "__main__":
    main()
