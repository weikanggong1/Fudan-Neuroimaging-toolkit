#!/usr/bin/env python3
"""Create the public OpenNeuro FLIRT comparison figure and JSON record."""

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


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def voxel_to_fsl(affine, shape, voxel_sizes):
    basis = np.diag([*voxel_sizes, 1.0]).astype(np.float64)
    if np.linalg.det(np.asarray(affine, dtype=np.float64)[:3, :3]) > 0:
        flip = np.eye(4, dtype=np.float64)
        flip[0, 0] = -1
        flip[0, 3] = (int(shape[0]) - 1) * float(voxel_sizes[0])
        basis = flip @ basis
    return basis


def fsl_centre(image):
    data = np.asarray(image.dataobj, dtype=np.float64)
    weights = data - data.min()
    denominator = max(float(weights.sum(dtype=np.float64)), 1e-5)
    voxel = np.asarray([
        np.dot(
            weights.sum(axis=tuple(other for other in range(3) if other != axis)),
            np.arange(data.shape[axis], dtype=np.float64),
        ) / denominator
        for axis in range(3)
    ])
    basis = voxel_to_fsl(image.affine, data.shape, image.header.get_zooms()[:3])
    return (basis @ np.r_[voxel, 1.0])[:3]


def rmsdiff(first, second, centre, radius=80.0):
    difference = np.asarray(first) @ np.linalg.inv(np.asarray(second)) - np.eye(4)
    linear = difference[:3, :3]
    translation = difference[:3, 3] + linear @ centre
    return float(np.sqrt(
        translation @ translation
        + (float(radius) ** 2 / 5.0) * np.trace(linear.T @ linear)
    ))


def read_time(path):
    elapsed, rss = Path(path).read_text().split()[:2]
    return {"wall_seconds": float(elapsed), "maximum_resident_kib": int(rss)}


def canonical_data(path):
    image = nib.as_closest_canonical(nib.load(str(path)))
    return np.asarray(image.dataobj, dtype=np.float32)


def slice_2d(data, axis, index):
    return np.rot90(np.take(data, index, axis=axis))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--figure", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--cpu-model", required=True)
    args = parser.parse_args(argv)

    work = Path(args.work_dir)
    source = Path(args.source_root)
    paths = {
        "input": work / "sub-02_T1w.nii.gz",
        "reference": work / "sub-01_T1w.nii.gz",
        "fsl": work / "fsl_moved.nii.gz",
        "fnit": work / "fnit_cpu_moved.nii.gz",
        "fsl_matrix": work / "fsl.mat",
        "fnit_matrix": work / "fnit_cpu.mat",
    }
    images = {name: nib.load(str(path)) for name, path in paths.items()
              if name in {"input", "reference", "fsl", "fnit"}}
    reference = np.asarray(images["reference"].dataobj, dtype=np.float64)
    fsl = np.asarray(images["fsl"].dataobj, dtype=np.float64)
    fnit = np.asarray(images["fnit"].dataobj, dtype=np.float64)
    mask = reference > 0
    first = fsl[mask]
    second = fnit[mask]
    difference = second - first
    centred_first = first - first.mean()
    centred_second = second - second.mean()
    scale = float(np.percentile(first, 99) - np.percentile(first, 1))
    fsl_matrix = np.loadtxt(paths["fsl_matrix"])
    fnit_matrix = np.loadtxt(paths["fnit_matrix"])

    display = {name: canonical_data(paths[name]) for name in ("input", "reference", "fsl", "fnit")}
    display["difference"] = np.abs(display["fnit"] - display["fsl"])
    output_mask = display["reference"] > 0
    centre = np.rint(np.argwhere(output_mask).mean(axis=0)).astype(int)
    input_centre = np.rint(np.argwhere(display["input"] > 0).mean(axis=0)).astype(int)
    output_values = np.concatenate([display["fsl"][output_mask], display["fnit"][output_mask]])
    output_limits = np.percentile(output_values, [1, 99])
    difference_max = max(float(np.percentile(display["difference"][output_mask], 99.5)), 1e-6)
    columns = ["input", "reference", "fsl", "fnit", "difference"]
    titles = ["Input", "Reference", "FSL FLIRT", "FNIT TorchFLIRT", "Absolute difference"]
    axes_to_show = (2, 1, 0)
    row_names = ("Axial", "Coronal", "Sagittal")
    fig, axes = plt.subplots(3, 5, figsize=(13.5, 8.2), constrained_layout=True)
    for row, axis in enumerate(axes_to_show):
        for column, name in enumerate(columns):
            data = display[name]
            index = int(input_centre[axis] if name == "input" else centre[axis])
            if name == "difference":
                vmin, vmax, cmap = 0.0, difference_max, "magma"
            elif name in {"fsl", "fnit"}:
                vmin, vmax, cmap = float(output_limits[0]), float(output_limits[1]), "gray"
            else:
                values = data[data > 0]
                vmin, vmax = (float(x) for x in np.percentile(values, [1, 99]))
                cmap = "gray"
            axes[row, column].imshow(slice_2d(data, axis, index), cmap=cmap, vmin=vmin, vmax=vmax)
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(titles[column], fontsize=11)
            if column == 0:
                axes[row, column].text(
                    -0.04, 0.5, row_names[row], rotation=90, va="center", ha="right",
                    transform=axes[row, column].transAxes, fontsize=10,
                )
    fig.suptitle("OpenNeuro ds000114: 12-DOF correlation-ratio registration", fontsize=13)
    figure = Path(args.figure)
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    source_names = (
        "src/fnit/flirt/core.py", "src/fnit/flirt/types.py",
        "src/fnit/flirt/coordinates.py", "src/fnit/flirt/standalone.py",
        "src/fnit/flirt/__init__.py", "src/fnit/_nib.py", "src/fnit/_transforms.py",
    )
    report = {
        "schema_version": 1,
        "date": "2026-09-28",
        "dataset": {
            "name": "OpenNeuro ds000114 v1.0.2",
            "doi": "10.18112/openneuro.ds000114.v1.0.2",
            "license": "CC0",
            "input": "published defaced derivative sub-02 T1w",
            "reference": "published defaced derivative sub-01 T1w",
            "input_sha256": sha256(paths["input"]),
            "reference_sha256": sha256(paths["reference"]),
        },
        "profile": "12-DOF correlation ratio, default angular search",
        "candidate": {
            "software": "FNIT TorchFLIRT 0.14.0",
            "device": "CPU",
            "source_sha256": {name: sha256(source / name) for name in source_names},
            "command": "fnit-flirt -in sub-02_T1w.nii.gz -ref sub-01_T1w.nii.gz -out fnit_moved.nii.gz -omat fnit.mat -dof 12 -cost corratio --device cpu",
            "timing": read_time(work / "fnit_cpu.time"),
            "output_sha256": sha256(paths["fnit"]),
            "matrix_sha256": sha256(paths["fnit_matrix"]),
        },
        "reference_implementation": {
            "software": "FSL FLIRT 6.0.7.4",
            "device": "CPU",
            "command": "flirt -in sub-02_T1w.nii.gz -ref sub-01_T1w.nii.gz -out fsl_moved.nii.gz -omat fsl.mat -dof 12 -cost corratio",
            "timing": read_time(work / "fsl.time"),
            "output_sha256": sha256(paths["fsl"]),
            "matrix_sha256": sha256(paths["fsl_matrix"]),
        },
        "hardware": {"cpu": args.cpu_model, "same_host": True, "shared_node": True},
        "comparison": {
            "mask_definition": "nonzero voxels in the public reference T1w derivative",
            "mask_voxels": int(mask.sum()),
            "matrix_rmsdiff_mm": rmsdiff(fnit_matrix, fsl_matrix, fsl_centre(images["reference"])),
            "moved_pearson_r": float(centred_first.dot(centred_second) / (np.linalg.norm(centred_first) * np.linalg.norm(centred_second))),
            "moved_mae": float(np.abs(difference).mean()),
            "moved_rmse": float(np.sqrt(np.square(difference).mean())),
            "moved_normalized_mae": float(np.abs(difference).mean() / scale),
            "moved_normalized_rmse": float(np.sqrt(np.square(difference).mean()) / scale),
            "shape_equal": images["fnit"].shape == images["fsl"].shape,
            "affine_max_abs": float(np.max(np.abs(images["fnit"].affine - images["fsl"].affine))),
            "dtype_equal": str(images["fnit"].get_data_dtype()) == str(images["fsl"].get_data_dtype()),
            "numerically_equivalent": False,
        },
        "figure": "docs/flirt/figures/flirt_public_current.png",
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
