#!/usr/bin/env python3
"""Render complete outputs of the verified CC0 public T1 affine case."""
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
    for name in ("public_input", "reference", "official", "candidate", "output_dir"):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    parser.add_argument("--public-lineage-verified", action="store_true", required=True)
    args = parser.parse_args()
    public_input = Path(args.public_input)
    digest = hashlib.sha256(public_input.read_bytes()).hexdigest()
    if digest != "f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a":
        raise ValueError("Expected the independently verified ds000114 public T1")
    paths = {name: Path(getattr(args, name)) for name in ("reference", "official", "candidate")}
    images = {name: nib.load(path) for name, path in paths.items()}
    reference = images["reference"]
    for image in images.values():
        if image.shape != reference.shape or image.ndim != 3:
            raise ValueError("All complete 3D images must use the reference grid")
        if not np.allclose(image.affine, reference.affine, atol=1e-6, rtol=0):
            raise ValueError("Output affines must match")
    data = {name: np.asarray(image.dataobj, dtype=np.float64) for name, image in images.items()}
    if not all(np.isfinite(values).all() for values in data.values()):
        raise ValueError("All complete outputs must be finite")
    error = data["candidate"] - data["official"]
    orientation = nib.orientations.ornt_transform(
        nib.orientations.io_orientation(reference.affine),
        nib.orientations.axcodes2ornt(("R", "A", "S")),
    )
    affine = reference.affine @ nib.orientations.inv_ornt_aff(orientation, reference.shape)
    maps = [nib.orientations.apply_orientation(values, orientation)
            for values in (data["official"], data["candidate"], error)]
    world = np.array([0.0, -24.0, 12.0, 1.0])
    cuts = np.rint((np.linalg.inv(affine) @ world)[:3]).astype(int)
    cuts = np.clip(cuts, 0, np.array(maps[0].shape) - 1)
    upper = float(np.percentile(data["official"][data["official"] > 0], 99.5))
    error_limit = max(float(np.abs(error).max()), 1e-12)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.linewidth": 0.6, "figure.facecolor": "white",
                         "savefig.facecolor": "white", "svg.fonttype": "none"})
    figure, axes = plt.subplots(3, 3, figsize=(7.1, 6.4), constrained_layout=True)
    for row, (values, label) in enumerate(zip(maps, ("FSL", "FNIT CPU", "FNIT − FSL"))):
        for axis, name in enumerate(("Sagittal", "Coronal", "Axial")):
            select = [slice(None)] * 3
            select[axis] = int(cuts[axis])
            artist = axes[row, axis].imshow(
                np.rot90(values[tuple(select)]), interpolation="nearest",
                cmap="gray" if row < 2 else "RdBu_r",
                vmin=0 if row < 2 else -error_limit,
                vmax=upper if row < 2 else error_limit,
            )
            axes[row, axis].set_xticks([])
            axes[row, axis].set_yticks([])
            if row == 0:
                axes[row, axis].set_title(name)
            if axis == 0:
                axes[row, axis].set_ylabel(label)
            for spine in axes[row, axis].spines.values():
                spine.set_color("black")
                spine.set_linewidth(0.6)
        bar = figure.colorbar(artist, ax=list(axes[row]), fraction=0.025, pad=0.015)
        bar.set_label("Image intensity" if row < 2 else "Intensity difference")
        bar.outline.set_linewidth(0.5)
    figure.suptitle("Public ds000114 T1: complete affine resampling", fontsize=10)
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        figure.savefig(destination / f"public_applywarp_comparison.{suffix}", dpi=220)
    plt.close(figure)
    report = {
        "dataset_url": "https://openneuro.org/datasets/ds000114/versions/1.0.2",
        "license": "CC0", "public_input_sha256": digest,
        "scope": "Complete public T1 trilinear float32 output; no private subject image",
        "shape": list(reference.shape), "slice_ras_world_mm": world[:3].tolist(),
        "max_absolute_error": float(np.abs(error).max()),
        "mean_absolute_error": float(np.abs(error).mean()),
        "relative_l2_error": float(np.linalg.norm(error) / np.linalg.norm(data["official"])),
        "image_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for name, path in paths.items()},
    }
    (destination / "public_applywarp_comparison.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
