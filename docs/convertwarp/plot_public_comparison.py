#!/usr/bin/env python3
"""Plot a declared public reference and two complete public dense warp fields.

This documentation utility never runs original programs or downloads data.
The caller must first verify the original public lineage and licensing.
"""
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
    parser.add_argument("--reference", required=True)
    parser.add_argument("--official-warp", required=True)
    parser.add_argument("--fnit-warp", required=True)
    parser.add_argument("--brain-mask", required=True)
    parser.add_argument("--dataset-url", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--public-inputs-verified", action="store_true", required=True)
    args = parser.parse_args()
    if not args.dataset_url.startswith("https://openneuro.org/datasets/ds000114"):
        raise ValueError("This example is restricted to the verified CC0 ds000114 lineage")
    paths = {"reference": Path(args.reference), "brain_mask": Path(args.brain_mask), "official": Path(args.official_warp),
             "fnit": Path(args.fnit_warp)}
    images = {name: nib.load(path) for name, path in paths.items()}
    reference = images["reference"]
    for name in ("official", "fnit"):
        image = images[name]
        if image.shape != (*reference.shape, 3) or not np.allclose(image.affine, reference.affine, atol=1e-6, rtol=0):
            raise ValueError("Both complete fields must use the declared public reference grid")
    background = np.asarray(reference.dataobj, dtype=np.float64)
    official = np.asarray(images["official"].dataobj, dtype=np.float64)
    current = np.asarray(images["fnit"].dataobj, dtype=np.float64)
    if not all(np.isfinite(data).all() for data in (background, official, current)):
        raise ValueError("Public figure inputs must be finite")
    difference = current - official
    maps = [np.linalg.norm(official, axis=-1), np.linalg.norm(current, axis=-1),
            np.linalg.norm(difference, axis=-1) * 1000]
    transform = nib.orientations.ornt_transform(nib.orientations.io_orientation(reference.affine),
                                                nib.orientations.axcodes2ornt(("R", "A", "S")))
    ras_affine = reference.affine @ nib.orientations.inv_ornt_aff(transform, reference.shape)
    background = nib.orientations.apply_orientation(background, transform)
    maps = [nib.orientations.apply_orientation(data, transform) for data in maps]
    coordinates = np.array([0.0, -24.0, 12.0, 1.0])
    cuts = np.rint((np.linalg.inv(ras_affine) @ coordinates)[:3]).astype(int)
    cuts = np.clip(cuts, 0, np.array(background.shape) - 1)
    mask_image = images["brain_mask"]
    if mask_image.shape != reference.shape or not np.allclose(mask_image.affine, reference.affine, atol=1e-6, rtol=0):
        raise ValueError("The public template brain mask must use the declared reference grid")
    foreground = nib.orientations.apply_orientation(np.asarray(mask_image.dataobj) > 0, transform)
    limit = max(float(max(data[foreground].max() for data in maps[:2])), 1e-12)
    error_limit = max(float(maps[2][foreground].max()), 1e-12)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.linewidth": 0.6, "figure.facecolor": "white",
                         "savefig.facecolor": "white", "svg.fonttype": "none"})
    figure, axes = plt.subplots(3, 3, figsize=(7.1, 6.4), constrained_layout=True)
    titles = ["FSL displacement magnitude", "FNIT displacement magnitude", "Vector difference"]
    plane_names = ["Sagittal", "Coronal", "Axial"]
    image_artist = []
    for row, (values, title) in enumerate(zip(maps, titles)):
        for axis, plane in enumerate(plane_names):
            selection = [slice(None)] * 3
            selection[axis] = int(cuts[axis])
            selection = tuple(selection)
            panel = axes[row, axis]
            panel.imshow(np.rot90(background[selection]), cmap="gray", interpolation="nearest")
            masked = np.ma.masked_where(~foreground[selection], values[selection])
            artist = panel.imshow(np.rot90(masked), cmap="viridis" if row < 2 else "magma",
                                  vmin=0, vmax=limit if row < 2 else error_limit,
                                  alpha=0.65 if row < 2 else 0.85,
                                  interpolation="nearest")
            panel.set_xticks([]); panel.set_yticks([])
            panel.set_title(plane if row == 0 else "", fontsize=8)
            for spine in panel.spines.values():
                spine.set_color("black"); spine.set_linewidth(0.6)
            if axis == 0:
                panel.set_ylabel(title, fontsize=8)
        image_artist.append(artist)
    for row, artist in enumerate(image_artist):
        bar = figure.colorbar(artist, ax=list(axes[row]), fraction=0.025, pad=0.015)
        bar.set_label("mm" if row < 2 else "µm")
        bar.outline.set_linewidth(0.5)
    figure.suptitle("Public ds000114 sub-01: complete T1 → MNI 2 mm field", fontsize=10)
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        figure.savefig(destination / f"public_convertwarp_comparison.{suffix}", dpi=220)
    plt.close(figure)
    report = {"dataset_url": args.dataset_url, "dataset_license": "CC0",
              "scope": "All inputs derive from the independently verified public ds000114 lineage; no private subject field",
              "shape": list(reference.shape), "output_dtype": str(images["fnit"].get_data_dtype()),
              "max_component_error_mm": float(np.abs(difference).max()),
              "mean_component_error_mm": float(np.abs(difference).mean()),
              "max_vector_error_mm": float(np.linalg.norm(difference, axis=-1).max()),
              "input_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
              "slice_ras_world_mm": coordinates[:3].tolist()}
    (destination / "public_convertwarp_comparison.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
