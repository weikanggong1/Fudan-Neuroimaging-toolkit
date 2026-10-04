#!/usr/bin/env python3
"""Plot an independently verified public MNI152 template conversion.

Inputs are prepared complete outputs, not private subject images. This script
does not download or invoke reference programs. Verify their public lineage
before calling it; only figures and aggregate numeric metrics are saved.
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
    for name in ("reference", "official", "fnit", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--public-template-lineage-verified", action="store_true", required=True)
    args = parser.parse_args()
    paths = {name: getattr(args, name) for name in ("reference", "official", "fnit")}
    images = {name: nib.load(path) for name, path in paths.items()}
    reference = images["reference"]
    arrays = {name: np.asarray(image.dataobj, dtype=np.float32) for name, image in images.items()}
    for name in ("official", "fnit"):
        if images[name].shape != (*reference.shape, 2):
            raise ValueError("Expected two public template channels on the complete reference grid")
        if not np.allclose(images[name].affine, reference.affine, atol=1e-6, rtol=0):
            raise ValueError("Converted maps must share the public reference grid")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Figure inputs must be finite")
    delta = arrays["fnit"].astype(np.float64) - arrays["official"].astype(np.float64)
    orientation = nib.orientations.ornt_transform(nib.orientations.io_orientation(reference.affine),
                                                 nib.orientations.axcodes2ornt(("R", "A", "S")))
    affine = reference.affine @ nib.orientations.inv_ornt_aff(orientation, reference.shape)
    background = nib.orientations.apply_orientation(arrays["reference"], orientation)
    maps = [nib.orientations.apply_orientation(arrays[name][..., 0], orientation)
            for name in ("official", "fnit")]
    maps.append(nib.orientations.apply_orientation(np.abs(delta[..., 0]), orientation))
    coordinates = np.array([0.0, -24.0, 12.0, 1.0])
    cuts = np.clip(np.rint((np.linalg.inv(affine) @ coordinates)[:3]).astype(int),
                   0, np.array(background.shape) - 1)
    foreground = maps[0] != 0
    limit = max(float(np.percentile(np.abs(maps[0][foreground]), 99)), 1e-12)
    error_limit = max(float(maps[2].max()), 1e-6)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.linewidth": 0.6, "figure.facecolor": "white",
                         "savefig.facecolor": "white", "svg.fonttype": "none"})
    figure, panels = plt.subplots(3, 3, figsize=(7.1, 6.2), constrained_layout=True)
    titles = ("CBIG + Workbench", "FNIT", "Absolute difference")
    for row, values in enumerate(maps):
        for axis, plane in enumerate(("Sagittal", "Coronal", "Axial")):
            selection = [slice(None)] * 3
            selection[axis] = int(cuts[axis])
            selection = tuple(selection)
            panel = panels[row, axis]
            panel.imshow(np.rot90(background[selection]), cmap="gray", interpolation="nearest")
            shown = np.ma.masked_where(~foreground[selection], values[selection])
            artist = panel.imshow(np.rot90(shown), cmap="viridis" if row < 2 else "magma",
                                  vmin=0, vmax=limit if row < 2 else error_limit,
                                  alpha=0.7, interpolation="nearest")
            panel.set_xticks([])
            panel.set_yticks([])
            if row == 0:
                panel.set_title(plane)
            if axis == 0:
                panel.set_ylabel(titles[row])
            for spine in panel.spines.values():
                spine.set_color("black")
                spine.set_linewidth(0.6)
        bar = figure.colorbar(artist, ax=list(panels[row]), fraction=0.025, pad=0.015)
        bar.set_label("Template intensity" if row < 2 else "Absolute difference")
        bar.outline.set_linewidth(0.5)
    figure.suptitle("Public MNI152 template: fsLR32k → MNI 2 mm (channel 1 of 2)", fontsize=10)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        figure.savefig(args.output_dir / f"public_space_comparison.{suffix}", dpi=220)
    plt.close(figure)
    report = {"source": "Public FSL MNI152 T1 and brain-masked T1 template channels",
              "reference_url": "https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/fnirt/user_guide.html",
              "license_url": "https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html",
              "route": "fsLR32k to MNI152 2 mm; CBIG RF-ANTs plus Workbench",
              "channels": 2, "shape": list(images["fnit"].shape),
              "slice_ras_world_mm": coordinates[:3].tolist(),
              "all_finite": True, "max_absolute_error": float(np.abs(delta).max()),
              "mean_absolute_error": float(np.abs(delta).mean()),
              "exact_fraction": float((delta == 0).mean()),
              "input_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                                for name, path in paths.items()}}
    (args.output_dir / "public_space_comparison.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
