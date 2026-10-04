"""Render the CC0 public T1w FLIRT comparison from complete saved outputs."""

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
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {name: getattr(args, name) for name in ("reference", "official", "candidate")}
    images = {name: nib.as_closest_canonical(nib.load(path)) for name, path in paths.items()}
    reference = images["reference"]
    for name, image in images.items():
        if image.shape != reference.shape or not np.allclose(image.affine, reference.affine, atol=1e-5):
            raise ValueError(f"{name} does not match the reference physical grid")
    data = {name: image.get_fdata(dtype=np.float32) for name, image in images.items()}
    if not all(np.isfinite(array).all() for array in data.values()):
        raise ValueError("comparison inputs must contain finite values")
    foreground = data["official"] != 0
    locations = np.asarray(np.where(foreground))
    if locations.shape[1] == 0:
        raise ValueError("official foreground is empty")
    centre = np.rint((locations.min(axis=1) + locations.max(axis=1)) / 2).astype(int)
    world = reference.affine @ np.r_[centre, 1]
    difference = np.abs(data["candidate"] - data["official"])
    gray_max = float(np.percentile(data["official"][foreground], 99))
    difference_max = max(float(np.percentile(difference[foreground], 99.5)), 1.0)
    voxel_sizes = nib.affines.voxel_sizes(reference.affine)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
                         "axes.linewidth": 0.5, "pdf.fonttype": 42,
                         "savefig.facecolor": "white"})
    figure, axes = plt.subplots(3, 4, figsize=(7.1, 6.2), constrained_layout=True)
    columns = ("Reference T1w", "FSL FLIRT", "FNIT", "Absolute difference")
    views = (("Axial", 2, (0, 1)), ("Coronal", 1, (0, 2)), ("Sagittal", 0, (1, 2)))
    for row, (view, axis, display_axes) in enumerate(views):
        selector = [slice(None)] * 3
        selector[axis] = centre[axis]
        selector = tuple(selector)
        for column, values in enumerate((*data.values(), difference)):
            panel = axes[row, column]
            artist = panel.imshow(values[selector].T, origin="lower",
                                  aspect=voxel_sizes[display_axes[1]] / voxel_sizes[display_axes[0]],
                                  cmap="magma" if column == 3 else "gray", vmin=0,
                                  vmax=difference_max if column == 3 else gray_max,
                                  interpolation="nearest")
            panel.set_xticks([])
            panel.set_yticks([])
            if row == 0:
                panel.set_title(columns[column], fontweight="bold", pad=4)
            if column == 0:
                panel.set_ylabel(f"{view}\n{'XYZ'[axis]} = {world[axis]:.1f} mm", labelpad=4)
            if column == 3:
                figure.colorbar(artist, ax=panel, shrink=0.65, pad=0.02,
                                label="T1w intensity units", ticks=(0, difference_max))
    figure.suptitle("Public ds000114 sub-02 → sub-01 · 12 DOF, correlation ratio", fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300)
    figure.savefig(args.output.with_suffix(".pdf"))
    plt.close(figure)
    record = {"dataset": "OpenNeuro ds000114 v1.0.2", "license": "CC0",
              "license_metadata": "https://raw.githubusercontent.com/OpenNeuroDatasets/ds000114/master/dataset_description.json",
              "input_note": "FNIT defaced public examples; complete-volume registered outputs",
              "files": {name: {"filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                        for name, path in paths.items()},
              "slice_world_mm": world[:3].tolist(), "display_grayscale_max": gray_max,
              "display_difference_max": difference_max}
    args.output.with_suffix(".sources.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
