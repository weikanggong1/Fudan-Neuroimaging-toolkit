"""Compare saved labels with a shared local T1 percentile display window."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import nibabel as nib
from nibabel.processing import resample_from_to, resample_to_output
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t1", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-offset", type=int, default=10000)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-name", default="FNIT precision candidate")
    parser.add_argument("--baseline-name", default="FNIT 4178a48")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels", type=int, nargs="+", default=[10203, 10240, 10243, 10244])
    parser.add_argument("--names", nargs="+", default=["Parasubiculum", "CA3-body", "DG-head", "DG-body"])
    parser.add_argument("--slice-padding-mm", type=float, default=0,
                        help="Predeclared superior/inferior display padding around official labels")
    args = parser.parse_args()
    if len(args.labels) != len(args.names):
        parser.error("--labels and --names must have the same length")
    if not np.isfinite(args.slice_padding_mm) or args.slice_padding_mm < 0:
        parser.error("--slice-padding-mm must be finite and nonnegative")
    original = nib.load(args.t1)
    canonical = nib.as_closest_canonical(original)
    linear = canonical.affine[:3, :3]
    oblique = not np.allclose(linear, np.diag(np.diag(linear)), rtol=0, atol=1e-5)
    image = resample_to_output(canonical, voxel_sizes=np.linalg.norm(linear, axis=0), order=1) if oblique else canonical
    grid = (image.shape, image.affine)
    data = image.get_fdata(dtype=np.float32)
    arrays = []
    for path in (args.reference, args.baseline, args.candidate):
        arrays.append(np.asarray(resample_from_to(nib.load(path), grid, order=0).dataobj, dtype=np.int32))
    if args.reference_offset:
        arrays[0] = np.where(arrays[0] != 0, arrays[0] + args.reference_offset, 0)
    selected = [np.where(np.isin(array, args.labels), array, 0) for array in arrays]
    points = np.argwhere(np.logical_or.reduce([array != 0 for array in selected]))
    if not len(points):
        raise ValueError("the requested labels are absent from all images")
    low = np.maximum(points.min(0) - 6, 0)
    high = np.minimum(points.max(0) + 7, image.shape)
    crop = (slice(low[0], high[0]), slice(low[1], high[1]))
    reference_points = np.argwhere(selected[0] != 0)
    if not len(reference_points):
        raise ValueError("the requested official labels are absent; slice selection cannot use a candidate")
    z_spacing = float(np.linalg.norm(image.affine[:3, 2]))
    padding_slices = int(np.ceil(args.slice_padding_mm / z_spacing))
    z_low = max(0, int(reference_points[:, 2].min()) - padding_slices)
    z_high = min(image.shape[2] - 1, int(reference_points[:, 2].max()) + padding_slices)
    indices = np.linspace(z_low, z_high, 8).round().astype(int)[1:-1]
    palette = [(0.2, .9, .35), (.25, .5, 1), (1, .65, .15), (.95, .3, .7)]
    if len(args.labels) > len(palette):
        palette = [tuple(plt.get_cmap("tab10")(i % 10)[:3]) for i in range(len(args.labels))]
    fig, axes = plt.subplots(5, 6, figsize=(13.5, 10), facecolor="black")
    # Window derives solely from the fixed T1 display crop and six planes.
    display_t1 = np.stack([data[crop + (int(index),)] for index in indices], axis=-1)
    positive_t1 = display_t1[np.isfinite(display_t1) & (display_t1 > 0)]
    if not len(positive_t1):
        raise ValueError("no positive finite T1 values in the fixed display crop")
    vmin, vmax = map(float, np.percentile(positive_t1, [2, 98]))
    if vmax <= vmin:
        raise ValueError("local T1 display window has no intensity range")
    spacing = np.linalg.norm(image.affine[:3, :3], axis=0)
    for column, index in enumerate(indices):
        t1 = data[crop + (index,)].T
        for row in range(5):
            axes[row, column].imshow(t1, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        for row, array in enumerate(selected):
            plane = array[crop + (index,)].T
            rgba = np.zeros((*plane.shape, 4), np.float32)
            for label, color in zip(args.labels, palette):
                rgba[plane == label] = (*color, .82)
            axes[row, column].imshow(rgba, origin="lower")
        for row, array in ((3, selected[1]), (4, selected[2])):
            difference = (array[crop + (index,)] != selected[0][crop + (index,)]).T
            rgba = np.zeros((*difference.shape, 4), np.float32)
            rgba[difference] = (1, .2, .15, .85)
            axes[row, column].imshow(rgba, origin="lower")
        z = float((image.affine @ [0, 0, index, 1])[2])
        # Place titles in figure coordinates. Narrow anatomical crops resize
        # image axes to preserve voxel aspect; adjacent axes can otherwise
        # paint over the title that extends beyond a narrow image rectangle.
        x_center = .085 + (column + .5) * (.997 - .085) / 6
        fig.text(x_center, .985, f"z = {z:.1f} mm", color="white", fontsize=10,
                 ha="center", va="center")
    for row, name in enumerate(("FreeSurfer 8.2", args.baseline_name, args.candidate_name,
                                "Previous differences", "Candidate differences")):
        axes[row, 0].set_ylabel(name, color="white", fontsize=10)
    for axis in axes.flat:
        axis.set_aspect(float(spacing[1] / spacing[0]))
        axis.set_xticks([])
        axis.set_yticks([])
    fig.legend(handles=[Patch(color=color, label=name) for color, name in zip(palette, args.names)],
               loc="lower center", ncol=min(4, len(args.labels)), labelcolor="white",
               facecolor="black", edgecolor="none")
    fig.subplots_adjust(left=.085, right=.997, top=.97, bottom=.055, hspace=.025, wspace=.015)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, facecolor="black")
    plt.close(fig)
    def identity(path):
        content = path.read_bytes()
        return {"path": str(path), "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    metadata = {"kind": "saved_real_subregion_low_Dice_axial_comparison",
                "inputs": {key: identity(path) for key, path in
                           (("t1", args.t1), ("reference", args.reference),
                            ("baseline", args.baseline), ("candidate", args.candidate))},
                "plot_driver": identity(Path(__file__)), "output": identity(args.output),
                "reference_offset": args.reference_offset, "labels": args.labels, "names": args.names,
                "original_shape": [int(size) for size in original.shape], "original_affine": original.affine.tolist(),
                "display_shape": [int(size) for size in image.shape], "display_affine": image.affine.tolist(),
                "oblique_display_resampled": oblique, "display_pixel_aspect_y_over_x": float(spacing[1]/spacing[0]),
                "slices_display_indices": indices.tolist(),
                "slices_RAS_z_mm": [float((image.affine @ [0, 0, index, 1])[2]) for index in indices],
                "slice_selection": "six fixed interior planes across the official selected labels "
                                   "with predeclared superior/inferior padding",
                "slice_padding_mm": float(args.slice_padding_mm),
                "slice_padding_display_voxels": padding_slices,
                "slice_indices_unique": bool(len(np.unique(indices)) == len(indices)),
                "grayscale_window": {"vmin": vmin, "vmax": vmax,
                                     "percentiles": [2, 98], "positive_finite_T1_values": int(len(positive_t1)),
                                     "scope": "positive finite T1 intensities within fixed display XY crop and six displayed axial planes; same window for official/before/final/difference rows; no model-label intensity selection"},
                "interpolation": {"labels": "nearest neighbour", "T1": "linear if oblique"},
                "difference": "red marks differing selected fine label IDs, including internal label changes",
                "metric_scope": "Dice remains on the original benchmark measurement grid"}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
