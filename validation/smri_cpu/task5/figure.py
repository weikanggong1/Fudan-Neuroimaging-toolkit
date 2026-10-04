"""Plot actual subregion reference/candidate maps after measured execution."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--like", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--official-right", type=Path,
                        help="right HA reference, using its original labels before the +10000 mapping")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family", choices=("brainstem", "thalamus", "hippo-amygdala"), default="brainstem")
    args = parser.parse_args()
    native = nib.load(args.like)
    # Only display resampling. Numeric scoring uses the separate fixed-grid audit.
    grid = (native.shape[:3], native.affine)
    ref = resample_from_to(nib.load(args.official), grid, order=0)
    if args.family == "hippo-amygdala":
        if args.official_right is None:
            parser.error("--official-right is required for bilateral hippo-amygdala display")
        right = resample_from_to(nib.load(args.official_right), grid, order=0)
        left_values, right_values = np.asarray(ref.dataobj), np.asarray(right.dataobj)
        if np.any((left_values != 0) & (right_values != 0)):
            raise ValueError("left/right HA reference masks overlap on the display grid")
        ref = nib.Nifti1Image(np.where(right_values != 0, right_values + 10000,
                                      left_values), native.affine)
    cand = resample_from_to(nib.load(args.candidate), grid, order=0)
    native = nib.as_closest_canonical(native)
    ref = nib.as_closest_canonical(ref)
    cand = nib.as_closest_canonical(cand)
    image = np.asarray(native.dataobj)
    first = np.asarray(ref.dataobj)
    second = np.asarray(cand.dataobj)
    all_ids = tuple(int(label) for label in np.union1d(np.unique(first), np.unique(second)))
    if args.family == "brainstem":
        ids = (173, 174, 175, 178)
    elif args.family == "thalamus":
        ids = tuple(label for label in all_ids if 8100 < label < 8300)
    else:
        ids = tuple(label for label in all_ids if
                    200 < label % 10000 < 300 or 7000 < label % 10000 < 7020)
    first = np.where(np.isin(first, ids), first, 0)
    second = np.where(np.isin(second, ids), second, 0)
    color_ids = sorted({label % 10000 for label in ids}) if args.family == "hippo-amygdala" else list(ids)
    included = np.isin(first, ids) | np.isin(second, ids)
    coordinates = np.argwhere(included)
    if not len(coordinates):
        raise ValueError("empty subregion display")
    center = np.median(coordinates, axis=0).astype(int)
    if args.family == "hippo-amygdala":
        # The bilateral median can lie in the midline gap; select an actual ROI voxel.
        center = coordinates[np.argmin(np.sum((coordinates - center) ** 2, axis=1))]
    low = np.maximum(coordinates.min(axis=0) - 12, 0)
    high = np.minimum(coordinates.max(axis=0) + 13, image.shape)
    slices = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
    image, first, second = (array[slices] for array in (image, first, second))
    center -= low
    colormap = (ListedColormap(["#e69f00", "#56b4e9", "#009e73", "#cc79a7"])
                if args.family == "brainstem" else plt.get_cmap("turbo", len(color_ids)))
    upper = float(np.percentile(image[image > 0], 99))
    fig, axes = plt.subplots(3, 3, figsize=(10, 9), facecolor="white")
    for column, (axis, title) in enumerate(((0, "Sagittal"), (1, "Coronal"), (2, "Axial"))):
        plane = lambda array: np.take(array, center[axis], axis=axis).T
        gray = plane(image)
        for row, label_map in enumerate((first, second, first != second)):
            ax = axes[row, column]
            ax.imshow(gray, cmap="gray", vmin=0, vmax=upper, origin="lower")
            if row < 2:
                colored = np.full(label_map.shape, np.nan)
                for label in ids:
                    index = color_ids.index(label % 10000 if args.family == "hippo-amygdala" else label)
                    colored[label_map == label] = index
                ax.imshow(plane(colored), cmap=colormap, vmin=0, vmax=len(color_ids) - 1,
                          origin="lower", alpha=0.65, interpolation="nearest")
            else:
                different = plane(label_map).astype(float)
                different[different == 0] = np.nan
                ax.imshow(different, cmap=ListedColormap(["#ff3030"]),
                          vmin=0, vmax=1, origin="lower", interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(title)
            if column == 0:
                ax.set_ylabel(("Official", "FNIT CPU", "Label difference")[row])
    subtitle = ("Orange: Midbrain | Blue: Pons | Green: Medulla | Purple: SCP"
                if args.family == "brainstem" else
                "Bilateral hippocampal subfields and amygdala nuclei; matching colors"
                if args.family == "hippo-amygdala" else
                "Thalamic nuclei: matching label colors in both rows")
    fig.suptitle("Public T1, same norm/aseg/wmparc stage\n" + subtitle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    manifest = {"display_only": True, "family": args.family, "display_labels": list(ids),
                "right_ha_label_offset": 10000 if args.family == "hippo-amygdala" else None,
                "display_resampling": "nearest-neighbor to norm; scoring uses independent fixed grids",
                "canonical_center_voxels": (center + low).tolist(),
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "input_sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest()
                                 for key, path in (("like", args.like), ("official", args.official),
                                                   ("candidate", args.candidate))}}
    if args.official_right is not None:
        manifest["input_sha256"]["official_right"] = hashlib.sha256(args.official_right.read_bytes()).hexdigest()
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
