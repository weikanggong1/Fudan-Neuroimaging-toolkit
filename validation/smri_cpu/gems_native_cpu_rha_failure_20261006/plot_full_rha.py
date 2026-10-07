"""Plot real completed native-grid RHA outputs on fixed union-midpoint slices."""
import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("like", "candidate-artifacts", "official-labels", "scored-results", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    score = json.loads(args.scored_results.read_text())
    if score["status"] != "full_recipe_scored_after_fitting":
        raise RuntimeError("plot only completed and independently scored outputs")
    if sha(args.like) != score["like_sha256"]:
        raise RuntimeError("real reference image changed")
    report = json.loads((args.candidate_artifacts / "report.json").read_text())
    if sha(args.candidate_artifacts / "report.json") != score["fixed_grid_audit"]["candidate_report_sha256"]:
        raise RuntimeError("scored candidate report changed")
    metadata = {int(label): row for label, row in report["labels"].items()
                if row["source"] == "hippo-amygdala-right"}
    ids = sorted(metadata)
    like = nib.load(args.like)
    grid = (like.shape, like.affine)
    candidate_path = args.candidate_artifacts / "subregions_native.nii.gz"
    candidate_receipt = score["full_run_scalar_receipt"]["output_files"]["artifacts/subregions_native.nii.gz"]
    if candidate_path.stat().st_size != candidate_receipt["bytes"] or sha(candidate_path) != candidate_receipt["sha256"]:
        raise RuntimeError("scored candidate labels changed")
    official_sha = score["fixed_grid_audit"]["official_label_sha256"]["hippo-amygdala/rh.hippoAmygLabels.FSvoxelSpace.mgz"]
    if sha(args.official_labels) != official_sha:
        raise RuntimeError("scored official labels changed")
    args.output.mkdir(mode=0o700, exist_ok=False)
    official = resample_from_to(nib.load(args.official_labels), grid, order=0)
    candidate = resample_from_to(nib.load(candidate_path), grid, order=0)
    raw = np.asarray(official.dataobj, dtype=np.int32)
    official_values = np.where(np.isin(raw + 10000, ids), raw + 10000, 0)
    candidate_values = np.asarray(candidate.dataobj, dtype=np.int32).copy()
    candidate_values[~np.isin(candidate_values, ids)] = 0
    canonical = nib.as_closest_canonical(like)
    intensity = np.asarray(canonical.dataobj, dtype=np.float32)
    def ras(values):
        return np.asarray(nib.as_closest_canonical(nib.Nifti1Image(values, like.affine)).dataobj)
    official_values, candidate_values = ras(official_values), ras(candidate_values)
    foreground = (official_values != 0) | (candidate_values != 0)
    coordinates = np.array(np.where(foreground))
    if not coordinates.size:
        raise RuntimeError("both outputs empty; no example brain image")
    lower = np.maximum(coordinates.min(1) - 6, 0)
    upper = np.minimum(coordinates.max(1) + 7, intensity.shape)
    center = (coordinates.min(1) + coordinates.max(1)) // 2
    crop = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))
    colors = plt.get_cmap("turbo")(np.linspace(.02, .98, len(ids)))
    cmap = ListedColormap(np.vstack(([0, 0, 0, 0], colors)))
    encode = lambda values: np.searchsorted(ids, values).astype(np.int32) + 1
    old, new = encode(official_values), encode(candidate_values)
    old[official_values == 0] = 0
    new[candidate_values == 0] = 0
    mismatch = official_values != candidate_values
    limits = np.percentile(intensity[crop], [1, 99])
    if limits[1] <= limits[0]:
        limits = [0, float(intensity.max())]
    figure, axes = plt.subplots(3, 3, figsize=(11, 10), constrained_layout=True)
    for axis in range(3):
        plane = list(crop)
        plane[axis] = int(center[axis])
        plane = tuple(plane)
        mri = intensity[plane].T
        for column, values in enumerate((old, new, mismatch)):
            axes[axis, column].imshow(mri, cmap="gray", origin="lower", vmin=limits[0], vmax=limits[1])
            selected = values[plane].T
            if column == 2:
                overlay = np.zeros(selected.shape + (4,))
                overlay[..., 0] = 1
                overlay[..., 3] = selected * .85
                axes[axis, column].imshow(overlay, origin="lower")
            else:
                axes[axis, column].imshow(np.ma.masked_equal(selected, 0), cmap=cmap, origin="lower",
                                          vmin=0, vmax=len(ids), alpha=.75, interpolation="nearest")
            axes[axis, column].set_axis_off()
            if axis == 0:
                axes[axis, column].set_title(("Official", "FNIT native CPU", "Different label")[column])
        axes[axis, 0].text(.02, .02, ("Sagittal", "Coronal", "Axial")[axis],
                          transform=axes[axis, 0].transAxes, color="white")
    figure.suptitle("Right hippocampal and amygdala subregions: same real T1 native grid")
    figure.savefig(args.output / "RHA_native_overlay.png", dpi=180)
    plt.close(figure)
    legend, axis = plt.subplots(figsize=(10, 4))
    axis.axis("off")
    axis.legend(handles=[Patch(facecolor=colors[index], label=metadata[label]["name"])
                         for index, label in enumerate(ids)], ncol=3, loc="center", frameon=False)
    legend.savefig(args.output / "RHA_region_legend.png", dpi=160, bbox_inches="tight")
    plt.close(legend)
    receipt = {"status": "real_completed_RHA_native_outputs_plotted",
               "selection": "union foreground bbox midpoint; 6 voxel context; no best-slice selection",
               "grid": "fixed norm voxel grid, order0 for labels, canonical RAS display only",
               "RAS_center_voxel": center.tolist(), "RAS_crop_start": lower.tolist(),
               "RAS_crop_stop": upper.tolist(), "labels": ids,
               "like_sha256": sha(args.like), "candidate_label_sha256": sha(candidate_path),
               "official_label_sha256": sha(args.official_labels),
               "scoring_result_sha256": sha(args.scored_results), "program_sha256": sha(__file__),
               "additional_registration_or_fitting": False,
               "plots": {p.name: sha(p) for p in args.output.glob("*.png")}}
    (args.output / "BRAIN_PLOT.public.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "plots": receipt["plots"]}))


if __name__ == "__main__":
    main()
