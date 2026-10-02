"""Plot measured fixed-TCK weights on the actual native-DWI FA grid (CPU only)."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import nibabel as nib
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--component-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.checkpoint_dir
    values = np.load(args.component_report.with_suffix(".npz"), allow_pickle=False)
    baseline = values["baseline_weights"]
    candidate = values["weights"]
    if not (np.isfinite(baseline).all() and np.isfinite(candidate).all()):
        raise ValueError("Cannot plot nonfinite measured weights")
    affine = np.load(source / "geometry.npz", allow_pickle=False)["dwi_affine"]
    fa = nib.load(source / "fa.nii.gz").get_fdata(dtype=np.float32)
    paths = list(nib.streamlines.load(source / "tracks.tck").streamlines)
    if not (len(paths) == len(baseline) == len(candidate)):
        raise ValueError("TCK and measured weight counts differ")
    inverse = np.linalg.inv(affine)
    voxel_paths = [path @ inverse[:3, :3].T + inverse[:3, 3] for path in paths]
    # Choose a real axial section by streamline segment count. This changes only
    # the illustration, never the measured tracks, weights or benchmark inputs.
    counts = np.zeros(fa.shape[2], dtype=np.int64)
    for path in voxel_paths:
        indices = np.rint(path[:, 2]).astype(np.int64)
        valid = indices[(indices >= 0) & (indices < len(counts))]
        np.add.at(counts, valid, 1)
    section = int(np.argmax(counts))
    segments, owners = [], []
    for track, path in enumerate(voxel_paths):
        for left, right in zip(path[:-1], path[1:]):
            if abs((left[2] + right[2]) / 2 - section) <= 1.5:
                segments.append(np.stack((left[:2], right[:2])))
                owners.append(track)
    segments = np.asarray(segments)
    owners = np.asarray(owners, dtype=np.int64)
    difference = np.abs(candidate - baseline)
    shared_max = float(max(baseline.max(), candidate.max()))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    for axis, field, title, maximum in zip(
            axes, (baseline, candidate, difference),
            ("Frozen optimizer weights", "Candidate optimizer weights", "Absolute weight difference"),
            (shared_max, shared_max, float(difference.max()))):
        axis.imshow(fa[:, :, section].T, origin="lower", cmap="gray", vmin=0, vmax=1)
        if maximum > 0:
            lines = LineCollection(segments, cmap="viridis", linewidths=.35,
                                   norm=plt.Normalize(vmin=0, vmax=maximum))
            lines.set_array(field[owners])
            axis.add_collection(lines)
            fig.colorbar(lines, ax=axis, shrink=.7, label="SIFT2 weight")
        else:
            axis.text(.5, .04, "All differences are zero", transform=axis.transAxes,
                      ha="center", color="white", bbox=dict(facecolor="black", alpha=.8))
        axis.set(title=title, xlim=(0, fa.shape[0]-1), ylim=(0, fa.shape[1]-1),
                 xlabel="Native DWI voxel x", ylabel="Native DWI voxel y")
        axis.set_aspect("equal")
    fig.suptitle(f"Real ds001226 CON03 | {len(paths)} accepted tracks | axial voxel z={section}\n"
                 "FA background; segments within ±1.5 voxels of the selected section")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    inputs = [source/"tracks.tck", source/"geometry.npz", source/"fa.nii.gz",
              args.component_report, args.component_report.with_suffix(".npz")]
    metadata = dict(scope="Real-data illustration only; no benchmark calculation or resampling",
                    accepted_tracks=len(paths), section_voxel_z=section,
                    slab_half_width_voxels=1.5, displayed_segments=len(segments),
                    weight_difference_max=float(difference.max()),
                    input_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs})
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2)+"\n")


if __name__ == "__main__":
    main()
