#!/usr/bin/env python3
"""Plot saved public CC0 T1 CPU outputs; no private images are accepted.

The input SHA is a strict guard for the repository's published defaced ds000114
example. Figures use the saved final pair only, never rerun a registration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


PUBLIC_T1_SHA256 = "f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def volume(path):
    image = nib.as_closest_canonical(nib.load(path))
    return image, np.asarray(image.dataobj, dtype=np.float64)


def select_outputs(suite_path, threads):
    suite = json.loads(Path(suite_path).read_text())
    if suite["status"] != "executed_with_numeric_comparisons":
        raise ValueError("Figures require a completed real-data comparison")
    records = [record for record in suite["records"] if record["threads"] == threads]
    if len(records) != 1:
        raise ValueError("Select one completed thread budget")
    record = records[0]
    if not any(value["sha256"] == PUBLIC_T1_SHA256
               for value in record["input_metadata_before"].values()):
        raise ValueError("Comparison is not bound to the verified public T1")
    worker = record["worker_results"]["candidate"][-1]
    candidate = worker["outputs"]
    first_path = Path(candidate[next(iter(candidate))])
    # A worker saves under candidate/api_0; locate the actual backend root,
    # rather than assuming a fixed number of directories above the image.
    backend = next((parent for parent in first_path.parents if parent.name == "candidate"), None)
    if backend is None:
        raise ValueError("Saved outputs do not identify a candidate backend directory")
    official = {key: str(backend.parent / "official" / Path(path).name) for key, path in candidate.items()}
    for outputs, metadata in ((official, record["official_output_metadata"]),
                              (candidate, worker["output_metadata"])):
        for key, path in outputs.items():
            if sha256(path) != metadata[key]["sha256"]:
                raise ValueError("Saved public output changed after its comparison")
    return official, candidate


def plot_comparison(arrays, mask, image, destination, row_labels, units, scalar_map="gray", signed_difference=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.6,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "figure.facecolor": "white"})
    points = np.argwhere(mask)
    if not len(points):
        raise ValueError("Empty display region")
    cuts = np.rint((points.min(axis=0) + points.max(axis=0)) / 2).astype(int)
    world = nib.affines.apply_affine(image.affine, cuts)
    foreground = np.concatenate([array[mask] for array in arrays[:2]])
    foreground = foreground[np.isfinite(foreground)]
    low, high = np.percentile(foreground, [1, 99])
    if high <= low:
        high = low + 1
    difference = arrays[2][mask]
    difference = difference[np.isfinite(difference)]
    limit = max(float(np.percentile(np.abs(difference), 99)), np.finfo(float).eps)
    figure, axes = plt.subplots(3, 3, figsize=(7.2, 6.4), constrained_layout=True)
    panel_labels = ("Sagittal", "Coronal", "Axial")
    voxel_size = image.header.get_zooms()[:3]
    last_scalar = last_difference = None
    for row, array in enumerate(arrays):
        for column, axis in enumerate(axes[row]):
            displayed = np.where(mask, array, np.nan)
            section = np.take(displayed, cuts[column], axis=column).T
            if row < 2:
                last_scalar = axis.imshow(section, origin="lower", cmap=scalar_map, vmin=low, vmax=high, interpolation="nearest")
            elif signed_difference:
                last_difference = axis.imshow(section, origin="lower", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), interpolation="nearest")
            else:
                last_difference = axis.imshow(section, origin="lower", cmap="magma", vmin=0, vmax=limit, interpolation="nearest")
            remaining_axes = [index for index in range(3) if index != column]
            axis.set_aspect(voxel_size[remaining_axes[1]] / voxel_size[remaining_axes[0]])
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if row == 0:
                axis.set_title(f"{panel_labels[column]} ({world[column]:.1f} mm)")
            if column == 0:
                axis.set_ylabel(row_labels[row], labelpad=7)
    figure.colorbar(last_scalar, ax=axes[:2].ravel().tolist(), shrink=0.8, label=units)
    figure.colorbar(last_difference, ax=axes[2].ravel().tolist(), shrink=0.8, label="Difference (" + units + ")")
    figure.savefig(destination, dpi=220, facecolor="white")
    figure.savefig(destination.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)
    return {"cuts_voxel_ras": cuts.tolist(), "cuts_world_mm": world.tolist(),
            "scalar_display_range": [float(low), float(high)], "difference_display_limit": limit,
            "difference_values_outside_display_limit_are_clipped": True,
            "voxel_size_mm": [float(value) for value in voxel_size],
            "orientation": "Closest canonical RAS, neurological display, nearest slice; no volume resampling"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-run", required=True)
    parser.add_argument("--public-input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--fnirt-suite", help="Optional completed latest-source FNIRT suite using the same public input")
    parser.add_argument("--fnirt-only", action="store_true", help="Plot the completed FNIRT phase separately from InvWarp")
    parser.add_argument("--invwarp-suite", help="Optional completed single-observation InvWarp suite using the same public input")
    args = parser.parse_args()
    input_path = Path(args.public_input)
    if sha256(input_path) != PUBLIC_T1_SHA256:
        raise ValueError("Only the verified public CC0 defaced T1 may be visualized for publication")
    root = Path(args.public_run)
    fnirt_manifest = json.loads((root / "fnirt.manifest.private.json").read_text())
    if len(fnirt_manifest["cases"]) != 1 or sha256(fnirt_manifest["cases"][0]["input"]) != PUBLIC_T1_SHA256:
        raise ValueError("Registration is not bound to the public example")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fnirt_suite_path = Path(args.fnirt_suite) if args.fnirt_suite else root / "fnirt/suite.private.json"
    official, candidate = select_outputs(fnirt_suite_path, args.threads)
    mask_image, mask_array = volume(fnirt_manifest["cases"][0]["accuracy_mask"])
    mask = mask_array > 0
    provenance = {"public_input_sha256": PUBLIC_T1_SHA256, "threads": args.threads, "figures": {}}
    provenance["fnirt_source_tree_sha256"] = json.loads(fnirt_suite_path.read_text())["source_metadata"]["candidate"]["tree_sha256"]
    for name, key, units, color in (("fnirt_warped", "iout", "T1 intensity", "gray"), ("fnirt_jacobian", "jout", "Jacobian", "viridis")):
        image, reference = volume(official[key])
        moving_image, moving = volume(candidate[key])
        if reference.shape != mask.shape or moving.shape != reference.shape or not np.allclose(image.affine, mask_image.affine) or not np.allclose(image.affine, moving_image.affine):
            raise ValueError("Displayed scalar images and mask must have the same physical grid")
        differences = moving - reference if key == "iout" else np.abs(moving - reference)
        details = plot_comparison([reference, moving, differences], mask, image, output / (name + ".png"),
                                  ["Official FSL", "FNIT CPU", "FNIT − FSL" if key == "iout" else "Absolute difference"], units, color, key == "iout")
        details["source_sha256"] = {"official": sha256(official[key]), "candidate": sha256(candidate[key])}
        provenance["figures"][name] = details
    if args.fnirt_only:
        (output / "figures.public.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
        return
    invwarp_suite_path = Path(args.invwarp_suite) if args.invwarp_suite else root / "invwarp/suite.private.json"
    official_inverse, candidate_inverse = select_outputs(invwarp_suite_path, args.threads)
    provenance["invwarp_source_tree_sha256"] = json.loads(invwarp_suite_path.read_text())["source_metadata"]["candidate"]["tree_sha256"]
    image, inverse_reference = volume(official_inverse["inverse_warp"])
    moving_image, inverse_moving = volume(candidate_inverse["inverse_warp"])
    native_image, native = volume(input_path)
    if inverse_reference.shape != inverse_moving.shape or inverse_reference.shape[:3] != native.shape or not np.allclose(image.affine, native_image.affine) or not np.allclose(image.affine, moving_image.affine):
        raise ValueError("Inverse warps and native public input must have the same physical grid")
    arrays = [np.linalg.norm(inverse_reference, axis=-1), np.linalg.norm(inverse_moving, axis=-1), np.linalg.norm(inverse_moving - inverse_reference, axis=-1)]
    details = plot_comparison(arrays, native > 0, image, output / "invwarp_displacement.png",
                              ["Official inverse magnitude", "FNIT inverse magnitude", "Vector difference"], "mm", "viridis")
    details["region"] = "Positive defaced native T1 foreground, including nearby skull"
    details["source_sha256"] = {"official": sha256(official_inverse["inverse_warp"]), "candidate": sha256(candidate_inverse["inverse_warp"])}
    provenance["figures"]["invwarp_displacement"] = details
    (output / "figures.public.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
