#!/usr/bin/env python3
"""Draw brain-only MNI slices from completed independent public-data pipelines.

This is a display utility, not a registration or numerical benchmark. Input
arrays are only permuted/flipped to canonical RAS orientation; they are never
resampled. It accepts an axis-aligned MNI grid and chooses the nearest voxel
plane to the requested world z coordinate. All panels use nearest display
interpolation, a shared template-brain mask, and the fixed ranges below.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


MAP_NAMES = ("FA", "MD", "ICVF")
MAP_LIMITS = {"FA": (0.0, 1.0), "MD": (0.0, 0.003), "ICVF": (0.0, 1.0)}
DIFFERENCE_LIMITS = {"FA": (0.0, 0.25), "MD": (0.0, 0.00075), "ICVF": (0.0, 0.4)}
MAP_UNITS = {"FA": "dimensionless", "MD": "mm^2/s", "ICVF": "fraction"}
ROLES = ("FNIT", "Original", "Absolute difference")


def file_record(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def load_scalar(path):
    """Accept 3D maps and original TBSS's one-volume 4D maps."""
    image = nib.load(str(path))
    if image.ndim == 3:
        values = np.asanyarray(image.dataobj)
    elif image.ndim == 4 and image.shape[-1] == 1:
        values = np.asanyarray(image.dataobj)[..., 0]
    else:
        raise ValueError("a plotted map must be 3D or singleton 4D")
    if not np.isfinite(values).all() or not np.isfinite(image.affine).all():
        raise ValueError("a plotted map must contain finite values and geometry")
    return image, values


def canonical_values(image, values):
    """Canonical orientation uses array permutations/flips, never interpolation."""
    original = nib.orientations.io_orientation(image.affine)
    target = nib.orientations.axcodes2ornt(("R", "A", "S"))
    transform = nib.orientations.ornt_transform(original, target)
    oriented = nib.orientations.apply_orientation(values, transform)
    affine = image.affine @ nib.orientations.inv_ornt_aff(transform, image.shape[:3])
    if not np.allclose(affine[:3, :3], np.diag(np.diag(affine[:3, :3])),
                       rtol=0, atol=1e-5):
        raise ValueError("world-coordinate plotting requires an axis-aligned MNI grid")
    if np.any(np.diag(affine[:3, :3]) <= 0):
        raise ValueError("canonical plotting geometry must increase along R, A and S")
    return oriented, affine


def slice_geometry(shape, affine, requested_z_mm):
    if not np.isfinite(requested_z_mm):
        raise ValueError("requested z coordinate must be finite")
    centres = affine[2, 3] + np.arange(shape[2]) * affine[2, 2]
    half_voxel = affine[2, 2] / 2.0
    if requested_z_mm < centres[0] - half_voxel or requested_z_mm > centres[-1] + half_voxel:
        raise ValueError("requested z coordinate falls outside the image")
    index = int(np.argmin(np.abs(centres - requested_z_mm)))
    extent = [float(affine[0, 3] - affine[0, 0] / 2),
              float(affine[0, 3] + (shape[0] - 0.5) * affine[0, 0]),
              float(affine[1, 3] - affine[1, 1] / 2),
              float(affine[1, 3] + (shape[1] - 0.5) * affine[1, 1])]
    return index, float(centres[index]), extent


def clip_record(values, mask, limits):
    brain = np.asarray(values)[mask].astype(np.float64)
    if brain.size == 0:
        raise ValueError("selected world plane contains no template-brain voxels")
    below = int(np.count_nonzero(brain < limits[0]))
    above = int(np.count_nonzero(brain > limits[1]))
    return {"brain_voxels": int(brain.size), "display_min": float(limits[0]),
            "display_max": float(limits[1]), "slice_min": float(brain.min()),
            "slice_max": float(brain.max()), "below_range_voxels": below,
            "above_range_voxels": above, "below_range_fraction": below / brain.size,
            "above_range_fraction": above / brain.size,
            "colour_range_clipped_fraction": (below + above) / brain.size}


def original_map_path(root, branch, name):
    if branch == "tbss":
        return root / "tbss" / "stats" / f"all_{name}.nii.gz"
    return root / "mmorf" / "standard" / f"{name}.nii.gz"


def make_figure(*, candidate_dir, original_dir, branch, mask_template,
                output, caption_json=None, z_mm=16.0, case_label="case01"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    candidate_dir, original_dir = Path(candidate_dir), Path(original_dir)
    mask_template, output = Path(mask_template), Path(output)
    caption_json = output.with_suffix(".json") if caption_json is None else Path(caption_json)
    if branch not in ("tbss", "mmorf"):
        raise ValueError("branch must be tbss or mmorf")
    if not case_label or any(character in case_label for character in ("/", "\\", "\n", "\r")):
        raise ValueError("case label must be a short display label without a path")
    mask_image, mask_values = load_scalar(mask_template)
    canonical_mask, canonical_affine = canonical_values(mask_image, mask_values > 0)
    index, actual_z_mm, extent = slice_geometry(canonical_mask.shape, canonical_affine, z_mm)
    slice_mask = canonical_mask[..., index]
    if not slice_mask.any():
        raise ValueError("selected world plane contains no template brain")
    report = {
        "schema_version": 1, "case_label": case_label, "branch": branch,
        "scope": "one selected completed real public-data case; template-brain axial slices, not full-3D accuracy metrics",
        "caption": "FA, MD and ICVF: FNIT, independent original-software reference, and absolute difference. Every panel uses the same template-brain mask; colours outside the fixed displayed ranges are saturated.",
        "display": {"requested_world_z_mm": float(z_mm), "actual_world_z_mm": actual_z_mm,
                    "canonical_voxel_z_index": index, "world_extent_xy_mm": extent,
                    "canonical_shape": list(canonical_mask.shape),
                    "canonical_affine": canonical_affine.tolist(),
                    "orientation": "neurological: subject left at left, anterior at top; canonical RAS",
                    "array_operation": "voxel permutation and flips only; no image resampling",
                    "imshow_interpolation": "nearest", "brain_mask": "template > 0",
                    "spatial_units_in_template_header": mask_image.header.get_xyzt_units()[0]},
        "template_mask": {**file_record(mask_template),
                          "brain_voxels_3D": int(np.count_nonzero(mask_values > 0)),
                          "brain_voxels_shown": int(np.count_nonzero(slice_mask))},
        "source_script": file_record(__file__), "maps": {},
    }
    figure, axes = plt.subplots(3, 3, figsize=(12, 11.3), constrained_layout=True)
    figure.suptitle(f"{case_label} | {branch.upper()} | axial z = {actual_z_mm:g} mm", fontsize=15)
    for row, name in enumerate(MAP_NAMES):
        paths = (candidate_dir / "registration" / "standard" / f"{name}.nii.gz",
                 original_map_path(original_dir, branch, name))
        displayed = []
        sources = {}
        for role, path in zip(ROLES[:2], paths):
            image, values = load_scalar(path)
            if values.shape != mask_values.shape or not np.allclose(
                    image.affine, mask_image.affine, rtol=0, atol=1e-5):
                raise ValueError("all maps and the fixed mask must share one grid")
            oriented, affine = canonical_values(image, values)
            if not np.allclose(affine, canonical_affine, rtol=0, atol=1e-5):
                raise ValueError("canonical map geometry differs from template geometry")
            displayed.append(np.asarray(oriented[..., index], dtype=np.float64))
            sources[role] = {**file_record(path), "source_shape": list(image.shape),
                             "stored_dtype": str(image.get_data_dtype()),
                             "singleton_volume_squeezed": image.ndim == 4}
        displayed.append(np.abs(displayed[0] - displayed[1]))
        report["maps"][name] = {"units": MAP_UNITS[name], "sources": sources,
                                "panels": {}}
        for column, (role, values) in enumerate(zip(ROLES, displayed)):
            limits = DIFFERENCE_LIMITS[name] if column == 2 else MAP_LIMITS[name]
            cmap = matplotlib.colormaps["magma" if column == 2 else "gray"].with_extremes(bad="black")
            shown = np.ma.masked_where(~slice_mask, values).T
            axis = axes[row, column]
            image_artist = axis.imshow(shown, origin="lower", extent=extent,
                                       vmin=limits[0], vmax=limits[1], cmap=cmap,
                                       interpolation="nearest", aspect="equal")
            axis.set_title(f"{name} | {role}", fontsize=11)
            axis.set_xlabel("World x (mm)")
            axis.set_ylabel("World y (mm)")
            axis.text(0.02, 0.98, "L", transform=axis.transAxes,
                      color="white", va="top", ha="left", fontsize=9)
            axis.text(0.98, 0.98, "R", transform=axis.transAxes,
                      color="white", va="top", ha="right", fontsize=9)
            colour_bar = figure.colorbar(image_artist, ax=axis, fraction=0.046, pad=0.02)
            if name == "MD":
                colour_bar.formatter = FormatStrFormatter("%.1e")
                colour_bar.update_ticks()
            colour_bar.set_label("mm²/s" if name == "MD" else "fraction" if name == "ICVF" else "FA", fontsize=9)
            report["maps"][name]["panels"][role] = clip_record(values, slice_mask, limits)
    output.parent.mkdir(parents=True, exist_ok=True)
    caption_json.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(output, dpi=160, facecolor="white", metadata={"Software": "FNIT public10 validation"})
    finally:
        plt.close(figure)
    report["figure"] = file_record(output)
    caption_json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True,
                        help="completed FNIT output root containing registration/standard")
    parser.add_argument("--original-dir", type=Path, required=True,
                        help="independent original-software root containing tbss/stats or mmorf/standard")
    parser.add_argument("--branch", choices=("tbss", "mmorf"), required=True)
    parser.add_argument("--mask-template", type=Path, required=True,
                        help="same-grid standard-space template; brain mask is values > 0")
    parser.add_argument("--output", type=Path, required=True, help="brain-only PNG output")
    parser.add_argument("--caption-json", type=Path,
                        help="anonymous hashes, coordinates and clipping fractions; defaults to PNG basename.json")
    parser.add_argument("--z-mm", type=float, default=16.0)
    parser.add_argument("--case-label", default="case01")
    args = parser.parse_args(argv)
    make_figure(candidate_dir=args.candidate_dir, original_dir=args.original_dir,
                branch=args.branch, mask_template=args.mask_template, output=args.output,
                caption_json=args.caption_json, z_mm=args.z_mm, case_label=args.case_label)
    print(args.output)


if __name__ == "__main__":
    main()
