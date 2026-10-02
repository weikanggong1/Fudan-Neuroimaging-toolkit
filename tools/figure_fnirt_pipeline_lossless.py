#!/usr/bin/env python3
"""Plot anonymous baseline/candidate/difference inside one template brain mask.

For 4D inputs read only the requested frame through ArrayProxy slicing. This
illustration does not replace the independently completed full-series gate.
Only masked image slices are written; paths and subject labels are not saved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-image", type=Path, required=True)
    parser.add_argument("--candidate-image", type=Path, required=True)
    parser.add_argument("--template-mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="FNIRT pipeline: masked template-space example")
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args()

    import nibabel as nib
    import numpy as np
    from PIL import Image, ImageDraw

    images = [nib.load(str(path)) for path in (args.baseline_image, args.candidate_image)]
    mask_image = nib.load(str(args.template_mask))
    if len(mask_image.shape) != 3:
        raise ValueError("template mask must be a 3D image")
    if images[0].shape != images[1].shape:
        raise ValueError("baseline and candidate image shape differs")
    for image in images:
        if len(image.shape) not in (3, 4) or image.shape[:3] != mask_image.shape:
            raise ValueError("image and template mask spatial shapes differ")
        if not np.array_equal(image.affine, mask_image.affine):
            raise ValueError("image and template mask affines differ")
        if not np.array_equal(image.header.get_zooms()[:3], mask_image.header.get_zooms()[:3]):
            raise ValueError("image and template mask spatial voxel sizes differ")
    if args.frame < 0 or (images[0].ndim == 3 and args.frame != 0):
        raise ValueError("frame must select an existing volume; a 3D image uses 0")
    if images[0].ndim == 4 and args.frame >= images[0].shape[3]:
        raise ValueError("frame exceeds the 4D volume count")

    mask_values = np.asanyarray(mask_image.dataobj)
    if not np.isfinite(mask_values).all():
        raise ValueError("template mask contains NaN or infinity")
    mask = mask_values > 0
    if not mask.any():
        raise ValueError("template brain mask is empty")
    # Do not decode the complete 4D gzip series for a first-frame illustration.
    frames = [np.asanyarray(image.dataobj[..., args.frame]) if image.ndim == 4
              else np.asanyarray(image.dataobj) for image in images]
    if any(not np.isfinite(frame[mask]).all() for frame in frames):
        raise ValueError("displayed brain frame contains NaN or infinity")
    a, b = [np.where(mask, frame, 0) for frame in frames]
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    low = min(0.0, float(np.percentile(a[mask], 0.5)), float(np.percentile(b[mask], 0.5)))
    high = max(float(np.percentile(a[mask], 99.5)), float(np.percentile(b[mask], 99.5)), low + 1e-12)
    maximum_error = float(delta[mask].max())
    error_scale = float(np.percentile(delta[mask], 99.5))
    error_scale = max(error_scale, maximum_error * 1e-6, 1e-12)
    # Show brain-centered orthogonal slices in the existing image voxel grid.
    nonempty = np.argwhere(mask)
    indices = tuple(int((nonempty[:, axis].min() + nonempty[:, axis].max()) // 2)
                    for axis in range(3))
    canvas = Image.new("RGB", (900, 780), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), args.title, fill="black")
    draw.text((8, 28), f"Frame {args.frame}; template brain mask only; full-series gate is reported separately", fill="black")
    titles = ("Baseline", "Candidate", "Absolute difference")
    for column, title in enumerate(titles):
        draw.text((column * 300 + 8, 52), title, fill="black")
    for row, axis in enumerate((2, 1, 0)):
        mask_panel = np.rot90(np.take(mask, indices[axis], axis=axis))
        for column, volume in enumerate((a, b, delta)):
            panel = np.rot90(np.take(volume, indices[axis], axis=axis))
            scaled = panel / error_scale if column == 2 else (panel - low) / (high - low)
            pixels = (np.clip(scaled, 0, 1) * 255).astype(np.uint8)
            pixels[~mask_panel] = 0
            tile = Image.fromarray(pixels).convert("RGB")
            zoom = min(290 / tile.width, 220 / tile.height)
            tile = tile.resize(
                (max(1, round(tile.width * zoom)), max(1, round(tile.height * zoom))),
                resample=getattr(Image, "Resampling", Image).NEAREST,
            )
            canvas.paste(tile, (column * 300 + (300 - tile.width) // 2,
                                75 + row * 225 + (220 - tile.height) // 2))
    draw.text((8, 754), f"Intensity range: {low:.6g} to {high:.6g}; max brain |difference|: {maximum_error:.6g}", fill="black")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    same_dtype = frames[0].dtype == frames[1].dtype
    same_bits = same_dtype and np.ascontiguousarray(frames[0]).tobytes() == np.ascontiguousarray(frames[1]).tobytes()
    print(json.dumps({
        "scope": "masked PNG illustration; selected frame only, not complete 4D verification",
        "frame_index": args.frame,
        "input_shapes": [list(image.shape) for image in images],
        "input_dtypes": [str(image.get_data_dtype()) for image in images],
        "mask_shape": list(mask_image.shape),
        "mask_geometry_matches_both_images": True,
        "brain_mask_voxels": int(mask.sum()),
        "displayed_voxel_slice_indices": list(indices),
        "displayed_frame_bitwise_equal_including_signed_zero": bool(same_bits),
        "displayed_brain_maximum_absolute_error": maximum_error,
        "template_mask_array_sha256": hashlib.sha256(np.ascontiguousarray(mask_values).tobytes()).hexdigest(),
        "figure_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "figure_tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
