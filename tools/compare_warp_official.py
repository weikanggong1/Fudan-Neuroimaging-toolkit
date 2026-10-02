#!/usr/bin/env python3
"""Compare complete real 3D/4D warped images with offline official outputs.

Private JSON entries contain candidate, official, reference_mask, and optional
official_seconds. Only scalar metrics/hashes and masked template slices are
exported. This tool never invokes original software or estimates a transform.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def moments(first, second, mask):
    # Both gzip images are inflated once. Framewise float64 differences keep
    # additional memory bounded without subsampling the statistical domain.
    a = first[..., None] if first.ndim == 3 else first
    b = second[..., None] if second.ndim == 3 else second
    totals = np.zeros(8, dtype=np.float64)
    maximum = 0.0
    changed_bits = 0
    for frame in range(a.shape[-1]):
        x, y = [np.ascontiguousarray(v[..., frame][mask]) for v in (a, b)]
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("comparison inputs must be finite")
        if x.dtype == y.dtype:
            dtype = np.dtype(f"u{x.dtype.itemsize}")
            changed_bits += int(np.count_nonzero(x.view(dtype) != y.view(dtype)))
        else:
            changed_bits += int(np.count_nonzero(x != y))
        x, y = x.astype(np.float64), y.astype(np.float64)
        error = y - x
        maximum = max(maximum, float(np.max(np.abs(error))))
        totals += (x.size, x.sum(), y.sum(), np.dot(x, x), np.dot(y, y),
                   np.dot(x, y), np.dot(error, error), np.abs(error).sum())
    count, sx, sy, sxx, syy, sxy, error2, absolute_sum = totals
    covariance = sxy - sx * sy / count
    variance = max(0.0, sxx - sx * sx / count) * max(0.0, syy - sy * sy / count)
    return {"compared_values": int(count), "frames": a.shape[-1],
            "pearson_r": float(covariance / np.sqrt(variance)) if variance else None,
            "mae": absolute_sum / count, "rmse": float(np.sqrt(error2 / count)),
            "maximum_absolute_error": maximum,
            "changed_values_including_signed_zero": changed_bits if first.dtype == second.dtype else None,
            "changed_numerical_values_if_different_dtype": changed_bits if first.dtype != second.dtype else None}


def figure(official, candidate, mask, output):
    a, b = [v[..., 0] if v.ndim == 4 else v for v in (official, candidate)]
    a, b = np.where(mask, a, 0), np.where(mask, b, 0)
    canvas = Image.new("RGB", (900, 710), "white")
    draw = ImageDraw.Draw(canvas)
    for index, title in enumerate(("Official: first frame", "FNIT: first frame", "Absolute difference")):
        draw.text((index * 300 + 8, 8), title, fill="black")
    intensity = max(float(np.percentile(np.abs(a[mask]), 99.5)), 1e-12)
    delta = np.abs(a - b)
    error_scale = max(float(np.percentile(delta[mask], 99.5)), 1e-12)
    for row, axis in enumerate((2, 1, 0)):
        for column, volume in enumerate((a, b, delta)):
            panel = np.rot90(np.take(volume, volume.shape[axis] // 2, axis=axis))
            scale = error_scale if column == 2 else intensity
            pixels = (np.clip(np.abs(panel) / scale, 0, 1) * 255).astype(np.uint8)
            tile = Image.fromarray(pixels).convert("RGB")
            zoom = min(290 / tile.width, 220 / tile.height)
            tile = tile.resize(
                (max(1, round(tile.width * zoom)), max(1, round(tile.height * zoom))),
                resample=getattr(Image, "Resampling", Image).NEAREST,
            )
            canvas.paste(tile, (column * 300 + (300 - tile.width) // 2,
                                30 + row * 225 + (220 - tile.height) // 2))
    canvas.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configuration = json.loads(args.case_json.read_text())
    report = {"schema_version": 1, "benchmark_sha256": sha256(__file__),
              "scope": "complete saved 3D/4D images; no re-estimation or frame reduction",
              "official_provenance": configuration.get("official_provenance", {}), "cases": []}
    for index, case in enumerate(configuration["cases"], 1):
        images = {key: nib.load(case[key]) for key in ("candidate", "official", "reference_mask")}
        candidate, official, mask_image = [images[k] for k in ("candidate", "official", "reference_mask")]
        if candidate.ndim not in (3, 4) or official.ndim not in (3, 4):
            raise ValueError("only complete 3D/4D real images are supported")
        if candidate.get_data_dtype().kind not in "iuf" or official.get_data_dtype().kind not in "iuf":
            raise ValueError("only real-valued numeric images are supported")
        if candidate.shape != official.shape or candidate.shape[:3] != mask_image.shape:
            raise ValueError("shape mismatch")
        if not np.allclose(candidate.affine, official.affine, rtol=0, atol=1e-5):
            raise ValueError("physical grid mismatch")
        if not np.allclose(candidate.affine, mask_image.affine, rtol=0, atol=1e-5):
            raise ValueError("mask grid mismatch")
        a, b = [np.asanyarray(v.dataobj) for v in (official, candidate)]
        mask = np.asanyarray(mask_image.dataobj) > 0
        if not mask.any():
            raise ValueError("empty reference mask")
        anonymous = f"case_{index:03d}"
        record = {"case": anonymous, "label": case["label"], "shape": list(candidate.shape),
                  "input_sha256": {k: sha256(case[k]) for k in images},
                  "brain_mask_voxels": int(mask.sum()), "brain": moments(a, b, mask),
                  "whole_grid": moments(a, b, np.ones(mask.shape, bool)),
                  "official_dtype": str(official.get_data_dtype()),
                  "candidate_dtype": str(candidate.get_data_dtype()),
                  "affine_exact": bool(np.array_equal(candidate.affine, official.affine))}
        header_keys = ("dim", "pixdim", "datatype", "xyzt_units", "intent_code", "intent_p1", "intent_p2", "intent_p3", "qform_code", "sform_code")
        record["header_fields_equal"] = {key: candidate.header[key].tobytes() == official.header[key].tobytes() for key in header_keys}
        record["extensions_equal"] = [(e.get_code(), e.content) for e in candidate.header.extensions] == [(e.get_code(), e.content) for e in official.header.extensions]
        if candidate.ndim == 4:
            record["temporal_metadata"] = {k: {"tr": float(v.header.get_zooms()[3]),
                                               "units": list(v.header.get_xyzt_units())}
                                           for k, v in (("official", official), ("candidate", candidate))}
        if case.get("official_seconds"):
            record["official_command_seconds"] = float(Path(case["official_seconds"]).read_text().strip())
        figure(a, b, mask, args.output_dir / f"{anonymous}.png")
        report["cases"].append(record)
        (args.output_dir / "report.public.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
