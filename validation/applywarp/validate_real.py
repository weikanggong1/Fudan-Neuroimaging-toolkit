#!/usr/bin/env python3
"""Run one real-image TorchApplyWarp comparison against an existing FSL output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw
import torch

from fnit.applywarp import TorchApplyWarp


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(candidate: np.ndarray, reference: np.ndarray) -> dict:
    mask = np.isfinite(candidate) & np.isfinite(reference)
    mask &= (candidate != 0) | (reference != 0)
    first = candidate[mask].astype(np.float64, copy=False)
    second = reference[mask].astype(np.float64, copy=False)
    difference = first - second
    return {
        "voxels": int(first.size),
        "pearson": float(np.corrcoef(first, second)[0, 1]),
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "maximum_absolute_error": float(np.max(np.abs(difference))),
    }


def _plot(candidate: np.ndarray, reference: np.ndarray, output: Path) -> None:
    difference = np.abs(candidate - reference)
    finite_values = reference[np.isfinite(reference)]
    vmax = float(np.percentile(finite_values, 99.5))
    diff_values = difference[np.isfinite(difference)]
    diff_max = max(float(np.percentile(diff_values, 99.9)), 1e-12)
    slices = ((2, reference.shape[2] // 2), (1, reference.shape[1] // 2))
    rows = []
    for axis, index in slices:
        panels = (
            np.take(reference, index, axis=axis),
            np.take(candidate, index, axis=axis),
            np.take(difference, index, axis=axis),
        )
        rendered = []
        for column, panel in enumerate(panels):
            values = np.rot90(panel)
            if column < 2:
                scaled = np.clip(values / vmax, 0, 1)
                rgb = np.repeat((scaled * 255).astype(np.uint8)[..., None], 3, axis=2)
            else:
                scaled = np.clip(values / diff_max, 0, 1)
                red = (scaled * 255).astype(np.uint8)
                green = (np.sqrt(scaled) * 150).astype(np.uint8)
                rgb = np.stack((red, green, np.zeros_like(red)), axis=2)
            image = Image.fromarray(rgb).resize((320, 320), Image.Resampling.BILINEAR)
            rendered.append(image)
        strip = Image.new("RGB", (960, 320), "black")
        for column, image in enumerate(rendered):
            strip.paste(image, (column * 320, 0))
        rows.append(strip)
    canvas = Image.new("RGB", (960, 700), "white")
    draw = ImageDraw.Draw(canvas)
    for column, title in enumerate(("FSL applywarp", "FNIT TorchApplyWarp", "absolute difference")):
        draw.text((column * 320 + 10, 8), title, fill="black")
    canvas.paste(rows[0], (0, 30))
    canvas.paste(rows[1], (0, 360))
    draw.text((8, 338), "axial", fill="black")
    draw.text((8, 668), "coronal", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--warp", required=True)
    parser.add_argument("--official-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--figure", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument("--fsl-wall-seconds", type=float, action="append", default=[])
    args = parser.parse_args(argv)

    if args.repetitions < 2:
        raise ValueError("repetitions must include at least one warm-up and one measured run")
    paths = {
        name: Path(value)
        for name, value in {
            "input": args.input,
            "reference": args.reference,
            "warp": args.warp,
            "official_output": args.official_output,
        }.items()
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    warper = TorchApplyWarp(device=args.device)
    elapsed = []
    peak_memory = []
    result = None
    for _ in range(args.repetitions):
        if warper.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        result = warper(
            input=paths["input"],
            reference=paths["reference"],
            warp=paths["warp"],
            interpolation="trilinear",
            output_dtype="float",
        )
        if warper.device.type == "cuda":
            torch.cuda.synchronize()
            peak_memory.append(int(torch.cuda.max_memory_allocated()))
        elapsed.append(time.perf_counter() - started)
    assert result is not None
    result.save(output)

    candidate_image = nib.load(str(output))
    official_image = nib.load(str(paths["official_output"]))
    candidate = np.asarray(candidate_image.dataobj, dtype=np.float32).squeeze()
    official = np.asarray(official_image.dataobj, dtype=np.float32).squeeze()
    if candidate.ndim != 3 or official.ndim != 3:
        raise ValueError("candidate and official outputs must each contain one 3D map")
    _plot(candidate, official, Path(args.figure))

    core_file = Path(__import__("fnit.applywarp.core", fromlist=["x"]).__file__)
    measured = elapsed[1:]
    report = {
        "schema_version": 1,
        "data": {
            "kind": "one deidentified real UKB-format dMRI subject",
            "map": "FA",
            "subject_identifier_published": False,
        },
        "candidate": {
            "implementation": "FNIT TorchApplyWarp",
            "device": str(warper.device),
            "source_file": "src/fnit/applywarp/core.py",
            "source_sha256": _sha256(core_file),
            "tf32": result.qc["tf32"],
            "warp_representation": result.qc["warp_representation"],
            "warp_intent_code": result.qc["warp_intent_code"],
            "interpolation": result.qc["interpolation"],
            "valid_fraction": result.qc["valid_fraction"],
            "timing_scope": "NIfTI load plus transform plus output transfer to CPU; save excluded",
            "warmup_seconds": elapsed[0],
            "measured_seconds": measured,
            "median_seconds": statistics.median(measured),
            "peak_cuda_memory_bytes": max(peak_memory) if peak_memory else None,
        },
        "reference": {
            "implementation": "FSL 6.0.7.4 applywarp",
            "timing_scope": "external command including NIfTI load, transform, and save",
            "wall_seconds": args.fsl_wall_seconds,
            "median_seconds": (
                statistics.median(args.fsl_wall_seconds)
                if args.fsl_wall_seconds
                else None
            ),
        },
        "contract": {
            "same_shape": candidate.shape == official.shape,
            "same_affine": bool(np.allclose(candidate_image.affine, official_image.affine, atol=1e-5, rtol=0)),
            "same_dtype": candidate_image.get_data_dtype() == official_image.get_data_dtype(),
        },
        "accuracy_union_support": _metrics(candidate, official),
        "artifacts": {
            "candidate_output": output.name,
            "figure": Path(args.figure).name,
        },
        "input_sha256": {name: _sha256(path) for name, path in paths.items()},
        "limits": [
            "one real subject and one continuous FA map",
            "official FNIRT cubic coefficient warp was fixed; nonlinear registration was not rerun",
            "FSL and FNIT timing scopes differ because FNIT save time is excluded",
        ],
    }
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
