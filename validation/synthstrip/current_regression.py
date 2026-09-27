#!/usr/bin/env python3
"""Run the current SynthStrip CLI on the fixed 12-case real-data manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import nibabel as nib
import numpy as np
import torch

from fnit.synthstrip import SynthStrip


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def time_record(path: Path) -> dict:
    text = path.read_text(errors="replace")
    wall = re.search(
        r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([0-9:.]+)",
        text,
    )
    rss = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    exit_status = re.search(r"Exit status:\s*(\d+)", text)
    if wall is None or rss is None or exit_status is None:
        raise RuntimeError(f"incomplete time log: {path}")
    fields = [float(value) for value in wall.group(1).split(":")]
    seconds = sum(value * 60**power for power, value in enumerate(reversed(fields)))
    return {
        "wall_seconds": seconds,
        "maximum_rss_kbytes": int(rss.group(1)),
        "exit_status": int(exit_status.group(1)),
    }


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = left.astype(np.float64, copy=False).ravel()
    right = right.astype(np.float64, copy=False).ravel()
    left = left - left.mean()
    right = right - right.mean()
    denominator = np.sqrt(np.dot(left, left) * np.dot(right, right))
    return float(np.dot(left, right) / denominator)


def compare_images(candidate_dir: Path, reference_dir: Path) -> dict:
    result = {}
    for name in ("image", "mask", "distance"):
        candidate_image = nib.load(str(candidate_dir / f"{name}.nii.gz"))
        reference_image = nib.load(str(reference_dir / f"{name}.nii.gz"))
        candidate = np.asanyarray(candidate_image.dataobj)
        reference = np.asanyarray(reference_image.dataobj)
        record = {
            "shape_equal": candidate.shape == reference.shape,
            "affine_max_abs": float(np.max(np.abs(candidate_image.affine - reference_image.affine))),
            "candidate_dtype": str(candidate.dtype),
            "reference_dtype": str(reference.dtype),
        }
        if candidate.shape != reference.shape:
            raise RuntimeError(f"shape mismatch for {name}")
        if name == "mask":
            candidate = candidate != 0
            reference = reference != 0
            intersection = int(np.count_nonzero(candidate & reference))
            denominator = int(candidate.sum() + reference.sum())
            record.update({
                "different_voxels": int(np.count_nonzero(candidate != reference)),
                "dice": float(2 * intersection / denominator),
                "candidate_voxels": int(candidate.sum()),
                "reference_voxels": int(reference.sum()),
            })
        else:
            difference = candidate.astype(np.float64) - reference.astype(np.float64)
            record.update({
                "mae": float(np.mean(np.abs(difference))),
                "max_abs": float(np.max(np.abs(difference))),
                "pearson_r": pearson(candidate, reference),
            })
        result[name] = record
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--reuse-outputs",
        action="store_true",
        help="Reuse complete per-case outputs and timing logs, then rebuild the report.",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    cases = manifest["cases"]
    args.output_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(args.source_root / "src")
    env["OMP_NUM_THREADS"] = str(args.threads)
    env["MKL_NUM_THREADS"] = str(args.threads)
    env["OPENBLAS_NUM_THREADS"] = str(args.threads)
    records = []
    for case in cases:
        case_id = case["case_id"]
        output = args.output_root / case_id
        output.mkdir(parents=True, exist_ok=True)
        timing = output / "time.txt"
        command = [
            "/usr/bin/time", "-v", "-o", str(timing),
            args.python, "-m", "fnit.cli", "synthstrip",
            "--image", case["path"],
            "--out", str(output / "image.nii.gz"),
            "--mask", str(output / "mask.nii.gz"),
            "--sdt", str(output / "distance.nii.gz"),
            "--weights", str(args.weights),
            "--device", args.device,
            "--threads", str(args.threads),
        ]
        expected = tuple(output / f"{name}.nii.gz" for name in ("image", "mask", "distance"))
        reused = args.reuse_outputs and timing.is_file() and all(path.is_file() for path in expected)
        if reused:
            elapsed = time_record(timing)["wall_seconds"]
        else:
            started = time.perf_counter()
            completed = subprocess.run(command, env=env, capture_output=True, text=True)
            elapsed = time.perf_counter() - started
            if completed.returncode:
                raise RuntimeError(f"{case_id} failed: {completed.stderr}")
        candidate_timing = time_record(timing)
        if candidate_timing["exit_status"] != 0:
            raise RuntimeError(f"{case_id} has nonzero exit status in {timing}")
        candidate_timing["driver_wall_seconds"] = elapsed
        candidate_timing["reused_for_report"] = reused
        reference_dir = args.reference_root / case_id
        reference_metadata = json.loads((reference_dir / "result.json").read_text())
        records.append({
            "case_id": case_id,
            "input_sha256": case["sha256"],
            "metrics": compare_images(output, reference_dir),
            "candidate_timing": candidate_timing,
            "reference_timing": {
                "wall_seconds": reference_metadata["wall_seconds"],
                "maximum_rss_kbytes": reference_metadata["max_rss_kib"],
            },
        })

    torch.cuda.set_device(args.device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = SynthStrip(weights=args.weights, device=args.device, threads=args.threads)
    model(cases[0]["path"])
    torch.cuda.synchronize()
    peak = int(torch.cuda.max_memory_allocated())

    mask_dice = [record["metrics"]["mask"]["dice"] for record in records]
    mask_difference = [record["metrics"]["mask"]["different_voxels"] for record in records]
    distance_mae = [record["metrics"]["distance"]["mae"] for record in records]
    image_r = [record["metrics"]["image"]["pearson_r"] for record in records]
    image_mae = [record["metrics"]["image"]["mae"] for record in records]
    candidate_times = [record["candidate_timing"]["wall_seconds"] for record in records]
    reference_times = [record["reference_timing"]["wall_seconds"] for record in records]
    candidate_rss = [record["candidate_timing"]["maximum_rss_kbytes"] for record in records]
    reference_rss = [record["reference_timing"]["maximum_rss_kbytes"] for record in records]
    report = {
        "schema_version": 1,
        "date": "2026-09-27",
        "feature": "SynthStrip current nibabel/SciPy pipeline versus FreeSurfer 8.2.0",
        "data": {
            "subjects": len(records),
            "kind": "real clinical T1w",
            "selection": manifest["selection"],
            "identifiers_public": False,
            "input_sha256": {record["case_id"]: record["input_sha256"] for record in records},
        },
        "reference": {
            "software": "FreeSurfer 8.2.0 mri_synthstrip official source CUDA",
            "script_sha256": "bbc2ff8f8779862039401b05d5cd6039fb4f3583e0032a793ac9adb3f4521590",
            "tf32_override": 0,
        },
        "candidate": {
            "device": args.device,
            "tf32_default": True,
            "float16_used": False,
            "threads": args.threads,
            "peak_cuda_memory_bytes_case01": peak,
            "source_sha256": {
                str(path.relative_to(args.source_root)): sha256(path)
                for path in (
                    args.source_root / "src/fnit/synthstrip/model.py",
                    args.source_root / "src/fnit/synthstrip/pipeline.py",
                    args.source_root / "src/fnit/_nib.py",
                    args.source_root / "src/fnit/weights.py",
                )
            },
        },
        "environment": {
            "host": "gpucw1",
            "gpu": "NVIDIA H100 PCIe",
            "shared_node_timing": True,
        },
        "summary": {
            "minimum_mask_dice": float(min(mask_dice)),
            "maximum_mask_different_voxels": int(max(mask_difference)),
            "mean_distance_mae_mm": float(np.mean(distance_mae)),
            "maximum_distance_mae_mm": float(max(distance_mae)),
            "minimum_stripped_image_pearson_r": float(min(image_r)),
            "maximum_stripped_image_mae": float(max(image_mae)),
            "candidate_wall_seconds_median": float(np.median(candidate_times)),
            "reference_wall_seconds_median": float(np.median(reference_times)),
            "candidate_over_reference_median": float(np.median(candidate_times) / np.median(reference_times)),
            "candidate_maximum_rss_kbytes_median": int(np.median(candidate_rss)),
            "reference_maximum_rss_kbytes_median": int(np.median(reference_rss)),
        },
        "limitations": (
            "The 12 cases have no manual brain-mask ground truth. Metrics quantify agreement "
            "with FreeSurfer 8.2.0. The reference disabled TF32 while FNIT used its required "
            "TF32 default, and timing came from a shared node."
        ),
        "timing_boundary": "Each CLI run includes Python startup, weight and image loading, inference, post-processing, and three NIfTI writes.",
        "records": records,
    }
    (args.output_root / "report.public.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
