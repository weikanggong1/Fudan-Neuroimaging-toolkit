#!/usr/bin/env python3
"""Run current FNIT SynthMorph against fixed FreeSurfer real-data outputs."""

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
    status = re.search(r"Exit status:\s*(\d+)", text)
    if wall is None or rss is None or status is None:
        raise RuntimeError(f"incomplete time log: {path}")
    fields = [float(value) for value in wall.group(1).split(":")]
    seconds = sum(value * 60**power for power, value in enumerate(reversed(fields)))
    return {
        "wall_seconds": seconds,
        "maximum_rss_kbytes": int(rss.group(1)),
        "exit_status": int(status.group(1)),
    }


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = left.astype(np.float64, copy=False).ravel()
    right = right.astype(np.float64, copy=False).ravel()
    left -= left.mean()
    right -= right.mean()
    denominator = np.sqrt(np.dot(left, left) * np.dot(right, right))
    return float(np.dot(left, right) / denominator)


def load_warp(path: Path) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    if data.ndim == 5 and data.shape[-2] == 1 and data.shape[-1] == 3:
        data = data[..., 0, :]
    if data.ndim != 4 or data.shape[-1] != 3:
        raise RuntimeError(f"unexpected warp shape {data.shape}: {path}")
    return image, np.asarray(data, dtype=np.float32)


def compare(candidate_dir: Path, reference_dir: Path) -> dict:
    candidate_image = nib.load(str(candidate_dir / "moved.nii.gz"))
    reference_image = nib.load(str(reference_dir / "moved.nii.gz"))
    candidate = np.asanyarray(candidate_image.dataobj).astype(np.float32)
    reference = np.asanyarray(reference_image.dataobj).astype(np.float32)
    if candidate.shape != reference.shape:
        raise RuntimeError("moved image shape mismatch")
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    reference_rms = float(np.sqrt(np.mean(reference.astype(np.float64) ** 2)))
    moved = {
        "shape_equal": True,
        "affine_max_abs": float(
            np.max(np.abs(candidate_image.affine - reference_image.affine))
        ),
        "candidate_dtype": str(candidate.dtype),
        "reference_dtype": str(reference.dtype),
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "nrmse_reference_rms": float(
            np.sqrt(np.mean(difference**2)) / reference_rms
        ),
        "max_abs": float(np.max(np.abs(difference))),
        "pearson_r": pearson(candidate, reference),
    }

    candidate_warp_image, candidate_warp = load_warp(candidate_dir / "transform.nii.gz")
    reference_warp_image, reference_warp = load_warp(reference_dir / "transform.nii.gz")
    if candidate_warp.shape != reference_warp.shape:
        raise RuntimeError("transform shape mismatch")
    vector_error = np.linalg.norm(
        candidate_warp.astype(np.float64) - reference_warp.astype(np.float64),
        axis=-1,
    )
    warp = {
        "semantic_shape_equal_after_squeeze": True,
        "candidate_file_shape": list(candidate_warp_image.shape),
        "reference_file_shape": list(reference_warp_image.shape),
        "affine_max_abs": float(
            np.max(np.abs(candidate_warp_image.affine - reference_warp_image.affine))
        ),
        "component_mae_mm": float(
            np.mean(np.abs(candidate_warp.astype(np.float64) - reference_warp.astype(np.float64)))
        ),
        "vector_error_mean_mm": float(vector_error.mean()),
        "vector_error_p95_mm": float(np.percentile(vector_error, 95)),
        "vector_error_max_mm": float(vector_error.max()),
    }
    return {"moved": moved, "transform": warp}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--reuse-outputs",
        action="store_true",
        help="Reuse complete per-case outputs and timing logs, then rebuild the report.",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    cases = manifest["cases"][: args.limit]
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
        expected = (output / "moved.nii.gz", output / "transform.nii.gz")
        reused = args.reuse_outputs and timing.is_file() and all(path.is_file() for path in expected)
        command = [
            "/usr/bin/time", "-v", "-o", str(timing),
            args.python, "-m", "fnit.cli", "synthmorph",
            case["path"], str(args.fixed),
            "--model", "joint",
            "--device", args.device,
            "--weights", str(args.weights),
            "--extent", "256",
            "--steps", "7",
            "--hyper", "0.5",
            "--threads", str(args.threads),
            "--out-moving", str(output / "moved.nii.gz"),
            "--trans", str(output / "transform.nii.gz"),
        ]
        if reused:
            driver_wall = time_record(timing)["wall_seconds"]
        else:
            started = time.perf_counter()
            completed = subprocess.run(command, env=env, capture_output=True, text=True)
            driver_wall = time.perf_counter() - started
            (output / "stdout_stderr.log").write_text(
                completed.stdout + completed.stderr
            )
            if completed.returncode:
                raise RuntimeError(f"{case_id} failed: {completed.stderr}")
        candidate_timing = time_record(timing)
        if candidate_timing["exit_status"] != 0:
            raise RuntimeError(f"{case_id} has nonzero exit status in {timing}")
        candidate_timing["driver_wall_seconds"] = driver_wall
        candidate_timing["reused_for_report"] = reused
        reference_dir = args.reference_root / case_id
        reference_metadata = json.loads((reference_dir / "result.json").read_text())
        records.append({
            "case_id": case_id,
            "input_sha256": case["sha256"],
            "metrics": compare(output, reference_dir),
            "candidate_timing": candidate_timing,
            "reference_timing": {
                "wall_seconds": reference_metadata["wall_seconds"],
                "maximum_rss_kbytes": reference_metadata["max_rss_kib"],
            },
        })

    peak = None
    if args.device.startswith("cuda"):
        from fnit.synthmorph import SynthMorph

        torch.cuda.set_device(args.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = SynthMorph(
            weights=args.weights,
            device=args.device,
            model="joint",
            extent=256,
            hyper=0.5,
            steps=7,
        )
        model(cases[0]["path"], args.fixed)
        torch.cuda.synchronize()
        peak = int(torch.cuda.max_memory_allocated())

    moved_nrmse = [record["metrics"]["moved"]["nrmse_reference_rms"] for record in records]
    moved_r = [record["metrics"]["moved"]["pearson_r"] for record in records]
    warp_max = [record["metrics"]["transform"]["vector_error_max_mm"] for record in records]
    warp_mean = [record["metrics"]["transform"]["vector_error_mean_mm"] for record in records]
    candidate_times = [record["candidate_timing"]["wall_seconds"] for record in records]
    reference_times = [record["reference_timing"]["wall_seconds"] for record in records]
    source_files = (
        "src/fnit/synthmorph/models.py",
        "src/fnit/synthmorph/pipeline.py",
        "src/fnit/synthmorph/spatial.py",
        "src/fnit/_nib.py",
        "src/fnit/_transforms.py",
        "src/fnit/weights.py",
        "src/fnit/cli.py",
    )
    report = {
        "schema_version": 1,
        "date": "2026-09-27",
        "feature": "SynthMorph nibabel/PyTorch joint registration versus FreeSurfer 8.2.0",
        "data": {
            "subjects": len(records),
            "kind": "real clinical T1w to MNI152 T1 2 mm",
            "selection": manifest["selection"],
            "identifiers_public": False,
            "input_sha256": {
                record["case_id"]: record["input_sha256"] for record in records
            },
            "fixed_sha256": sha256(args.fixed),
        },
        "reference": {
            "software": "FreeSurfer 8.2.0 mri_synthmorph official source",
            "device_class": "GPU" if args.device.startswith("cuda") else "CPU",
            "script_sha256": "4fa7fb879243cbf55addef6cf253e4206fda90a9c82728f72a6050eaa80bf5c6",
            "tf32_override": 0,
        },
        "candidate": {
            "device": args.device,
            "tf32_default": args.device.startswith("cuda"),
            "float16_used": False,
            "threads": args.threads,
            "peak_cuda_memory_bytes_case01": peak,
            "source_sha256": {
                relative: sha256(args.source_root / relative)
                for relative in source_files
            },
        },
        "summary": {
            "maximum_moved_nrmse_reference_rms": float(max(moved_nrmse)),
            "minimum_moved_pearson_r": float(min(moved_r)),
            "mean_warp_vector_error_mm": float(np.mean(warp_mean)),
            "maximum_warp_vector_error_mm": float(max(warp_max)),
            "candidate_wall_seconds_median": float(np.median(candidate_times)),
            "reference_wall_seconds_median": float(np.median(reference_times)),
            "candidate_over_reference_median": float(
                np.median(candidate_times) / np.median(reference_times)
            ),
        },
        "transform_contract": (
            "Both files encode target-grid target-to-source world-RAS displacement "
            "in millimetres. FreeSurfer writes [X,Y,Z,1,3], FNIT writes the NIfTI "
            "vector shape [X,Y,Z,3]; comparisons squeeze only the singleton frame axis."
        ),
        "timing_boundary": (
            "Each CLI run includes Python startup, HDF5 loading and fixed-hypernetwork "
            "specialization, image loading, inference, resampling, and two NIfTI writes."
        ),
        "records": records,
    }
    (args.output_root / "report.public.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
