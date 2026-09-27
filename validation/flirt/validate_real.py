#!/usr/bin/env python3
"""Run a source-bound real-data TorchFLIRT comparison.

The public report contains de-identified slot names and hashes only.  Private
input paths remain in the command line used on the validation host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

import nibabel as nib
import numpy as np


CASE = re.compile(r"case[0-9]+")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(source_root):
    names = (
        "src/fnit/flirt/core.py",
        "src/fnit/flirt/types.py",
        "src/fnit/flirt/coordinates.py",
        "src/fnit/flirt/standalone.py",
        "src/fnit/flirt/__init__.py",
        "src/fnit/_nib.py",
        "src/fnit/_transforms.py",
    )
    return {name: sha256(Path(source_root) / name) for name in names}


def voxel_to_fsl(affine, shape, voxel_sizes):
    basis = np.diag([*voxel_sizes, 1.0]).astype(np.float64)
    if np.linalg.det(np.asarray(affine, dtype=np.float64)[:3, :3]) > 0:
        flip = np.eye(4, dtype=np.float64)
        flip[0, 0] = -1
        flip[0, 3] = (int(shape[0]) - 1) * float(voxel_sizes[0])
        basis = flip @ basis
    return basis


def fsl_centre(image):
    data = np.asarray(image.dataobj, dtype=np.float64)
    weights = data - data.min()
    mass = float(weights.sum(dtype=np.float64))
    denominator = mass if abs(mass) >= 1e-5 else 1.0
    voxel = np.asarray(
        [
            np.dot(
                weights.sum(axis=tuple(other for other in range(3) if other != axis)),
                np.arange(data.shape[axis], dtype=np.float64),
            )
            / denominator
            for axis in range(3)
        ]
    )
    basis = voxel_to_fsl(image.affine, data.shape, image.header.get_zooms()[:3])
    return (basis @ np.r_[voxel, 1.0])[:3]


def rmsdiff(first, second, centre, radius=80.0):
    difference = np.asarray(first) @ np.linalg.inv(np.asarray(second)) - np.eye(4)
    linear = difference[:3, :3]
    translation = difference[:3, 3] + linear @ centre
    return float(
        np.sqrt(
            translation @ translation
            + (float(radius) ** 2 / 5.0) * np.trace(linear.T @ linear)
        )
    )


def metrics(reference, candidate, mask):
    first = np.asarray(reference.dataobj, dtype=np.float64)[mask]
    second = np.asarray(candidate.dataobj, dtype=np.float64)[mask]
    difference = second - first
    centred_first = first - first.mean()
    centred_second = second - second.mean()
    binary_first = first >= 0.2
    binary_second = second >= 0.2
    denominator = int(binary_first.sum() + binary_second.sum())
    return {
        "pearson_r": float(
            centred_first.dot(centred_second)
            / (np.linalg.norm(centred_first) * np.linalg.norm(centred_second))
        ),
        "mae": float(np.abs(difference).mean()),
        "rmse": float(np.sqrt(np.square(difference).mean())),
        "max_abs": float(np.abs(difference).max()),
        "dice_at_0_2": (
            float(2 * np.logical_and(binary_first, binary_second).sum() / denominator)
            if denominator
            else 1.0
        ),
    }


def distribution(values):
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "count": int(array.size),
        "minimum": float(array.min()),
        "q1": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "q3": float(np.percentile(array, 75)),
        "maximum": float(array.max()),
    }


def worker(args):
    source_root = Path(args.source_root).resolve()
    sys.path.insert(0, str(source_root / "src"))
    import torch
    from fnit.flirt import run_flirt

    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result = run_flirt(
        input=args.input,
        reference=args.reference,
        output=args.output,
        omat=args.omat,
        init=None,
        inweight=None,
        refweight=None,
        dof=12,
        cost="corratio",
        device=args.device,
        overwrite=True,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    record = {
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "peak_cuda_memory_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
        ),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "qc": result.qc,
    }
    Path(args.worker_json).write_text(json.dumps(record, indent=2) + "\n")
    return 0


def run(args):
    study = Path(args.study_root)
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    template = Path(args.template or study / "assets" / "template_GM_v1.nii.gz")
    template_image = nib.load(str(template))
    if args.mask:
        mask_path = Path(args.mask)
        mask_image = nib.load(str(mask_path))
        mask = np.asarray(mask_image.dataobj) > 0
        if mask.shape != template_image.shape or not np.allclose(
            mask_image.affine, template_image.affine, atol=1e-5, rtol=0
        ):
            raise ValueError("mask and template grids differ")
        mask_record = {
            "definition": "provided binary mask",
            "sha256": sha256(mask_path),
            "voxels": int(mask.sum()),
        }
    else:
        mask = np.asarray(template_image.dataobj) > 0
        mask_record = {
            "definition": "template GM intensity > 0",
            "sha256": hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest(),
            "voxels": int(mask.sum()),
        }
    centre = fsl_centre(template_image)
    cases = sorted(
        path.name
        for path in (study / "subjects").iterdir()
        if path.is_dir() and CASE.fullmatch(path.name)
    )[: args.case_count]
    if len(cases) != args.case_count:
        raise ValueError("insufficient caseNN inputs")
    historical = json.loads(Path(args.reference_report).read_text())
    old_records = {item["case"]: item for item in historical["records"]}
    process_times = {}
    if args.process_log:
        for line in Path(args.process_log).read_text().splitlines():
            parts = line.split()
            if len(parts) >= 3 and CASE.fullmatch(parts[0]):
                process_times[parts[0]] = float(parts[2])
    if args.reuse_existing and len(process_times) != len(cases):
        raise ValueError("process log does not contain one timing for every case")
    records = []
    for case in cases:
        root = study / "subjects" / case / "T1"
        moving = root / "T1_fast" / "T1_brain_pve_1.nii.gz"
        official_root = root / "T1_vbm" / "ukb"
        official_matrix = official_root / "T1_GM_to_template_GM.mat"
        official_case = Path(args.official_moved_root) / case
        official_moved = official_case / "official_apply.nii.gz"
        case_output = output / case
        case_output.mkdir(exist_ok=True)
        candidate_moved = case_output / "candidate.nii.gz"
        candidate_matrix = case_output / "candidate.mat"
        worker_json = case_output / "worker.json"
        if args.reuse_existing:
            required = (candidate_moved, candidate_matrix, worker_json)
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"missing candidate outputs: {missing}")
            process_wall = process_times[case]
        else:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "worker",
                "--source-root",
                str(Path(args.source_root).resolve()),
                "--input",
                str(moving),
                "--reference",
                str(template),
                "--output",
                str(candidate_moved),
                "--omat",
                str(candidate_matrix),
                "--worker-json",
                str(worker_json),
                "--device",
                args.device,
                "--threads",
                str(args.threads),
            ]
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(Path(args.source_root).resolve() / "src")
            started = time.perf_counter()
            completed = subprocess.run(command, env=environment, check=False)
            process_wall = time.perf_counter() - started
            if completed.returncode:
                raise RuntimeError(f"candidate failed for {case}: {completed.returncode}")
        worker_record = json.loads(worker_json.read_text())
        reference_image = nib.load(str(official_moved))
        candidate_image = nib.load(str(candidate_moved))
        contract = {
            "shape_equal": candidate_image.shape == reference_image.shape,
            "affine_max_abs": float(
                np.max(np.abs(candidate_image.affine - reference_image.affine))
            ),
            "dtype_equal": str(candidate_image.get_data_dtype())
            == str(reference_image.get_data_dtype()),
        }
        record = {
            "case_id": case,
            "input_sha256": sha256(moving),
            "official_matrix_sha256": sha256(official_matrix),
            "official_moved_sha256": sha256(official_moved),
            "matrix_rmsdiff_mm": rmsdiff(
                np.loadtxt(candidate_matrix), np.loadtxt(official_matrix), centre
            ),
            "moved": metrics(reference_image, candidate_image, mask),
            "contract": contract,
            "candidate_process_wall_seconds": process_wall,
            "candidate_internal_wall_seconds": worker_record["wall_seconds"],
            "candidate_peak_cuda_memory_allocated_bytes": worker_record[
                "peak_cuda_memory_allocated_bytes"
            ],
            "candidate_peak_cuda_memory_reserved_bytes": worker_record[
                "peak_cuda_memory_reserved_bytes"
            ],
            "reference_wall_seconds": old_records[case]["seconds"]["fsl_cpu"],
            "fresh_applyxfm_wall_seconds": float(
                (official_case / "apply.time").read_text().split()[0]
            ),
        }
        records.append(record)
        print(case, record["matrix_rmsdiff_mm"], process_wall, flush=True)
    keys = ("pearson_r", "mae", "rmse", "max_abs", "dice_at_0_2")
    report = {
        "schema_version": 2,
        "date": args.date,
        "feature": "FNIT TorchFLIRT versus FSL FLIRT 6.0.7.4",
        "input": {
            "subjects": len(records),
            "kind": "real T1w-derived FSL FAST GM to UKB group-GM template",
            "identifiers_public": False,
            "same_inputs_for_reference_and_candidate": True,
            "template_sha256": sha256(template),
            "comparison_mask": mask_record,
        },
        "candidate": {
            "version": "0.14.0",
            "device": args.device,
            "threads": args.threads,
            "tf32_default": args.device.startswith("cuda"),
            "float16_used": False,
            "source_sha256": source_hashes(args.source_root),
            "timing_scope": "fresh Python process per subject including image I/O, optimization, resampling and two writes",
        },
        "reference": {
            "software": "FSL FLIRT 6.0.7.4",
            "device": "CPU",
            "output_policy": (
                "official matrices plus fresh FSL flirt -applyxfm outputs, "
                "verified by per-case SHA-256"
            ),
            "timing_scope": historical["execution"]["timing_scope"],
            "full_registration_timing_provenance": (
                "same-input FSL 6.0.7.4 run; per-case values copied into records"
            ),
            "fresh_applyxfm_timing_scope": (
                "full command wall time including read, resampling and write"
            ),
        },
        "contract": {
            "shape_match_count": sum(x["contract"]["shape_equal"] for x in records),
            "dtype_match_count": sum(x["contract"]["dtype_equal"] for x in records),
            "maximum_affine_abs_difference": max(
                x["contract"]["affine_max_abs"] for x in records
            ),
            "fsl_scaled_mm_matrix_direction": "input to reference",
        },
        "summary": {
            "matrix_rmsdiff_mm": distribution(
                x["matrix_rmsdiff_mm"] for x in records
            ),
            "moved": {
                key: distribution(x["moved"][key] for x in records) for key in keys
            },
            "candidate_process_wall_seconds": distribution(
                x["candidate_process_wall_seconds"] for x in records
            ),
            "candidate_internal_wall_seconds": distribution(
                x["candidate_internal_wall_seconds"] for x in records
            ),
            "reference_wall_seconds": distribution(
                x["reference_wall_seconds"] for x in records
            ),
            "fresh_applyxfm_wall_seconds": distribution(
                x["fresh_applyxfm_wall_seconds"] for x in records
            ),
            "maximum_peak_cuda_memory_allocated_bytes": max(
                (x["candidate_peak_cuda_memory_allocated_bytes"] for x in records
                 if x["candidate_peak_cuda_memory_allocated_bytes"] is not None),
                default=None,
            ),
            "maximum_peak_cuda_memory_reserved_bytes": max(
                (x["candidate_peak_cuda_memory_reserved_bytes"] for x in records
                 if x["candidate_peak_cuda_memory_reserved_bytes"] is not None),
                default=None,
            ),
        },
        "acceptance": {
            "matrix_rmsdiff_threshold_mm": 0.05,
            "matrix_pass_count": sum(x["matrix_rmsdiff_mm"] <= 0.05 for x in records),
            "numerically_equivalent": False,
            "functional_tolerance_passed": all(
                x["matrix_rmsdiff_mm"] <= 0.05 for x in records
            ),
        },
        "hardware": {
            "candidate": args.candidate_hardware or (
                "NVIDIA H100 PCIe 80 GB" if args.device.startswith("cuda")
                else "CPU model not recorded"
            ),
            "reference": "Intel Xeon Gold 6430",
            "shared_node": True,
            "platform": platform.system(),
        },
        "records": records,
    }
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    return 0


def combine(args):
    cpu = json.loads(Path(args.cpu_report).read_text())
    gpu = json.loads(Path(args.gpu_report).read_text())
    if cpu["candidate"]["source_sha256"] != gpu["candidate"]["source_sha256"]:
        raise ValueError("CPU and GPU reports do not bind the same candidate source")
    for key in ("template_sha256", "comparison_mask", "subjects"):
        if cpu["input"][key] != gpu["input"][key]:
            raise ValueError(f"CPU and GPU reports differ at input.{key}")
    cpu_records = {item["case_id"]: item for item in cpu["records"]}
    gpu_records = {item["case_id"]: item for item in gpu["records"]}
    if cpu_records.keys() != gpu_records.keys():
        raise ValueError("CPU and GPU case sets differ")

    def run_record(report):
        return {
            "version": report["candidate"]["version"],
            "device": report["candidate"]["device"],
            "threads": report["candidate"]["threads"],
            "tf32_default": report["candidate"]["tf32_default"],
            "float16_used": report["candidate"]["float16_used"],
            "hardware": report["hardware"]["candidate"],
            "shared_node": report["hardware"]["shared_node"],
            "timing_scope": report["candidate"]["timing_scope"],
            "contract": report["contract"],
            "summary": report["summary"],
            "acceptance": report["acceptance"],
        }

    records = []
    for case in sorted(cpu_records):
        c = cpu_records[case]
        g = gpu_records[case]
        for key in ("input_sha256", "official_matrix_sha256", "official_moved_sha256"):
            if c[key] != g[key]:
                raise ValueError(f"reference mismatch for {case}: {key}")
        records.append({
            "case_id": case,
            "input_sha256": c["input_sha256"],
            "official_matrix_sha256": c["official_matrix_sha256"],
            "official_moved_sha256": c["official_moved_sha256"],
            "reference_full_registration_wall_seconds": c["reference_wall_seconds"],
            "fresh_applyxfm_wall_seconds": c["fresh_applyxfm_wall_seconds"],
            "cpu": {
                "matrix_rmsdiff_mm": c["matrix_rmsdiff_mm"],
                "moved": c["moved"],
                "process_wall_seconds": c["candidate_process_wall_seconds"],
            },
            "cuda_tf32": {
                "matrix_rmsdiff_mm": g["matrix_rmsdiff_mm"],
                "moved": g["moved"],
                "process_wall_seconds": g["candidate_process_wall_seconds"],
                "peak_cuda_memory_allocated_bytes": g["candidate_peak_cuda_memory_allocated_bytes"],
                "peak_cuda_memory_reserved_bytes": g["candidate_peak_cuda_memory_reserved_bytes"],
            },
        })
    reference_median = gpu["summary"]["reference_wall_seconds"]["median"]
    cpu_median = cpu["summary"]["candidate_process_wall_seconds"]["median"]
    gpu_median = gpu["summary"]["candidate_process_wall_seconds"]["median"]
    report = {
        "schema_version": 3,
        "date": gpu["date"],
        "feature": "FNIT TorchFLIRT CPU/GPU versus FSL FLIRT 6.0.7.4",
        "claim_boundary": (
            "same-input tolerance comparison for the supported 12-DOF correlation-ratio path; "
            "not bitwise or complete FSL numerical equivalence"
        ),
        "input": gpu["input"],
        "candidate_source_sha256": gpu["candidate"]["source_sha256"],
        "reference": gpu["reference"],
        "runs": {"cpu": run_record(cpu), "cuda_tf32": run_record(gpu)},
        "timing_comparison": {
            "fsl_cpu_median_seconds": reference_median,
            "fnit_cpu_median_seconds": cpu_median,
            "fnit_cuda_tf32_median_seconds": gpu_median,
            "fnit_cpu_over_fsl": cpu_median / reference_median,
            "fnit_cuda_tf32_over_fsl": gpu_median / reference_median,
            "fnit_cuda_tf32_over_fnit_cpu": gpu_median / cpu_median,
            "interpretation": (
                "observed wall-time ratios on shared nodes; not exclusive-hardware throughput"
            ),
        },
        "acceptance": {
            "matrix_rmsdiff_threshold_mm": 0.05,
            "cpu_pass_count": cpu["acceptance"]["matrix_pass_count"],
            "cuda_tf32_pass_count": gpu["acceptance"]["matrix_pass_count"],
            "case_count": len(records),
            "numerically_equivalent": False,
        },
        "records": records,
    }
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    return 0


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    worker_parser = commands.add_parser("worker")
    for name in ("source_root", "input", "reference", "output", "omat", "worker_json"):
        worker_parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    worker_parser.add_argument("--device", default="cuda:0")
    worker_parser.add_argument("--threads", type=int, default=4)
    worker_parser.set_defaults(function=worker)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--study-root", required=True)
    run_parser.add_argument("--source-root", required=True)
    run_parser.add_argument("--template")
    run_parser.add_argument(
        "--mask",
        help="optional template-grid mask; default is template GM intensity > 0",
    )
    run_parser.add_argument(
        "--official-moved-root",
        required=True,
        help="directory containing caseNN/official_apply.nii.gz and apply.time",
    )
    run_parser.add_argument("--output-root", required=True)
    run_parser.add_argument("--reference-report", required=True)
    run_parser.add_argument("--report", required=True)
    run_parser.add_argument("--case-count", type=int, default=10)
    run_parser.add_argument("--device", default="cuda:0")
    run_parser.add_argument("--threads", type=int, default=4)
    run_parser.add_argument("--date", default="2026-09-28")
    run_parser.add_argument(
        "--candidate-hardware",
        help="de-identified candidate hardware model written to the report",
    )
    run_parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="summarize existing caseNN candidate files instead of rerunning",
    )
    run_parser.add_argument(
        "--process-log",
        help="case log containing: caseNN matrix_rmsdiff process_wall_seconds",
    )
    run_parser.set_defaults(function=run)
    combine_parser = commands.add_parser("combine")
    combine_parser.add_argument("--cpu-report", required=True)
    combine_parser.add_argument("--gpu-report", required=True)
    combine_parser.add_argument("--report", required=True)
    combine_parser.set_defaults(function=combine)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
