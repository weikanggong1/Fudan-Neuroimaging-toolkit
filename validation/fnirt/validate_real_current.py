#!/usr/bin/env python3
"""Build the public matched-input real-FA TorchFNIRT validation report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw


SOURCE_FILES = (
    "src/fnit/__init__.py",
    "src/fnit/_nib.py",
    "src/fnit/_transforms.py",
    "src/fnit/flirt/coordinates.py",
    "src/fnit/fnirt/__init__.py",
    "src/fnit/fnirt/io.py",
    "src/fnit/fnirt/optimizer.py",
    "src/fnit/fnirt/registration.py",
    "src/fnit/fnirt/spline.py",
    "src/fnit/fnirt/topology.py",
    "src/fnit/fnirt/_topology_triton.py",
    "src/fnit/dmri_pipeline/tbss.py",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--official-dir", required=True, type=Path)
    parser.add_argument("--prior-official-dir", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--registration-weight", required=True, type=Path)
    parser.add_argument("--affine", required=True, type=Path)
    parser.add_argument("--config-prefix", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-snapshot-sha256", required=True)
    parser.add_argument("--candidate-run-script", required=True, type=Path)
    parser.add_argument("--official-run-script", required=True, type=Path)
    parser.add_argument("--candidate-run-report", required=True, type=Path)
    parser.add_argument("--candidate-time", required=True, type=Path)
    parser.add_argument("--candidate-gpu-baseline", required=True, type=Path)
    parser.add_argument("--candidate-gpu-processes", required=True, type=Path)
    parser.add_argument("--official-total-time", required=True, type=Path)
    parser.add_argument("--official-cpu-baseline", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-figure", required=True, type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_duration(value: str) -> float:
    parts = [float(item) for item in value.strip().split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"unsupported elapsed duration: {value}")


def _gpu_occupancy(baseline_path: Path, processes_path: Path) -> dict:
    baseline_rows = [line for line in baseline_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not baseline_rows:
        raise ValueError("GPU baseline is empty")
    fields = [field.strip() for field in baseline_rows[0].split(",")]
    if len(fields) != 6:
        raise ValueError("unexpected GPU baseline format")
    gpu_uuid = fields[1]
    resident_count = 0
    resident_memory_mib = 0
    for line in processes_path.read_text(encoding="utf-8").splitlines():
        parts = [field.strip() for field in line.split(",")]
        if len(parts) >= 4 and parts[0] == gpu_uuid:
            resident_count += 1
            resident_memory_mib += int(parts[-1].removesuffix(" MiB"))
    return {
        "physical_index": int(fields[0]),
        "model": fields[2],
        "memory_used_mib": int(fields[3].removesuffix(" MiB")),
        "memory_free_mib": int(fields[4].removesuffix(" MiB")),
        "utilization_percent": int(fields[5].removesuffix(" %")),
        "shared_resident_process_count": resident_count,
        "shared_resident_memory_mib": resident_memory_mib,
    }


def _parse_time(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    elapsed = re.search(
        r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\S+)", text
    )
    rss = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    status = re.search(r"Exit status:\s*(\d+)", text)
    if not (elapsed and rss and status):
        raise ValueError(f"could not parse GNU time output: {path}")
    return {
        "wall_seconds": _parse_duration(elapsed.group(1)),
        "maximum_rss_kib": int(rss.group(1)),
        "exit_status": int(status.group(1)),
    }


def _image_contract(left: nib.spatialimages.SpatialImage, right: nib.spatialimages.SpatialImage) -> dict:
    return {
        "same_shape": left.shape == right.shape,
        "same_affine": bool(np.allclose(left.affine, right.affine, atol=1e-5, rtol=0)),
        "same_dtype": left.get_data_dtype() == right.get_data_dtype(),
        "candidate_shape": list(left.shape),
        "official_shape": list(right.shape),
        "candidate_dtype": str(left.get_data_dtype()),
        "official_dtype": str(right.get_data_dtype()),
        "candidate_intent_code": int(left.header["intent_code"]),
        "official_intent_code": int(right.header["intent_code"]),
    }


def _metrics(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> dict:
    mask = mask & np.isfinite(left) & np.isfinite(right)
    a = left[mask].astype(np.float64, copy=False)
    b = right[mask].astype(np.float64, copy=False)
    if a.size < 2:
        raise ValueError("comparison mask contains fewer than two voxels")
    difference = a - b
    return {
        "voxels": int(a.size),
        "pearson": float(np.corrcoef(a, b)[0, 1]),
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "maximum_absolute_error": float(np.max(np.abs(difference))),
    }


def _comparison(candidate: Path, official: Path, mask_kind: str, reference_mask: np.ndarray) -> dict:
    candidate_image = nib.load(str(candidate))
    official_image = nib.load(str(official))
    candidate_data = np.asarray(candidate_image.dataobj, dtype=np.float64)
    official_data = np.asarray(official_image.dataobj, dtype=np.float64)
    if candidate_data.shape != official_data.shape:
        raise ValueError(f"shape mismatch: {candidate} vs {official}")
    if mask_kind == "all":
        mask = np.ones(candidate_data.shape, dtype=bool)
    elif mask_kind == "union":
        mask = (np.abs(candidate_data) > 1e-6) | (np.abs(official_data) > 1e-6)
    elif mask_kind == "reference":
        mask = reference_mask
    else:
        raise ValueError(mask_kind)
    return {
        "contract": _image_contract(candidate_image, official_image),
        "metrics": _metrics(candidate_data, official_data, mask),
        "candidate_sha256": _sha256(candidate),
        "official_sha256": _sha256(official),
    }


def _scaled_panel(values: np.ndarray, lower: float, upper: float, *, difference: bool) -> Image.Image:
    scaled = np.clip((values - lower) / max(upper - lower, np.finfo(np.float32).eps), 0, 1)
    scaled = np.flipud(scaled)
    if difference:
        red = np.asarray(255 * scaled, dtype=np.uint8)
        green = np.asarray(210 * np.sqrt(scaled), dtype=np.uint8)
        blue = np.asarray(45 * scaled * scaled, dtype=np.uint8)
        pixels = np.stack((red, green, blue), axis=-1)
        panel = Image.fromarray(pixels, mode="RGB")
    else:
        pixels = np.asarray(255 * scaled, dtype=np.uint8)
        panel = Image.fromarray(pixels, mode="L").convert("RGB")
    return panel.resize((300, 250), Image.Resampling.BILINEAR)


def _make_figure(candidate_dir: Path, official_dir: Path, output: Path, iout_r: float, jac_r: float) -> None:
    candidate_iout = np.asarray(
        nib.load(str(candidate_dir / "dti_FA_to_MNI.nii.gz")).dataobj,
        dtype=np.float32,
    )
    official_iout = np.asarray(
        nib.load(str(official_dir / "dti_FA_to_MNI.nii.gz")).dataobj,
        dtype=np.float32,
    )
    candidate_jac = np.asarray(
        nib.load(str(candidate_dir / "jacobian_nonlinear.nii.gz")).dataobj,
        dtype=np.float32,
    )
    official_jac = np.asarray(
        nib.load(str(official_dir / "jacobian_nonlinear.nii.gz")).dataobj,
        dtype=np.float32,
    )
    slices = (
        (official_iout[:, :, 90].T, candidate_iout[:, :, 90].T, "warped FA axial", 0.0, 1.0),
        (official_iout[:, 109, :].T, candidate_iout[:, 109, :].T, "warped FA coronal", 0.0, 1.0),
        (official_jac[:, :, 90].T, candidate_jac[:, :, 90].T, "nonlinear Jacobian axial", 0.2, 2.2),
        (official_jac[:, 109, :].T, candidate_jac[:, 109, :].T, "nonlinear Jacobian coronal", 0.2, 2.2),
    )
    canvas = Image.new("RGB", (960, 1160), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 12), f"Matched real FA | warped r={iout_r:.6f} | nonlinear Jacobian r={jac_r:.6f}", fill="black")
    for column, title in enumerate(("FSL 6.0.7.4", "TorchFNIRT", "absolute difference (99.5%)")):
        draw.text((20 + column * 315, 38), title, fill="black")
    for row, (official, candidate, label, lower, upper) in enumerate(slices):
        difference = np.abs(candidate - official)
        finite = difference[np.isfinite(difference)]
        difference_upper = float(np.percentile(finite, 99.5)) if finite.size else 1.0
        difference_upper = max(difference_upper, np.finfo(np.float32).eps)
        top = 75 + row * 270
        draw.text((20, top - 15), label, fill="black")
        panels = (
            _scaled_panel(official, lower, upper, difference=False),
            _scaled_panel(candidate, lower, upper, difference=False),
            _scaled_panel(difference, 0.0, difference_upper, difference=True),
        )
        for column, panel in enumerate(panels):
            canvas.paste(panel, (20 + column * 315, top))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def main() -> None:
    args = _arguments()
    run_report = json.loads(args.candidate_run_report.read_text(encoding="utf-8"))
    reference = nib.load(str(args.reference))
    reference_mask = np.asarray(reference.dataobj) != 0
    files = {
        "coefficient": (
            args.candidate_dir / "dti_FA_to_MNI_warp.nii.gz",
            args.official_dir / "dti_FA_to_MNI_warp.nii.gz",
            "all",
        ),
        "iout": (
            args.candidate_dir / "dti_FA_to_MNI.nii.gz",
            args.official_dir / "dti_FA_to_MNI.nii.gz",
            "union",
        ),
        "jacobian_nonlinear": (
            args.candidate_dir / "jacobian_nonlinear.nii.gz",
            args.official_dir / "jacobian_nonlinear.nii.gz",
            "reference",
        ),
        "jacobian_with_affine": (
            args.candidate_dir / "jacobian_full_pull.nii.gz",
            args.official_dir / "jacobian_withaff.nii.gz",
            "reference",
        ),
    }
    accuracy = {
        name: _comparison(candidate, official, mask, reference_mask)
        for name, (candidate, official, mask) in files.items()
    }
    _make_figure(
        args.candidate_dir,
        args.official_dir,
        args.output_figure,
        accuracy["iout"]["metrics"]["pearson"],
        accuracy["jacobian_nonlinear"]["metrics"]["pearson"],
    )

    official_stage_times = {
        f"stage{stage}": _parse_time(args.official_dir / f"stage{stage}.time.txt")
        for stage in (1, 2, 3)
    }
    official_fnirt_wall = sum(item["wall_seconds"] for item in official_stage_times.values())
    prior_reproducibility = {
        "coefficient": _comparison(
            args.official_dir / "dti_FA_to_MNI_warp.nii.gz",
            args.prior_official_dir / "dti_FA_to_MNI_warp.nii.gz",
            "all",
            reference_mask,
        ),
        "iout": _comparison(
            args.official_dir / "dti_FA_to_MNI.nii.gz",
            args.prior_official_dir / "dti_FA_to_MNI.nii.gz",
            "union",
            reference_mask,
        ),
    }
    contract_passed = all(
        item["contract"]["same_shape"]
        and item["contract"]["same_affine"]
        and item["contract"]["same_dtype"]
        for item in accuracy.values()
    ) and (
        accuracy["coefficient"]["contract"]["candidate_intent_code"]
        == accuracy["coefficient"]["contract"]["official_intent_code"]
        == 2007
    )
    report = {
        "schema_version": 1,
        "data": {
            "subjects": 1,
            "kind": "one deidentified real UKB-format diffusion FA image",
            "subject_identifier_published": False,
            "input_sha256": _sha256(args.input),
            "reference_sha256": _sha256(args.reference),
            "affine_sha256": _sha256(args.affine),
            "registration_weight_sha256": _sha256(args.registration_weight),
            "registration_weight_role": "FLIRT input weight only; neither FNIRT implementation receives it",
            "config_sha256": {
                f"oxford_s{stage}.cnf": _sha256(
                    args.config_prefix.with_name(args.config_prefix.name + f"_s{stage}.cnf")
                )
                for stage in (1, 2, 3)
            },
        },
        "boundary": {
            "matched": True,
            "shared": "identical preprocessed FA, FMRIB58_FA_1mm reference, FSL scaled-mm affine, implicit masks, and Oxford s1/s2/s3 schedule",
            "candidate": "all six schedule levels and three process stages run in one TorchFNIRT process",
            "official": "FSL 6.0.7.4 runs three FNIRT processes with coefficient and intensity handoff",
        },
        "candidate": {
            "implementation": "current TorchFNIRT with TBSSConfig().fnirt",
            "source_snapshot_tar_sha256": args.source_snapshot_sha256,
            "source_sha256": {
                relative: _sha256(args.source_root / relative)
                for relative in SOURCE_FILES
            },
            "run_script_sha256": _sha256(args.candidate_run_script),
            "validator_sha256": _sha256(Path(__file__)),
            "command": "PYTHONPATH=<SOURCE_SNAPSHOT>/src CUDA_VISIBLE_DEVICES=0 CUDA_MODULE_LOADING=LAZY PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python validation/fnirt/run_current_matched.py --input <FA> --reference <FMRIB58_FA_1mm> --affine <FSL_AFFINE> --output-dir <OUTPUT> --device cuda",
            "device": "NVIDIA H100 PCIe GPU 0",
            "dtype": "float32 images and outputs / float64 FNIRT solver",
            "tf32": True,
            "qc": {
                "topology_projection_required": run_report["qc"]["topology_projection_required"],
                "process_stages": run_report["qc"]["process_stages"],
                "control_grid_shape": run_report["qc"]["control_grid_shape"],
            },
        },
        "reference": {
            "implementation": "FSL FNIRT 6.0.7.4",
            "run_script_sha256": _sha256(args.official_run_script),
            "command": "FSLDIR=<FSL_6.0.7.4> validation/fnirt/run_official_matched.sh <OUTPUT> <FA> <FMRIB58_FA_1mm> <FSL_AFFINE> <OXFORD_CONFIG_PREFIX>",
            "fresh_rerun_matches_prior_official": prior_reproducibility,
        },
        "execution": {
            "run_date": "2026-09-28",
            "host": "gpucw1",
            "candidate_software": {
                "pytorch": run_report["execution"]["torch"],
                "pytorch_cuda": run_report["execution"]["torch_cuda"],
                "nibabel": nib.__version__,
                "numpy": np.__version__,
            },
            "candidate_gpu_baseline": _gpu_occupancy(
                args.candidate_gpu_baseline, args.candidate_gpu_processes
            ),
            "candidate_timing_isolated": False,
            "candidate_timing_note": "GPU 0 was idle at launch but another process retained 47.5 GiB; 32.7 GiB remained free",
            "official_cpu_baseline": args.official_cpu_baseline.read_text(encoding="utf-8").strip(),
        },
        "accuracy": accuracy,
        "timing": {
            "candidate_external": _parse_time(args.candidate_time),
            "candidate_synchronized_core_seconds": run_report["execution"]["synchronized_wall_seconds"],
            "official_stage_external": official_stage_times,
            "official_three_stage_fnirt_wall_seconds": official_fnirt_wall,
            "official_total_including_full_jacobian_utility": _parse_time(args.official_total_time),
            "observed_nonisolated_wall_ratio_official_stages_over_candidate": official_fnirt_wall / _parse_time(args.candidate_time)["wall_seconds"],
            "speedup_reported": False,
            "comparison_note": "candidate external time includes Python startup and four output writes; official stage sum includes three process startups and writes coefficient, iout, and nonlinear Jacobian. The GPU shared resident memory and the CPU baseline load average was high, so the observed ratio is not an isolated speedup estimate; full-affine Jacobian utility is excluded from it",
        },
        "memory": {
            "candidate_peak_cuda_bytes": run_report["execution"]["peak_cuda_memory_bytes"],
            "candidate_external_maximum_rss_kib": _parse_time(args.candidate_time)["maximum_rss_kib"],
            "official_maximum_stage_rss_kib": max(
                item["maximum_rss_kib"] for item in official_stage_times.values()
            ),
        },
        "status": {
            "run_completed": True,
            "same_input_config_and_affine": True,
            "output_contract_passed": contract_passed,
            "numerical_equivalence_passed": False,
            "reason": "all output contracts match, but coefficient, warped image, and Jacobian differences exceed floating-point-only error",
        },
        "artifacts": {
            "figure": "docs/fnirt/figures/fnirt_real_current.png",
            "figure_sha256": _sha256(args.output_figure),
        },
        "limits": [
            "one real subject",
            "GPU timing was not process-isolated because an idle process retained memory on the same GPU",
            "the FSL CPU and Torch GPU implementations use different process boundaries",
            "the official run started at a high CPU load average and is not an isolated CPU timing",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
