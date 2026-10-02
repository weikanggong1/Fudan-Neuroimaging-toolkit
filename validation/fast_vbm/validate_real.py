#!/usr/bin/env python3
"""Run one current-source FastVBM branch on a real T1w and compare FSL VBM maps."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import resource
import statistics
import time

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw
import torch

from fnit.fast_vbm import FastVBM
from fnit.fast_vbm.pipeline import OUTPUT_FILENAMES


SOURCE_FILES = (
    "src/fnit/fast_vbm/pipeline.py",
    "src/fnit/fast_vbm/registration.py",
    "src/fnit/fast_vbm/synthmorph_backend.py",
    "src/fnit/fast/pipeline.py",
    "src/fnit/synthstrip/pipeline.py",
    "src/fnit/synthmorph/pipeline.py",
    "src/fnit/flirt/core.py",
    "src/fnit/fnirt/registration.py",
    "src/fnit/applywarp/core.py",
    "src/fnit/_nib.py",
    "src/fnit/_transforms.py",
)
MAPS = {
    "warped_gm": "T1_GM_to_template_GM.nii.gz",
    "jacobian": "T1_GM_JAC_nl.nii.gz",
    "modulated_gm": "T1_GM_to_template_GM_mod.nii.gz",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(candidate: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> dict:
    valid = mask & np.isfinite(candidate) & np.isfinite(reference)
    first = candidate[valid].astype(np.float64, copy=False)
    second = reference[valid].astype(np.float64, copy=False)
    difference = first - second
    if first.size < 2:
        raise ValueError("accuracy mask contains fewer than two valid voxels")
    return {
        "voxels": int(first.size),
        "pearson": float(np.corrcoef(first, second)[0, 1]),
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "dice_at_0.2": float(
            2 * np.count_nonzero((first > 0.2) & (second > 0.2))
            / max(np.count_nonzero(first > 0.2) + np.count_nonzero(second > 0.2), 1)
        ),
    }


def _contract(candidate: nib.spatialimages.SpatialImage, reference: nib.spatialimages.SpatialImage) -> dict:
    return {
        "same_shape": tuple(candidate.shape) == tuple(reference.shape),
        "same_affine": bool(np.allclose(candidate.affine, reference.affine, atol=1e-5, rtol=0)),
        "candidate_dtype": str(candidate.get_data_dtype()),
        "reference_dtype": str(reference.get_data_dtype()),
    }


def check_outputs(output_dir: Path, image_path: Path, template_path: Path) -> dict:
    """Check all saved images, PVE normalization and the modulation formula."""
    native = nib.load(image_path)
    template = nib.load(template_path)
    arrays = {}
    contracts = {}
    for name, filename in OUTPUT_FILENAMES.items():
        image = nib.load(output_dir / filename)
        reference = template if name in MAPS else native
        contract = _contract(image, reference)
        data = np.asanyarray(image.dataobj)
        if not contract["same_shape"] or not contract["same_affine"] or not np.isfinite(data).all():
            raise ValueError(f"invalid saved output: {name}")
        arrays[name] = data
        contracts[name] = {**contract, "all_finite": True}
    brain_mask = arrays["brain_mask"] > 0
    mask = brain_mask & (arrays["brain"] > 0)
    pves = np.stack([arrays[f"pve_{tissue}"] for tissue in ("csf", "gm", "wm")])[:, mask]
    if not mask.any() or np.min(pves) < 0 or np.max(pves) > 1:
        raise ValueError("empty brain mask or PVE outside [0, 1]")
    pve_error = float(np.max(np.abs(pves.sum(axis=0) - 1)))
    modulation_error = float(np.max(np.abs(arrays["modulated_gm"] -
                                           arrays["warped_gm"] * arrays["jacobian"])))
    if pve_error > 1e-5 or modulation_error > 1e-6:
        raise ValueError("PVE sum or modulation identity failed")
    return {"images": contracts, "pve_sum_max_absolute_error": pve_error,
            "pve_support_definition": "brain_mask > 0 and brain intensity > 0, as used by TorchFAST",
            "pve_support_voxels": int(mask.sum()),
            "brain_mask_nonpositive_intensity_voxels": int(np.count_nonzero(brain_mask & ~mask)),
            "modulation_max_absolute_error": modulation_error,
            "nonpositive_jacobian_voxels": int(np.count_nonzero(arrays["jacobian"] <= 0)),
            "jacobian_range": [float(arrays["jacobian"].min()), float(arrays["jacobian"].max())]}


def _render_row(reference: np.ndarray, candidate: np.ndarray, title: str) -> Image.Image:
    difference = np.abs(candidate - reference)
    z = reference.shape[2] // 2
    arrays = [np.rot90(values[:, :, z]) for values in (reference, candidate, difference)]
    vmax = max(float(np.percentile(reference[np.isfinite(reference)], 99.5)), 1e-8)
    dmax = max(float(np.percentile(difference[np.isfinite(difference)], 99.9)), 1e-8)
    panels = []
    for index, values in enumerate(arrays):
        scale = vmax if index < 2 else dmax
        scaled = np.clip(values / scale, 0, 1)
        if index < 2:
            rgb = np.repeat((scaled * 255).astype(np.uint8)[..., None], 3, axis=2)
        else:
            red = (scaled * 255).astype(np.uint8)
            green = (np.sqrt(scaled) * 150).astype(np.uint8)
            rgb = np.stack((red, green, np.zeros_like(red)), axis=2)
        panels.append(Image.fromarray(rgb).resize((320, 300), Image.Resampling.BILINEAR))
    row = Image.new("RGB", (960, 330), "white")
    draw = ImageDraw.Draw(row)
    draw.text((8, 6), f"{title}; FSL/FNIT: 0..{vmax:.3g}; absolute difference: 0..{dmax:.3g}", fill="black")
    for column, panel in enumerate(panels):
        row.paste(panel, (column * 320, 30))
    return row


def _plot(candidate_dir: Path, references: dict[str, np.ndarray], output: Path, backend: str) -> None:
    canvas = Image.new("RGB", (960, 1030), "white")
    draw = ImageDraw.Draw(canvas)
    for column, title in enumerate(("FSL reference", f"FNIT {backend}", "absolute difference")):
        draw.text((column * 320 + 10, 6), title, fill="black")
    for row, name in enumerate(MAPS):
        candidate = np.asanyarray(nib.load(candidate_dir / MAPS[name]).dataobj, dtype=np.float32)
        rendered = _render_row(references[name], candidate, name)
        canvas.paste(rendered, (0, 25 + row * 330))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--reference-mask", required=True)
    parser.add_argument("--official-warped", required=True)
    parser.add_argument("--official-jacobian", required=True)
    parser.add_argument("--official-native-gm")
    parser.add_argument("--official-brain-mask")
    parser.add_argument("--synthstrip-weights", required=True)
    parser.add_argument("--synthmorph-weights")
    parser.add_argument("--backend", choices=("fnirt", "synthmorph"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--figure-out", required=True)
    parser.add_argument("--source-root", required=True)
    args = parser.parse_args(argv)

    image_path = Path(args.image)
    template_path = Path(args.template)
    mask_path = Path(args.reference_mask)
    official_warped_path = Path(args.official_warped)
    official_jacobian_path = Path(args.official_jacobian)
    output_dir = Path(args.output_dir)
    source_root = Path(args.source_root)

    official_images = {
        "warped_gm": nib.load(official_warped_path),
        "jacobian": nib.load(official_jacobian_path),
    }
    official_data = {
        name: np.asanyarray(image.dataobj, dtype=np.float32)
        for name, image in official_images.items()
    }
    official_data["modulated_gm"] = official_data["warped_gm"] * official_data["jacobian"]
    mask_image = nib.load(mask_path)
    accuracy_mask = np.asanyarray(mask_image.dataobj) > 0

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")

    process_start = time.perf_counter()
    model = FastVBM(
        device=args.device,
        threads=args.threads,
        synthstrip_weights=args.synthstrip_weights,
        synthmorph_weights=args.synthmorph_weights,
        registration_backend=args.backend,
    )
    if str(args.device).startswith("cuda"):
        torch.cuda.synchronize(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
    compute_start = time.perf_counter()
    result = model(
        image_path,
        template_path,
        brain_mask=None,
        reference_mask=mask_path,
    )
    if str(args.device).startswith("cuda"):
        torch.cuda.synchronize(args.device)
    compute_seconds = time.perf_counter() - compute_start
    save_start = time.perf_counter()
    result.save(output_dir, overwrite=False)
    save_seconds = time.perf_counter() - save_start
    in_process_seconds = time.perf_counter() - process_start
    peak_cuda = (
        int(torch.cuda.max_memory_allocated(args.device))
        if str(args.device).startswith("cuda")
        else 0
    )

    output_checks = check_outputs(output_dir, image_path, template_path)

    accuracy = {}
    contracts = {}
    output_hashes = {}
    for name, filename in MAPS.items():
        candidate_path = output_dir / filename
        candidate_image = nib.load(candidate_path)
        candidate_data = np.asanyarray(candidate_image.dataobj, dtype=np.float32)
        reference_image = official_images.get(name, official_images["warped_gm"])
        contract = _contract(candidate_image, reference_image)
        if not contract["same_shape"] or not contract["same_affine"]:
            raise ValueError(f"candidate and FSL grids differ: {name}")
        accuracy[name] = _metrics(candidate_data, official_data[name], accuracy_mask)
        contracts[name] = contract
        output_hashes[name] = _sha256(candidate_path)

    upstream = {}
    if args.official_native_gm:
        official_native = nib.load(args.official_native_gm)
        candidate_native = nib.load(output_dir / "T1_brain_pve_1.nii.gz")
        contract = _contract(candidate_native, official_native)
        upstream["native_gm_contract"] = contract
        if contract["same_shape"] and contract["same_affine"]:
            first = np.asanyarray(candidate_native.dataobj, dtype=np.float32)
            second = np.asanyarray(official_native.dataobj, dtype=np.float32)
            upstream["native_gm_accuracy_union_support"] = _metrics(
                first, second, (first != 0) | (second != 0)
            )
    if args.official_brain_mask:
        official_brain = nib.load(args.official_brain_mask)
        candidate_brain = nib.load(output_dir / "brain_mask.nii.gz")
        contract = _contract(candidate_brain, official_brain)
        upstream["brain_mask_contract"] = contract
        if contract["same_shape"] and contract["same_affine"]:
            first = np.asanyarray(candidate_brain.dataobj) > 0
            second = np.asanyarray(official_brain.dataobj) > 0
            upstream["brain_mask_dice"] = float(
                2 * np.count_nonzero(first & second)
                / max(np.count_nonzero(first) + np.count_nonzero(second), 1)
            )

    figure_path = Path(args.figure_out)
    _plot(output_dir, official_data, figure_path, args.backend)

    source_hashes = {}
    for relative in SOURCE_FILES:
        path = source_root / relative
        if path.is_file():
            source_hashes[relative] = _sha256(path)

    input_hashes = {
        "raw_t1w": _sha256(image_path),
        "gm_template": _sha256(template_path),
        "reference_mask": _sha256(mask_path),
        "official_warped_gm": _sha256(official_warped_path),
        "official_jacobian": _sha256(official_jacobian_path),
        "synthstrip_weights": _sha256(Path(args.synthstrip_weights)),
    }
    if args.synthmorph_weights:
        input_hashes["synthmorph_weights"] = _sha256(Path(args.synthmorph_weights))

    pipeline_report = json.loads((output_dir / "fast_vbm_report.json").read_text())
    report = {
        "schema_version": 1,
        "data": {
            "kind": "one deidentified real clinical T1w",
            "subjects": 1,
            "subject_identifier_published": False,
        },
        "candidate": {
            "implementation": "FNIT FastVBM",
            "backend": args.backend,
            "device": args.device,
            "source_sha256": source_hashes,
            "tf32": {
                "matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                "cudnn": bool(torch.backends.cudnn.allow_tf32),
                "reduced_precision_tensor_dtype": False,
            },
        },
        "reference": {
            "implementation": "existing FSL/UKB VBM outputs",
            "warped_gm": "fsl_reg/FNIRT output",
            "jacobian": "FNIRT nonlinear-only Jacobian",
            "modulated_gm": "warped GM multiplied by Jacobian in float32",
        },
        "accuracy_mask": {
            "definition": "explicit template-grid reference mask > 0",
            "voxels": int(np.count_nonzero(accuracy_mask)),
        },
        "contracts": contracts,
        "output_checks": output_checks,
        "accuracy": accuracy,
        "upstream_diagnosis": upstream,
        "timing_seconds": {
            "compute_including_input_load_and_lazy_weights": compute_seconds,
            "nifti_save_13_outputs_and_report": save_seconds,
            "in_process_total": in_process_seconds,
            "pipeline_stage_report": pipeline_report.get("timing_sec", {}),
        },
        "memory": {
            "peak_cuda_allocated_bytes": peak_cuda,
            "process_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
        "environment": {
            "host": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(args.device) if peak_cuda else None,
            "cpu_threads": args.threads,
        },
        "input_sha256": input_hashes,
        "output_sha256": output_hashes,
        "artifacts": {
            "figure": figure_path.name,
            "pipeline_report": "fast_vbm_report.json",
        },
        "limits": [
            "one real subject",
            "the official maps were existing FSL/UKB outputs and were not regenerated inside this script",
            "FastVBM and FSL/UKB use different brain extraction, tissue segmentation, affine, and nonlinear optimizers",
            "the timings are one cold in-process run and are not a repeated benchmark",
        ],
    }
    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "report": str(report_path),
        "backend": args.backend,
        "compute_seconds": compute_seconds,
        "save_seconds": save_seconds,
        "peak_cuda_allocated_bytes": peak_cuda,
        "accuracy": accuracy,
    }, indent=2))


if __name__ == "__main__":
    main()
