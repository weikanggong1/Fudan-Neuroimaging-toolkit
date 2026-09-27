#!/usr/bin/env python3
"""Validate one current-source raw-to-standard TBSS run on real AP/PA dMRI."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw


MAPS = ("FA", "MD", "L1", "L2", "L3", "MO", "ICVF", "OD", "ISOVF")
NATIVE_FILES = {
    "FA": "dti_FA.nii.gz",
    "MD": "dti_MD.nii.gz",
    "L1": "dti_L1.nii.gz",
    "L2": "dti_L2.nii.gz",
    "L3": "dti_L3.nii.gz",
    "MO": "dti_MO.nii.gz",
    "ICVF": "NODDI_ICVF.nii.gz",
    "OD": "NODDI_OD.nii.gz",
    "ISOVF": "NODDI_ISOVF.nii.gz",
}
SOURCE_FILES = (
    "src/fnit/__init__.py",
    "src/fnit/cli.py",
    "src/fnit/_dmri.py",
    "src/fnit/dmri_pipeline/pipeline.py",
    "src/fnit/dmri_pipeline/tbss.py",
    "src/fnit/topup/core.py",
    "src/fnit/topup/io.py",
    "src/fnit/topup/ukb.py",
    "src/fnit/eddy/core.py",
    "src/fnit/eddy/ukb.py",
    "src/fnit/dtifit/core.py",
    "src/fnit/amico_noddi/core.py",
    "src/fnit/amico_noddi/kernels.py",
    "src/fnit/amico_noddi/solver.py",
    "src/fnit/flirt/core.py",
    "src/fnit/fnirt/registration.py",
    "src/fnit/applywarp/core.py",
    "src/fnit/_nib.py",
    "src/fnit/_transforms.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path):
    image = nib.load(path)
    data = np.asanyarray(image.dataobj, dtype=np.float32).squeeze()
    if data.ndim != 3:
        raise ValueError(f"expected one 3D map: {path}")
    return image, data


def _compare(candidate_path: Path, reference_path: Path) -> dict:
    candidate_image, candidate = _load(candidate_path)
    reference_image, reference = _load(reference_path)
    if candidate.shape != reference.shape:
        return {
            "same_shape": False,
            "same_affine": False,
            "same_dtype": candidate_image.get_data_dtype() == reference_image.get_data_dtype(),
            "metrics": None,
        }
    valid = np.isfinite(candidate) & np.isfinite(reference)
    valid &= (candidate != 0) | (reference != 0)
    first = candidate[valid].astype(np.float64, copy=False)
    second = reference[valid].astype(np.float64, copy=False)
    difference = first - second
    return {
        "same_shape": True,
        "same_affine": bool(np.allclose(candidate_image.affine, reference_image.affine, atol=1e-5, rtol=0)),
        "same_dtype": candidate_image.get_data_dtype() == reference_image.get_data_dtype(),
        "metrics": {
            "voxels": int(first.size),
            "pearson": float(np.corrcoef(first, second)[0, 1]),
            "mae": float(np.mean(np.abs(difference))),
            "rmse": float(np.sqrt(np.mean(np.square(difference)))),
            "maximum_absolute_error": float(np.max(np.abs(difference))),
        },
    }


def _external_time(path: Path) -> dict:
    text = path.read_text()
    elapsed = re.search(r"Elapsed \(wall clock\) time .*: ([0-9:.]+)", text)
    maximum_rss = re.search(r"Maximum resident set size \(kbytes\): ([0-9]+)", text)
    exit_status = re.search(r"Exit status: ([0-9]+)", text)
    if not (elapsed and maximum_rss and exit_status):
        raise ValueError(f"cannot parse GNU time output: {path}")
    seconds = 0.0
    for value in elapsed.group(1).split(":"):
        seconds = seconds * 60 + float(value)
    return {
        "wall_seconds": seconds,
        "maximum_rss_kib": int(maximum_rss.group(1)),
        "exit_status": int(exit_status.group(1)),
    }


def _panel(values: np.ndarray, maximum: float, size=(360, 380), difference=False):
    scaled = np.clip(values / max(maximum, np.finfo(np.float32).eps), 0, 1)
    if difference:
        red = np.rint(255 * scaled)
        green = np.rint(150 * np.sqrt(scaled))
        rgb = np.stack((red, green, np.zeros_like(red)), axis=2).astype(np.uint8)
    else:
        rgb = np.repeat(np.rint(255 * scaled)[..., None].astype(np.uint8), 3, axis=2)
    return Image.fromarray(rgb).resize(size, Image.Resampling.BILINEAR)


def _plot(candidate_path: Path, reference_path: Path, output: Path, pearson: float, mae: float):
    _, candidate = _load(candidate_path)
    _, reference = _load(reference_path)
    difference = np.abs(candidate - reference)
    support = (candidate != 0) | (reference != 0)
    vmax = max(float(np.percentile(reference[support], 99.5)), 1e-8)
    dmax = max(float(np.percentile(difference[support], 99.5)), 1e-8)
    canvas = Image.new("RGB", (1080, 860), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), f"real FA, n=1: r={pearson:.6f}, MAE={mae:.6f}", fill="black")
    for column, label in enumerate(("FSL/UKB TBSS", "FNIT current", "absolute difference")):
        draw.text((column * 360 + 10, 34), label, fill="black")
    for row, (axis, index, label) in enumerate(((2, reference.shape[2] // 2, "axial"), (1, reference.shape[1] // 2, "coronal"))):
        top = 60 + row * 400
        draw.text((10, top), label, fill="black")
        arrays = (
            np.rot90(np.take(reference, index, axis=axis)),
            np.rot90(np.take(candidate, index, axis=axis)),
            np.rot90(np.take(difference, index, axis=axis)),
        )
        for column, values in enumerate(arrays):
            canvas.paste(_panel(values, dmax if column == 2 else vmax, difference=column == 2), (column * 360, top + 20))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--official-native", required=True)
    parser.add_argument("--official-tbss", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--source-snapshot-tar-sha256", required=True)
    parser.add_argument("--run-script-sha256", required=True)
    parser.add_argument("--pytorch-cuda-alloc-conf", required=True)
    parser.add_argument("--gpu-used-before-mib", type=float, required=True)
    parser.add_argument("--gpu-util-before-percent", type=float, required=True)
    parser.add_argument("--fa-template", required=True)
    parser.add_argument("--fa-skeleton", required=True)
    parser.add_argument("--external-time", required=True)
    parser.add_argument("--official-time", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--figure-out", required=True)
    args = parser.parse_args(argv)

    candidate_root = Path(args.candidate_root)
    candidate_native = candidate_root / "native"
    registration = candidate_root / "registration"
    official_native = Path(args.official_native)
    official_tbss = Path(args.official_tbss)
    source_root = Path(args.source_root)
    pipeline_report_path = candidate_root / "dmri_pipeline_report.json"
    pipeline_report = json.loads(pipeline_report_path.read_text())

    native = {
        name: _compare(candidate_native / NATIVE_FILES[name], official_native / NATIVE_FILES[name])
        for name in MAPS
    }
    standard = {}
    skeleton = {}
    for name in MAPS:
        reference_name = "all_FA.nii.gz" if name == "FA" else f"all_{name}.nii.gz"
        standard[name] = _compare(
            registration / "standard" / f"{name}.nii.gz",
            official_tbss / "stats" / reference_name,
        )
        skeleton[name] = _compare(
            registration / "skeleton" / f"{name}.nii.gz",
            official_tbss / "stats" / f"all_{name}_skeletonised.nii.gz",
        )

    external = _external_time(Path(args.external_time))
    official_external = _external_time(Path(args.official_time))
    source_relatives = set(SOURCE_FILES)
    for directory in (
        "src/fnit/dmri_pipeline",
        "src/fnit/topup",
        "src/fnit/eddy",
        "src/fnit/dtifit",
        "src/fnit/amico_noddi",
        "src/fnit/flirt",
        "src/fnit/fnirt",
        "src/fnit/applywarp",
    ):
        source_relatives.update(
            path.relative_to(source_root).as_posix()
            for path in (source_root / directory).glob("*.py")
        )
    source_hashes = {
        relative: _sha256(source_root / relative)
        for relative in sorted(source_relatives)
        if (source_root / relative).is_file()
    }
    raw_hashes = {
        path.name: _sha256(path)
        for path in sorted(Path(args.raw_dir).iterdir())
        if path.is_file()
    }
    reference_hashes = {}
    for name in MAPS:
        reference_name = "all_FA.nii.gz" if name == "FA" else f"all_{name}.nii.gz"
        reference_hashes[f"standard_{name}"] = _sha256(official_tbss / "stats" / reference_name)
        reference_hashes[f"skeleton_{name}"] = _sha256(official_tbss / "stats" / f"all_{name}_skeletonised.nii.gz")

    fa = standard["FA"]["metrics"]
    figure_path = Path(args.figure_out)
    _plot(
        registration / "standard" / "FA.nii.gz",
        official_tbss / "stats" / "all_FA.nii.gz",
        figure_path,
        fa["pearson"],
        fa["mae"],
    )
    component_peaks = {
        "eddy": pipeline_report["eddy"]["peak_cuda_memory_bytes"],
        "dtifit": pipeline_report["dtifit"]["peak_cuda_memory_bytes"],
        "noddi": pipeline_report["noddi"]["peak_cuda_memory_bytes"],
        "tbss": pipeline_report["registration"]["peak_cuda_memory_bytes"],
    }
    report = {
        "schema_version": 1,
        "data": {
            "subjects": 1,
            "kind": "one deidentified real UKB-format AP/PA diffusion MRI",
            "subject_identifier_published": False,
            "raw_file_sha256": raw_hashes,
            "template_sha256": {
                "FMRIB58_FA_1mm": _sha256(Path(args.fa_template)),
                "FMRIB58_FA_skeleton_1mm": _sha256(Path(args.fa_skeleton)),
            },
        },
        "boundary": {
            "candidate": "raw AP/PA through TOPUP, EDDY, DTIFIT, AMICO-NODDI, weighted FLIRT, three-stage FNIRT, nine map propagations, and skeleton multiplication",
            "reference_native": "existing official UKB DTI/NODDI maps",
            "reference_standard": "official FSL 6.0.7.4 weighted FLIRT, three FNIRT stages, applywarp, and skeleton multiplication from prepared official maps",
            "timing_match": "candidate timing is full raw-to-standard; official timing starts from prepared FA and parameter maps",
        },
        "candidate": {
            "implementation": "FNIT DMRIPipeline registration_backend=tbss",
            "source_snapshot_tar_sha256": args.source_snapshot_tar_sha256,
            "source_sha256": source_hashes,
            "pipeline_report_sha256": _sha256(pipeline_report_path),
            "validator_sha256": _sha256(Path(__file__)),
            "device": pipeline_report["device"],
            "dtype": pipeline_report["dtype"],
            "tf32": pipeline_report["tf32"],
            "output_contract": {
                "nine_native_maps": pipeline_report["native_maps"],
                "nine_standard_maps": pipeline_report["standard_maps"],
                "common_standard_output_contract": pipeline_report["common_standard_output_contract"],
            },
        },
        "reference": {
            "implementation": "FSL 6.0.7.4 / UKB TBSS registration",
            "sha256": reference_hashes,
        },
        "execution": {
            "run_date": "2026-09-28",
            "host": "gpucw1",
            "gpu": "NVIDIA H100 PCIe",
            "software": {
                "pytorch": "2.5.1",
                "pytorch_cuda": "11.8",
                "nibabel": "5.4.2",
                "numpy": "1.26.4",
            },
            "run_script_sha256": args.run_script_sha256,
            "candidate_command": "python -m fnit.cli dmri-pipeline --raw-dir <RAW_DIR> --output-dir <OUTPUT_DIR> --registration-backend tbss --fa-template <FMRIB58_FA_1mm.nii.gz> --fa-skeleton <FMRIB58_FA-skeleton_1mm.nii.gz> --device cuda",
            "environment": {
                "PYTHONPATH": "<SOURCE_SNAPSHOT>/src",
                "CUDA_VISIBLE_DEVICES": "1",
                "PYTORCH_CUDA_ALLOC_CONF": args.pytorch_cuda_alloc_conf,
                "CUDA_MODULE_LOADING": "LAZY",
            },
            "gpu_baseline_before_candidate": {
                "memory_used_mib": args.gpu_used_before_mib,
                "utilization_percent": args.gpu_util_before_percent,
                "exclusive_timing": False,
                "note": "a separate long-running GPU training process was active",
            },
            "reference_command": "validation/dmri_pipeline/run_official_tbss.sh <OUTPUT_DIR> <PREPROCESSED_FA> <INPUT_WEIGHT> <MAP_DIR> <FMRIB58_FA_1mm.nii.gz> <FMRIB58_FA-skeleton_1mm.nii.gz> <OXFORD_CONFIG_PREFIX>",
        },
        "accuracy": {
            "native_maps": native,
            "standard_maps": standard,
            "skeleton_maps": skeleton,
        },
        "timing": {
            "candidate_external": external,
            "candidate_internal_total_seconds": pipeline_report["elapsed_seconds"],
            "candidate_stages_seconds": pipeline_report["timings_seconds"],
            "official_registration_external": official_external,
            "comparison_scope_matched": False,
            "candidate_timing_isolated": False,
            "reason_no_speedup": "candidate timing includes raw preprocessing; official timing starts from prepared native maps",
        },
        "memory": {
            "component_peak_cuda_bytes": component_peaks,
            "maximum_reported_component_cuda_bytes": max(component_peaks.values()),
        },
        "status": {
            "run_completed": external["exit_status"] == 0,
            "standard_grid_contract_passed": all(v["same_shape"] and v["same_affine"] for v in standard.values()),
            "standard_dtype_contract_passed": all(v["same_dtype"] for v in standard.values()),
            "skeleton_grid_contract_passed": all(v["same_shape"] and v["same_affine"] for v in skeleton.values()),
            "skeleton_dtype_contract_passed": all(v["same_dtype"] for v in skeleton.values()),
            "numerical_equivalence_passed": False,
            "reason": "one-case raw-to-standard outputs are not near voxelwise equality",
        },
        "artifacts": {"figure": figure_path.name},
        "limits": [
            "one real subject",
            "candidate and official timing scopes differ",
            "candidate wall timing includes contention from a separate GPU training process",
            "native-map differences are part of the raw-to-standard comparison",
        ],
    }
    output = Path(args.report_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "report": str(output),
        "candidate_wall_seconds": external["wall_seconds"],
        "maximum_reported_component_cuda_bytes": report["memory"]["maximum_reported_component_cuda_bytes"],
        "standard_map_pearson": {name: standard[name]["metrics"]["pearson"] for name in MAPS},
        "skeleton_map_pearson": {name: skeleton[name]["metrics"]["pearson"] for name in MAPS},
    }, indent=2))


if __name__ == "__main__":
    main()
