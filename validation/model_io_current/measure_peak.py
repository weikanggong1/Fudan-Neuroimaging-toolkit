#!/usr/bin/env python3
"""Measure one real-image FNIT inference and describe every saved image."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch


FEATURE_MODULES = {
    "wmh-synthseg": "wmh_synthseg",
    "synthseg": "synthseg_parc",
    "synthsr": "synthsr",
    "fast": "fast",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest(root: Path, feature: str) -> dict:
    names = ["_dmri.py", "_nib.py", "weights.py"]
    paths = [root / "fnit" / name for name in names]
    paths.extend(sorted((root / "fnit" / FEATURE_MODULES[feature]).glob("*.py")))
    records = {
        str(path.relative_to(root)): sha256(path)
        for path in paths if path.is_file()
    }
    payload = "".join(f"{records[name]}  {name}\n" for name in sorted(records))
    return {
        "files": records,
        "tree_sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }


def image_record(path: Path) -> dict:
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    return {
        "file": path.name,
        "shape": list(image.shape),
        "dtype": str(image.get_data_dtype()),
        "affine": np.asarray(image.affine, dtype=float).tolist(),
        "finite": bool(np.isfinite(data).all()),
        "sha256": sha256(path),
    }


def save_result(feature: str, result, source: Path, output_dir: Path) -> list[Path]:
    if feature == "wmh-synthseg":
        paths = [output_dir / "segmentation.nii.gz",
                 output_dir / "lesion_probability.nii.gz"]
        result.segmentation.save(paths[0])
        result.lesion_probability.save(paths[1])
        return paths
    if feature == "synthseg":
        paths = [output_dir / "segmentation.nii.gz"]
        result.segmentation.save(paths[0])
        result.write_volumes_csv(source, output_dir / "volumes.csv")
        return paths
    if feature == "synthsr":
        paths = [output_dir / "synthsr.nii.gz"]
        result.image.save(paths[0])
        return paths
    fields = {
        "pve_csf.nii.gz": "pve_csf",
        "pve_gm.nii.gz": "pve_gm",
        "pve_wm.nii.gz": "pve_wm",
        "hard_segmentation.nii.gz": "hard_segmentation",
        "pve_segmentation.nii.gz": "pve_segmentation",
        "mixel_type.nii.gz": "mixel_type",
        "bias_field.nii.gz": "bias_field",
        "restored.nii.gz": "restored",
    }
    paths = []
    for name, field in fields.items():
        path = output_dir / name
        getattr(result, field).save(path)
        paths.append(path)
    return paths


def run(args):
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    if args.feature == "wmh-synthseg":
        from fnit.wmh_synthseg import WMHSynthSeg
        model = WMHSynthSeg(weights=args.weights, device=args.device, threads=args.threads)
    elif args.feature == "synthseg":
        from fnit.synthseg_parc import SynthSeg
        model = SynthSeg(weights=args.weights, device=args.device, threads=args.threads)
    elif args.feature == "synthsr":
        from fnit.synthsr import SynthSR
        model = SynthSR(weights=args.weights, device=args.device, threads=args.threads)
    else:
        from fnit.fast import TorchFAST
        model = TorchFAST(device=args.device, threads=args.threads)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    loaded = time.perf_counter()

    if args.feature == "wmh-synthseg":
        result = model(args.input, crop=True, save_lesion_probabilities=True)
    elif args.feature == "synthseg":
        result = model(args.input, keep_geometry=False)
    else:
        result = model(args.input)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inferred = time.perf_counter()
    paths = save_result(args.feature, result, args.input, args.output_dir)
    saved = time.perf_counter()

    report = {
        "schema": "fnit_model_io_single_case/v1",
        "feature": args.feature,
        "case_id": args.case_id,
        "input_sha256": sha256(args.input),
        "source": source_manifest(args.source_root, args.feature),
        "runtime": {
            "device": str(device),
            "threads": torch.get_num_threads(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        },
        "seconds": {
            "model_load": loaded - started,
            "inference": inferred - loaded,
            "save": saved - inferred,
            "python_api_total": saved - started,
        },
        "cuda_peak_mib": {
            "allocated": (torch.cuda.max_memory_allocated(device) / 2 ** 20
                          if device.type == "cuda" else None),
            "reserved": (torch.cuda.max_memory_reserved(device) / 2 ** 20
                         if device.type == "cuda" else None),
        },
        "outputs": [image_record(path) for path in paths],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature", required=True, choices=tuple(FEATURE_MODULES))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--weights", type=Path,
                        help="official checkpoint file or directory; omit for FAST")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--case-id", default="case01")
    parser.add_argument("--source-root", required=True, type=Path,
                        help="directory containing the fnit package")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    args.input = args.input.resolve()
    args.source_root = args.source_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = run(args)
    print(json.dumps({"feature": report["feature"],
                      "seconds": report["seconds"],
                      "cuda_peak_mib": report["cuda_peak_mib"]}))


if __name__ == "__main__":
    main()
