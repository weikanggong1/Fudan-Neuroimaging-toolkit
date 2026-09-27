"""Differential validation against an installed FreeSurfer SynthStrip executable.

Run after ``module load freesurfer`` using the package's Python environment.
The reference classes are extracted from source without executing its CLI.
"""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import surfa as sf
import torch
from torch import nn

from fnit.synthstrip import StripModel, SynthStrip


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def comparison(reference, actual):
    first, second = reference.data.astype(np.float64), actual.data.astype(np.float64)
    error = np.abs(first - second)
    return {
        "shape": list(first.shape),
        "shape_equal": first.shape == second.shape,
        "max_abs_error": float(error.max()),
        "mean_abs_error": float(error.mean()),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "unequal_voxels": int(np.count_nonzero(first != second)),
        "affine_max_abs_error": float(np.max(np.abs(
            reference.geom.vox2world.matrix - actual.affine
        ))),
    }


def network_check(source, weights, device):
    tree = ast.parse(source.read_text())
    classes = [node for node in ast.walk(tree)
               if isinstance(node, ast.ClassDef) and node.name in ("StripModel", "ConvBlock")]
    assert len(classes) == 2, "expected the two official SynthStrip classes"
    namespace = {"torch": torch, "nn": nn, "np": np}
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(source), "exec"), namespace)
    reference = namespace["StripModel"]().eval().to(device)
    candidate = StripModel().eval().to(device)
    state = torch.load(weights, map_location=device, weights_only=True)["model_state_dict"]
    reference.load_state_dict(state, strict=True)
    candidate.load_state_dict(state, strict=True)
    torch.manual_seed(904)
    sample = torch.rand(1, 1, 64, 64, 64, device=device)
    with torch.no_grad():
        first, second = reference(sample), candidate(sample)
    difference = (first - second).abs()
    return {
        "parameter_count": sum(value.numel() for value in candidate.parameters()),
        "state_dict_keys_identical": list(reference.state_dict()) == list(candidate.state_dict()),
        "input_shape": list(sample.shape),
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "bitwise_identical": bool(torch.equal(first, second)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reference-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--reference", default="mri_synthstrip")
    parser.add_argument("--source")
    parser.add_argument("--cases", nargs="+", default=["default", "border8", "no_csf", "4d_fill"])
    args = parser.parse_args()
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    fs_home = Path(os.environ["FREESURFER_HOME"])
    source = Path(args.source) if args.source else fs_home / "python/scripts/mri_synthstrip"
    weights = fs_home / "models/synthstrip.1.pt"
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    report = {
        "reference_source": str(source),
        "reference_source_sha256": sha256(source),
        "weights_sha256": sha256(weights),
        "input": str(Path(args.image).resolve()),
        "input_sha256": sha256(args.image),
        "input_role": "validation image; performance does not establish population accuracy",
        "torch": torch.__version__,
        "surfa": getattr(sf, "__version__", "unknown"),
        "device": args.device,
        "reference_device": args.reference_device,
        "network": network_check(source, weights, args.device),
        "cases": {},
    }
    for case in args.cases:
        if case not in ("default", "border8", "no_csf", "4d_fill"):
            parser.error(f"unknown case {case}")
        image_path = Path(args.image)
        border, fill, no_csf = 1, None, case == "no_csf"
        if case == "border8":
            border = 8
        if case == "4d_fill":
            original = sf.load_volume(args.image)
            volume = original.data.astype(np.float32)
            original.new(np.stack([volume, volume * 1.7], axis=-1)).save(root / "input4d.nii.gz")
            image_path = root / "input4d.nii.gz"
            fill = -10
        directory = root / case
        directory.mkdir(exist_ok=True)
        command = [args.reference, "-i", str(image_path), "-t", str(args.threads),
                   "-b", str(border), "-o", str(directory / "reference_image.nii.gz"),
                   "-m", str(directory / "reference_mask.nii.gz"),
                   "-d", str(directory / "reference_distance.nii.gz")]
        if args.reference_device == "cuda":
            command.append("-g")
        if no_csf:
            command.append("--no-csf")
        if fill is not None:
            command += ["-f", str(fill)]
        start = time.perf_counter()
        with (directory / "reference.log").open("w") as logfile:
            subprocess.run(command, stdout=logfile, stderr=subprocess.STDOUT, check=True)
        reference_seconds = time.perf_counter() - start
        start = time.perf_counter()
        model = SynthStrip(device=args.device, no_csf=no_csf, threads=args.threads)
        model_load_seconds = time.perf_counter() - start
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = model(image_path, border=border, fill=fill)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        candidate_seconds = time.perf_counter() - start
        metrics = {}
        for name in ("image", "mask", "distance"):
            volume = getattr(result, name)
            volume.save(directory / f"torch_{name}.nii.gz")
            reference = sf.load_volume(directory / f"reference_{name}.nii.gz")
            metrics[name] = comparison(reference, volume)
            if name == "mask":
                first, second = reference.data != 0, volume.data != 0
                denominator = int(first.sum() + second.sum())
                metrics[name]["dice"] = 1.0 if denominator == 0 else float(
                    2 * np.count_nonzero(first & second) / denominator
                )
        metrics.update({
            "reference_process_seconds_including_startup_and_io": reference_seconds,
            "candidate_model_load_seconds": model_load_seconds,
            "candidate_call_seconds_excluding_output_io": candidate_seconds,
            "candidate_peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
                if args.device.startswith("cuda") else None,
        })
        report["cases"][case] = metrics
        (root / "synthstrip_validation.json").write_text(json.dumps(report, indent=2) + "\n")
        print(case, json.dumps(metrics), flush=True)
        del model, result
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    print(root / "synthstrip_validation.json")


if __name__ == "__main__":
    main()
