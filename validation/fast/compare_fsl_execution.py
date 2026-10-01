"""Compare both FNIT execution paths with archived native FAST outputs.

The private manifest supplies the input and native output paths. This script
does not launch FSL. Image outputs stay under --private-output; --report contains
only anonymous metrics, hashes, timing boundaries and software versions.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch

from fnit.fast import TorchFAST
from fnit.fast import algorithm, pipeline
from fnit.fast import _fsl_scan


FIELDS = {
    "pve_csf": "pve_0", "pve_gm": "pve_1", "pve_wm": "pve_2",
    "hard_segmentation": "seg", "pve_segmentation": "pveseg",
    "mixel_type": "mixeltype", "bias_field": "bias", "restored": "restore",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(candidate, reference, mask, labels=False):
    left = np.asarray(candidate, dtype=np.float64)[mask]
    right = np.asarray(reference, dtype=np.float64)[mask]
    delta = left - right
    record = {
        "pearson": float(np.corrcoef(left, right)[0, 1]),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "mae": float(np.mean(np.abs(delta))),
        "max_abs": float(np.max(np.abs(delta))),
        "changed_voxels": int(np.count_nonzero(delta)),
    }
    if labels:
        record["agreement_fraction"] = float(np.mean(left == right))
    else:
        for threshold in (0.5, 0.8):
            a, b = left >= threshold, right >= threshold
            denominator = int(a.sum() + b.sum())
            record[f"dice_{threshold}"] = (float(2 * np.count_nonzero(a & b) / denominator)
                                             if denominator else 1.0)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--execution", nargs="+", choices=("tensor", "fsl"),
                        default=("tensor", "fsl"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    image = nib.load(manifest["input"])
    input_data = np.asarray(image.dataobj, dtype=np.float32)
    brain = input_data > 0
    references = {field: nib.load(manifest["native"][suffix])
                  for field, suffix in FIELDS.items()}
    args.private_output.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "One real brain-extracted T1; identical native FAST input, without priors; three classes, T1 defaults",
        "original_command": "fast -t 1 -n 3 -I 4 -W 15 -O 4 -f 0.02 -l 20 -H 0.1 -R 0.3 -B -b -o OUTPUT INPUT",
        "native_wall_seconds": manifest.get("native_wall_seconds"),
        "native_exit_code": manifest.get("native_exit_code"),
        "native_outputs_validated_after_abnormal_exit": manifest.get("native_outputs_validated_after_abnormal_exit", False),
        "native_software": manifest.get("native_software"),
        "timing_boundary": "FNIT: loaded image to all output images returned, includes model construction, H2D, computation and D2H; excludes process startup, input file read, gzip output writes and comparisons; first run includes any Triton compilation",
        "runtime_original_calls": False,
        "input_sha256": sha256(manifest["input"]),
        "native_output_sha256": {suffix: sha256(manifest["native"][suffix]) for suffix in FIELDS.values()},
        "source_sha256": {f"src/fnit/fast/{Path(module.__file__).name}": sha256(module.__file__)
                          for module in (algorithm, pipeline, _fsl_scan)},
        "driver_sha256": sha256(__file__),
        "shape": list(image.shape), "header_pixdim_mm": list(map(float, image.header.get_zooms()[:3])),
        "positive_mask_voxels": int(brain.sum()),
        "software": {"torch": torch.__version__, "nibabel": nib.__version__,
                     "numpy": np.__version__, "cuda": torch.version.cuda},
        "device": args.device,
        "tf32": {"matmul": torch.backends.cuda.matmul.allow_tf32,
                 "cudnn": torch.backends.cudnn.allow_tf32},
        "execution": {},
    }
    if torch.device(args.device).type == "cuda":
        report["gpu"] = torch.cuda.get_device_name(torch.device(args.device))
    for execution in args.execution:
        print(f"Starting execution={execution}", flush=True)
        if torch.device(args.device).type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(torch.device(args.device))
            torch.cuda.synchronize(torch.device(args.device))
        start = time.perf_counter()
        model = TorchFAST(device=args.device, threads=1, execution=execution)
        result = model(image)
        if torch.device(args.device).type == "cuda":
            torch.cuda.synchronize(torch.device(args.device))
        elapsed = time.perf_counter() - start
        record = {"wall_seconds": elapsed, "outputs": {},
                  "config": vars(model.config), "tissue_means": result.tissue_means,
                  "tissue_variances": result.tissue_variances}
        if torch.device(args.device).type == "cuda":
            record["peak_torch_allocated_bytes"] = torch.cuda.max_memory_allocated(torch.device(args.device))
        output_dir = args.private_output / execution
        output_dir.mkdir(exist_ok=True)
        for field, suffix in FIELDS.items():
            output = getattr(result, field)
            reference = references[field]
            output_path = output_dir / f"fast_{suffix}.nii.gz"
            nib.save(output, output_path)
            record["outputs"][suffix] = {
                **metrics(np.asarray(output.dataobj), np.asarray(reference.dataobj), brain,
                          field in {"hard_segmentation", "pve_segmentation", "mixel_type"}),
                "shape_equal": output.shape == reference.shape,
                "affine_max_abs": float(np.max(np.abs(output.affine - reference.affine))),
                "dtype_equal": output.get_data_dtype() == reference.get_data_dtype(),
                "qform_code_equal": int(output.header["qform_code"]) == int(reference.header["qform_code"]),
                "sform_code_equal": int(output.header["sform_code"]) == int(reference.header["sform_code"]),
                "output_sha256": sha256(output_path),
            }
        report["execution"][execution] = record
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"execution": execution, "wall_seconds": elapsed,
                          "pve_r": {suffix: record["outputs"][suffix]["pearson"]
                                    for suffix in ("pve_0", "pve_1", "pve_2")}}), flush=True)
    report["tf32"] = {"matmul": torch.backends.cuda.matmul.allow_tf32,
                      "cudnn": torch.backends.cudnn.allow_tf32}
    args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
