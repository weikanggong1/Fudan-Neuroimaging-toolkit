"""Real-data, fixed-input benchmark for lossless dMRI changes.

Run this script in separate processes with the baseline/candidate PYTHONPATH.
It does not download data or run any original neuroimaging executable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import nibabel as nib
import numpy as np
import torch


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_summary(value):
    array = np.asarray(value)
    return {
        "shape": list(array.shape), "dtype": str(array.dtype),
        "finite": bool(np.isfinite(array).all()),
        "sha256_decoded": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
    }


def image_summary(image):
    result = array_summary(np.asanyarray(image.dataobj))
    result.update(affine=np.asarray(image.affine).tolist(),
                  zooms=list(map(float, image.header.get_zooms())),
                  qform_code=int(image.header["qform_code"]),
                  sform_code=int(image.header["sform_code"]))
    return result


def gpu_state():
    return subprocess.check_output([
        "nvidia-smi", "--query-gpu=uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True).strip().splitlines()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("eddy", "amico", "classic"), required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--eddy-preparation-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory-limit-bytes", type=int, default=20_000_000_000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    total_memory = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(args.memory_limit_bytes / total_memory, device)
    report = {
        "stage": args.stage, "input_scope": "one real full-brain DWI",
        "torch": torch.__version__, "numpy": np.__version__, "nibabel": nib.__version__,
        "device": torch.cuda.get_device_name(device), "threads": torch.get_num_threads(),
        "memory_limit_bytes": args.memory_limit_bytes,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_before": gpu_state(), "source_manifest": {}, "inputs": {},
    }
    import fnit
    source_root = Path(fnit.__file__).parent
    prefixes = ("eddy", "topup", "_dmri.py") if args.stage == "eddy" else ("amico_noddi", "_dmri.py")
    for prefix in prefixes:
        path = source_root / prefix
        files = path.rglob("*.py") if path.is_dir() else (path,)
        for path in sorted(files):
            report["source_manifest"][str(path.relative_to(source_root))] = sha256_file(path)
    if args.stage == "eddy":
        from fnit.eddy import TorchEDDY
        prep = args.eddy_preparation_dir
        if prep is None:
            parser.error("EDDY requires --eddy-preparation-dir")
        inputs = {
            "imain": args.input_dir / "AP.nii.gz",
            "bvals": args.input_dir / "AP.bval", "bvecs": args.input_dir / "AP.bvec",
            "mask": prep / "eddy/nodif_brain_mask.nii.gz",
            "acqp": prep / "topup/acqparams.txt", "index": prep / "eddy/eddy_index.txt",
        }
        topup = prep / "topup/fieldmap_out"
        for path in topup.parent.glob("fieldmap_out*"):
            if path.is_file():
                report["inputs"][path.name] = sha256_file(path)
        runner = TorchEDDY(device=device)
        report["gp_seed"] = args.seed
        def compute():
            return runner.run(**inputs, topup=topup, ref_scan_no=0,
                              gp_seed=args.seed, out=args.output_dir / "data")
    else:
        from fnit.amico_noddi import TorchAMICONODDI
        inputs = {
            "data": args.input_dir / "DWI.nii.gz", "mask": args.input_dir / "mask.nii.gz",
            "bvecs": args.input_dir / "bvecs", "bvals": args.input_dir / "bvals",
        }
        runner = TorchAMICONODDI(device=device, fit_method=args.stage)
        def compute():
            return runner.run(**inputs, output_dir=args.output_dir, naming="amico")
    for key, path in inputs.items():
        report["inputs"][key] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    dwi = nib.load(str(inputs.get("imain", inputs.get("data"))))
    report["input_shape"] = list(dwi.shape)
    report["mask_voxels"] = int(np.count_nonzero(nib.load(str(inputs["mask"])).dataobj))
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    allocator_before = torch.cuda.memory_stats(device)
    started = time.perf_counter()
    result = compute()
    torch.cuda.synchronize(device)
    report["wall_seconds_including_io"] = time.perf_counter() - started
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    allocator_after = torch.cuda.memory_stats(device)
    report["allocation_retries"] = allocator_after["num_alloc_retries"] - allocator_before["num_alloc_retries"]
    report["allocator_ooms"] = allocator_after["num_ooms"] - allocator_before["num_ooms"]
    report["tf32_actual_after"] = {"matmul": torch.backends.cuda.matmul.allow_tf32,
                                   "cudnn": torch.backends.cudnn.allow_tf32}
    report["gpu_after"] = gpu_state()
    report["qc"] = result.qc
    if args.stage == "eddy":
        report["outputs"] = {"corrected": image_summary(result.corrected)}
        for name in ("rotated_bvecs", "parameters", "movement_rms", "restricted_movement_rms",
                     "outlier_map", "outlier_n_stdev_map", "outlier_n_sqr_stdev_map"):
            report["outputs"][name] = array_summary(getattr(result, name))
        report["outlier_report_sha256"] = hashlib.sha256(
            "\n".join(result.outlier_report_lines).encode()).hexdigest()
    else:
        report["outputs"] = {name: image_summary(getattr(result, name))
                             for name in ("ndi", "odi", "fwf", "directions", "rmse")}
    (args.output_dir / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "stage", "wall_seconds_including_io", "peak_allocated_bytes", "mask_voxels", "outputs")}), flush=True)


if __name__ == "__main__":
    main()
