"""Run the current MMORF path on real NIfTI inputs with bounded GPU memory."""

import argparse
import json
from pathlib import Path
import subprocess
import time

import torch

from fnit.mmorf import run_mmorf


def _snapshot(device):
    if not str(device).startswith("cuda"):
        return None
    index = int(str(device).split(":")[1]) if ":" in str(device) else 0
    result = subprocess.run(
        ["nvidia-smi", "--id", str(index),
         "--query-gpu=memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True,
    )
    used, total, utilisation = result.stdout.strip().split(", ")
    return {"used_mib": int(used), "total_mib": int(total),
            "utilization_percent": int(utilisation)}


def _affines(values):
    if values is None:
        return None
    values = [None if value.upper() == "AUTO" else value for value in values]
    return values[0] if len(values) == 1 else values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mov-scalar", action="append", required=True)
    parser.add_argument("--ref-scalar", action="append", required=True)
    parser.add_argument("--mov-tensor", required=True)
    parser.add_argument("--ref-tensor", required=True)
    parser.add_argument("--aff-mov-scalar", action="append")
    parser.add_argument("--aff-ref-scalar", action="append")
    parser.add_argument("--aff-mov-tensor")
    parser.add_argument("--aff-ref-tensor")
    parser.add_argument("--scalar-weight", type=float, action="append")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-fraction", type=float, default=0.24)
    args = parser.parse_args(argv)
    if len(args.mov_scalar) != len(args.ref_scalar):
        parser.error("scalar modalities must be paired in order")
    if not 0 < args.memory_fraction <= 1:
        parser.error("memory-fraction must lie in (0, 1]")
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(
            args.memory_fraction, device=torch.device(args.device)
        )
    before = _snapshot(args.device)
    started = time.perf_counter()
    result = run_mmorf(
        moving_scalar=args.mov_scalar,
        reference_scalar=args.ref_scalar,
        moving_tensor=args.mov_tensor,
        reference_tensor=args.ref_tensor,
        output_dir=args.output_dir,
        moving_scalar_affine=_affines(args.aff_mov_scalar),
        reference_scalar_affine=_affines(args.aff_ref_scalar),
        moving_tensor_affine=args.aff_mov_tensor,
        reference_tensor_affine=args.aff_ref_tensor,
        scalar_weights=args.scalar_weight,
        device=args.device,
    )
    report = {
        "wall_seconds_with_write": time.perf_counter() - started,
        "gpu_before": before,
        "gpu_after": _snapshot(args.device),
        "memory_fraction_limit": args.memory_fraction,
        "scalar_pair_count": len(args.mov_scalar),
        "qc": result.qc,
    }
    path = Path(args.output_dir) / "mmorf_benchmark.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
