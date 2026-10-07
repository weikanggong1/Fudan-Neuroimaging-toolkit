"""One complete FNIRT stage using fixed, saved official GM and FLIRT."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import time

import nibabel as nib
import numpy as np
import torch

from fnit.flirt.coordinates import flirt_to_world_affine
from fnit.fnirt import GMFNIRTConfig, TorchFNIRT
from fnit.fnirt import registration, spline, optimizer, topology, io


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("gm", "template", "mask", "affine", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(min(1., 20_000_000_000 / torch.cuda.get_device_properties(device).total_memory), device)
        torch.cuda.reset_peak_memory_stats(device)
    gm, template, mask = nib.load(args.gm), nib.load(args.template), nib.load(args.mask)
    initial = flirt_to_world_affine(np.loadtxt(args.affine), gm.affine, template.affine,
                                   gm.shape, template.shape, gm.header.get_zooms()[:3],
                                   template.header.get_zooms()[:3])
    runner = TorchFNIRT(config=GMFNIRTConfig(), device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    fit = runner(gm, template, initial, reference_mask=mask)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    images = {"coefficients": fit.coefficient_image, "warped": fit.moved,
              "pull": fit.pull_transform, "nonlinear_jacobian": fit.nonlinear_jacobian,
              "full_pull_jacobian": fit.full_pull_jacobian}
    for name, image in images.items():
        nib.save(image, args.output / (name + ".nii.gz"))
    report = {"status": "complete", "scope": "one complete same-input GM-FNIRT stage; no raw-T1 pipeline",
              "host": socket.gethostname(), "device": str(device),
              "affinity": sorted(os.sched_getaffinity(0)), "torch_threads": torch.get_num_threads(),
              "stage_seconds": elapsed,
              "timing_boundary": "TorchFNIRT call incl. image decoding and all optimization/topology/pull/Jacobian/QC; excludes imports, header loading, persistent output saving, hashes and comparison",
              "initial_flirt": np.loadtxt(args.affine).tolist(), "initial_world": initial.tolist(),
              "inputs": {name: digest(path) for name, path in (("gm", args.gm), ("template", args.template),
                                                               ("mask", args.mask), ("flirt", args.affine))},
              "source": {name: digest(Path(module.__file__)) for name, module in
                         (("registration", registration), ("spline", spline), ("optimizer", optimizer),
                          ("topology", topology), ("io", io))},
              "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
              "tf32_cudnn": torch.backends.cudnn.allow_tf32,
              "allocator_budget_bytes": 20_000_000_000 if device.type == "cuda" else None,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
              "fit_qc": fit.qc,
              "outputs": {name: digest(args.output / (name + ".nii.gz")) for name in images}}
    (args.output / "report.public.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "complete", "seconds": elapsed, "output_count": len(images)}))


if __name__ == "__main__":
    main()
