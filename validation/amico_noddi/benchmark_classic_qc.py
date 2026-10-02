"""Paired real full-brain classic refinement with one fixed AMICO initialization."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch

from fnit.amico_noddi import AMICONODDIConfig
from fnit.amico_noddi.classic import fit_classic_noddi
from fnit.amico_noddi.kernels import build_noddi_kernels, load_raw_bvecs, principal_directions
from fnit.amico_noddi.solver import fit_noddi
from fnit._dmri import load_bvals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(20_000_000_000 / torch.cuda.get_device_properties(device).total_memory, device)
    torch.backends.cuda.matmul.allow_tf32 = True
    cfg = AMICONODDIConfig()
    values = np.asarray(nib.load(args.input_dir / "DWI.nii.gz").dataobj, dtype=np.float32)
    mask = np.asarray(nib.load(args.input_dir / "mask.nii.gz").dataobj, dtype=np.uint8) == 1
    flat = np.flatnonzero(mask.reshape(-1))
    bvals = load_bvals(args.input_dir / "bvals")
    bvecs = load_raw_bvecs(args.input_dir / "bvecs", bvals.size)
    kernels = build_noddi_kernels(bvals, bvecs, cfg.ic_ods, cfg.ic_vfs,
                                 d_par=cfg.d_par, d_iso=cfg.d_iso,
                                 b0_threshold=cfg.b0_threshold, b_step=cfg.b_step)
    b0 = kernels["b0"]
    norm = values[..., b0].mean(axis=3)
    positive = norm > 0
    cutoff = 0.0 * norm[positive].mean() if np.any(positive) else 0.0
    invalid = norm <= cutoff
    norm[invalid] = 1
    norm = 1 / norm
    norm[invalid] = 0
    signal = values.reshape(-1, values.shape[3])[flat]
    signal *= norm.reshape(-1)[flat, None]
    signal = signal.astype(np.float64)
    signal[signal < 0] = 0
    directions = principal_directions(signal, kernels["raw"])
    estimates, _, _, _ = fit_noddi(
        signal, directions, kernels, device=device, lambda1=cfg.lambda1,
        lambda2=cfg.lambda2, kkt_tolerance=cfg.kkt_tolerance,
        cg_tolerance=cfg.cg_tolerance, maximum_active_steps=cfg.maximum_active_steps,
        lut_batch_size=cfg.lut_batch_size,
    )
    bvals = bvals.copy(); bvals[b0] = 0
    bvecs = bvecs.T.copy(); bvecs[b0] = (1, 0, 0)
    bvecs[~b0] /= np.linalg.norm(bvecs[~b0], axis=1, keepdims=True)
    report = {"scope": "real full-brain classic refinement; fixed unrounded AMICO initialization; no image IO or initialization in timers",
              "mask_voxels": int(flat.size), "runs": [], "all_gates_passed": True}
    previous = None
    # Interleave only the pure QC synchronization flag in the same process.
    for route in ("host_qc", "device_qc", "device_qc", "host_qc"):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        result = fit_classic_noddi(signal, bvals, bvecs, b0, estimates, directions,
                                  d_par=cfg.d_par, d_iso=cfg.d_iso, device=device,
                                  _defer_qc_count=route == "device_qc")
        torch.cuda.synchronize(device)
        row = {"route": route, "seconds": time.perf_counter() - started,
               "output_sha256": [hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest() for array in result[:3]],
               "qc": result[3]}
        signature = (row["output_sha256"], row["qc"])
        if previous is None:
            previous = signature
        row["exact"] = signature == previous
        report["all_gates_passed"] &= row["exact"]
        report["runs"].append(row)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    raise SystemExit(0 if report["all_gates_passed"] else 1)


if __name__ == "__main__":
    main()
