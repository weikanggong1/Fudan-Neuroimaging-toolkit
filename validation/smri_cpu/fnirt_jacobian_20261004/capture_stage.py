"""Fit only FNIRT on existing real GM and the recorded affine, once."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket

import nibabel as nib
import numpy as np
import torch

from fnit._nib import new_image
from fnit.fast_vbm.registration import (
    _common_applywarp, _fsl_dense_nonlinear_jacobian, _pull_ras_to_fsl_fields,
)
from fnit.flirt.coordinates import world_to_flirt_affine
from fnit.fnirt import GMFNIRTConfig, TorchFNIRT


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--reference-mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    moving_path = args.candidate / "T1_brain_pve_1.nii.gz"
    moving, fixed = nib.load(moving_path), nib.load(args.template)
    mask = nib.load(args.reference_mask)
    prior = json.loads((args.candidate / "fast_vbm_report.json").read_text())
    initial = np.linalg.inv(prior["registration"]["pull_world_affine"])
    flirt = world_to_flirt_affine(
        initial, moving.affine, fixed.affine, moving.shape, fixed.shape,
        moving.header.get_zooms()[:3], fixed.header.get_zooms()[:3])
    fit = TorchFNIRT(device="cpu", config=GMFNIRTConfig())(
        moving, fixed, initial, reference_mask=mask)
    for name, image in [("coefficients", fit.coefficient_image), ("pull", fit.pull_transform),
                        ("analytic", fit.nonlinear_jacobian), ("full_pull_jacobian", fit.full_pull_jacobian),
                        ("estimator_warped", fit.moved)]:
        nib.save(image, args.output / (name + ".nii.gz"))
    residual, dense, fixed_fsl = _pull_ras_to_fsl_fields(
        fit.pull_transform, moving, fixed, flirt, device=torch.device("cpu"))
    dense_jacobian = _fsl_dense_nonlinear_jacobian(residual, fixed_fsl).numpy()
    warped, _ = _common_applywarp(np.asanyarray(moving.dataobj), np.asanyarray(fixed.dataobj),
                                moving.affine, fixed.affine, dense, device=torch.device("cpu"))
    for name, data in [("old_warped", warped), ("old_dense_jacobian", dense_jacobian),
                       ("old_modulated", warped * dense_jacobian)]:
        nib.save(new_image(data, fixed), args.output / (name + ".nii.gz"))
    report = {"status": "complete", "scope": "one existing-real-GM FNIRT fit; no SynthStrip, FAST or FLIRT refit",
              "host": socket.gethostname(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
              "torch_threads": torch.get_num_threads(), "device": "cpu",
              "initial_forward_world": initial.tolist(), "initial_flirt": flirt.tolist(),
              "inputs": {"gm": digest(moving_path), "template": digest(args.template),
                         "reference_mask": digest(args.reference_mask),
                         "prior_pipeline_report": digest(args.candidate / "fast_vbm_report.json")},
              "fit_qc": fit.qc,
              "outputs": {p.name: digest(p) for p in args.output.glob("*.nii.gz")}}
    (args.output / "capture.public.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "complete", "output_count": len(report["outputs"])}))


if __name__ == "__main__":
    main()
