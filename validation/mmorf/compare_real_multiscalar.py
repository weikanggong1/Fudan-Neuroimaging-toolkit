"""Compare real T1+FA MMORF registrations with one matched sampler."""

import argparse
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch

from compare_mmorf import contract, load, metrics
from fnit.mmorf import apply_mmorf_warp


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fnit-dir", type=Path, required=True)
    parser.add_argument("--official-dir", type=Path, required=True)
    parser.add_argument("--mov-t1", required=True)
    parser.add_argument("--mov-fa", required=True)
    parser.add_argument("--ref-t1", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-fraction", type=float, default=0.24)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(
            args.memory_fraction, device=torch.device(args.device)
        )
    fnit_qc = json.loads((args.fnit_dir / "mmorf_report.json").read_text())
    moving_matrices = fnit_qc["linear_alignment"]["moving_scalar"]
    official_warp_path = args.official_dir / "official_warp.nii.gz"
    official_jacobian_path = args.official_dir / "official_jacobian.nii.gz"
    started = time.perf_counter()
    for name, moving, matrix in (
        ("T1", args.mov_t1, moving_matrices[0]["matrix"]),
        ("FA", args.mov_fa, moving_matrices[1]["matrix"]),
    ):
        sampled = apply_mmorf_warp(
            image=moving,
            reference=args.ref_t1,
            warp=official_warp_path,
            affine=np.asarray(matrix, dtype=np.float64),
            device=args.device,
            interpolation="cubic",
        )
        nib.save(sampled, str(args.official_dir / f"sampled_{name}.nii.gz"))

    _, brain = load(args.ref_t1)
    brain_mask = brain > 0
    fnit_warp_image, fnit_warp = load(args.fnit_dir / "mmorf_warp.nii.gz")
    official_warp_image, official_warp = load(official_warp_path)
    fnit_jac_image, fnit_jac = load(args.fnit_dir / "mmorf_jacobian.nii.gz")
    official_jac_image, official_jac = load(official_jacobian_path)
    _, fnit_t1 = load(args.fnit_dir / "mmorf_warped_scalar.nii.gz")
    _, official_t1 = load(args.official_dir / "sampled_T1.nii.gz")
    _, fnit_fa = load(args.fnit_dir / "mmorf_warped_scalar_2.nii.gz")
    _, official_fa = load(args.official_dir / "sampled_FA.nii.gz")
    fa_mask = brain_mask & (fnit_fa != 0) & (official_fa != 0)
    report = {
        "boundary": "one real subject; identical T1, FA, tensor, templates, and FLIRT matrices",
        "sampler": "FNIT cubic sampler for both estimated warps and both moving scalar images",
        "warp": metrics(fnit_warp, official_warp,
                        np.broadcast_to(brain_mask[..., None], fnit_warp.shape)),
        "warp_component_pearson": [
            metrics(fnit_warp[..., axis], official_warp[..., axis], brain_mask)["pearson"]
            for axis in range(3)
        ],
        "jacobian": metrics(fnit_jac, official_jac, brain_mask),
        "warped_T1": metrics(fnit_t1, official_t1, brain_mask),
        "warped_FA": metrics(fnit_fa, official_fa, fa_mask),
        "contracts": {
            "warp": contract(fnit_warp_image, official_warp_image),
            "jacobian": contract(fnit_jac_image, official_jac_image),
        },
        "seconds_including_sampling": time.perf_counter() - started,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(args.device)
            if args.device.startswith("cuda") else None
        ),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
