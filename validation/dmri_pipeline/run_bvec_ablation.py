"""Refit one fixed EDDY output with original AP b-vectors and run TBSS."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from fnit.amico_noddi import TorchAMICONODDI
from fnit.dmri_pipeline.tbss import TorchTBSS
from fnit.dtifit import TorchDTIFIT, select_shell


MAPS = ("FA", "MD", "L1", "L2", "L3", "MO", "ICVF", "OD", "ISOVF")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--fa-template", type=Path, required=True)
    parser.add_argument("--fa-skeleton", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    native = args.output_dir / "native"
    native.mkdir()
    eddy = args.baseline_root / "eddy"
    corrected = eddy / "data.nii.gz"
    mask = eddy / "nodif_brain_mask.nii.gz"
    bval = args.raw_dir / "AP.bval"
    bvec = args.raw_dir / "AP.bvec"
    shell_image, shell_bval, shell_bvec = select_shell(
        corrected, bval, bvec, native / "data_1_shell",
        shell=1000, tolerance=100,
    )
    TorchDTIFIT(device=args.device).run(
        shell_image, mask, shell_bvec, shell_bval,
        output_prefix=native / "dti", save_tensor=True,
    )
    TorchAMICONODDI(device=args.device).run(
        corrected, mask, bvec, bval, output_dir=native, naming="ukb",
    )
    files = {
        name: f"{'dti' if name in MAPS[:6] else 'NODDI'}_{name}.nii.gz"
        for name in MAPS
    }
    if torch.device(args.device).type == "cuda":
        torch.empty(1, device=args.device)  # initialise CUDA memory statistics
    TorchTBSS(device=args.device).run(
        {name: native / files[name] for name in MAPS},
        args.fa_template,
        args.fa_skeleton,
        output_dir=args.output_dir / "registration",
    )


if __name__ == "__main__":
    main()
