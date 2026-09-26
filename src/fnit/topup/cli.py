"""Single-subject command line interface for PyTorch TOPUP."""

from __future__ import annotations

import argparse

from .core import TorchTOPUP
from .ukb import run_ukb_topup


def _add_arguments(parser):
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--raw-dir",
        help="UKB-format directory containing AP/PA .nii.gz, .bval, and .json",
    )
    mode.add_argument("--imain", help="FSL-style merged b0 image")
    parser.add_argument("--datain", help="FSL-style acquisition parameters")
    parser.add_argument("--output-dir", help="output directory for --raw-dir")
    parser.add_argument("--out", help="FSL-style TOPUP output basename")
    parser.add_argument("--fout", help="field in Hz")
    parser.add_argument("--iout", help="corrected b0 series")
    parser.add_argument("--jacout", help="Jacobian output basename")
    parser.add_argument("--config", default="b02b0.cnf", choices=("b02b0.cnf",))
    parser.add_argument("--device")
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(_fnit_handler=run_args)
    return parser


def add_parser(commands):
    parser = commands.add_parser(
        "topup",
        help="PyTorch/CUDA TOPUP b02b0 field estimation for one acquisition",
        allow_abbrev=False,
    )
    return _add_arguments(parser)


def run_args(args):
    if args.raw_dir:
        if not args.output_dir:
            raise ValueError("--raw-dir requires --output-dir")
        result, prepared = run_ukb_topup(
            args.raw_dir,
            args.output_dir,
            device=args.device,
            overwrite=args.overwrite,
        )
        print(prepared["imain"])
        print(prepared["datain"])
        print(f"AP b0 index: {prepared['ap_index']}")
        print(f"PA b0 index: {prepared['pa_index']}")
        return result
    if not args.datain or not args.out:
        raise ValueError("--imain requires --datain and --out")
    return TorchTOPUP(device=args.device).run(
        args.imain,
        args.datain,
        out=args.out,
        fout=args.fout,
        iout=args.iout,
        jacout=args.jacout,
        overwrite=args.overwrite,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fnit-topup", allow_abbrev=False)
    _add_arguments(parser)
    args = parser.parse_args(argv)
    return args._fnit_handler(args)


__all__ = ["add_parser", "main", "run_args"]
