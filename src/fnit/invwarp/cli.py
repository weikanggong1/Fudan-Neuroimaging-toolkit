"""Single-subject FSL invwarp-compatible entry point."""

import argparse
from pathlib import Path

from .core import TorchInvWarp


def _add_arguments(parser):
    parser.add_argument("--ref", required=True, help="diffusion output reference image")
    parser.add_argument("--warp", required=True, help="forward composite warp")
    parser.add_argument("--out", required=True, help="inverse dense warp output")
    convention = parser.add_mutually_exclusive_group()
    convention.add_argument("--rel", action="store_true", help="relative input and output (default)")
    convention.add_argument("--abs", action="store_true", help="absolute input and output")
    parser.add_argument("--niter", type=int, default=30, help="maximum fixed-point iterations")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(_fnit_handler=run_args)
    return parser


def add_parser(commands):
    return _add_arguments(commands.add_parser("invwarp", allow_abbrev=False))


def run_args(args):
    output = Path(args.out)
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)
    result = TorchInvWarp(args.device).run(
        reference=args.ref, warp=args.warp, output=output,
        warp_convention="absolute" if args.abs else "relative" if args.rel else "auto",
        output_convention="absolute" if args.abs else "relative",
        iterations=args.niter)
    print(output)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fnit-invwarp", allow_abbrev=False)
    return run_args(_add_arguments(parser).parse_args(argv))


if __name__ == "__main__":
    main()
