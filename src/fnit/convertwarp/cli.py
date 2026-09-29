"""Single-subject FSL convertwarp-compatible entry point."""

import argparse
from pathlib import Path

from .core import TorchConvertWarp


def _add_arguments(parser):
    parser.add_argument("--ref", required=True, help="MNI output reference image")
    warp = parser.add_mutually_exclusive_group(required=True)
    warp.add_argument("--warp1", help="FSL dense or FNIRT nonlinear warp")
    warp.add_argument("--mmorf-warp", help="FNIT MMORF reference-axis millimetre field")
    parser.add_argument("--source", help="native FA or diffusion image; required with --mmorf-warp")
    parser.add_argument("--premat", help="diffusion-to-structural FLIRT matrix")
    parser.add_argument("--postmat", help="warp-reference-to-output FLIRT matrix")
    parser.add_argument("--out", required=True, help="composite dense warp output")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--relout", action="store_true", help="relative output (default)")
    output.add_argument("--absout", action="store_true", help="absolute output")
    convention = parser.add_mutually_exclusive_group()
    convention.add_argument("--rel", action="store_true", help="relative input dense warp")
    convention.add_argument("--abs", action="store_true", help="absolute input dense warp")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(_fnit_handler=run_args)
    return parser


def add_parser(commands):
    return _add_arguments(commands.add_parser("convertwarp", allow_abbrev=False))


def run_args(args):
    output = Path(args.out)
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)
    convention = "relative" if args.rel else "absolute" if args.abs else "auto"
    model = TorchConvertWarp(args.device)
    if args.mmorf_warp:
        if (args.source is None or args.premat is None or args.postmat is not None
                or args.rel or args.abs):
            raise ValueError("--mmorf-warp requires --source and --premat, "
                             "without --postmat, --rel, or --abs")
        result = model.run_mmorf(
            reference=args.ref, source=args.source, mmorf_warp=args.mmorf_warp,
            affine=args.premat, output=output,
            output_convention="absolute" if args.absout else "relative")
    else:
        if args.source is not None:
            raise ValueError("--source applies only to --mmorf-warp")
        result = model.run(
            reference=args.ref, warp1=args.warp1, output=output,
            premat=args.premat, postmat=args.postmat,
            warp_convention=convention,
            output_convention="absolute" if args.absout else "relative")
    print(output)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fnit-convertwarp", allow_abbrev=False)
    return run_args(_add_arguments(parser).parse_args(argv))


if __name__ == "__main__":
    main()
