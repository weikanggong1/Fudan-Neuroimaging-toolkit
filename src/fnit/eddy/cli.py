"""Command-line interface for TorchEDDY."""

import argparse
from .core import TorchEDDY
from .ukb import run_ukb_eddy


def _arguments(parser):
    parser.add_argument("--raw-dir")
    parser.add_argument("--topup-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--imain")
    parser.add_argument("--mask")
    parser.add_argument("--acqp")
    parser.add_argument("--index")
    parser.add_argument("--bvecs")
    parser.add_argument("--bvals")
    parser.add_argument("--topup")
    parser.add_argument("--out")
    parser.add_argument("--ref-scan-no", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--overwrite", action="store_true")


def run(args):
    if args.raw_dir:
        if not args.topup_dir or not args.output_dir:
            raise ValueError("--raw-dir requires --topup-dir and --output-dir")
        result, _ = run_ukb_eddy(
            args.raw_dir,
            args.topup_dir,
            args.output_dir,
            device=args.device,
            overwrite=args.overwrite,
        )
    else:
        required = (
            args.imain,
            args.mask,
            args.acqp,
            args.index,
            args.bvecs,
            args.bvals,
            args.topup,
            args.out,
        )
        if any(v is None for v in required):
            raise ValueError(
                "direct mode requires --imain --mask --acqp --index --bvecs --bvals --topup --out"
            )
        result = TorchEDDY(args.device).run(
            args.imain,
            args.mask,
            args.acqp,
            args.index,
            args.bvecs,
            args.bvals,
            topup=args.topup,
            out=args.out,
            ref_scan_no=args.ref_scan_no,
            overwrite=args.overwrite,
        )
    print(result.qc)


def add_parser(commands):
    parser = commands.add_parser("eddy", help="UKB EDDY correction on PyTorch GPU")
    _arguments(parser)
    parser.set_defaults(_fnit_handler=run)
    return parser


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fnit-eddy")
    _arguments(parser)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
