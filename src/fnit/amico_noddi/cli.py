"""Command-line interface for TorchAMICONODDI."""

import argparse
from .core import TorchAMICONODDI


def add_parser(commands):
    parser = commands.add_parser(
        "amico-noddi", help="AMICO 2.0.3-compatible NODDI fitting"
    )
    for flag, dest in (
        ("-k", "data"),
        ("-m", "mask"),
        ("-r", "bvecs"),
        ("-b", "bvals"),
    ):
        parser.add_argument(flag, f"--{dest}", required=True)
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--naming", choices=("ukb", "amico"), default="ukb")
    parser.add_argument("--fit-method", choices=("amico", "classic"), default="amico")
    parser.add_argument("--device")
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(_fnit_handler=run)
    return parser


def run(args):
    result = TorchAMICONODDI(device=args.device, fit_method=args.fit_method).run(
        args.data,
        args.mask,
        args.bvecs,
        args.bvals,
        output_dir=args.output_dir,
        naming=args.naming,
        overwrite=args.overwrite,
    )
    print(args.output_dir)
    print(result.qc)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fnit-amico-noddi")
    parser.add_argument("-k", "--data", required=True)
    parser.add_argument("-m", "--mask", required=True)
    parser.add_argument("-r", "--bvecs", required=True)
    parser.add_argument("-b", "--bvals", required=True)
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--naming", choices=("ukb", "amico"), default="ukb")
    parser.add_argument("--fit-method", choices=("amico", "classic"), default="amico")
    parser.add_argument("--device")
    parser.add_argument("--overwrite", action="store_true")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
