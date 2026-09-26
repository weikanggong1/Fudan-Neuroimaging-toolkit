"""Command-line interface for TorchDTIFIT."""

import argparse

from .core import TorchDTIFIT


def _arguments(parser):
    parser.add_argument("-k", "--data", required=True)
    parser.add_argument("-m", "--mask", required=True)
    parser.add_argument("-r", "--bvecs", required=True)
    parser.add_argument("-b", "--bvals", required=True)
    parser.add_argument(
        "-o", "--out", required=True, help="extensionless FSL output basename"
    )
    parser.add_argument("--device")
    parser.add_argument(
        "--save_tensor",
        action="store_true",
        help="write <out>_tensor.nii.gz, matching FSL --save_tensor",
    )
    parser.add_argument("--overwrite", action="store_true")


def add_parser(commands):
    parser = commands.add_parser(
        "dtifit", help="FSL-compatible diffusion tensor fitting"
    )
    _arguments(parser)
    parser.set_defaults(_fnit_handler=run)
    return parser


def run(args):
    result = TorchDTIFIT(device=args.device).run(
        args.data,
        args.mask,
        args.bvecs,
        args.bvals,
        output_prefix=args.out,
        save_tensor=args.save_tensor,
        overwrite=args.overwrite,
    )
    names = (
        result.maps
        if args.save_tensor
        else {name: image for name, image in result.maps.items() if name != "tensor"}
    )
    for name in names:
        print(f"{args.out}_{name}.nii.gz")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fnit-dtifit")
    _arguments(parser)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
