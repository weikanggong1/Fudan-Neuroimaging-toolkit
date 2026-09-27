"""Single-subject command line interface for PyTorch MMORF."""

import argparse

from .standalone import run_mmorf


def _arguments(parser):
    parser.add_argument("--mov-scalar", required=True, help="moving scalar image")
    parser.add_argument("--ref-scalar", required=True, help="common-space scalar reference")
    parser.add_argument("--mov-tensor", required=True, help="moving FSL six-frame tensor")
    parser.add_argument("--ref-tensor", required=True, help="reference FSL six-frame tensor")
    parser.add_argument("--aff-mov-scalar", help="moving scalar to common FLIRT matrix")
    parser.add_argument("--aff-mov-tensor", help="moving tensor to common FLIRT matrix")
    parser.add_argument("--aff-ref-tensor", help="reference tensor to common FLIRT matrix")
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--device")
    parser.add_argument("--overwrite", action="store_true")


def run(args):
    result = run_mmorf(
        args.mov_scalar,
        args.ref_scalar,
        args.mov_tensor,
        args.ref_tensor,
        moving_scalar_affine=args.aff_mov_scalar,
        moving_tensor_affine=args.aff_mov_tensor,
        reference_tensor_affine=args.aff_ref_tensor,
        device=args.device,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(args.output_dir)
    print(result.qc)


def add_parser(commands):
    parser = commands.add_parser(
        "mmorf", help="joint scalar/tensor nonlinear registration"
    )
    _arguments(parser)
    parser.set_defaults(_fnit_handler=run)
    return parser


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fnit-mmorf")
    _arguments(parser)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
