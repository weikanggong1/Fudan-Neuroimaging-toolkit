"""Single-subject command line interface for PyTorch MMORF."""

import argparse

from .standalone import run_mmorf


def _arguments(parser):
    parser.add_argument("--mov-scalar", required=True, action="append", help="moving scalar; repeat in pair order")
    parser.add_argument("--ref-scalar", required=True, action="append", help="reference scalar; repeat in pair order")
    parser.add_argument("--mov-tensor", required=True, help="moving FSL six-frame tensor")
    parser.add_argument("--ref-tensor", required=True, help="reference FSL six-frame tensor")
    parser.add_argument("--aff-mov-scalar", action="append", help="moving scalar to common FLIRT matrix; repeat per pair, use AUTO for missing")
    parser.add_argument("--aff-ref-scalar", action="append", help="reference scalar to common FLIRT matrix; repeat per pair, use AUTO for missing")
    parser.add_argument("--aff-mov-tensor", help="moving tensor to common FLIRT matrix")
    parser.add_argument("--aff-ref-tensor", help="reference tensor to common FLIRT matrix")
    parser.add_argument("--scalar-weight", action="append", type=float, help="scalar pair weight; repeat per pair")
    parser.add_argument("--no-auto-linear", action="store_true", help="use identity for unspecified linear matrices")
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--device")
    parser.add_argument("--overwrite", action="store_true")


def run(args):
    def single_or_sequence(values):
        return values[0] if len(values) == 1 else values

    def affines(values):
        if values is None:
            return None
        selected = [None if value.upper() == "AUTO" else value for value in values]
        return single_or_sequence(selected)

    result = run_mmorf(
        single_or_sequence(args.mov_scalar),
        single_or_sequence(args.ref_scalar),
        args.mov_tensor,
        args.ref_tensor,
        moving_scalar_affine=affines(args.aff_mov_scalar),
        reference_scalar_affine=affines(args.aff_ref_scalar),
        moving_tensor_affine=args.aff_mov_tensor,
        reference_tensor_affine=args.aff_ref_tensor,
        scalar_weights=args.scalar_weight,
        auto_linear=not args.no_auto_linear,
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
