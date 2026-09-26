"""Single-subject BEDPOSTX command line interface."""

import argparse

from .core import TorchBEDPOSTX


def _add_arguments(parser):
    parser.add_argument("--subject-dir", required=True,
                        help="directory with data.nii.gz, mask, bvals and bvecs")
    parser.add_argument("--output-dir", help="output directory; default is <subject>.bedpostX")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--nfibres", type=int, default=3)
    parser.add_argument("--model", type=int, choices=(1, 2), default=2)
    parser.add_argument("--burnin", type=int, default=1000)
    parser.add_argument("--njumps", type=int, default=1250)
    parser.add_argument("--sample-every", type=int, default=25)
    parser.add_argument("--ard-weight", type=float, default=1.0)
    parser.add_argument("--chunk-size", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=8665904)
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(_fnit_handler=run_args)
    return parser


def add_parser(commands):
    parser = commands.add_parser(
        "bedpostx", help="PyTorch Bayesian crossing-fibre fit for one DWI subject",
        allow_abbrev=False,
    )
    return _add_arguments(parser)


def run_args(args):
    model = TorchBEDPOSTX(
        device=args.device, threads=args.threads, nfibres=args.nfibres,
        model=args.model, burnin=args.burnin, njumps=args.njumps,
        sample_every=args.sample_every, ard_weight=args.ard_weight,
        chunk_size=args.chunk_size, seed=args.seed,
    )
    result = model(args.subject_dir, output_dir=args.output_dir, overwrite=args.overwrite)
    print(result.output_dir)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fnit-bedpostx", allow_abbrev=False)
    _add_arguments(parser)
    return run_args(parser.parse_args(argv))


__all__ = ["add_parser", "main", "run_args"]
