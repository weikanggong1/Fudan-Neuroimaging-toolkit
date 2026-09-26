"""Single-subject volume tractography command line interface."""

import argparse
from pathlib import Path

from .pipeline import TorchProbtrackX


def _add_arguments(parser):
    parser.add_argument("--samples-dir", required=True, help="FSL-compatible bedpostX directory")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--seed", help="one diffusion-space NIfTI seed mask")
    mode.add_argument("--roi-list", help="text file listing >=2 diffusion-space NIfTI ROIs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--nsamples", type=int, default=5000)
    parser.add_argument("--nsteps", type=int, default=2000)
    parser.add_argument("--steplength", type=float, default=0.5)
    parser.add_argument("--cthr", type=float, default=0.2)
    parser.add_argument("--fibthresh", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--rseed", type=int, default=12345)
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(_fnit_handler=run_args)
    return parser


def add_parser(commands):
    parser = commands.add_parser("probtrackx", allow_abbrev=False,
                                 help="seed-to-voxel and ROI network tractography")
    return _add_arguments(parser)


def run_args(args):
    regions = None
    if args.roi_list:
        listing = Path(args.roi_list).resolve()
        regions = [Path(line) if Path(line).is_absolute() else listing.parent / line
                   for line in listing.read_text().splitlines() if line.strip()]
    model = TorchProbtrackX(device=args.device, nsamples=args.nsamples,
                            nsteps=args.nsteps, steplength=args.steplength,
                            cthr=args.cthr, fibthresh=args.fibthresh,
                            batch_size=args.batch_size, seed=args.rseed)
    result = model.run(args.samples_dir, args.output_dir, seed=args.seed,
                       regions=regions, overwrite=args.overwrite)
    print(result.paths)
    print(result.waytotal)
    if result.network_matrix:
        print(result.network_matrix)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fnit-probtrackx", allow_abbrev=False)
    _add_arguments(parser)
    return run_args(parser.parse_args(argv))


__all__ = ["add_parser", "main", "run_args"]


if __name__ == "__main__":
    main()
