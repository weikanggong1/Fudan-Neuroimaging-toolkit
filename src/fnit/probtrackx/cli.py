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
    parser.add_argument("--mask", help="tracking mask in diffusion space; defaults to bedpostX mask")
    parser.add_argument("--avoid", help="reject half paths entering this volume mask")
    parser.add_argument("--stop", help="stop half paths upon entering this volume mask")
    parser.add_argument("--wtstop", help="volume mask or text list; stop after leaving an entered mask")
    parser.add_argument("--waypoints", help="volume mask or text list of waypoint masks")
    parser.add_argument("--waycond", choices=("AND", "OR"), default="AND")
    parser.add_argument("--wayorder", action="store_true")
    parser.add_argument("--onewaycondition", action="store_true")
    parser.add_argument("--forcefirststep", action="store_true")
    parser.add_argument("--omatrix1", action="store_true", help="seed voxel by seed voxel")
    parser.add_argument("--omatrix2", action="store_true", help="seed voxel by target2 voxel")
    parser.add_argument("--target2", help="target2 volume on the diffusion grid")
    parser.add_argument("--omatrix3", action="store_true", help="target3 voxel co-visitation")
    parser.add_argument("--target3", help="target3 volume on the diffusion grid")
    parser.add_argument("--lrtarget3", help="optional matrix3 column target volume")
    parser.add_argument("--distthresh1", type=float, default=0.0)
    parser.add_argument("--distthresh3", type=float, default=0.0)
    parser.add_argument("--targetmasks", help="text file listing target ROI volumes")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--nsamples", type=int, default=5000)
    parser.add_argument("--nsteps", type=int, default=2000)
    parser.add_argument("--steplength", type=float, default=0.5)
    parser.add_argument("--cthr", type=float, default=0.2)
    parser.add_argument("--fibthresh", type=float, default=0.01)
    parser.add_argument("--distthresh", type=float, default=0.0)
    parser.add_argument("--sampvox", type=float, default=0.0)
    parser.add_argument("--fibst", type=int)
    parser.add_argument("--randfib", type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument("--usef", action="store_true")
    parser.add_argument("--pd", action="store_true", help="weight path density by length")
    parser.add_argument("--ompl", action="store_true", help="save mean path lengths")
    parser.add_argument("--batch-size", type=int, default=16384)
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
    if args.omatrix2 != bool(args.target2):
        raise ValueError("--omatrix2 and --target2 must be supplied together")
    if args.omatrix3 != bool(args.target3):
        raise ValueError("--omatrix3 and --target3 must be supplied together")
    targetmasks = None
    if args.targetmasks:
        listing = Path(args.targetmasks).resolve()
        targetmasks = [Path(line) if Path(line).is_absolute() else listing.parent / line
                       for line in listing.read_text().splitlines() if line.strip()]
        if not targetmasks:
            raise ValueError("--targetmasks list is empty")
    model = TorchProbtrackX(device=args.device, nsamples=args.nsamples,
                            nsteps=args.nsteps, steplength=args.steplength,
                            cthr=args.cthr, fibthresh=args.fibthresh,
                            batch_size=args.batch_size, seed=args.rseed,
                            distthresh=args.distthresh, sampvox=args.sampvox,
                            fibst=args.fibst, usef=args.usef, randfib=args.randfib,
                            pathdist=args.pd, mean_path_length=args.ompl)
    result = model.run(args.samples_dir, args.output_dir, seed=args.seed,
                       regions=regions, mask=args.mask, avoid=args.avoid,
                       stop=args.stop, forcefirststep=args.forcefirststep,
                       wtstop=args.wtstop, waypoints=args.waypoints,
                       waycond=args.waycond, wayorder=args.wayorder,
                       onewaycondition=args.onewaycondition,
                       matrix1=args.omatrix1, target2=args.target2,
                       target3=args.target3, lrtarget3=args.lrtarget3,
                       distthresh1=args.distthresh1, distthresh3=args.distthresh3,
                       targetmasks=targetmasks, overwrite=args.overwrite)
    print(result.paths)
    print(result.waytotal)
    if result.network_matrix:
        print(result.network_matrix)
    if result.lengths:
        print(result.lengths)
    if result.network_lengths:
        print(result.network_lengths)
    for path in (result.matrix1, result.matrix2, result.matrix3,
                 result.network_probability, result.network_symmetric,
                 *result.seed_to_targets):
        if path:
            print(path)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fnit-probtrackx", allow_abbrev=False)
    _add_arguments(parser)
    return run_args(parser.parse_args(argv))


__all__ = ["add_parser", "main", "run_args"]


if __name__ == "__main__":
    main()
