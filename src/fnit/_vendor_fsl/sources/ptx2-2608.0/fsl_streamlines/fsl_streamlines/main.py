"""
fsl-streamlines: Turns the results of XTRACT/PROBTRACKX into streamlines for
 visualisation.

Before running this, user must run PROBTRACKX or XTRACT with the --opathdir
option.

E.g.
echo '--opathdir' > options.txt
xtract -ptx_options options.txt <rest of command>
"""

import argparse
import datetime
import os
import os.path as op

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("fsl_streamlines")
except PackageNotFoundError:
    __version__ = '<unknown>'

import numpy as np

from fsl.data.image     import Image
from fsl.scripts.imtest import imtest

from fsl_streamlines.streamliner import Streamliner, NoSeedsError

try:
    DEFAULT_NUM_JOBS = int(os.environ.get('FSL_NUM_THREADS', 1))
except Exception:
    DEFAULT_NUM_JOBS = 1


class ArgumentParser(argparse.ArgumentParser):

    def __init__(self, **kwargs):
        argparse.ArgumentParser.__init__(self, prog="fsl-streamlines",
                                         add_help=True, **kwargs)
        self.add_argument("input", nargs="+",
                          help="XTRACT or PROBTRACKX tract directory/"
                               "directories, or orientation file")
        self.add_argument("-o", "--output-prefix", default='streamlines',
                          help="Output file prefix (default: streamlines)")
        self.add_argument("-f", "--output-format", choices=("trk", "vtk", "trx"),
                          default="trk",
                          help="Output format - one of trk (default), vtk, or trx")
        self.add_argument("-p", "--ptx2-prefix", default='fdt_paths',
                          help="File prefix for PROBTRACKX inputs (default: "
                               "fdt_paths)")
        self.add_argument("-t", "--density-threshold", type=float,
                          default=1e-3,
                          help="Threshold for streamline density (default: "
                               "1e-3). Used for defining seeds, and for "
                               "terminating streamlines. If density image "
                               "is not present and --seed-mask is not "
                               "provided, all voxels within the tract are "
                               "used as seeds (but --subsample is still "
                               "applied).")
        self.add_argument('-m', '--seed-mask',
                          help="Binary mask defining seed voxels. Can be "
                               "used instead of --density-threshold for "
                               "defining seeds.")
        self.add_argument("-min", "--min-steps", type=int, default=2,
                          help="Minimum number of steps in a streamline "
                               "(default: 2)")
        self.add_argument("-max", "--max-steps", type=int, default=300,
                          help="Maximum number of steps in a streamline "
                               "(default: 300)")
        self.add_argument("-s", "--step", type=float, default=0.4,
                          help="Step size relative to voxel size (default: "
                               "0.4)")
        self.add_argument("-j", "--jitter", action="store_true", default=False,
                          help="Randomize seed position within voxel")
        self.add_argument("-spv", "--seeds-per-voxel", type=int, default=1,
                          help="If --jitter specified, number of random seeds "
                               "to use per voxel (default: 1)")
        self.add_argument("-ss", "--subsample", type=int, default=1,
                          help="Subsample seed voxels, e.g. if 10 consider "
                               "only every 10th seed voxel (default: 1)")
        self.add_argument("-nj", "--num-jobs", type=int,
                          default=DEFAULT_NUM_JOBS,
                          help="Number of processes to use to parallelise "
                               f"streamlining (default: {DEFAULT_NUM_JOBS})")
        self.add_argument("-i", "--interp", default='nn',
                          choices=('nn', 'trilinear', 'spline'),
                          help="Interpolation method for density values "
                               "(default: nn)")
        self.add_argument("--no-density", action="store_true", default=False,
                          help="Do not colour streamlines by density")
        self.add_argument("-xf", "--xflip", choices=("auto", "true", "false"),
                          default="auto", help="Flip vectors about the X "
                          "voxel axis (default: true for data with "
                          "neurological storage order, false otherwise)")
        self.add_argument('-so', "--save-overlay",
                          action='append', metavar='IMAGE',
                          help="With --output-format=vtk, also save following "
                               "image in compatible .vti format. Can be "
                               "specified more than once.")
        self.add_argument("-rs", "--seed", type=int,
                          help="Seed for random number generator, used with "
                               "--jitter (default: random seed)")
        self.add_argument("-v", "--version", action="version",
                          version=f"%(prog)s {__version__}")


def parse_args(argv):
    parser = ArgumentParser()
    opts   = parser.parse_args(argv)

    if opts.min_steps < 2:
        parser.error(f'min-steps cannot be less than 2 ({opts.min_steps})!')

    if opts.max_steps < opts.min_steps:
        parser.error(f'max-steps ({opts.max_steps}) must be '
                     f'greater than min-steps ({opts.min_steps})!')

    if opts.seed_mask is not None:
        opts.seed_mask = Image(opts.seed_mask)

    if   opts.interp == 'nn':        opts.interp = 0
    elif opts.interp == 'trilinear': opts.interp = 1
    elif opts.interp == 'spline':    opts.interp = 3

    if   opts.xflip == 'true':  opts.xflip = True
    elif opts.xflip == 'false': opts.xflip = False

    if opts.save_overlay is None:
        opts.save_overlay = []

    return opts


def load_input(tract_input, ptx2_prefix):

    # Input may be one of:
    #   - streamline orientation file
    #   - xtract tract directory
    #   - ptx2 output directory

    # streamline orientation file
    if imtest(tract_input):
        return Image(tract_input), None

    # xtract/ptx2 must have been run with the
    # ptx2 --opathdir option, to produce a _localdir
    # file containing streamline orientations.

    # - xtract names its primary density output "densityNorm"
    # - user specifies ptx2 prefix (default fdt_path)
    # - boolean controls whether to normalise by waytotal
    candidates = [('density_localdir',        'densityNorm',    False),
                  (f'{ptx2_prefix}_localdir', f'{ptx2_prefix}', True)]

    orientation = None
    density     = None

    for (orntfile, densfile, normalise) in candidates:
        orntfile = op.join(tract_input, orntfile)
        densfile = op.join(tract_input, densfile)
        wayfile  = op.join(tract_input, 'waytotal')

        if imtest(orntfile):
            orientation = Image(orntfile)
            if imtest(densfile):
                density = Image(densfile)

                # xtract saves normalised density values
                # (densityNorm) but ptx2 doesn't, so we
                # normalise here
                if normalise and op.exists(wayfile):
                    # If multiple seed masks were used, waytotal
                    # will contain more than one value. We can
                    # just take the sum to get the total number
                    # of streamlines.
                    waytotal   = np.loadtxt(wayfile).sum()
                    density[:] = density[:] / waytotal
            break

    return orientation, density


def main(argv=None):

    options = parse_args(argv)

    if options.seed:
        np.random.seed(options.seed)

    tract_inputs = options.input

    for tract_input in tract_inputs:

        orientation, density = load_input(tract_input, options.ptx2_prefix)

        # L/R flip for neuro images
        xflip = options.xflip
        if xflip == 'auto':
            xflip = orientation.isNeurological()
        if xflip:
            orientation[..., 0] = -orientation[..., 0]

        if orientation is None:
            print(f"Could not find orientation file in {tract_input}! "
                  "Did you run xtract/probtrackx2 with the --opathdir "
                  "option? Skipping {tract_input}...")
            continue

        if options.no_density:
            density = None

        print(f'----- Streamlining {tract_input} -----')

        try:
            S = Streamliner(
                orientation,
                density=density,
                threshold=options.density_threshold,
                seed_mask=options.seed_mask,
                stepsize=options.step,
                nsteps=options.max_steps,
                jitter=options.jitter,
                n_per_vox=options.seeds_per_voxel,
                min_n_steps=options.min_steps,
                subsample=options.subsample,
                order=options.interp,
                num_jobs=options.num_jobs)
        except NoSeedsError as e:
            print(f"{e} Skipping {tract_input}...")
            continue

        start = datetime.datetime.now()
        streamlines, density_values = S.streamline()
        end = datetime.datetime.now()

        elapsed = (end - start)
        secs    = elapsed.seconds
        millis  = elapsed.microseconds // 1000

        print(f'Finished in {secs}.{millis} seconds: '
              f'{len(streamlines)} streamlines')

        # single file input
        if imtest(tract_input):
            basename = op.join(op.dirname(tract_input), options.output_prefix)
        # xtract/ptx2 folder input
        else:
            basename = op.join(tract_input, options.output_prefix)

        if len(streamlines) == 0:
            print('No streamlines generated! Output file will not be created')
        elif options.output_format == 'trk':
            S.save_trk(basename, streamlines, orientation, density_values)
        elif options.output_format == 'trx':
            S.save_trx(basename, streamlines, orientation, density_values)
        elif options.output_format == 'vtk':
            S.save_vtk(basename, streamlines, density_values, options.save_overlay)

if __name__ == "__main__":
    main()
