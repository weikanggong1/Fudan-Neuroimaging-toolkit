"""Command line interface for the supported PyTorch FNIRT path."""

import argparse
import sys

from .standalone import SUPPORTED_CONFIG, run_fnirt


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m fnit.fnirt",
        description=(
            "Run the source-derived PyTorch implementation of FSL's supported "
            "GM_2_MNI152GM_2mm FNIRT path. Other FNIRT configurations and "
            "options are rejected."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--in", dest="input", required=True, help="input/moving GM NIfTI")
    parser.add_argument("--ref", required=True, help="reference/fixed GM NIfTI")
    parser.add_argument(
        "--aff",
        help=(
            "input-to-reference FLIRT matrix in FSL scaled-mm coordinates; "
            "default: identity"
        ),
    )
    parser.add_argument(
        "--cout",
        help=(
            "output cubic coefficient NIfTI (intent 2007); default: "
            "<input>_warpcoef with FSLOUTPUTTYPE extension"
        ),
    )
    parser.add_argument("--iout", help="output warped input on the reference grid")
    parser.add_argument("--jout", help="output nonlinear-only Jacobian determinant")
    parser.add_argument(
        "--refmask",
        required=True,
        help="explicit reference-grid binary mask",
    )
    parser.add_argument(
        "--config",
        default=SUPPORTED_CONFIG,
        help=f"only {SUPPORTED_CONFIG} is accepted (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="PyTorch device, for example cpu, cuda, or cuda:1; default: CUDA when available",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing output files; outputs are otherwise protected",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        run_fnirt(
            args.input,
            args.ref,
            args.aff,
            cout=args.cout,
            iout=args.iout,
            jout=args.jout,
            refmask=args.refmask,
            config=args.config,
            device=args.device,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, NotImplementedError, TypeError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
