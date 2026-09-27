"""Command line interface for the supported PyTorch FLIRT path."""

import argparse
import sys

from .standalone import run_flirt


def build_parser(prog="fnit-flirt"):
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Run the source-derived PyTorch implementation of FLIRT's supported "
            "12-DOF correlation-ratio or 6-DOF normalized-mutual-information "
            "registration path. Other profiles are rejected."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("-in", "--in", dest="input", required=True,
                        help="input/moving 3D image")
    parser.add_argument("-ref", "--ref", dest="reference", required=True,
                        help="reference/fixed image defining the output grid")
    parser.add_argument(
        "-out", "--out", dest="output",
        help="warped input on the reference grid",
    )
    parser.add_argument(
        "-omat", "--omat",
        help="input-to-reference 4x4 matrix in FSL scaled-mm coordinates",
    )
    parser.add_argument(
        "-init", "--init",
        help="initial input-to-reference FSL scaled-mm matrix",
    )
    parser.add_argument(
        "-inweight", "--inweight",
        help="input-space voxelwise registration weight",
    )
    parser.add_argument(
        "-refweight", "--refweight",
        help="reference-space voxelwise registration weight",
    )
    parser.add_argument("-dof", type=int, choices=(6, 12), default=12,
                        help="degrees of freedom; choose 6 with -cost normmi")
    parser.add_argument(
        "-cost", choices=("corratio", "normmi"), default="corratio",
        help="cost function; choose normmi with -dof 6",
    )
    parser.add_argument(
        "--device", default=None,
        help="PyTorch device, for example cpu, cuda, or cuda:1; default: CUDA when available",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace existing output files; outputs are otherwise protected",
    )
    return parser


def main(argv=None, *, prog="fnit-flirt"):
    parser = build_parser(prog=prog)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        run_flirt(
            args.input,
            args.reference,
            output=args.output,
            omat=args.omat,
            init=args.init,
            inweight=args.inweight,
            refweight=args.refweight,
            dof=args.dof,
            cost=args.cost,
            device=args.device,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, NotImplementedError, TypeError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
