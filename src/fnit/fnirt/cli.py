"""Command line interface for the supported PyTorch FNIRT path."""

import argparse
from dataclasses import replace
import sys

from .standalone import _validate_config, run_fnirt


def _csv(text, cast=float):
    try:
        return tuple(cast(value) for value in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid comma-separated values: {text}") from error


def _bits(text):
    values = _csv(text, int)
    if any(value not in (0, 1) for value in values):
        raise argparse.ArgumentTypeError("expected comma-separated 0 or 1 values")
    return tuple(bool(value) for value in values)


def _triplet(text):
    values = _csv(text)
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated values")
    return values


def _bit(text):
    values = _bits(text)
    if len(values) != 1:
        raise argparse.ArgumentTypeError("expected one 0 or 1 value")
    return values[0]


def _pair(text):
    values = _csv(text)
    if len(values) != 2:
        raise argparse.ArgumentTypeError("expected two comma-separated values")
    return values


def _effective_config(args):
    config = _validate_config(args.config)
    if args.ssqlambda is False and args.lambda_values is None:
        raise ValueError("--ssqlambda=0 requires --lambda; FSL changes its default lambda")
    names = {
        "subsamp": "subsampling",
        "miter": "maximum_iterations",
        "infwhm": "input_fwhm_mm",
        "reffwhm": "reference_fwhm_mm",
        "lambda_values": "regularization",
        "estint": "estimate_intensity",
        "applyrefmask": "apply_reference_mask",
        "warpres": "warp_resolution_mm",
        "jacrange": "jacobian_range",
        "intmod": "intensity_model",
        "intorder": "intensity_order",
        "biasres": "bias_resolution_mm",
        "biaslambda": "bias_regularization",
        "ssqlambda": "ssd_weighted_lambda",
        "imprefm": "implicit_reference_mask",
        "impinm": "implicit_input_mask",
    }
    changes = {
        field: value for option, field in names.items()
        if (value := getattr(args, option)) is not None
    }
    if args.minmet is not None:
        changes["minimization_methods"] = (
            args.minmet * len(changes.get("subsampling", config.subsampling))
            if len(args.minmet) == 1 else args.minmet
        )
    if args.warpres is not None and config.warp_resolution_schedule_mm is not None:
        changes["warp_resolution_schedule_mm"] = (
            args.warpres,
        ) * len(changes.get("subsampling", config.subsampling))
    return replace(config, **changes)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m fnit.fnirt",
        description=(
            "Run PyTorch FNIRT with the FSL no-config defaults or the GM, T1, "
            "and UKB TBSS presets. Only implemented FSL options can be overridden."
        ),
        allow_abbrev=False,
    )
    return add_arguments(parser)


def add_arguments(parser):
    parser.add_argument("--in", dest="input", required=True, help="input/moving 3D NIfTI")
    parser.add_argument("--ref", required=True, help="reference/fixed 3D NIfTI")
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
        help="reference-grid binary mask; required by GM and T1 presets",
    )
    parser.add_argument(
        "--config",
        default="default",
        help="default, gm, t1, tbss, or unmodified FSL GM/T1 config (default: %(default)s)",
    )
    for name, converter in (
        ("subsamp", lambda value: _csv(value, int)),
        ("miter", lambda value: _csv(value, int)),
        ("infwhm", _csv),
        ("reffwhm", _csv),
        ("lambda", _csv),
        ("estint", _bits),
        ("applyrefmask", _bits),
        ("warpres", _triplet),
        ("biasres", _triplet),
    ):
        parser.add_argument("--" + name, type=converter, dest="lambda_values" if name == "lambda" else name)
    parser.add_argument("--jacrange", type=_pair)
    parser.add_argument("--intmod", choices=("global_linear", "global_non_linear_with_bias"))
    parser.add_argument("--intorder", type=int)
    parser.add_argument("--biaslambda", type=float)
    parser.add_argument("--ssqlambda", type=_bit)
    parser.add_argument("--imprefm", type=_bit)
    parser.add_argument("--impinm", type=_bit)
    parser.add_argument("--minmet", type=lambda value: _csv(value, str))
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
            config=_effective_config(args),
            device=args.device,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, NotImplementedError, TypeError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
