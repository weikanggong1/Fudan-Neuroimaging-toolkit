#!/usr/bin/env python3
"""Compare FNIT and official MMORF outputs on one matched subject."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def load(path):
    image = nib.load(path)
    return image, np.asarray(image.dataobj, dtype=np.float64)


def metrics(candidate, reference, mask):
    valid = mask & np.isfinite(candidate) & np.isfinite(reference)
    a = candidate[valid]
    b = reference[valid]
    if a.size < 2:
        raise ValueError("comparison mask contains fewer than two finite values")
    error = a - b
    return {
        "voxels": int(a.size),
        "pearson": float(np.corrcoef(a, b)[0, 1]),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "maximum_absolute_error": float(np.max(np.abs(error))),
    }


def contract(candidate, reference):
    return {
        "shape_match": candidate.shape == reference.shape,
        "affine_maximum_absolute_difference": float(
            np.max(np.abs(candidate.affine - reference.affine))
        ),
        "dtype_match": candidate.get_data_dtype() == reference.get_data_dtype(),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fnit-warp", required=True)
    parser.add_argument("--official-warp", required=True)
    parser.add_argument("--fnit-jacobian", required=True)
    parser.add_argument("--official-jacobian", required=True)
    parser.add_argument("--brain-mask", required=True)
    parser.add_argument("--fnit-scalar")
    parser.add_argument("--official-scalar")
    parser.add_argument(
        "--map",
        action="append",
        nargs=3,
        metavar=("NAME", "FNIT", "OFFICIAL_WARP_SAMPLED"),
        default=[],
        help="repeat for maps sampled with the same affine/interpolator",
    )
    parser.add_argument(
        "--map-mask",
        nargs=2,
        metavar=("FNIT_FA", "OFFICIAL_FA"),
        help="FA pair whose common non-zero support masks repeated --map comparisons",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    brain_image, brain = load(args.brain_mask)
    brain_mask = brain > 0

    fnit_warp_image, fnit_warp = load(args.fnit_warp)
    official_warp_image, official_warp = load(args.official_warp)
    if fnit_warp.shape != official_warp.shape or fnit_warp.shape[-1] != 3:
        raise ValueError("warp inputs must have one matching [X,Y,Z,3] shape")
    warp_mask = np.broadcast_to(brain_mask[..., None], fnit_warp.shape)

    fnit_jacobian_image, fnit_jacobian = load(args.fnit_jacobian)
    official_jacobian_image, official_jacobian = load(args.official_jacobian)
    report = {
        "comparison_boundary": (
            "one matched subject; identical registration inputs and initial affines"
        ),
        "mask": "brain_mask > 0",
        "warp": metrics(fnit_warp, official_warp, warp_mask),
        "warp_component_pearson": [
            metrics(fnit_warp[..., axis], official_warp[..., axis], brain_mask)[
                "pearson"
            ]
            for axis in range(3)
        ],
        "jacobian": metrics(fnit_jacobian, official_jacobian, brain_mask),
        "contracts": {
            "warp": contract(fnit_warp_image, official_warp_image),
            "jacobian": contract(fnit_jacobian_image, official_jacobian_image),
            "mask_affine_maximum_absolute_difference": float(
                np.max(np.abs(brain_image.affine - fnit_jacobian_image.affine))
            ),
        },
    }

    if bool(args.fnit_scalar) != bool(args.official_scalar):
        raise ValueError("provide both --fnit-scalar and --official-scalar")
    if args.fnit_scalar:
        fnit_scalar_image, fnit_scalar = load(args.fnit_scalar)
        official_scalar_image, official_scalar = load(args.official_scalar)
        report["warped_scalar"] = metrics(
            fnit_scalar, official_scalar, brain_mask
        )
        report["contracts"]["warped_scalar"] = contract(
            fnit_scalar_image, official_scalar_image
        )

    if args.map:
        if not args.map_mask:
            raise ValueError("--map-mask is required when --map is used")
        _, fnit_mask_data = load(args.map_mask[0])
        _, official_mask_data = load(args.map_mask[1])
        map_mask = (fnit_mask_data != 0) & (official_mask_data != 0)
        report["standard_maps"] = {}
        report["map_comparison_boundary"] = (
            "same FNIT sampler, affine and native input; only the estimated warp differs"
        )
        for name, fnit_path, official_path in args.map:
            _, fnit_map = load(fnit_path)
            _, official_map = load(official_path)
            report["standard_maps"][name] = metrics(
                fnit_map, official_map, map_mask
            )

    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
