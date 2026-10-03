"""Audit every saved gradient frame before accepting angular comparisons."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="official rotated bvec file")
    parser.add_argument("--bvals", type=Path, required=True, help="same-input raw bval file")
    parser.add_argument("--run", action="append", required=True, help="label=rotated bvec file")
    parser.add_argument("--output", type=Path, required=True, help="new JSON receipt")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    bvals = np.loadtxt(args.bvals).reshape(-1)
    if not np.isfinite(bvals).all() or np.any(bvals < 0):
        raise ValueError("invalid raw bvals")
    expected_nonzero = bvals >= 100
    report = {"scope": "read-only saved gradient acceptance; every frame retained",
              "script_SHA256": digest(__file__), "bvals_SHA256": digest(args.bvals),
              "reference_SHA256": digest(args.reference), "frames": int(bvals.size),
              "b0_frame_indices": np.flatnonzero(~expected_nonzero).tolist(), "runs": {}}
    for label, path in [("official", args.reference)] + [spec.split("=", 1) for spec in args.run]:
        gradients = np.loadtxt(path)
        if gradients.shape != (3, bvals.size) or not np.isfinite(gradients).all():
            raise ValueError(f"nonfinite or wrong-shape gradients: {label}")
        norms = np.linalg.norm(gradients, axis=0)
        if not np.array_equal(norms > 0, expected_nonzero):
            raise ValueError(f"gradient zero support differs from raw bvals: {label}")
        deviation = float(np.max(np.abs(norms[expected_nonzero] - 1)))
        if deviation > 2e-6:
            raise ValueError(f"nonunit diffusion gradients: {label}: {deviation}")
        report["runs"][label] = {"SHA256": digest(path), "all_finite": True,
                                  "shape_matches": True, "zero_support_matches_bvals": True,
                                  "zero_count": int((~expected_nonzero).sum()),
                                  "max_nonzero_norm_deviation": deviation}
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
