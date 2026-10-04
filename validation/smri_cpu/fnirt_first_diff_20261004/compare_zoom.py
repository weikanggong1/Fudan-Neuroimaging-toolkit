"""Compare one same-coefficient real 4-to-2 FNIRT transition."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from fnit.fnirt.registration import _pack, _unpack
from fnit.fnirt.spline import zoom_coefficients


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    coefficients, scale = _unpack(torch.from_numpy(np.loadtxt(args.input)), (7,8,7), True)
    resized = zoom_coefficients(coefficients, (46,55,46), (5,5,5), (8.,8.,8.), (4.,4.,4.))
    official = np.loadtxt(args.official)
    candidate = _pack(resized, scale).numpy()
    difference = official - candidate
    report = {"scope": "coefficient-only 4-to-2 transition with shared real first-accepted official FP64 parameters; no fit",
              "old_matrix": [24,28,24], "new_matrix": [46,55,46],
              "old_control_grid": [7,8,7], "new_control_grid": list(resized.shape[1:]),
              "knot_spacing": [5,5,5], "old_voxels_mm": [8,8,8], "new_voxels_mm": [4,4,4],
              "different_values": int(np.count_nonzero(difference)),
              "max_abs_mm": float(np.abs(difference).max()),
              "rmse_mm": float(np.sqrt(np.mean(difference ** 2))),
              "relative_l2": float(np.linalg.norm(difference) / np.linalg.norm(official)),
              "scale_exact": bool(official[-1] == candidate[-1]),
              "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
              "official_output_sha256": hashlib.sha256(args.official.read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
