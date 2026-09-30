"""比较同主机同输入 N4 的量化前 float32 与量化后 uint8 结果。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference_float", "candidate_float", "reference_uint8",
                 "candidate_uint8", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    reference_image = nib.load(str(args.reference_float))
    shape = reference_image.shape
    reference = np.asarray(reference_image.dataobj, dtype=np.float32)
    candidate = np.fromfile(args.candidate_float, dtype=np.float32).reshape(shape, order="F")
    reference_uint8 = np.asarray(nib.load(str(args.reference_uint8)).dataobj)
    candidate_uint8 = np.asarray(nib.load(str(args.candidate_uint8)).dataobj)
    float_error = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    different = np.argwhere(reference_uint8 != candidate_uint8)
    round_to_uint8 = lambda values: np.floor(np.clip(values, 0, 255) + 0.5).astype(np.uint8)
    report = {
        "code_commit": args.code_commit, "shape": [int(axis) for axis in shape],
        "inputs_sha256": {key: _hash(getattr(args, key)) for key in
                          ("reference_float", "candidate_float", "reference_uint8",
                           "candidate_uint8")},
        "float_different_voxels": int(np.count_nonzero(float_error)),
        "float_max_absolute_difference": float(float_error.max()),
        "float_p99_absolute_difference": float(np.percentile(float_error, 99)),
        "float_p9999_absolute_difference": float(np.percentile(float_error, 99.99)),
        "reference_float_to_uint8_different_voxels": int(np.count_nonzero(
            round_to_uint8(reference) != reference_uint8)),
        "candidate_float_to_uint8_different_voxels": int(np.count_nonzero(
            round_to_uint8(candidate) != candidate_uint8)),
        "uint8_different_voxels": int(len(different)),
        "uint8_max_absolute_difference": int(np.max(np.abs(
            reference_uint8.astype(np.int16) - candidate_uint8.astype(np.int16)))),
        "uint8_differences": [
            {"voxel": point.tolist(), "reference_float": float(reference[tuple(point)]),
             "candidate_float": float(candidate[tuple(point)]),
             "reference_uint8": int(reference_uint8[tuple(point)]),
             "candidate_uint8": int(candidate_uint8[tuple(point)])}
            for point in different],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("float_different_voxels", "float_max_absolute_difference",
                       "uint8_different_voxels", "uint8_max_absolute_difference")}, indent=2))


if __name__ == "__main__":
    main()
