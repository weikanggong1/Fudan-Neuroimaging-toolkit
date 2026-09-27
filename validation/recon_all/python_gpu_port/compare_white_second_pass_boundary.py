"""Compare RAM fields at the installed white first/second-pass boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


VERTEX_DTYPE = np.dtype({
    "names": ("xyz", "orig", "normal", "val", "target", "val2", "d", "mean", "marked", "rip"),
    "formats": (("<f4", (3,)), ("<f4", (3,)), ("<f4", (3,)), "<f4",
                ("<f4", (3,)), "<f4", "<f4", "<f4", "<i2", "u1"),
    "offsets": (24, 36, 48, 140, 184, 332, 368, 428, 452, 460),
    "itemsize": 464,
})


def _read(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) != 106622 * VERTEX_DTYPE.itemsize:
        raise ValueError(f"unexpected VERTEX buffer length: {path}: {len(data)}")
    return np.frombuffer(data, dtype=VERTEX_DTYPE)


def _float_diff(left: np.ndarray, right: np.ndarray) -> dict:
    difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return {
        "exact_components": int(np.count_nonzero(left == right)),
        "total_components": int(left.size),
        "max_abs": float(difference.max()),
        "p99_abs": float(np.percentile(difference, 99)),
        "count_gt_1e4": int(np.count_nonzero(difference > 1e-4)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-raw", type=Path, required=True)
    parser.add_argument("--second-raw", type=Path, required=True)
    parser.add_argument("--python-first-npz", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    first, second = _read(args.first_raw), _read(args.second_raw)
    with np.load(args.python_first_npz) as prior:
        python_xyz = prior["step17_after_collision"]
        python_rip = prior["ripped"]
        python_val = prior["target_values"]
    report = {
        "first_raw_sha256": hashlib.sha256(args.first_raw.read_bytes()).hexdigest(),
        "second_raw_sha256": hashlib.sha256(args.second_raw.read_bytes()).hexdigest(),
        "python_npz_sha256": hashlib.sha256(args.python_first_npz.read_bytes()).hexdigest(),
        "first_ripped": int(np.count_nonzero(first["rip"])),
        "second_ripped": int(np.count_nonzero(second["rip"])),
        "newly_ripped": np.flatnonzero((first["rip"] == 0) & (second["rip"] != 0)).astype(int).tolist(),
        "first_to_second": {name: _float_diff(first[name], second[name])
                            for name in ("xyz", "orig", "normal", "val", "target", "val2", "d", "mean")},
        "python_to_official_first_xyz": _float_diff(python_xyz, first["xyz"]),
        "python_to_official_first_val": _float_diff(python_val, first["val"]),
        "python_to_official_first_rip_mismatch": int(np.count_nonzero(python_rip != (first["rip"] != 0))),
        "second_marked": int(np.count_nonzero(second["marked"])),
        "second_nonripped_values": {
            "minimum": float(second["val"][second["rip"] == 0].min()),
            "maximum": float(second["val"][second["rip"] == 0].max()),
        },
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("newly_ripped", "first_to_second")}, indent=2))
    print("newly_ripped_count", len(report["newly_ripped"]))
    print("first_to_second", json.dumps(report["first_to_second"]))


if __name__ == "__main__":
    main()
