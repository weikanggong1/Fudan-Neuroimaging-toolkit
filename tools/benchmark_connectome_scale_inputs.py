"""Verify numerical image and affine equality of converted MRtrix/FNIT inputs."""

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fnit-fod", "mrtrix-fod", "fnit-five", "mrtrix-five",
                 "fnit-gmwmi", "mrtrix-gmwmi", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report = {}
    for name in ("fod", "five", "gmwmi"):
        fnit_path = getattr(args, "fnit_" + name)
        mrtrix_path = getattr(args, "mrtrix_" + name)
        fnit = nib.load(str(fnit_path))
        mrtrix = nib.load(str(mrtrix_path))
        if fnit.shape != mrtrix.shape:
            raise ValueError(f"{name} image shapes differ: {fnit.shape} vs {mrtrix.shape}")
        a = np.asarray(fnit.dataobj, dtype=np.float32)
        b = np.asarray(mrtrix.dataobj, dtype=np.float32)
        corners = np.asarray([(x, y, z, 1.) for x in (0, a.shape[0] - 1)
                              for y in (0, a.shape[1] - 1)
                              for z in (0, a.shape[2] - 1)])
        displaced = corners @ (fnit.affine - mrtrix.affine).T
        report[name] = {
            "shape": list(a.shape),
            "geometry_equal_at_1e-5": bool(np.allclose(fnit.affine, mrtrix.affine, atol=1e-5)),
            "affine_max_abs_difference": float(np.max(np.abs(fnit.affine - mrtrix.affine))),
            "max_corner_world_displacement_mm": float(np.linalg.norm(displaced[:, :3], axis=1).max()),
            "voxel_max_abs_difference": float(np.max(np.abs(a - b))),
            "voxels_above_1e-6": int(np.count_nonzero(np.abs(a - b) > 1e-6)),
            "fnit_sha256": hashlib.sha256(fnit_path.read_bytes()).hexdigest(),
            "mrtrix_converted_sha256": hashlib.sha256(mrtrix_path.read_bytes()).hexdigest(),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
