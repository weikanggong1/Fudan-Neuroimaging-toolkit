"""Summarize three same-input TorchFLIRT runs on the moving image grid."""

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--moving", type=Path, required=True)
    parser.add_argument("--reports", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = [json.loads(path.read_text()) for path in args.reports]
    assert all(run["input_sha256"] == runs[0]["input_sha256"] for run in runs)
    moving = nib.load(str(args.moving))
    assert runs[0]["input_sha256"][args.moving.name] == hashlib.sha256(args.moving.read_bytes()).hexdigest()
    axes = [np.linspace(0, size - 1, 13) for size in moving.shape[:3]]
    voxels = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    points = voxels @ moving.affine[:3, :3].T + moving.affine[:3, 3]
    transformed = [points @ (matrix := np.asarray(run["torch_world"]))[:3, :3].T
                   + matrix[:3, 3] for run in runs]
    pairs = []
    for left, right in ((0, 1), (0, 2), (1, 2)):
        distances = np.linalg.norm(transformed[left] - transformed[right], axis=1)
        pairs.append({
            "runs": [left + 1, right + 1],
            "mean": float(distances.mean()),
            "p95": float(np.percentile(distances, 95)),
            "max": float(distances.max()),
            "rms": float(np.sqrt(np.mean(distances**2))),
        })
    report = {
        "input_sha256": runs[0]["input_sha256"],
        "report_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in args.reports},
        "runs": [{"torch_seconds": run["torch_seconds"],
                  "torch_matrix": run["torch_matrix"],
                  "fsl_grid_displacement_mm": run["grid_displacement_mm"]}
                 for run in runs],
        "pairwise_13cubed_grid_displacement_mm": pairs,
        "maximum_pairwise_displacement_mm": max(item["max"] for item in pairs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pairwise": pairs, "maximum": report["maximum_pairwise_displacement_mm"]}))


if __name__ == "__main__":
    main()
