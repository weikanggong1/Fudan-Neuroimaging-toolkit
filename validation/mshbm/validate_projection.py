"""用真实 BOLD 的前八帧检查 GPU 三线性采样与 SciPy 插值。"""

import argparse
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import map_coordinates

from fnit.mshbm import load_assets, project_volume
from fnit.mshbm.volume import _surface_points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("volume", "left-surface", "right-surface", "private-output", "report"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    image = nib.load(args.volume)
    data = np.asarray(image.dataobj[..., :8], dtype=np.float32)
    output = Path(args.private_output)
    output.mkdir(parents=True, exist_ok=True)
    subset = output / "eight_real_frames.nii.gz"
    nib.save(nib.Nifti1Image(data, image.affine), subset)
    assets = load_assets()
    points = _surface_points(args.left_surface, args.right_surface)[assets["cortex_mask"]]
    inverse = np.linalg.inv(image.affine)
    coordinates = points @ inverse[:3, :3].T + inverse[:3, 3]
    expected = np.stack([map_coordinates(data[..., i], coordinates.T, order=1,
                                         mode="constant", prefilter=False) for i in range(8)])
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    actual = project_volume(subset, args.left_surface, args.right_surface,
                            assets=assets, device=args.device)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    difference = actual.astype(np.float64) - expected
    report = {"frames": 8, "values": int(actual.size), "sampling_seconds": elapsed,
              "r": float(np.corrcoef(actual.ravel(), expected.ravel())[0, 1]),
              "mae": float(np.mean(np.abs(difference))),
              "max_absolute_difference": float(np.abs(difference).max()),
              "cuda_peak_allocated_gib": (torch.cuda.max_memory_allocated() / 2**30
                                          if args.device.startswith("cuda") else None),
              "oracle": "SciPy map_coordinates order=1; real frames; same RAS coordinates"}
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
