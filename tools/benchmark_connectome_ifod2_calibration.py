"""真实 WM FOD 几何的 iFOD2 校准方向和拒绝比例对照。"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from fnit.connectome.tracking import _ifod2_calibration


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fod", type=Path, required=True, help="真实归一化 WM FOD NIfTI")
    parser.add_argument("--official", type=Path, required=True, help="官方校准方向参考文本")
    parser.add_argument("--output", type=Path, required=True, help="输出指标 JSON")
    parser.add_argument("--device", default="cuda:0", help="PyTorch 输出设备")
    args = parser.parse_args()
    image = nib.load(str(args.fod))
    if image.ndim != 4 or image.shape[-1] != 45:
        raise ValueError("expected real lmax=8 WM FOD")
    spacing = image.header.get_zooms()[:3]
    voxel_mm = float(np.prod(spacing) ** (1 / 3))
    step_mm = voxel_mm / 2
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.empty(1, device=device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    directions, ratio = _ifod2_calibration(8, step_mm, voxel_mm, 45., .5, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    with args.official.open() as stream:
        official_ratio, official_count = stream.readline().split()
        official_directions = np.loadtxt(stream)
    candidate = directions.cpu().numpy()
    if candidate.shape != official_directions.shape or len(candidate) != int(official_count):
        raise ValueError("official and FNIT calibration direction counts differ")
    report = {
        "dataset": "OpenNeuro ds004666 sub-01 ses-2mm real normalised WM FOD geometry",
        "mrtrix_commit": "eeab681d3e0cb004cf1d1d31579d3892197ef5b6",
        "input_sha256": {"fod": _sha256(args.fod), "official": _sha256(args.official)},
        "fod_shape": image.shape, "voxel_mm": voxel_mm, "step_mm": step_mm,
        "device": str(device), "tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "official_count": int(official_count), "fnit_count": len(candidate),
        "official_ratio": float(official_ratio), "fnit_ratio": ratio,
        "ratio_abs_error": abs(ratio - float(official_ratio)),
        "direction_max_abs_error": float(np.abs(candidate - official_directions).max()),
        "core_seconds": elapsed,
        "torch_peak_allocated_gib": (torch.cuda.max_memory_allocated(device) / 2**30
                                     if device.type == "cuda" else None),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    np.savetxt(args.output.with_suffix(".csv"),
               np.column_stack((official_directions, candidate)), delimiter=",")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
