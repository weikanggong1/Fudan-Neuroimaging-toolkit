"""Time source-derived SIFT2 angular quadrature on a real MRtrix FOD."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from fnit.connectome.fod import real_sh


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("asset", "fod", "proc_mask", "reference_target", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    with np.load(args.asset) as asset:
        directions = torch.from_numpy(asset["directions"]).to(args.device)
        weights = torch.from_numpy(asset["integration_weights"]).to(args.device)
    fod_image = nib.load(args.fod)
    fod = np.asarray(fod_image.dataobj, dtype=np.float32)
    mask_image = nib.load(args.proc_mask)
    mask = np.asarray(mask_image.dataobj, dtype=np.float32) > 0
    target_image = nib.load(args.reference_target)
    target = np.asarray(target_image.dataobj, dtype=np.float32)
    if fod.shape[:3] != mask.shape or mask.shape != target.shape:
        raise ValueError("FOD, ACT processing mask, and MRtrix target grids differ")
    if not np.allclose(fod_image.affine, mask_image.affine) or not np.allclose(fod_image.affine, target_image.affine):
        raise ValueError("FOD, ACT processing mask, and MRtrix target affines differ")
    sh = torch.from_numpy(fod[mask].astype(np.float64)).to(args.device)
    lmax = int(math.sqrt(2 * fod.shape[-1] + 0.25) - 1.5)
    if (lmax + 1) * (lmax + 2) // 2 != fod.shape[-1]:
        raise ValueError("unexpected SH coefficient count")
    basis = real_sh(directions, lmax)
    torch.backends.cuda.matmul.allow_tf32 = True

    def integrate() -> torch.Tensor:
        return torch.cat([part @ basis.T @ weights for part in sh.split(8192)])

    integrate()
    torch.cuda.synchronize()
    times = []
    for _ in range(3):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result = integrate()
        stop.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(stop) / 1000)
    observed = result.cpu().numpy()
    expected = fod[mask, 0].astype(np.float64) * math.sqrt(4 * math.pi)
    error = np.abs(observed - expected)
    official = target[mask]
    valid = np.isfinite(official)
    report = {
        "method": "Float64 GPU angular quadrature on actual corrected ds004666 MRtrix WM FOD; inputs resident on GPU; excludes NIfTI load and transfer",
        "reference_target_note": "MRtrix before_target sums thresholded positive FMLS lobes; signed full-FOD quadrature is a different quantity, so this is diagnostic only",
        "input_sha256": {name: _sha256(getattr(args, name)) for name in ("asset", "fod", "proc_mask", "reference_target")},
        "fod_shape": list(fod.shape),
        "lmax": lmax,
        "active_voxels": int(mask.sum()),
        "official_target_finite_voxels": int(valid.sum()),
        "gpu": torch.cuda.get_device_name(args.device),
        "seconds_repeats": times,
        "seconds_median": statistics.median(times),
        "analytic_sh_integral_mae": float(error.mean()),
        "analytic_sh_integral_max_abs": float(error.max()),
        "official_positive_target_vs_signed_integral_pearson": float(np.corrcoef(observed[valid], official[valid])[0, 1]),
        "official_positive_target_vs_signed_integral_mae": float(np.abs(observed[valid] - official[valid]).mean()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
