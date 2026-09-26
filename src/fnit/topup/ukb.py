"""Prepare the AP/PA b0 pair expected by UK Biobank TOPUP."""

from __future__ import annotations

import json
import math
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from .core import TorchTOPUP, _grid, _rigid_coordinates, _sample


_PE_VECTORS = {
    "i": (1, 0, 0),
    "i-": (-1, 0, 0),
    "j": (0, 1, 0),
    "j-": (0, -1, 0),
    "k": (0, 0, 1),
    "k-": (0, 0, -1),
}


def _load_b0_candidates(raw_dir, stem):
    raw_dir = Path(raw_dir)
    image = nib.load(str(raw_dir / f"{stem}.nii.gz"))
    data = np.asarray(image.dataobj, dtype=np.float32)
    bvals = np.loadtxt(raw_dir / f"{stem}.bval", dtype=np.float64).reshape(-1)
    if data.ndim != 4 or data.shape[3] != bvals.size:
        raise ValueError(f"{stem}.bval does not match {stem}.nii.gz")
    indices = np.flatnonzero(bvals < 100)
    if not indices.size:
        raise ValueError(f"{stem} contains no b<100 volume")
    return image, data[..., indices], indices


def _registered_correlation(moving, fixed, voxel_sizes, device):
    moving = torch.as_tensor(moving.copy(), dtype=torch.float32, device=device)
    fixed = torch.as_tensor(fixed.copy(), dtype=torch.float32, device=device)
    factor = 2 if min(moving.shape) >= 16 else 1
    if factor > 1:
        def pool(value):
            value = value[None, None].permute(0, 1, 4, 3, 2)
            return F.avg_pool3d(value, factor, factor)[0, 0].permute(2, 1, 0)
        moving, fixed = pool(moving), pool(fixed)
    sizes = tuple(float(value) * factor for value in voxel_sizes)
    grid = _grid(fixed.shape, device=device, dtype=torch.float32)
    parameters = torch.zeros(6, device=device, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [parameters],
        lr=0.5,
        max_iter=20,
        history_size=8,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-6,
        tolerance_change=1e-7,
    )
    latest = {}

    def closure():
        optimizer.zero_grad(set_to_none=True)
        coordinates = _rigid_coordinates(grid, parameters, sizes)
        sampled, valid = _sample(moving, coordinates)
        mask = valid & (fixed != 0) & (sampled != 0)
        left = sampled[mask] - sampled[mask].mean()
        right = fixed[mask] - fixed[mask].mean()
        correlation = torch.dot(left, right) / (
            torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
        ).clamp_min(torch.finfo(torch.float32).eps)
        (1 - correlation).backward()
        latest["correlation"] = float(correlation.detach())
        return 1 - correlation

    optimizer.step(closure)
    return latest["correlation"]


def _best_b0(candidates, voxel_sizes, device):
    """Apply UKB's pairwise-rigid first-if-0.98-else-best rule."""
    count = candidates.shape[3]
    if count == 1:
        return 0, np.ones(1, dtype=np.float64)
    scores = np.zeros(count, dtype=np.float64)
    for left in range(count):
        for right in range(left + 1, count):
            correlation = _registered_correlation(
                candidates[..., left],
                candidates[..., right],
                voxel_sizes,
                device,
            )
            scores[left] += correlation
            scores[right] += correlation
    scores /= count - 1
    return (0 if scores[0] >= 0.98 else int(np.argmax(scores))), scores


def _metadata(raw_dir, stem, shape):
    with (Path(raw_dir) / f"{stem}.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    direction = metadata.get("PhaseEncodingDirection")
    if direction not in _PE_VECTORS:
        raise ValueError(f"unsupported {stem} PhaseEncodingDirection: {direction!r}")
    if "TotalReadoutTime" in metadata:
        readout = float(metadata["TotalReadoutTime"])
    elif "EffectiveEchoSpacing" in metadata:
        axis = int(np.flatnonzero(np.abs(_PE_VECTORS[direction]))[0])
        readout = float(metadata["EffectiveEchoSpacing"]) * (shape[axis] - 1)
    else:
        raise ValueError(
            f"{stem}.json needs TotalReadoutTime or EffectiveEchoSpacing"
        )
    readout = math.floor(readout * 10000.0) / 10000.0
    return _PE_VECTORS[direction], readout


def prepare_ukb_topup(raw_dir, output_dir, *, device=None, overwrite=False):
    """Create ``B0_AP_PA.nii.gz`` and ``acqparams.txt`` from UKB raw dMRI."""
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    required = [raw_dir / f"{stem}.{suffix}" for stem in ("AP", "PA")
                for suffix in ("nii.gz", "bval", "json")]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    ap_image, ap_candidates, ap_indices = _load_b0_candidates(raw_dir, "AP")
    pa_image, pa_candidates, pa_indices = _load_b0_candidates(raw_dir, "PA")
    if ap_candidates.shape[:3] != pa_candidates.shape[:3]:
        raise ValueError("AP and PA images must have the same matrix size")
    if not np.allclose(ap_image.affine, pa_image.affine, atol=5e-4, rtol=0):
        raise ValueError("AP and PA images must use the same voxel-to-world geometry")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    voxel_sizes = tuple(float(value) for value in ap_image.header.get_zooms()[:3])
    ap_best, ap_scores = _best_b0(
        ap_candidates, voxel_sizes, device
    )
    pa_best, pa_scores = _best_b0(
        pa_candidates, voxel_sizes, device
    )
    selected = np.stack(
        (ap_candidates[..., ap_best], pa_candidates[..., pa_best]), axis=3
    )
    if selected.shape[2] % 2:
        selected = selected[:, :, :-1]
    ap_pe, ap_readout = _metadata(raw_dir, "AP", selected.shape[:3])
    pa_pe, pa_readout = _metadata(raw_dir, "PA", selected.shape[:3])
    acquisition = np.asarray(
        ((*ap_pe, ap_readout), (*pa_pe, pa_readout)), dtype=np.float64
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "B0_AP_PA.nii.gz"
    acquisition_path = output_dir / "acqparams.txt"
    if not overwrite:
        existing = [path for path in (image_path, acquisition_path) if path.exists()]
        if existing:
            raise FileExistsError(f"output exists: {existing[0]}; pass overwrite=True")
    header = ap_image.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(selected.astype(np.float32), ap_image.affine, header), image_path)
    np.savetxt(acquisition_path, acquisition, fmt=("%g", "%g", "%g", "%.7g"))
    return {
        "imain": image_path,
        "datain": acquisition_path,
        "ap_index": int(ap_indices[ap_best]),
        "pa_index": int(pa_indices[pa_best]),
        "ap_scores": ap_scores.tolist(),
        "pa_scores": pa_scores.tolist(),
    }


def run_ukb_topup(
    raw_dir,
    output_dir,
    *,
    device=None,
    overwrite=False,
):
    """Prepare and run one UKB-format AP/PA acquisition."""
    output_dir = Path(output_dir)
    prepared = prepare_ukb_topup(
        raw_dir, output_dir, device=device, overwrite=overwrite
    )
    result = TorchTOPUP(device=device).run(
        prepared["imain"],
        prepared["datain"],
        out=output_dir / "fieldmap_out",
        fout=output_dir / "fieldmap_fout",
        iout=output_dir / "fieldmap_iout",
        jacout=output_dir / "fieldmap_jacout",
        overwrite=overwrite,
    )
    return result, prepared


__all__ = ["prepare_ukb_topup", "run_ukb_topup"]
