"""Compare fixed-response PyTorch MSMT-CSD with MRtrix on the same DWI.

Example (on the benchmark host, with paths bound to one subject):
python tools/benchmark_connectome_response_fod.py --dwi corrected.nii.gz
    --grad grad.b --wm-response response_wm.txt --gm-response response_gm.txt
    --csf-response response_csf.txt --mask brain_mask.nii.gz
    --ref-wm wm_fod.nii.gz --ref-gm gm.nii.gz --ref-csf csf.nii.gz
    --device cuda:0 --out response_fod.json

MRtrix reference:
dwi2fod msmt_csd corrected.mif response_wm.txt wm_fod.mif
    response_gm.txt gm.mif response_csf.txt csf.mif -mask brain_mask.nii.gz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import nibabel as nib
from nibabel.orientations import apply_orientation, inv_ornt_aff, io_orientation, ornt_transform
import numpy as np
import torch

from fnit.connectome.fod import fit_mrtrix_msmt_csd


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _response(path: Path) -> tuple[np.ndarray, np.ndarray]:
    first = path.open().readline().strip()
    if not first.startswith("# Shells:"):
        raise ValueError(f"missing MRtrix shell header in {path}")
    shells = np.fromstring(first.split(":", 1)[1], sep=",")
    values = np.loadtxt(path, comments="#", ndmin=2)
    return shells, values


def _metrics(candidate: np.ndarray, reference: np.ndarray) -> dict:
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    return {
        "n_values": int(difference.size),
        "mae": float(np.mean(np.abs(difference))),
        "max_abs": float(np.max(np.abs(difference))),
        "fraction_abs_le_1e-5": float(np.mean(np.abs(difference) <= 1e-5)),
        "pearson": float(np.corrcoef(candidate.ravel(), reference.ravel())[0, 1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dwi", "grad", "wm-response", "gm-response", "csf-response",
                 "mask", "ref-wm", "ref-gm", "ref-csf", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset-label", default="OpenNeuro ds004666 sub-01 ses-2mm, FSL TOPUP/EDDY corrected AP DWI")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-voxels", type=int, default=0,
                        help="0 fits the entire mask; otherwise evenly spaced mask voxels")
    args = parser.parse_args()
    paths = {name.replace("-", "_"): getattr(args, name.replace("-", "_"))
             for name in ("dwi", "grad", "wm-response", "gm-response", "csf-response",
                          "mask", "ref-wm", "ref-gm", "ref-csf")}
    start = time.perf_counter()
    dwi_image = nib.load(paths["dwi"])
    data = np.asanyarray(dwi_image.dataobj).astype(np.float32, copy=False)
    mask_image = nib.load(paths["mask"])
    mask = np.asanyarray(mask_image.dataobj).astype(bool)
    if mask.ndim == 4 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    orientation = ornt_transform(
        io_orientation(mask_image.affine), io_orientation(dwi_image.affine),
    )
    reoriented_affine = mask_image.affine @ inv_ornt_aff(orientation, mask.shape)
    affine_error = float(np.max(np.abs(reoriented_affine - dwi_image.affine)))
    if affine_error > 5e-5:
        raise ValueError("mask cannot be aligned to DWI by axis orientation alone")
    mask = np.ascontiguousarray(apply_orientation(mask, orientation))
    if mask.shape != data.shape[:3]:
        raise ValueError("reoriented mask shape differs from DWI")
    eligible = np.flatnonzero(mask)
    if args.max_voxels:
        eligible = eligible[np.linspace(
            0, len(eligible) - 1, min(args.max_voxels, len(eligible)), dtype=np.int64,
        )]
        mask = np.zeros(mask.size, dtype=bool)
        mask[eligible] = True
        mask = mask.reshape(data.shape[:3])
    shell, wm_response = _response(paths["wm_response"])
    gm_shell, gm_response = _response(paths["gm_response"])
    csf_shell, csf_response = _response(paths["csf_response"])
    if not (np.array_equal(shell, gm_shell) and np.array_equal(shell, csf_shell)):
        raise ValueError("MRtrix response shell headers differ")
    gradients = np.loadtxt(paths["grad"])
    device = torch.device(args.device)
    dwi_gpu = torch.as_tensor(data, device=device)
    mask_gpu = torch.as_tensor(mask, device=device)
    load_seconds = time.perf_counter() - start
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    compute_start = time.perf_counter()
    wm, gm, csf = fit_mrtrix_msmt_csd(
        dwi_gpu, gradients, shell, wm_response, gm_response, csf_response,
        mask_gpu, batch_size=args.batch_size,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    compute_seconds = time.perf_counter() - compute_start
    peak_gib = (torch.cuda.max_memory_allocated() / 1024**3
                if device.type == "cuda" else None)
    indices = np.flatnonzero(mask)
    results = {}
    for name, candidate, path in (
        ("wm", wm, paths["ref_wm"]),
        ("gm", gm, paths["ref_gm"]),
        ("csf", csf, paths["ref_csf"]),
    ):
        reference_image = nib.load(path)
        if not np.allclose(dwi_image.affine, reference_image.affine, atol=1e-5):
            raise ValueError(f"{name} reference affine differs from DWI")
        reference = np.asanyarray(reference_image.dataobj)
        if name != "wm" and reference.ndim == 4:
            reference = reference[..., 0]
        if reference.shape != tuple(candidate.shape):
            raise ValueError(f"{name} reference shape differs from candidate")
        candidate_sample = candidate.cpu().numpy().reshape(-1, candidate.shape[-1] if name == "wm" else 1)[indices]
        reference_sample = reference.reshape(-1, reference.shape[-1] if name == "wm" else 1)[indices]
        results[name] = _metrics(candidate_sample, reference_sample)
    report = {
        "dataset": args.dataset_label,
        "scope": "fixed official Dhollander responses; raw dwi2fod msmt_csd output before mtnormalise",
        "reference_command": "dwi2fod msmt_csd DWI response_wm.txt wm_fod.mif response_gm.txt gm.mif response_csf.txt csf.mif -mask MASK",
        "device": str(device),
        "torch": torch.__version__,
        "tf32_enabled": bool(torch.backends.cuda.matmul.allow_tf32),
        "dtype_signal": "float32",
        "dtype_solver": "float64",
        "shape": list(data.shape),
        "selected_voxels": int(len(indices)),
        "mask_orientation_transform": orientation.tolist(),
        "reoriented_mask_affine_max_abs_difference": affine_error,
        "batch_size": args.batch_size,
        "shell_bvals": shell.tolist(),
        "timing": {"load_seconds": load_seconds, "compute_seconds": compute_seconds},
        "peak_torch_allocated_gib": peak_gib,
        "input_sha256": {name: _sha(path) for name, path in paths.items()},
        "metrics": results,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected_voxels": len(indices), "compute_seconds": compute_seconds,
                      "peak_gib": peak_gib, "metrics": results}, indent=2))


if __name__ == "__main__":
    main()
