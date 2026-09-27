"""Compare PyTorch and MRtrix mtnormalise on identical real three-tissue FODs.

Convert the six MRtrix inputs/outputs using ``mrconvert input.mif out.nii.gz``
without resampling, then pass their NIfTI paths below. The reference command is
``mtnormalise wm.mif wm_norm.mif gm.mif gm_norm.mif csf.mif csf_norm.mif
-mask mask.mif``. Runtime measures the already-loaded GPU function only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from fnit.connectome.mtnormalise import normalise_mrtrix_three_tissue


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(candidate: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> dict:
    error = candidate[mask].astype(np.float64) - reference[mask].astype(np.float64)
    a = candidate[mask].astype(np.float64)
    b = reference[mask].astype(np.float64)
    return {
        "n": int(len(error)),
        "mae": float(np.abs(error).mean()),
        "max_absolute_error": float(np.abs(error).max()),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "pearson": float(np.corrcoef(a, b)[0, 1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("wm", "gm", "csf", "wm_ref", "gm_ref", "csf_ref", "mask"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--example-png", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    paths = {name: getattr(args, name) for name in
             ("wm", "gm", "csf", "wm_ref", "gm_ref", "csf_ref", "mask")}
    images = {name: nib.load(str(path)) for name, path in paths.items()}
    geometry = images["wm"].affine
    if any(not np.allclose(image.affine, geometry, atol=1e-5) for image in images.values()):
        raise ValueError("all input and reference images must share an affine")
    arrays = {name: np.asarray(image.dataobj, dtype=np.float32) for name, image in images.items()}
    for name in ("gm", "csf", "gm_ref", "csf_ref"):
        if arrays[name].ndim == 4 and arrays[name].shape[-1] == 1:
            arrays[name] = arrays[name][..., 0]
    shape = arrays["wm"].shape[:3]
    if arrays["wm"].ndim != 4 or any(arrays[name].shape != shape for name in
                                      ("gm", "csf", "gm_ref", "csf_ref", "mask")):
        raise ValueError("wrong three-tissue image dimensions")
    if arrays["wm_ref"].shape != arrays["wm"].shape:
        raise ValueError("WM input and reference SH channels differ")
    mask = arrays["mask"] > 0
    tensors = {name: torch.from_numpy(arrays[name].copy()).to(device)
               for name in ("wm", "gm", "csf")}
    mask_tensor = torch.from_numpy(mask).to(device)
    affine = torch.as_tensor(geometry, device=device, dtype=torch.float64)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = normalise_mrtrix_three_tissue(
        tensors["wm"], tensors["gm"], tensors["csf"], mask_tensor, affine
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    outputs = {name: getattr(result, name).cpu().numpy() for name in ("wm", "gm", "csf", "field")}
    within = {"wm": np.broadcast_to(mask[..., None], outputs["wm"].shape),
              "gm": mask, "csf": mask}
    report = {
        "comparison": "same raw real-data FODs and mask; official MRtrix mtnormalise defaults",
        "inputs_sha256": {name: _hash(path) for name, path in paths.items()},
        "mask_voxels": int(mask.sum()),
        "device": str(device),
        "dtype": "float32 inputs/outputs; float64 optimizer; GPU TF32 permitted",
        "pytorch_seconds_loaded": elapsed,
        "pytorch_peak_cuda_gib": (torch.cuda.max_memory_allocated(device) / 2**30
                                  if device.type == "cuda" else None),
        "balance_factors": result.balance_factors.cpu().tolist(),
        "accepted_voxels": int(result.accepted_mask.sum()),
        "metrics_masked": {name: _metrics(outputs[name], arrays[name + "_ref"], within[name])
                           for name in ("wm", "gm", "csf")},
        "metrics_full_volume": {name: _metrics(outputs[name], arrays[name + "_ref"],
                                              np.ones_like(outputs[name], dtype=bool))
                                for name in ("wm", "gm", "csf")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.example_png is not None:
        import matplotlib.pyplot as plt
        z = shape[2] // 2
        reference = arrays["wm_ref"][:, :, z, 0]
        candidate = outputs["wm"][:, :, z, 0]
        limit = np.quantile(reference[mask[:, :, z]], .99)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        for ax, image, title in zip(axes, (reference, candidate, candidate-reference),
                                    ("MRtrix WM c00", "PyTorch WM c00", "PyTorch minus MRtrix")):
            ax.imshow(image.T, origin="lower", cmap="gray" if ax != axes[2] else "coolwarm",
                      vmin=0 if ax != axes[2] else -limit * .05,
                      vmax=limit if ax != axes[2] else limit * .05)
            ax.set_title(title)
            ax.axis("off")
        args.example_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.example_png, dpi=160)
        plt.close(fig)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
