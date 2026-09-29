"""Shared image loading, timing and edge metrics for real connectome benchmarks."""

import hashlib
from pathlib import Path

import nibabel as nib
import numpy as np
import torch


NAMES = ("count", "sift2_fbc", "mean_length", "mean_fa")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load(path: Path, device: torch.device, dtype=torch.float32):
    image = nib.load(str(path))
    data = torch.as_tensor(np.asarray(image.dataobj).copy(), dtype=dtype, device=device)
    affine = torch.as_tensor(image.affine, dtype=torch.float64, device=device)
    return data, affine


def _metrics(candidate: np.ndarray, reference: np.ndarray,
             support: tuple[np.ndarray, np.ndarray]) -> dict:
    edge = np.triu_indices(candidate.shape[0], 1)
    a, b = candidate[edge], reference[edge]
    common = support[0][edge] & support[1][edge]
    return {
        "pearson_upper": float(np.corrcoef(a, b)[0, 1]),
        "relative_l1_upper": float(np.abs(a - b).sum() / np.abs(b).sum()),
        "normalized_mae_common_support": (
            float(np.abs(a[common] - b[common]).mean() / np.abs(b[common]).mean())
            if common.any() and np.abs(b[common]).sum() else None
        ),
        "support_dice": float(2 * common.sum() / (support[0][edge].sum() + support[1][edge].sum())),
    }
