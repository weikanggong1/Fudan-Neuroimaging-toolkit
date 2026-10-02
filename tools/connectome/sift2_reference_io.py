"""CPU readers and unchanged comparison formulas for fixed-TCK reference tools."""
from __future__ import annotations
import hashlib
import numpy as np
import nibabel as nib
import torch

def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def read_geometry(path):
    """Read pipeline-exported matrices without a NIfTI-1 header round trip."""
    with np.load(path, allow_pickle=False) as geometry:
        matrices = []
        for key in ("dwi_affine", "five_tissue_affine", "dwi_affine"):
            matrix = geometry[key]
            if matrix.dtype != np.float64 or matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                raise ValueError(f"{key} must be finite float64 [4,4]")
            matrices.append(matrix.copy())
    # Both normalized FOD and FA use the pipeline's original DWI affine.
    return matrices

def read_actual_inputs(tracking_path, geometry_path, wm_path, five_path, fa_path):
    checkpoint = torch.load(tracking_path, map_location="cpu", weights_only=True)
    matrices = read_geometry(geometry_path)
    for key, matrix in zip(("fod_affine", "five_tissue_affine"), matrices):
        tensor = checkpoint[key]
        if tensor.dtype != torch.float64 or not np.array_equal(tensor.numpy().view(np.uint64), matrix.view(np.uint64)):
            raise ValueError(f"PT and geometry.npz {key} do not match exactly")
    arrays = []
    for key, path in (("wm_sh", wm_path), ("five_tissue", five_path)):
        tensor = checkpoint[key]
        if tensor.dtype != torch.float32:
            raise ValueError(f"{key} must retain original float32 dtype")
        exported = nib.load(str(path)).get_fdata(dtype=np.float32)
        actual = tensor.numpy()
        if not np.array_equal(actual.view(np.uint32), exported.view(np.uint32)):
            raise ValueError(f"{key} export differs from original PT tensor")
        arrays.append(actual)
    # Current pipeline samples FA after tracking, so tracking PT may have fa=None.
    # NIfTI stores the scalar voxel values exactly; its header affine is unused.
    exported_fa = nib.load(str(fa_path)).get_fdata(dtype=np.float32)
    if checkpoint.get("fa") is not None:
        actual_fa = checkpoint["fa"].numpy()
        if not np.array_equal(actual_fa.view(np.uint32), exported_fa.view(np.uint32)):
            raise ValueError("FA export differs from original PT tensor")
        arrays.append(actual_fa)
    else:
        arrays.append(exported_fa)
    spacing = checkpoint.get("five_tissue_spacing_mm")
    return arrays, matrices, spacing

def metrics(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"comparison shape mismatch: {a.shape} vs {b.shape}")
    finite = np.isfinite(a) & np.isfinite(b)
    error = np.abs(a[finite].astype(np.float64) - b[finite].astype(np.float64))
    left = a[finite].astype(np.float64)
    right = b[finite].astype(np.float64)
    correlation = (float(np.corrcoef(left, right)[0, 1])
                   if len(left) > 1 and np.std(left) > 0 and np.std(right) > 0 else None)
    return dict(corr=correlation, correlation_finite_pairs=int(np.count_nonzero(finite)),
                neq=int(np.count_nonzero(~((a == b) | (np.isnan(a) & np.isnan(b))))),
                nonfinite_mismatch=int(np.count_nonzero(np.isfinite(a) != np.isfinite(b))),
                max=float(error.max(initial=0)), p99=float(np.percentile(error, 99)) if len(error) else 0.,
                rmse=float(np.sqrt(np.mean(error ** 2))) if len(error) else 0.)
