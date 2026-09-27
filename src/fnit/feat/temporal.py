"""FEAT-compatible grand-mean scaling and temporal high-pass filtering.

Inputs and outputs are 4D arrays in NIfTI ``(x, y, z, time)`` order. No FSL
executable or Python neuroimaging workflow package is called at runtime.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch


def grand_mean_scale(data, mask, *, target_median=10000.0):
    """Scale an entire 4D run by ``target_median / masked_50th_percentile``.

    FEAT obtains its scale from ``fslstats 4D -k 3D_mask -p 50``. This pools
    all masked voxels at all time points; it does not take temporal medians
    first. The mask includes values strictly greater than 0.5. FSL's 50th
    percentile is the item at sorted index ``floor(n / 2)``, including when
    ``n`` is even (unlike NumPy's average-of-two median).

    Returns the float32 scaled 4D array and the scalar multiplication factor.
    """
    array = np.asarray(data)
    region = np.asarray(mask)
    if array.ndim != 4 or region.shape != array.shape[:3]:
        raise ValueError("data must be 4D and mask must match its 3D grid")
    if not np.isfinite(target_median) or target_median <= 0:
        raise ValueError("target_median must be positive and finite")
    selected = array[region > 0.5].reshape(-1).copy()
    if selected.size == 0:
        raise ValueError("mask contains no voxels")
    selected.partition(selected.size // 2)
    median = float(selected[selected.size // 2])
    if not np.isfinite(median) or median <= 0:
        raise ValueError("masked 50th percentile must be positive and finite")
    factor = float(target_median) / median
    return np.asarray(array, dtype=np.float32) * np.float32(factor), factor


def gaussian_highpass_matrix(length, sigma_volumes, *, preserve_mean=True):
    """Return the temporal projection for FSL ``-bptf sigma -1``.

    At each time point, a Gaussian-weighted local straight line is fit over
    ``[t - int(3*sigma), t + int(3*sigma)]`` clipped to run boundaries. The
    fitted intercept is subtracted from that point. FSL then removes the
    residual's temporal mean. ``preserve_mean=True`` adds the original voxel
    mean, matching FEAT's subsequent ``-add tempMean`` operation.
    """
    if not isinstance(length, (int, np.integer)) or length < 2:
        raise ValueError("length must be at least 2")
    if not np.isfinite(sigma_volumes) or sigma_volumes <= 0:
        raise ValueError("sigma_volumes must be positive and finite")
    sigma = float(sigma_volumes)
    radius = int(3 * sigma)
    fitted = np.zeros((length, length), dtype=np.float64)
    for t in range(length):
        indices = np.arange(max(0, t - radius), min(length, t + radius + 1))
        offset = indices - t
        weights = np.exp(-0.5 * (offset / sigma) ** 2)
        a = np.sum(weights * offset)
        c = np.sum(weights * offset * offset)
        n = np.sum(weights)
        denominator = c * n - a * a
        if denominator == 0:
            continue
        fitted[t, indices] = weights * (c - a * offset) / denominator
    residual = np.eye(length, dtype=np.float64) - fitted
    residual -= residual.mean(axis=0, keepdims=True)
    if preserve_mean:
        residual += 1.0 / length
    return residual.astype(np.float32)


def gaussian_highpass(data, *, sigma_volumes, device="cpu", voxel_chunk=8192,
                      preserve_mean=True):
    """Filter a 4D run in spatial chunks; return float32 4D data.

    CUDA enables the package TF32 default. The temporal projection itself
    accumulates in float64: TF32 rounds the roughly 10,000-unit baseline and
    changes small BOLD residuals. ``voxel_chunk`` bounds temporary device
    storage; the full image is never copied to the GPU.
    """
    array = np.asarray(data, dtype=np.float32)
    if array.ndim != 4:
        raise ValueError("data must be a 4D array")
    if not isinstance(voxel_chunk, int) or voxel_chunk <= 0:
        raise ValueError("voxel_chunk must be a positive integer")
    computing_device = torch.device(device)
    if computing_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    transform = torch.as_tensor(
        gaussian_highpass_matrix(array.shape[-1], sigma_volumes,
                                 preserve_mean=False),
        dtype=torch.float64, device=computing_device,
    )
    input_flat = array.reshape(-1, array.shape[-1])
    output = np.empty(array.shape, dtype=np.float32)
    output_flat = output.reshape(-1, array.shape[-1])
    with torch.no_grad():
        for start in range(0, input_flat.shape[0], voxel_chunk):
            end = min(start + voxel_chunk, input_flat.shape[0])
            chunk = torch.as_tensor(input_flat[start:end], device=computing_device,
                                    dtype=torch.float64)
            mean = chunk.mean(dim=1, keepdim=True)
            residual = (transform @ (chunk - mean).T).T
            if preserve_mean:
                residual += mean
            output_flat[start:end] = residual.float().cpu().numpy()
    return output


def _save_like(data, reference, output_path):
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    header.set_slope_inter(1.0, 0.0)
    image = nib.Nifti1Image(data, reference.affine, header)
    image.set_qform(reference.get_qform(), int(reference.header["qform_code"]))
    image.set_sform(reference.get_sform(), int(reference.header["sform_code"]))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(path))
    return path


def scale_nifti(input_path, mask_path, output_path, *, target_median=10000.0):
    """Save FEAT grand-mean-scaled NIfTI and return its multiplication factor."""
    source = nib.load(str(input_path))
    mask = nib.load(str(mask_path))
    if not np.allclose(source.affine, mask.affine, atol=1e-4):
        raise ValueError("input and mask must have the same voxel geometry")
    scaled, factor = grand_mean_scale(
        np.asarray(source.dataobj), np.asarray(mask.dataobj),
        target_median=target_median,
    )
    _save_like(scaled, source, output_path)
    return factor


def highpass_nifti(input_path, output_path, *, cutoff_seconds,
                   tr_seconds=None, device="cpu", voxel_chunk=8192,
                   preserve_mean=True):
    """Save a FEAT high-pass-filtered NIfTI and return its output path.

    FEAT's cutoff in seconds maps to the FSL sigma in volumes as
    ``sigma_volumes = cutoff_seconds / (2 * tr_seconds)``.
    """
    source = nib.load(str(input_path))
    time_unit = source.header.get_xyzt_units()[1]
    seconds_per_unit = {"msec": 0.001, "usec": 0.000001}.get(time_unit, 1.0)
    tr = float(source.header.get_zooms()[3] * seconds_per_unit
               if tr_seconds is None else tr_seconds)
    if not np.isfinite(tr) or tr <= 0 or not np.isfinite(cutoff_seconds) or cutoff_seconds <= 0:
        raise ValueError("TR and cutoff_seconds must be positive and finite")
    filtered = gaussian_highpass(
        np.asarray(source.dataobj), sigma_volumes=float(cutoff_seconds) / (2 * tr),
        device=device, voxel_chunk=voxel_chunk, preserve_mean=preserve_mean,
    )
    return _save_like(filtered, source, output_path)
