"""Tissue and motion nuisance regression for 4D functional MRI.

Masks must already be in the BOLD grid. Temporal filtering and regression use
one joint projection, so a removed frequency cannot return after regression.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch


def _load_matrix(value: str | Path | np.ndarray, name: str) -> np.ndarray:
    matrix = np.loadtxt(value, ndmin=2) if isinstance(value, (str, Path)) else np.asarray(value)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite two-dimensional matrix")
    return np.asarray(matrix, dtype=np.float64)


def motion_regressors(motion: str | Path | np.ndarray, model: int = 24) -> np.ndarray:
    """Return the 6, 12, or Friston-24 model from T x 6 realignment parameters.

    The 12-column model adds first differences with a zero first row. The
    24-column model follows the cited MATLAB implementation: current and
    preceding six parameters, then the squares of both sets.
    """
    six = _load_matrix(motion, "motion")
    if six.shape[1] != 6:
        raise ValueError("motion must have six columns")
    if model == 6:
        return six.copy()
    if model == 12:
        previous = np.vstack((six[:1], six[:-1]))
        return np.column_stack((six, six - previous))
    if model == 24:
        previous = np.vstack((np.zeros((1, 6)), six[:-1]))
        return np.column_stack((six, previous, six**2, previous**2))
    raise ValueError("motion model must be 6, 12, or 24")


def _load_bold(input_bold: str | Path) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    image = nib.load(str(input_bold))
    if len(image.shape) != 4 or image.shape[3] < 4:
        raise ValueError("input_bold must be a 4D NIfTI with at least four volumes")
    return image, np.asarray(image.dataobj, dtype=np.float32)


def _mask_on_bold_grid(mask: str | Path, bold: nib.spatialimages.SpatialImage) -> np.ndarray:
    image = nib.load(str(mask))
    if image.shape != bold.shape[:3] or not np.allclose(image.affine, bold.affine, atol=1e-3):
        raise ValueError(f"mask {mask} must already be aligned to the BOLD grid")
    result = np.asarray(image.dataobj) > 0
    if not result.any():
        raise ValueError(f"mask {mask} contains no voxels")
    return result


def _save_bold(output_bold: str | Path, data: np.ndarray, reference: nib.spatialimages.SpatialImage) -> Path:
    output = Path(output_bold)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(data, reference.affine, header), str(output))
    return output


def _device(device: str | torch.device | None) -> torch.device:
    selected = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if selected.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return selected


def _bandpass(matrix: torch.Tensor, keep: torch.Tensor | None) -> torch.Tensor:
    if keep is None:
        return matrix
    spectrum = torch.fft.rfft(matrix, dim=0)
    spectrum[~keep, :] = 0
    return torch.fft.irfft(spectrum, n=matrix.shape[0], dim=0)


def clean_confounds(
    input_bold: str | Path,
    output_bold: str | Path,
    *,
    wm_mask: str | Path | None = None,
    csf_mask: str | Path | None = None,
    brain_mask: str | Path | None = None,
    motion: str | Path | np.ndarray | None = None,
    motion_model: int = 24,
    bandpass: tuple[float, float] | None = None,
    tr: float | None = None,
    global_signal: bool = False,
    device: str | torch.device | None = None,
    chunk_size: int = 4096,
) -> Path:
    """Regress selected confounds, quadratic drift, and optional stopband.

    WM and CSF inputs are already intersected tissue masks on the BOLD grid;
    their mean time series are used. `global_signal=True` additionally uses
    the mean within `brain_mask`. The returned file has the input shape and
    affine and contains mean-zero float32 residuals. It mirrors an uncensored
    single-run 3dTproject call with default ``-polort 2`` and ``-ort``: drift,
    tissue signals, motion, and frequencies are projected out together.
    ``bandpass=(low, high)`` uses Hz and a discrete Fourier-bin projector.
    Projection arithmetic is float64 to avoid TF32 residual drift on long BOLD
    runs; NIfTI input and output remain float32. Nuisance columns are centered
    and normalized before solving, so motion units and tissue baselines do not
    determine which columns survive the numerical rank cutoff.
    """
    image, data = _load_bold(input_bold)
    nt = data.shape[3]
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    regressors: list[np.ndarray] = []
    for tissue in (wm_mask, csf_mask):
        if tissue is not None:
            mask = _mask_on_bold_grid(tissue, image)
            regressors.append(data[mask].mean(axis=0, dtype=np.float64))
    if global_signal:
        if brain_mask is None:
            raise ValueError("brain_mask is required for global_signal=True")
        mask = _mask_on_bold_grid(brain_mask, image)
        regressors.append(data[mask].mean(axis=0, dtype=np.float64))
    if motion is not None:
        motion_columns = motion_regressors(motion, motion_model)
        if motion_columns.shape[0] != nt:
            raise ValueError("motion row count must equal the number of BOLD volumes")
        regressors.extend(motion_columns.T)

    t = np.linspace(-1.0, 1.0, nt)
    design = np.column_stack((np.ones(nt), t, (3 * t**2 - 1) / 2, *regressors))
    if not np.isfinite(design).all():
        raise ValueError("confound design must contain only finite values")
    # Keep the intercept separately. Constant nuisance columns add no new
    # direction; centering other columns preserves the span with the intercept.
    varying = np.r_[True, np.ptp(design[:, 1:], axis=0) > 0]
    design = design[:, varying]
    design[:, 1:] -= design[:, 1:].mean(axis=0)
    column_norms = np.linalg.norm(design, axis=0)
    nonzero = column_norms > 0
    design = design[:, nonzero] / column_norms[nonzero]
    keep = None
    if bandpass is not None:
        if tr is None:
            time_unit = image.header.get_xyzt_units()[1]
            if time_unit not in (None, "unknown", "sec", "msec", "usec"):
                raise ValueError("NIfTI time unit is not a duration; pass tr in seconds")
            scale = {"msec": 1e-3, "usec": 1e-6}.get(time_unit, 1.0)
            tr_seconds = float(image.header.get_zooms()[3]) * scale
        else:
            tr_seconds = tr
        low, high = bandpass
        if tr_seconds <= 0 or not (0 < low < high < 0.5 / tr_seconds):
            raise ValueError("bandpass must satisfy 0 < low < high < Nyquist")
        frequencies = np.fft.rfftfreq(nt, tr_seconds)
        keep_np = (frequencies >= low) & (frequencies <= high)
        if not keep_np.any():
            raise ValueError("bandpass contains no Fourier bin for this time series")

    selected = _device(device)
    if bandpass is not None:
        keep = torch.as_tensor(keep_np, device=selected)
    design_tensor = torch.as_tensor(design, dtype=torch.float64, device=selected)
    filtered_design = _bandpass(design_tensor, keep)
    # A stopband-only column can leave FFT roundoff. Do not normalize that
    # roundoff into an additional passband regressor. Input columns have unit
    # norm, making this tolerance independent of their physical units.
    filtered_norms = torch.linalg.vector_norm(filtered_design, dim=0)
    effective = filtered_norms > np.finfo(np.float64).eps * max(design.shape)
    filtered_design = filtered_design[:, effective] / filtered_norms[effective]
    pseudoinverse = torch.as_tensor(
        np.linalg.pinv(filtered_design.cpu().numpy(), rcond=1e-8),
        dtype=torch.float64,
        device=selected,
    )

    flat = data.reshape((-1, nt))
    output = np.empty_like(flat)
    for start in range(0, flat.shape[0], chunk_size):
        stop = min(start + chunk_size, flat.shape[0])
        series = torch.as_tensor(flat[start:stop].T.copy(), dtype=torch.float64, device=selected)
        filtered = _bandpass(series, keep)
        residual = filtered - filtered_design @ (pseudoinverse @ filtered)
        output[start:stop] = residual.T.cpu().numpy()
    return _save_bold(output_bold, output.reshape(data.shape), image)
