"""PyTorch Fourier slice timing, following AFNI 3dTshift's default method."""
from pathlib import Path
import math

import nibabel as nib
import numpy as np
import torch


def _fft_length(length):
    candidates = []
    factor = 2
    while factor < 2 * length:
        candidates.extend(factor * odd for odd in (1, 3, 5, 15)
                          if factor * odd >= length)
        factor *= 2
    return min(candidates)


def _fourier_phase(length, shift, device):
    """AFNI uses float32 recursive phase multiplication, not exp(k*angle)."""
    angle = np.float32(np.float32(shift) * np.float32(2 * math.pi / length))
    cosine = np.float32(math.cos(float(angle)))
    sine = np.float32(math.sin(float(angle)))
    real, imaginary = np.float32(1), np.float32(0)
    phase = np.empty(length // 2 + 1, dtype=np.complex64)
    phase[0] = 1
    for index in range(1, len(phase)):
        next_real = np.float32(cosine * real - sine * imaginary)
        next_imaginary = np.float32(cosine * imaginary + sine * real)
        real, imaginary = next_real, next_imaginary
        phase[index] = complex(real, imaginary)
    return torch.as_tensor(phase, device=device)


def slice_timing_correct(source, output, metadata, *, reference_fraction=0.5,
                         ignore=0, device="cpu", voxel_batch_size=4096):
    """Correct inherited BIDS SliceTiming; keep ignored initial volumes unchanged.

    The default target is halfway across the slice acquisition range, rounded
    to milliseconds, as in fMRIPrep 25.2.4. Signal/trend ranges are preserved.
    """
    image = nib.load(str(source))
    tr = float(metadata["RepetitionTime"])
    direction = metadata.get("SliceEncodingDirection", "k")
    if direction not in ("i", "i-", "j", "j-", "k", "k-"):
        raise ValueError("invalid BIDS SliceEncodingDirection")
    axis = "ijk".index(direction[0])
    timings = np.asarray(metadata["SliceTiming"], dtype=np.float64)
    if (image.ndim != 4 or timings.shape != (image.shape[axis],)
            or not np.isfinite(timings).all() or not np.isfinite(tr) or tr <= 0
            or np.any(timings < 0) or np.any(timings >= tr)):
        raise ValueError("SliceTiming must contain one finite acquisition time in [0,TR) per slice")
    if direction.endswith("-"):
        timings = timings[::-1]
    if not 0 <= reference_fraction <= 1 or ignore < 0 or image.shape[3] - ignore < 5:
        raise ValueError("invalid reference_fraction/ignore or fewer than five usable frames")
    if voxel_batch_size < 1:
        raise ValueError("voxel_batch_size must be positive")
    target = float(np.round(timings.min() + reference_fraction * np.ptp(timings), 3))
    data = np.asarray(image.dataobj, dtype=np.float32).copy()
    if not np.isfinite(data).all():
        raise ValueError("BOLD contains nonfinite values")
    view = np.moveaxis(data, axis, 0)
    n = image.shape[3] - ignore
    nfft = _fft_length(image.shape[3] + 4)
    t = torch.arange(n, device=device, dtype=torch.float32)
    for index, timing in enumerate(timings):
        # 3dTshift stores TR, onsets and its fractional shift as float32.
        delta = np.float32(np.float32(np.float32(target) - np.float32(timing)) / np.float32(tr))
        if abs(delta) < .001:
            continue
        phase = _fourier_phase(nfft, delta, device)
        rows = view[index].reshape(-1, image.shape[3]).copy()
        for start in range(0, len(rows), voxel_batch_size):
            stop = min(start + voxel_batch_size, len(rows))
            original = torch.as_tensor(rows[start:stop, ignore:], device=device)
            # OLS trend accumulated in double, with AFNI's float trend storage.
            x0 = original.double().sum(-1, keepdim=True)
            x1 = (original * t).double().sum(-1, keepdim=True)
            intercept = (2 / (n + 1) / n * ((2 * n - 1) * x0 - 3 * x1)).float()
            slope = (-6 / (n * n - 1) / n * ((n - 1) * x0 - 2 * x1)).float()
            trend = intercept + slope * t
            residual = original - trend
            corrected = torch.fft.irfft(torch.fft.rfft(residual, n=nfft) * phase,
                                         n=nfft)[..., :n]
            corrected = corrected.clamp(residual.amin(-1, keepdim=True),
                                          residual.amax(-1, keepdim=True)) + trend
            corrected = corrected.clamp(original.amin(-1, keepdim=True),
                                          original.amax(-1, keepdim=True))
            rows[start:stop, ignore:] = corrected.cpu().numpy()
        view[index] = rows.reshape(view[index].shape)
    header = image.header.copy()
    header.set_data_dtype(np.float32)
    header.set_zooms((*header.get_zooms()[:3], tr))
    header.set_xyzt_units(xyz="mm", t="sec")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, image.affine, header), destination)
    return destination, {"SliceTimingCorrected": True, "StartTime": target,
                         "SliceTimeReference": reference_fraction,
                         "Method": "Fourier, linear detrend/restore, range clipping",
                         "IgnoredInitialVolumes": ignore}
