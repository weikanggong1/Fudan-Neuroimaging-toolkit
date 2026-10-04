"""Build a BOLD reference using audited NiWorkflows selection and intensity steps.

Reference registration reuses FNIT's TorchMCFLIRT. It is deliberately identified
separately from NiWorkflows' default AFNI Fourier registration; matching reference
selection does not establish numerical equivalence of these motion backends.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch

from ..mcflirt import TorchMCFLIRT


@dataclass(frozen=True)
class ReferenceSelection:
    selected_indices: tuple[int, ...]
    algorithm_dummy_scans: int
    skip_vols: int
    global_signal: tuple[float, ...]


@dataclass(frozen=True)
class BoldReferenceResult:
    reference: Path
    selected_volumes: Path
    metadata: Path
    selected_indices: tuple[int, ...]
    algorithm_dummy_scans: int
    skip_vols: int
    drift: tuple[float, ...]
    timing_seconds: dict[str, float]


def _image(value):
    image = value if isinstance(value, nib.spatialimages.SpatialImage) else nib.load(str(value))
    if image.ndim not in (3, 4) or min(image.shape[:3]) < 1:
        raise ValueError("bold must be a nonempty 3D or 4D NIfTI image")
    if image.ndim == 4 and image.shape[3] < 1:
        raise ValueError("bold must contain at least one frame")
    if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-12:
        raise ValueError("bold must have a finite, invertible voxel-to-world affine")
    return image


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _clipped(values, nonnegative):
    # NumPy float32 reductions/percentile order intentionally match the audited
    # upstream intensity predicates; geometry and reference motion use PyTorch.
    return np.clip(values, 0.0 if nonnegative else np.percentile(values, .2),
                   np.percentile(values, 99.8))


def _initial_outlier_count(global_signal):
    # Keep the upstream float32 Euclidean-distance operation order, including
    # sqrt(square), rather than substituting abs near the decision threshold.
    distance = np.sqrt((global_signal - np.median(global_signal)) ** 2)
    mad = np.median(distance)
    if mad == 0:
        # Upstream divides by zero with no epsilon: both Inf and NaN fail its
        # `score <= 3.5` stopping condition. Preserve that selection explicitly.
        return len(global_signal)
    score = .6745 * distance / mad
    count = 0
    for value in score:
        if value <= 3.5:
            break
        count += 1
    return count


def select_reference_volumes(bold, *, dummy_scans=None, n_volumes=40,
                             zero_dummy_masked=20, nonnegative=True):
    """Select reference frames; ``dummy_scans`` only overrides ``skip_vols``.

    Clip the first ``n_volumes`` float32 frames, detect consecutive initial
    global-signal MAD outliers, then select those initial frames when the count
    is at least two. Otherwise select the last ``zero_dummy_masked`` frames of
    that detection window. This mask is not an instruction to discard BOLD.
    """
    _positive_integer(n_volumes, "n_volumes")
    _positive_integer(zero_dummy_masked, "zero_dummy_masked")
    if not isinstance(nonnegative, (bool, np.bool_)):
        raise ValueError("nonnegative must be boolean")
    image = _image(bold)
    frames = image.shape[3] if image.ndim == 4 else 1
    if dummy_scans is not None and (
        isinstance(dummy_scans, bool) or not isinstance(dummy_scans, (int, np.integer))
        or dummy_scans < 0 or dummy_scans > frames
    ):
        raise ValueError("dummy_scans must be None or an integer between 0 and the frame count")
    if frames == 1:
        indices, count, signal = (0,), 1, ()
    else:
        values = np.asarray(image.dataobj[..., :n_volumes], dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("the nonsteady-state detection window contains nonfinite values")
        signal_array = np.mean(_clipped(values, nonnegative), axis=(0, 1, 2))
        count = _initial_outlier_count(signal_array)
        stop = count if count >= 2 else min(frames, n_volumes)
        start = 0 if count >= 2 else max(0, stop - zero_dummy_masked)
        indices = tuple(range(start, stop))
        signal = tuple(float(value) for value in signal_array)
    return ReferenceSelection(indices, count, count if dummy_scans is None else int(dummy_scans), signal)


def _save(values, source, path):
    header = source.header.copy()
    header.extensions.clear()
    header.set_data_dtype(np.float32)
    header.set_slope_inter(1, 0)
    nib.save(nib.Nifti1Image(np.asarray(values, dtype=np.float32), source.affine, header), str(path))


def _median(values, device, spatial_chunk_size):
    frames = values.shape[3]
    flat = values.reshape(-1, frames)
    output = np.empty(flat.shape[0], dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(flat), spatial_chunk_size):
            chunk = torch.as_tensor(flat[start:start + spatial_chunk_size], device=device)
            ordered = torch.sort(chunk, dim=1).values
            middle = ordered[:, frames // 2]
            if frames % 2 == 0:
                # torch.median alone picks the lower middle; NumPy/upstream uses
                # the arithmetic average of the two central observations.
                middle = (ordered[:, frames // 2 - 1] + middle) / 2
            output[start:start + len(chunk)] = middle.cpu().numpy()
    return output.reshape(values.shape[:3])


def _synchronized_time(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def prepare_bold_reference(bold, output_dir, *, device=None, dummy_scans=None,
                           n_volumes=40, zero_dummy_masked=20, nonnegative=True,
                           motion_correction=True, stage_iterations=(1, 1, 1),
                           spatial_chunk_size=262144, overwrite=False):
    """Save a robust raw-BOLD HMC target and its selected-frame provenance.

    The default motion backend is FNIT TorchMCFLIRT, referencing the first
    selected, drift-normalized frame with cubic-spline output. ``False`` skips
    reference motion and is useful for isolating selection/intensity statistics.
    No frame is removed from the input series. SBRef is a separate coregistration
    reference in fMRIPrep and must not silently replace this raw-BOLD HMC target.
    """
    _positive_integer(spatial_chunk_size, "spatial_chunk_size")
    if not isinstance(motion_correction, (bool, np.bool_)):
        raise ValueError("motion_correction must be boolean")
    iterations = tuple(stage_iterations)
    if len(iterations) != 3 or any(isinstance(v, bool) or not isinstance(v, (int, np.integer))
                                  or v < 0 for v in iterations):
        raise ValueError("stage_iterations must contain three nonnegative integers")
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    image = _image(bold)
    output = Path(output_dir).expanduser().resolve()
    reference = output / "bold_reference.nii.gz"
    selected_volumes = output / "bold_reference_selected.nii.gz"
    metadata = output / "bold_reference.json"
    paths = (reference, selected_volumes, metadata)
    if any(path.is_symlink() for path in paths):
        raise ValueError("reference output files must not be symlinks")
    source_filename = (image.get_filename()
                       if isinstance(bold, nib.spatialimages.SpatialImage) else bold)
    if source_filename is not None:
        source_path = Path(source_filename).expanduser().resolve()
        if source_path in paths or any(
            path.exists() and source_path.exists() and path.samefile(source_path)
            for path in paths
        ):
            raise ValueError("reference output must not overwrite its input image")
    if any(path.exists() for path in paths) and not overwrite:
        raise FileExistsError("reference outputs already exist; use overwrite=True")
    # There is no directory-exists cache shortcut. An interrupted overwrite
    # must not leave a former request's complete manifest looking current.
    if overwrite and metadata.is_file():
        metadata.unlink()
    started = _synchronized_time(selected_device)
    selection = select_reference_volumes(image, dummy_scans=dummy_scans, n_volumes=n_volumes,
                                         zero_dummy_masked=zero_dummy_masked, nonnegative=nonnegative)
    data = np.asarray(image.dataobj, dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError("bold must contain only finite values")
    if image.ndim == 3:
        data = data[..., None]
    # concat_images preserves the selected-volume layout used by the audited
    # upstream. Different strides can change float32 spatial mean reductions.
    sliced = nib.concat_images(
        nib.Nifti1Image(data[..., index], image.affine, image.header.copy())
        for index in selection.selected_indices
    )
    raw_selected = sliced.get_fdata(dtype="float32")
    finished = _synchronized_time(selected_device)
    timing = {"selection_and_loading": finished - started}
    if data.shape[3] == 1:
        # NiWorkflows returns a 3D/squeezed single-frame image directly, before
        # clipping or normalization. In particular an all-zero 3D image is valid.
        values, drift = raw_selected.copy(), np.ones(1, dtype=np.float32)
    else:
        values = _clipped(raw_selected, nonnegative)
        drift = np.mean(values, axis=(0, 1, 2))
        maximum = drift.max()
        if maximum == 0 or np.any(drift == 0):
            raise ValueError("selected reference frames have zero global signal; drift normalization is undefined")
        drift /= maximum
        values /= drift
        values = np.clip(values, 0.0 if nonnegative else values.min(), values.max())
    normalized = nib.Nifti1Image(values, image.affine, image.header.copy())
    normalized.set_data_dtype(np.float32)
    timing["clipping_and_drift"] = _synchronized_time(selected_device) - finished
    finished = _synchronized_time(selected_device)
    matrices, parameters, cost_evaluations = None, None, 0
    if len(selection.selected_indices) == 1:
        # Upstream's single-selected-volume output is the original sliced image,
        # even though its saved out_volumes is clipped/normalized beforehand.
        final = raw_selected[..., 0]
        drift = np.ones(1, dtype=np.float32)
    else:
        if motion_correction:
            target = nib.Nifti1Image(values[..., 0], image.affine, image.header.copy())
            target.set_data_dtype(np.float32)
            fit = TorchMCFLIRT(device=selected_device).run(
                normalized, reference=target, stages=3, stage_iterations=iterations,
                resample=True, interpolation="spline",
            )
            values = np.asarray(fit.corrected.dataobj, dtype=np.float32)
            matrices, parameters = fit.matrices.tolist(), fit.parameters.tolist()
            cost_evaluations = int(fit.cost_evaluations)
        if values.shape != raw_selected.shape or not np.isfinite(values).all():
            raise ValueError("reference motion produced an invalid selected series")
        values = np.clip(values, 0.0 if nonnegative else values.min(), values.max())
        final = _median(values, selected_device, spatial_chunk_size)
    timing["reference_motion_and_median"] = _synchronized_time(selected_device) - finished
    output.mkdir(parents=True, exist_ok=True)
    _save(np.asarray(normalized.dataobj), image, selected_volumes)
    _save(final, image, reference)
    timing["total"] = _synchronized_time(selected_device) - started
    description = {
        "status": "complete", "strategy": "robust", "input_frames": data.shape[3],
        "selected_indices": list(selection.selected_indices),
        "algorithm_dummy_scans": selection.algorithm_dummy_scans,
        "skip_vols": selection.skip_vols, "discarded_input_frames": 0,
        "global_signal_detection_window": list(selection.global_signal),
        "drift": [float(value) for value in drift],
        "parameters": {"n_volumes": int(n_volumes), "zero_dummy_masked": int(zero_dummy_masked),
                       "nonnegative": bool(nonnegative),
                       "dummy_scans": None if dummy_scans is None else int(dummy_scans),
                       "motion_correction": bool(motion_correction),
                       "stage_iterations": [int(value) for value in iterations],
                       "spatial_chunk_size": int(spatial_chunk_size)},
        "reference_motion": {"backend": "fnit.TorchMCFLIRT" if motion_correction and len(selection.selected_indices) > 1 else None,
                             "reference_selected_index": 0, "interpolation": "spline",
                             "matrices": matrices, "parameters": parameters,
                             "cost_evaluations": cost_evaluations},
        "upstream": {"fmriprep": "25.2.4", "niworkflows": "1.14.4",
                     "default_motion": "AFNI Volreg Fourier, twopass, zpad=4",
                     "motion_backend_equivalence": "not_assessed"},
        "device": str(selected_device), "dtype": "float32", "timing_seconds": timing,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "outputs": {path.name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                "bytes": path.stat().st_size} for path in paths[:2]},
    }
    metadata.write_text(json.dumps(description, indent=2) + "\n")
    return BoldReferenceResult(reference, selected_volumes, metadata, selection.selected_indices,
                               selection.algorithm_dummy_scans, selection.skip_vols,
                               tuple(float(value) for value in drift), timing)
