"""Pinned MRtrix iFOD2/ACT tracking, built and managed independently by FNIT.

The native executable is resolved from FNIT's verified cache, never from PATH.
PyTorch remains responsible for the surrounding FOD, SIFT2 and matrix stages.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

import nibabel as nib
import numpy as np
import torch


@dataclass
class Tractogram:
    """World-mm paths ``[Pi,3]`` and their quantities in TCK record order.

    ``endpoints`` is float32 ``[T,2,3]``; ``lengths_mm`` is float32 ``[T]``.
    ``mean_fa`` is optional float32 ``[T]`` from precise voxel crossing.
    ``seeds_attempted`` retains the API's requested seed budget, rather than
    the TCK ``total_count`` of generated candidates. Native TCK does not store
    the original seed points: ``accepted_seeds=None`` represents unknown data.
    ``native_provenance`` records the verified binary, parameters and geometry.
    """

    paths: tuple[torch.Tensor, ...]
    endpoints: torch.Tensor
    lengths_mm: torch.Tensor
    mean_fa: torch.Tensor | None
    seeds_attempted: int
    accepted_seeds: torch.Tensor | None = None
    native_provenance: dict[str, Any] | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _affine(value) -> np.ndarray:
    result = np.asarray(_array(value), dtype=np.float64)
    if (result.shape != (4, 4) or not np.isfinite(result).all()
            or abs(np.linalg.det(result[:3, :3])) < 1e-10
            or not np.allclose(result[3], [0., 0., 0., 1.], rtol=0, atol=1e-12)):
        raise ValueError("affine must be a finite nonsingular voxel-to-RAS-mm [4,4] matrix")
    return result


def write_tracking_image(
    value, affine, output_path: str | Path, *, voxel_spacing_mm=None,
) -> Path:
    """Write float32 voxels with float64 NIfTI-2 sform, without resampling.

    ``value`` is a tensor/array or a nibabel image; ``affine`` maps voxel
    centres to RAS millimetres. ``voxel_spacing_mm`` optionally preserves the
    independent 5TT header spacing (three positive values). NIfTI stores the
    first three sform rows; the bottom affine row must be canonical within
    1e-12. The uncompressed ``.nii`` output is reproducible and can also be
    supplied to an independently built reference executable for comparison.
    """
    if isinstance(value, nib.spatialimages.SpatialImage):
        if affine is None:
            affine = value.affine
        value = np.asanyarray(value.dataobj)
    data = np.asarray(_array(value), dtype=np.float32)
    geometry = _affine(affine)
    image = nib.Nifti2Image(data, geometry)
    image.set_sform(geometry, code=1)
    image.set_qform(None, code=0)
    image.header.set_xyzt_units("mm")
    if voxel_spacing_mm is not None:
        spacing = np.asarray(_array(voxel_spacing_mm), dtype=np.float64)
        if (spacing.shape != (3,) or not np.isfinite(spacing).all()
                or np.any(spacing <= 0)):
            raise ValueError("voxel_spacing_mm must contain three positive finite values")
        image.header.set_zooms(tuple(spacing) + (1.,) * (data.ndim - 3))
    path = Path(output_path)
    if path.suffix != ".nii":
        raise ValueError("tracking tensor serialization requires uncompressed NIfTI-2 .nii")
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, path)
    return path


def _prepare_image(value, affine, destination: Path, *, spacing=None):
    """Preserve input files; serialize arrays without changing their grid."""
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser().resolve(strict=True)
        image = nib.load(str(path))
        if affine is not None and not np.array_equal(_affine(affine)[:3], image.affine[:3]):
            raise ValueError("file input affine differs from its stored sform")
        if spacing is not None and not np.allclose(
                np.asarray(_array(spacing)), image.header.get_zooms()[:3], rtol=0, atol=1e-12):
            raise ValueError("file input spacing differs from its stored voxel spacing")
        return path, image, False
    if isinstance(value, nib.spatialimages.SpatialImage):
        if affine is None:
            affine = value.affine
        if spacing is None:
            spacing = value.header.get_zooms()[:3]
    if affine is None:
        raise ValueError("tensor/array inputs require an explicit voxel-to-RAS-mm affine")
    path = write_tracking_image(value, affine, destination, voxel_spacing_mm=spacing)
    return path, nib.load(str(path)), True


def _image_record(path: Path, image, serialized: bool) -> dict:
    return dict(sha256=_sha256(path), size_bytes=path.stat().st_size,
                shape=list(image.shape), affine=image.affine.tolist(),
                voxel_spacing_mm=[float(v) for v in image.header.get_zooms()[:3]],
                serialized_nifti2=serialized)


def _load_tracks(path: Path, device: torch.device):
    """Load native float32 world points and accumulate polyline lengths."""
    loaded = nib.streamlines.load(str(path), lazy_load=False)
    streamlines = loaded.tractogram.streamlines
    counts = np.asarray([len(points) for points in streamlines], dtype=np.int64)
    if counts.size and np.any(counts < 2):
        raise RuntimeError("native tckgen produced a streamline with fewer than two points")
    raw = np.asarray(streamlines._data, dtype=np.float32)
    if counts.size == 0:
        raw = np.empty((0, 3), dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1:] != (3,) or not np.isfinite(raw).all():
        raise RuntimeError("native tckgen produced malformed or nonfinite world coordinates")
    # One packed transfer avoids a GPU allocation/H2D copy for every track.
    points = torch.from_numpy(np.ascontiguousarray(raw)).to(device)
    offsets = np.r_[np.int64(0), counts.cumsum(dtype=np.int64)]
    paths = tuple(points[int(begin):int(end)]
                  for begin, end in zip(offsets[:-1], offsets[1:]))
    endpoints = np.empty((len(paths), 2, 3), dtype=np.float32)
    lengths = np.empty(len(paths), dtype=np.float32)
    for index, track in enumerate(streamlines):
        endpoints[index] = track[[0, -1]]
        delta = np.diff(track, axis=0).astype(np.float32, copy=False)
        squared = delta[:, 0] * delta[:, 0] + delta[:, 1] * delta[:, 1]
        squared = squared + delta[:, 2] * delta[:, 2]
        segment = np.sqrt(squared)
        # MRtrix length() is a float32 left-to-right accumulation, rather
        # than a double/reordered sum of the segment lengths.
        lengths[index] = np.cumsum(segment, dtype=np.float32)[-1]
    header = {str(key): (value.item() if isinstance(value, np.generic) else value)
              for key, value in loaded.header.items()
              if key in {"count", "total_count", "step_size", "min_dist", "max_dist",
                         "max_angle", "samples_per_step", "fod_power", "threshold",
                         "init_threshold", "output_step_size"}}
    return (paths, torch.from_numpy(endpoints).to(device),
            torch.from_numpy(lengths).to(device), header)


@torch.inference_mode()
def probabilistic_tractography(
    wm_sh,
    fod_affine=None,
    five_tissue=None,
    five_tissue_affine=None,
    gmwmi=None,
    *,
    n_seeds: int,
    lmax: int = 8,
    five_tissue_spacing_mm=None,
    fa: torch.Tensor | None = None,
    seed: int = 0,
    tracking_threads: int = 8,
    max_length_mm: float = 250.,
    min_length_mm: float | None = None,
    step_mm: float | None = None,
    max_angle_degrees: float = 45.,
    cutoff: float = 0.1,
    power: float = 0.5,
    output_tck: str | Path | None = None,
    device: str | torch.device | None = None,
    batch_size=None,
    arc_proposals=None,
    compile_arc: bool = False,
) -> Tractogram:
    """Execute the pinned official iFOD2/ACT algorithm without MRtrix install.

    FOD float32 ``[X,Y,Z,C]``, 5TT float32 ``[A,B,C,5]`` and GMWMI
    float32 ``[A,B,C]`` may be tensors, nibabel images or NIfTI paths.
    Tensor grids require their two affines; file grids retain their headers.
    ``lmax`` must match the FOD coefficient count (default 8: 45 columns).
    Independent FOD and anatomy grids remain independent throughout tracking.

    ``n_seeds`` is ``tckgen -seeds`` with ``-select 0``. ``seed`` sets
    ``MRTRIX_RNG_SEED``; for an exactly repeatable trajectory sequence choose
    ``tracking_threads=1``. Multiple workers retain official MRtrix scheduling.
    ``max_length_mm``, ``min_length_mm``, ``step_mm``, ``max_angle_degrees``,
    ``cutoff`` and ``power`` map directly to the same official options;
    samples per arc is fixed at 3. Omitted step/minimum length use the pinned
    official defaults (half/twice the geometric mean FOD voxel spacing).
    ``five_tissue_spacing_mm`` preserves the separate tensor 5TT header spacing.

    The verified FNIT binary uses CPU threads; output tensors use the input
    FOD device, or explicit ``device`` for files. Optional ``fa`` is sampled
    through FNIT's mature precise voxel-crossing method on the FOD grid.
    ``output_tck`` retains the native TCK; absent it, temporary files are
    removed after loading. Existing outputs are never overwritten.
    Retired PyTorch ``batch_size``/``arc_proposals`` options are rejected when
    supplied; ``compile_arc=True`` is rejected. No PyTorch tracker is retained.
    """
    if compile_arc or batch_size is not None or arc_proposals is not None:
        raise ValueError("PyTorch tracking options compile_arc/batch_size/arc_proposals are retired; use tracking_threads")
    if (type(n_seeds) is not int or n_seeds < 1 or type(seed) is not int or seed < 0
            or type(tracking_threads) is not int or tracking_threads < 1):
        raise ValueError("n_seeds/tracking_threads must be positive integers and seed nonnegative")
    if type(lmax) is not int or lmax < 0 or lmax % 2:
        raise ValueError("lmax must be a nonnegative even integer")
    for name, value, lower, upper in (
        ("max_length_mm", max_length_mm, 0, math.inf),
        ("max_angle_degrees", max_angle_degrees, 0, 90),
        ("cutoff", cutoff, 0, math.inf), ("power", power, 0, math.inf),
    ):
        if not math.isfinite(value) or not lower < value <= upper:
            raise ValueError(f"{name} is outside the valid positive finite range")
    if step_mm is not None and (not math.isfinite(step_mm) or step_mm <= 0 or step_mm >= max_length_mm):
        raise ValueError("step_mm must be positive and smaller than max_length_mm")
    if min_length_mm is not None and (not math.isfinite(min_length_mm) or not 0 <= min_length_mm <= max_length_mm):
        raise ValueError("min_length_mm must be finite, nonnegative and no greater than max_length_mm")
    if five_tissue is None or gmwmi is None:
        raise ValueError("five_tissue and gmwmi are required")
    selected_device = torch.device(device or (wm_sh.device if isinstance(wm_sh, torch.Tensor) else "cpu"))
    if fa is not None and (fa.dtype != torch.float32 or fa.device != selected_device):
        raise ValueError("fa must be float32 on the requested output device")
    destination = Path(output_tck).expanduser().resolve() if output_tck is not None else None
    if destination is not None:
        if destination.suffix != ".tck" or destination.exists():
            raise ValueError("output_tck must be a new .tck file")
        destination.parent.mkdir(parents=True, exist_ok=True)
    from .native_runtime import ensure_tckgen, native_runtime_manifest
    binary = ensure_tckgen()
    manifest = native_runtime_manifest(binary)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="fnit-tckgen-") as temporary:
        work = Path(temporary)
        fod_path, fod_image, fod_serialized = _prepare_image(wm_sh, fod_affine, work / "wm_fod.nii")
        five_path, five_image, five_serialized = _prepare_image(
            five_tissue, five_tissue_affine, work / "five_tissue.nii", spacing=five_tissue_spacing_mm)
        gmwmi_affine = (five_image.affine if five_tissue_affine is None else five_tissue_affine)
        if isinstance(gmwmi, (str, Path, nib.spatialimages.SpatialImage)):
            gmwmi_affine = None
        gmwmi_spacing = (five_image.header.get_zooms()[:3] if five_tissue_spacing_mm is None
                         else five_tissue_spacing_mm)
        gmwmi_path, gmwmi_image, gmwmi_serialized = _prepare_image(
            gmwmi, gmwmi_affine, work / "gmwmi.nii", spacing=gmwmi_spacing)
        width = (lmax + 1) * (lmax + 2) // 2
        if len(fod_image.shape) != 4 or fod_image.shape[-1] != width:
            raise ValueError("FOD must have shape [X,Y,Z,C] matching lmax")
        if (len(five_image.shape) != 4 or five_image.shape[-1] != 5
                or tuple(gmwmi_image.shape) != tuple(five_image.shape[:3])
                or not np.array_equal(gmwmi_image.affine, five_image.affine)):
            raise ValueError("5TT [A,B,C,5] and GMWMI [A,B,C] must share voxel centres")
        if fa is not None and tuple(fa.shape) != tuple(fod_image.shape[:3]):
            raise ValueError("FA must match the FOD voxel grid")
        tck_path = destination or work / "tracks.tck"
        options = ["-algorithm", "iFOD2", "-seed_gmwmi", str(gmwmi_path),
                   "-act", str(five_path), "-seeds", str(n_seeds), "-select", "0",
                   "-maxlength", str(max_length_mm), "-angle", str(max_angle_degrees),
                   "-cutoff", str(cutoff), "-samples", "3", "-power", str(power),
                   "-nthreads", str(tracking_threads), "-config", "RealignTransform", "false",
                   "-config", "NIfTIUseSform", "true", "-config", "NIfTIAutoLoadJSON", "false",
                   "-config", "TckgenEarlyExit", "false"]
        if step_mm is not None:
            options += ["-step", str(step_mm)]
        if min_length_mm is not None:
            options += ["-minlength", str(min_length_mm)]
        environment = os.environ.copy()
        environment.update(MRTRIX_RNG_SEED=str(seed), MRTRIX_NTHREADS=str(tracking_threads),
                           MRTRIX_CONFIGFILE=os.devnull, FNIT_MRTRIX_ISOLATED_CONFIG="1")
        environment.pop("LD_PRELOAD", None)
        # The verified companion library takes precedence over an unrelated
        # installed MRtrix library, while retaining Conda's other libraries.
        library_paths = [str(binary.parent.parent / "lib"), str(Path(sys.prefix) / "lib"),
                         *environment.get("LD_LIBRARY_PATH", "").split(os.pathsep)]
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(dict.fromkeys(
            path for path in library_paths if path))
        command_started = time.perf_counter()
        command = [str(binary.resolve()), str(fod_path), str(tck_path), *options]
        completed = subprocess.run(command, cwd=work, env=environment, check=False,
                                   capture_output=True, text=True)
        command_seconds = time.perf_counter() - command_started
        if completed.returncode != 0:
            raise RuntimeError(f"FNIT native tckgen failed with exit code {completed.returncode}: {completed.stderr[-6000:]}")
        if not tck_path.is_file():
            raise RuntimeError("FNIT native tckgen completed without producing its TCK output")
        paths, endpoints, lengths, header = _load_tracks(tck_path, selected_device)
        provenance = dict(
            backend="fnit-pinned-mrtrix-tckgen", runtime=manifest,
            scientific_parameters=dict(algorithm="iFOD2", requested_seed_budget=n_seeds,
                select=0, rng_seed=seed, tracking_threads=tracking_threads, lmax=lmax,
                max_length_mm=max_length_mm, min_length_mm=min_length_mm, step_mm=step_mm,
                max_angle_degrees=max_angle_degrees, cutoff=cutoff, samples=3, power=power,
                realign_transform=False, nifti_use_sform=True, nifti_auto_load_json=False,
                tckgen_early_exit=False, isolated_config=True),
            input_images={"wm_fod": _image_record(fod_path, fod_image, fod_serialized),
                          "five_tissue": _image_record(five_path, five_image, five_serialized),
                          "gmwmi": _image_record(gmwmi_path, gmwmi_image, gmwmi_serialized)},
            tck_sha256=_sha256(tck_path), tck_header=header,
            accepted_seed_coordinates="not_stored_in_tck", command_seconds=command_seconds,
            adapter_seconds=time.perf_counter() - started)
    mean_fa = None
    if fa is not None and paths:
        from .tcksample_precise import sample_streamline_mean_precise
        mean_fa = sample_streamline_mean_precise(paths, fa, torch.as_tensor(
            fod_image.affine, dtype=torch.float64, device=selected_device))
    return Tractogram(paths, endpoints, lengths, mean_fa, n_seeds, None, provenance)
