"""T1-to-MNI registration and one-pass BOLD resampling in world coordinates."""

from dataclasses import dataclass, field
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch

from ..flirt import TorchFLIRT
from .._world_resampling import resample_world_image


@dataclass(frozen=True)
class T1MNIResult:
    """Saved forward affine and fixed-grid MNI-to-T1 RAS pull field."""

    affine: Path
    pull_ras: Path
    backend: str
    moving_to_fixed_world: np.ndarray
    qc: dict | None = None
    timing_seconds: dict[str, float] = field(default_factory=dict)


def _completed_time(device):
    """Finish GPU work only at a reported phase boundary."""
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def register_t1_to_mni(
    t1_brain,
    mni_brain,
    output_dir,
    *,
    backend="synthmorph",
    synthmorph_weights=None,
    reference_mask=None,
    fnirt_config=None,
    fnirt_execution="optimized",
    device=None,
):
    """Register one T1 brain to a 2-mm MNI brain with an existing FNIT backend.

    ``pull_ras`` has shape MNI-X/Y/Z x 3. At each MNI voxel it stores
    ``T1_world - MNI_world`` in RAS millimetres. It can be composed with BBR
    without inverting a nonlinear field or resampling the BOLD twice.
    """
    if backend not in ("synthmorph", "fnirt"):
        raise ValueError("backend must be 'synthmorph' or 'fnirt'")
    if fnirt_execution not in ("reference", "optimized"):
        raise ValueError("fnirt_execution must be 'reference' or 'optimized'")
    if backend != "fnirt" and fnirt_config is not None:
        raise ValueError("fnirt_config requires backend='fnirt'")
    if backend != "fnirt" and fnirt_execution != "optimized":
        raise ValueError("fnirt_execution requires backend='fnirt'")
    selected = device or ("cuda" if torch.cuda.is_available() else "cpu")
    moving = nib.load(str(t1_brain))
    fixed = nib.load(str(mni_brain))
    if len(moving.shape) != 3 or len(fixed.shape) != 3:
        raise ValueError("T1 and MNI template must each be 3D")
    timing = {}
    started = _completed_time(selected)
    linear = TorchFLIRT(device=selected)(moving, fixed)
    finished = _completed_time(selected)
    timing["t1_to_mni_affine"] = finished - started
    initial = np.asarray(linear.moving_to_fixed_world, dtype=np.float64)
    started = finished
    if backend == "synthmorph":
        from ..synthmorph import SynthMorph

        model = SynthMorph(
            weights=synthmorph_weights, device=selected, model="deform"
        )
        nonlinear = model(moving, fixed, init=initial)
        qc = None
    else:
        from ..fnirt import TorchFNIRT, resolve_fnirt_config

        config = resolve_fnirt_config(fnirt_config, default="t1")
        nonlinear = TorchFNIRT(device=selected, config=config, execution=fnirt_execution)(
            moving, fixed, initial, reference_mask=reference_mask
        )
        qc = getattr(nonlinear, "qc", None)
    finished = _completed_time(selected)
    timing["t1_to_mni_nonlinear"] = finished - started
    started = finished
    pull = nonlinear.transform if backend == "synthmorph" else nonlinear.pull_transform
    field = np.asarray(pull.dataobj, dtype=np.float32)
    if field.shape != (*fixed.shape[:3], 3) or not np.isfinite(field).all():
        raise ValueError("registration produced an invalid MNI-to-T1 pull field")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    affine_path = output / "T1_to_MNI152_2mm_affine.mat"
    field_path = output / "MNI152_2mm_to_T1_pull_ras.nii.gz"
    np.savetxt(affine_path, np.asarray(linear.matrix), fmt="%.12g")
    header = fixed.header.copy()
    header.set_data_dtype(np.float32)
    header.set_slope_inter(1, 0)
    image = nib.Nifti1Image(field, fixed.affine, header)
    image.header.set_intent("vector")
    nib.save(image, str(field_path))
    timing["warp_conversion"] = _completed_time(selected) - started
    return T1MNIResult(
        affine=affine_path,
        pull_ras=field_path,
        backend=backend,
        moving_to_fixed_world=np.asarray(linear.moving_to_fixed_world),
        qc=qc,
        timing_seconds=timing,
    )


def resample_world(
    source, reference, reference_to_source_world, output, *,
    pre_affine_pull_ras=None, output_mask=None, interpolation="linear",
    boundary="grid-constant", motion_pull_world=None,
    coordinate_precision="float64", spatial_chunk_size=262144,
    batch_size=8, device=None,
):
    """Compatibility file interface to the shared one-pass world sampler.

    Image values, geometry, temporal metadata, and default parameters retain
    their existing contract. Volume's final routing uses the public
    TorchApplyWarp/apply_transform interfaces; other callers can retain this
    existing helper. RAS fields are explicit and never inferred as FSL warps.
    """
    image = resample_world_image(
        source, reference, reference_to_source_world,
        pre_affine_pull_ras=pre_affine_pull_ras, output_mask=output_mask,
        interpolation=interpolation, boundary=boundary,
        motion_pull_world=motion_pull_world,
        coordinate_precision=coordinate_precision,
        spatial_chunk_size=spatial_chunk_size, batch_size=batch_size,
        device=device,
    )
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(output_path))
    return output_path
