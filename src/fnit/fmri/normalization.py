"""T1-to-MNI registration and one-pass BOLD resampling in world coordinates."""

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from ..flirt import TorchFLIRT


@dataclass(frozen=True)
class T1MNIResult:
    """Saved forward affine and fixed-grid MNI-to-T1 RAS pull field."""

    affine: Path
    pull_ras: Path
    backend: str
    moving_to_fixed_world: np.ndarray
    qc: dict | None = None


def register_t1_to_mni(
    t1_brain,
    mni_brain,
    output_dir,
    *,
    backend="synthmorph",
    synthmorph_weights=None,
    reference_mask=None,
    fnirt_config=None,
    device=None,
):
    """Register one T1 brain to a 2-mm MNI brain with an existing FNIT backend.

    ``pull_ras`` has shape MNI-X/Y/Z x 3. At each MNI voxel it stores
    ``T1_world - MNI_world`` in RAS millimetres. It can be composed with BBR
    without inverting a nonlinear field or resampling the BOLD twice.
    """
    if backend not in ("synthmorph", "fnirt"):
        raise ValueError("backend must be 'synthmorph' or 'fnirt'")
    if backend != "fnirt" and fnirt_config is not None:
        raise ValueError("fnirt_config requires backend='fnirt'")
    selected = device or ("cuda" if torch.cuda.is_available() else "cpu")
    moving = nib.load(str(t1_brain))
    fixed = nib.load(str(mni_brain))
    if len(moving.shape) != 3 or len(fixed.shape) != 3:
        raise ValueError("T1 and MNI template must each be 3D")
    linear = TorchFLIRT(device=selected)(moving, fixed)
    initial = np.asarray(linear.moving_to_fixed_world, dtype=np.float64)
    if backend == "synthmorph":
        from ..synthmorph import SynthMorph

        model = SynthMorph(
            weights=synthmorph_weights, device=selected, model="deform"
        )
        pull = model(moving, fixed, init=initial).transform
        qc = None
    else:
        from ..fnirt import TorchFNIRT, resolve_fnirt_config

        config = resolve_fnirt_config(fnirt_config, default="t1")
        nonlinear = TorchFNIRT(device=selected, config=config)(
            moving, fixed, initial, reference_mask=reference_mask
        )
        pull = nonlinear.pull_transform
        qc = getattr(nonlinear, "qc", None)
    field = np.asarray(pull.dataobj, dtype=np.float32)
    if field.shape != (*fixed.shape[:3], 3) or not np.isfinite(field).all():
        raise ValueError("registration produced an invalid MNI-to-T1 pull field")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    affine_path = output / "T1_to_MNI152_2mm_affine.mat"
    field_path = output / "MNI152_2mm_to_T1_pull_ras.nii.gz"
    np.savetxt(affine_path, np.asarray(linear.matrix), fmt="%.12g")
    image = nib.Nifti1Image(field, fixed.affine)
    image.header.set_intent("vector")
    nib.save(image, str(field_path))
    return T1MNIResult(
        affine=affine_path,
        pull_ras=field_path,
        backend=backend,
        moving_to_fixed_world=np.asarray(linear.moving_to_fixed_world),
        qc=qc,
    )


def _periodic_cubic_coefficients(values):
    """Invert the separable cubic B-spline kernel without filtering time."""
    coefficients = values
    for axis in (-3, -2, -1):
        size = values.shape[axis]
        frequency = torch.fft.rfftfreq(size, device=values.device, dtype=values.dtype)
        response = (2 + torch.cos(2 * torch.pi * frequency)) / 3
        shape = [1] * values.ndim
        shape[axis] = len(frequency)
        coefficients = torch.fft.irfft(
            torch.fft.rfft(coefficients, dim=axis) / response.reshape(shape),
            n=size, dim=axis,
        )
    return coefficients


def resample_world(
    source,
    reference,
    reference_to_source_world,
    output,
    *,
    pre_affine_pull_ras=None,
    output_mask=None,
    interpolation="linear",
    batch_size=8,
    device=None,
):
    """Write a 3D/4D source on the reference grid after one interpolation.

    ``reference_to_source_world`` maps reference RAS millimetres into source
    RAS millimetres. An optional fixed-grid pull displacement is added to the
    reference RAS coordinate *before* that affine. For MNI BOLD resampling,
    pass inverse(EPI-to-T1 BBR) and the MNI-to-T1 pull field.
    ``spline`` prefilters each frame into cubic B-spline coefficients on
    the selected device; time is never filtered. Source boundaries are
    periodic, and samples outside the source or output mask are zero.
    """
    image = nib.load(str(source))
    target = nib.load(str(reference))
    if image.ndim not in (3, 4) or target.ndim != 3:
        raise ValueError("source must be 3D/4D and reference must be 3D")
    if batch_size < 1 or interpolation not in ("linear", "nearest", "spline"):
        raise ValueError("invalid batch_size or interpolation")
    matrix = np.asarray(reference_to_source_world, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("reference_to_source_world must be a finite 4x4 matrix")
    selected = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if selected.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    shape = target.shape
    axes = torch.meshgrid(
        *(torch.arange(n, dtype=torch.float64, device=selected) for n in shape),
        indexing="ij",
    )
    voxels = torch.stack(axes).reshape(3, -1)
    target_affine = torch.as_tensor(target.affine, dtype=torch.float64, device=selected)
    world = target_affine[:3, :3] @ voxels + target_affine[:3, 3:4]
    if pre_affine_pull_ras is not None:
        pull_image = nib.load(str(pre_affine_pull_ras))
        if pull_image.shape != (*shape, 3) or not np.allclose(
            pull_image.affine, target.affine, atol=1e-4
        ):
            raise ValueError("pull field must use the reference grid")
        pull = torch.as_tensor(
            np.moveaxis(np.asarray(pull_image.dataobj, dtype=np.float32), -1, 0).copy(),
            dtype=torch.float64, device=selected,
        ).reshape(3, -1)
        world += pull
    transform = torch.as_tensor(
        np.linalg.inv(image.affine) @ matrix,
        dtype=torch.float64, device=selected,
    )
    coords = transform[:3, :3] @ world + transform[:3, 3:4]
    if interpolation == "spline":
        from ..eddy.fsl2111_strict.spline import sample_cubic_periodic_fast
        spline_coords = coords.reshape(1, 3, *shape).float()
    grid = torch.stack(
        [2 * coords[axis] / max(image.shape[axis] - 1, 1) - 1
         for axis in (2, 1, 0)], dim=-1,
    ).reshape(1, *shape, 3).float()
    valid = torch.ones(coords.shape[1], dtype=torch.bool, device=selected)
    for axis in range(3):
        valid &= (coords[axis] >= 0) & (coords[axis] <= image.shape[axis] - 1)
    valid = valid.reshape(1, *shape)
    if output_mask is not None:
        mask_image = nib.load(str(output_mask))
        if mask_image.shape != shape or not np.allclose(
            mask_image.affine, target.affine, atol=1e-4
        ):
            raise ValueError("output_mask must use the reference grid")
        valid &= torch.as_tensor(
            np.asarray(mask_image.dataobj) > 0.5, device=selected
        )[None]
    data = np.asarray(image.dataobj, dtype=np.float32)
    frames = 1 if image.ndim == 3 else image.shape[3]
    result = np.empty((*shape, frames), dtype=np.float32)
    for start in range(0, frames, batch_size):
        stop = min(start + batch_size, frames)
        source_batch = data if image.ndim == 3 else data[..., start:stop]
        if source_batch.ndim == 3:
            source_batch = source_batch[..., None]
        source_tensor = torch.as_tensor(
            np.moveaxis(source_batch, -1, 0).copy(),
            dtype=torch.float32, device=selected,
        )[:, None]
        if interpolation == "spline":
            coefficients = _periodic_cubic_coefficients(source_tensor[:, 0])
            sampled = sample_cubic_periodic_fast(
                coefficients, spline_coords.expand(stop - start, -1, -1, -1, -1),
            )
        else:
            sampled = F.grid_sample(
                source_tensor, grid.expand(stop - start, *grid.shape[1:]),
                mode="bilinear" if interpolation == "linear" else "nearest",
                padding_mode="zeros", align_corners=True,
            )[:, 0]
        sampled *= valid
        result[..., start:stop] = np.moveaxis(sampled.cpu().numpy(), 0, -1)
    if image.ndim == 3:
        result = result[..., 0]
    header = target.header.copy()
    header.set_data_dtype(np.float32)
    output_image = nib.Nifti1Image(result, target.affine, header)
    if image.ndim == 4:
        output_image.header.set_zooms((*target.header.get_zooms()[:3], image.header.get_zooms()[3]))
        output_image.header.set_xyzt_units(
            xyz=target.header.get_xyzt_units()[0],
            t=image.header.get_xyzt_units()[1],
        )
    output_image.set_qform(target.affine, code=int(target.header["qform_code"]))
    output_image.set_sform(target.affine, code=int(target.header["sform_code"]))
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(output_image, str(output_path))
    return output_path
