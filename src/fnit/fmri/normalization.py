"""T1-to-MNI registration and one-pass BOLD resampling in world coordinates."""

from dataclasses import dataclass, field
from pathlib import Path
import time

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


def _grid_constant_cubic_coefficients(values):
    """Match SciPy's cubic prefilter with its 12-voxel constant prepad.

    A mirrored column is embedded in a periodic column before deconvolution.
    This solves the same symmetric spline system on the GPU without filtering
    time. Like SciPy, spline coefficients and query coordinates use float64;
    source images and saved outputs remain float32.
    """
    coefficients = F.pad(values.to(torch.float64), (12, 12, 12, 12, 12, 12), value=0)
    for axis in (-3, -2, -1):
        size = coefficients.shape[axis]
        tail = coefficients.narrow(axis, 1, size - 2).flip((axis,))
        extended = torch.cat((coefficients, tail), dim=axis)
        frequency = torch.fft.rfftfreq(extended.shape[axis], device=values.device,
                                      dtype=coefficients.dtype)
        response = (2 + torch.cos(2 * torch.pi * frequency)) / 3
        shape = [1] * values.ndim
        shape[axis] = len(response)
        filtered = torch.fft.irfft(torch.fft.rfft(extended, dim=axis)
                                  / response.reshape(shape),
                                  n=extended.shape[axis], dim=axis)
        coefficients = filtered.narrow(axis, 0, size).contiguous()
    return coefficients


def _mirror_cubic_coefficients(values):
    """Cubic spline coefficients for an unpadded mirrored array, on device."""
    coefficients = values.to(torch.float64)
    for axis in (-3, -2, -1):
        size = coefficients.shape[axis]
        if size < 2:
            continue
        tail = coefficients.narrow(axis, 1, size - 2).flip((axis,))
        extended = torch.cat((coefficients, tail), dim=axis)
        frequency = torch.fft.rfftfreq(extended.shape[axis], device=values.device,
                                      dtype=coefficients.dtype)
        response = (2 + torch.cos(2 * torch.pi * frequency)) / 3
        layout = [1] * values.ndim
        layout[axis] = len(response)
        filtered = torch.fft.irfft(torch.fft.rfft(extended, dim=axis)
                                   / response.reshape(layout),
                                   n=extended.shape[axis], dim=axis)
        coefficients = filtered.narrow(axis, 0, size).contiguous()
    return coefficients


def _fmriprep_dense_world(world, pull_image, pull, voxels, spatial_chunk_size):
    """Match DenseFieldTransform's f4 query and on-grid deformation lookup.

    A fixed-grid field stores exact float64 grid-world locations plus its
    displacements. Rounded queries select those deformations; adding the
    displacement to a rounded target-world point would change this behavior.
    Off-grid queries use cubic interpolation with constant boundaries and
    retain the input point for out-of-domain field components.
    """
    field_affine = torch.as_tensor(pull_image.affine, dtype=torch.float64,
                                   device=world.device)
    query_world = world.to(torch.float32).to(torch.float64)
    inverse = torch.as_tensor(np.linalg.inv(pull_image.affine), dtype=torch.float64,
                               device=world.device)
    index = inverse[:3, :3] @ query_world + inverse[:3, 3:4]
    rounded = index.round()
    on_grid = torch.sqrt(((index - rounded) ** 2).sum(dim=0)) < 1e-3
    field = field_affine[:3, :3] @ voxels + field_affine[:3, 3:4] + pull
    if bool(on_grid.all()):
        selected = rounded.to(torch.int64)
        if any(bool(((selected[axis] < 0)
                      | (selected[axis] >= pull_image.shape[axis])).any())
               for axis in range(3)):
            raise ValueError("rounded deformation query is outside its reference grid")
        flat = ((selected[0] * pull_image.shape[1] + selected[1])
                * pull_image.shape[2] + selected[2])
        return field[:, flat]
    from ..eddy.fsl2111_strict.spline import _pad_cubic_coefficients, sample_cubic_periodic_fast
    coefficients = _mirror_cubic_coefficients(field.reshape(3, *pull_image.shape[:3]))
    padded = _pad_cubic_coefficients(coefficients, "mirror")
    mapped = torch.empty_like(world)
    for start in range(0, world.shape[1], spatial_chunk_size):
        stop = min(start + spatial_chunk_size, world.shape[1])
        coordinates = index[:, start:stop].reshape(1, 3, -1, 1, 1).expand(3, -1, -1, -1, -1)
        sampled = sample_cubic_periodic_fast(
            coefficients, coordinates, boundary="mirror", padded_coeff=padded,
        ).reshape(3, -1)
        inside = torch.ones(stop - start, dtype=torch.bool, device=world.device)
        for axis in range(3):
            inside &= ((index[axis, start:stop] >= 0)
                       & (index[axis, start:stop] <= pull_image.shape[axis] - 1))
        mapped[:, start:stop] = torch.where(inside[None], sampled,
                                           query_world[:, start:stop])
    return mapped


def resample_world(
    source,
    reference,
    reference_to_source_world,
    output,
    *,
    pre_affine_pull_ras=None,
    output_mask=None,
    interpolation="linear",
    boundary="grid-constant",
    motion_pull_world=None,
    coordinate_precision="float64",
    spatial_chunk_size=262144,
    batch_size=8,
    device=None,
):
    """Write a 3D/4D source on the reference grid after one interpolation.

    ``reference_to_source_world`` maps reference RAS millimetres into source
    RAS millimetres. An optional fixed-grid pull displacement is added to the
    reference RAS coordinate *before* that affine. For MNI BOLD resampling,
    pass inverse(EPI-to-T1 BBR) and the MNI-to-T1 pull field.
    ``motion_pull_world`` is one reference-world-to-original-frame-world
    matrix per frame. It is composed after the fixed transform, allowing
    raw/minimal BOLD to be sampled directly into T1w or MNI in one pass.
    ``spline`` uses cubic B-splines on the selected device. The default
    ``grid-constant`` boundary matches fMRIPrep 25.2.4 (cval=0); ``periodic``
    explicitly selects the earlier FSL-style boundary. Time is not filtered.
    ``coordinate_precision="fmriprep"`` follows the fixed fMRIPrep 25.2.4
    resampler: target world points round to float32, dense fields return their
    float64 deformation values, and HMC is composed in source voxel space.
    ``float64`` retains the earlier world-coordinate composition.
    """
    image = nib.load(str(source))
    target = nib.load(str(reference))
    if image.ndim not in (3, 4) or target.ndim != 3:
        raise ValueError("source must be 3D/4D and reference must be 3D")
    if batch_size < 1 or spatial_chunk_size < 1 or interpolation not in ("linear", "nearest", "spline"):
        raise ValueError("invalid batch_size or interpolation")
    if boundary not in ("grid-constant", "periodic"):
        raise ValueError("boundary must be grid-constant or periodic")
    if coordinate_precision not in ("float64", "fmriprep"):
        raise ValueError("coordinate_precision must be float64 or fmriprep")
    matrix = np.asarray(reference_to_source_world, dtype=np.float64)
    from ..flirt.coordinates import _numpy_affine
    matrix = _numpy_affine(matrix, "reference_to_source_world")
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
            np.moveaxis(np.asarray(pull_image.dataobj, dtype=np.float64), -1, 0).copy(),
            dtype=torch.float64, device=selected,
        ).reshape(3, -1)
        if not bool(torch.isfinite(pull).all()):
            raise ValueError("pull field contains nonfinite values")
        if coordinate_precision == "fmriprep":
            world = _fmriprep_dense_world(world, pull_image, pull, voxels, spatial_chunk_size)
        else:
            world += pull
    elif coordinate_precision == "fmriprep":
        world = world.to(torch.float32).to(torch.float64)
    if coordinate_precision == "fmriprep":
        # nitransforms Affine.map uses _as_homogeneous(dtype='float32') at
        # each affine entry, including the output of a dense transform.
        world = world.to(torch.float32).to(torch.float64)
    fixed = torch.as_tensor(matrix, dtype=torch.float64, device=selected)
    reference_world = fixed[:3, :3] @ world + fixed[:3, 3:4]
    world_to_source = torch.as_tensor(np.linalg.inv(image.affine),
                                     dtype=torch.float64, device=selected)
    reference_voxels = None
    if coordinate_precision == "fmriprep":
        reference_voxels = (world_to_source[:3, :3] @ reference_world.to(torch.float32).to(torch.float64)
                            + world_to_source[:3, 3:4])
    valid = torch.ones((1, *shape), dtype=torch.bool, device=selected)
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
    if not np.isfinite(data).all():
        raise ValueError("source contains nonfinite values")
    frames = 1 if image.ndim == 3 else image.shape[3]
    motion = np.broadcast_to(np.eye(4), (frames, 4, 4)) if motion_pull_world is None \
        else np.asarray(motion_pull_world, dtype=np.float64)
    if motion.shape != (frames, 4, 4) or not np.isfinite(motion).all() \
            or not np.allclose(motion[:, 3], (0, 0, 0, 1), rtol=0, atol=1e-8):
        raise ValueError("motion_pull_world must have one finite affine per frame")
    result = np.empty((*shape, frames), dtype=np.float32)
    voxel_motion = None
    if coordinate_precision == "fmriprep":
        # Tiny matrix products follow the installed CPU reference's order;
        # the large coordinate maps and all interpolation remain on device.
        inverse_source = np.linalg.inv(image.affine)
        voxel_motion = np.asarray([inverse_source @ frame @ image.affine
                                   for frame in motion], dtype=np.float64)
    for start in range(0, frames, batch_size):
        stop = min(start + batch_size, frames)
        if coordinate_precision == "fmriprep":
            pull = torch.as_tensor(voxel_motion[start:stop].copy(), dtype=torch.float64, device=selected)
            coords = (pull[:, :3, :3] @ reference_voxels
                       + pull[:, :3, 3:4]).reshape(stop - start, 3, *shape)
        else:
            pull = torch.as_tensor(motion[start:stop].copy(), dtype=torch.float64, device=selected)
            frame_world = pull[:, :3, :3] @ reference_world + pull[:, :3, 3:4]
            coords = (world_to_source[:3, :3] @ frame_world
                      + world_to_source[:3, 3:4]).reshape(stop - start, 3, *shape)
        if boundary == "periodic":
            # Preserve the clean-volume edge correction. Grid-constant uses
            # unmodified coordinates, including its outside-grid support.
            for axis in range(3):
                upper = image.shape[axis] - 1
                within = (coords[:, axis] >= -1e-6) & (coords[:, axis] <= upper + 1e-6)
                coords[:, axis] = torch.where(
                    within, coords[:, axis].clamp(0, upper), coords[:, axis]
                )
        source_batch = data if image.ndim == 3 else data[..., start:stop]
        if source_batch.ndim == 3:
            source_batch = source_batch[..., None]
        source_tensor = torch.as_tensor(
            np.moveaxis(source_batch, -1, 0).copy(),
            dtype=torch.float32, device=selected,
        )[:, None]
        if interpolation == "spline":
            from ..eddy.fsl2111_strict.spline import _pad_cubic_coefficients, sample_cubic_periodic_fast
            coefficients = (_grid_constant_cubic_coefficients(source_tensor[:, 0])
                            if boundary == "grid-constant" else
                            _periodic_cubic_coefficients(source_tensor[:, 0]))
            padded_coeff = _pad_cubic_coefficients(
                coefficients, "mirror" if boundary == "grid-constant" else "periodic",
            )
            flat_coords = coords.reshape(stop - start, 3, -1)
            sampled = torch.empty((stop - start, flat_coords.shape[2]),
                                  dtype=coefficients.dtype, device=selected)
            for query_start in range(0, flat_coords.shape[2], spatial_chunk_size):
                query_stop = min(query_start + spatial_chunk_size, flat_coords.shape[2])
                query = flat_coords[:, :, query_start:query_stop].reshape(stop - start, 3, -1, 1, 1)
                sample_coords = query + 12 if boundary == "grid-constant" else query.float()
                sampled[:, query_start:query_stop] = sample_cubic_periodic_fast(
                    coefficients, sample_coords,
                    boundary="mirror" if boundary == "grid-constant" else "periodic",
                    padded_coeff=padded_coeff,
                ).reshape(stop - start, -1)
            sampled = sampled.reshape(stop - start, *shape)
            if boundary == "grid-constant":
                # Beyond the prepad there is only zero extension. Prevent the
                # interpolation helper's mirrored coefficients re-entering FOV.
                inside = torch.ones_like(valid).expand(stop - start, *shape).clone()
                for axis in range(3):
                    inside &= ((coords[:, axis] >= -12)
                               & (coords[:, axis] <= image.shape[axis] + 11))
                sampled *= inside
        else:
            grid = torch.stack([2 * coords[:, axis] / max(image.shape[axis] - 1, 1) - 1
                                for axis in (2, 1, 0)], dim=-1).float()
            sampled = F.grid_sample(
                source_tensor, grid,
                mode="bilinear" if interpolation == "linear" else "nearest",
                padding_mode="zeros", align_corners=True,
            )[:, 0]
        if boundary == "periodic":
            inside = torch.ones_like(valid).expand(stop - start, *shape).clone()
            for axis in range(3):
                inside &= ((coords[:, axis] >= 0)
                           & (coords[:, axis] <= image.shape[axis] - 1))
            sampled *= inside
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
