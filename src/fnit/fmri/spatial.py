"""Single-interpolation BOLD resampling for per-volume motion and a fixed warp."""

import os
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import map_coordinates

from ..applywarp.core import (
    FSL_CUBIC_SPLINE_COEFFICIENTS,
    _expand_cubic_coefficients,
    _fsl_voxel_matrix,
    _load_dense_warp,
    _load_nifti,
    _sample_linear,
    _spatial_grid,
)


def apply_motion_warp(
    input_bold,
    reference,
    motion_matrices,
    *,
    warp=None,
    postmat=None,
    warp_convention="auto",
    interpolation="linear",
    batch_size=16,
    device=None,
    mcflirt=False,
):
    """Apply each FLIRT motion matrix and one shared FSL warp in one sampling.

    ``motion_matrices[t]`` maps the raw input frame to the warp-source grid.
    ``warp`` maps warp source to warp reference; ``postmat`` maps warp reference
    to the final reference. All matrices follow FSL scaled-mm coordinates.
    The returned NIfTI uses the 3D ``reference`` grid and input time axis.
    ``mcflirt=True`` selects the MCFLIRT Constant spline and extraslice
    boundary for motion-only sampling. The shared warp contract otherwise
    keeps zero padding and uses SciPy for spline sampling.
    """
    image = _load_nifti(input_bold, "input_bold")
    target = _load_nifti(reference, "reference")
    if image.ndim != 4 or target.ndim != 3:
        raise ValueError("input_bold must be 4D and reference must be 3D")
    matrices = np.asarray(motion_matrices, dtype=np.float64)
    if matrices.shape != (image.shape[3], 4, 4) or not np.isfinite(matrices).all():
        raise ValueError("motion_matrices must contain one finite 4x4 matrix per frame")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if interpolation not in ("linear", "spline"):
        raise ValueError("interpolation must be linear or spline")
    if postmat is None:
        post = np.eye(4)
    elif isinstance(postmat, (str, os.PathLike)):
        post = np.loadtxt(postmat, dtype=np.float64)
    else:
        post = np.asarray(postmat, dtype=np.float64)
    if post.shape != (4, 4) or not np.isfinite(post).all():
        raise ValueError("postmat must be one finite 4x4 matrix")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if mcflirt and (warp is not None or postmat is not None):
        raise ValueError("mcflirt sampling requires motion-only transforms")
    if mcflirt:
        from ..mcflirt.sampling import sample_motion_frame

        data = np.asarray(image.dataobj, dtype=np.float32)
        output = np.empty((*target.shape[:3], image.shape[3]), dtype=np.float32)
        frame_sampler = None
        if device.type == "cuda" and interpolation == "spline":
            from ..mcflirt._sampling_cuda import CudaMotionFrameSampler

            frame_sampler = CudaMotionFrameSampler(
                image, target, device=device, interpolation=interpolation)
        for frame, matrix in enumerate(matrices):
            if frame_sampler is not None:
                output[..., frame] = frame_sampler.sample_numpy(data[..., frame], matrix)
            else:
                output[..., frame] = sample_motion_frame(
                    data[..., frame], image, target, matrix, device=device,
                    interpolation=interpolation,
                ).cpu().numpy()
        header = target.header.copy()
        header.set_data_dtype(np.float32)
        result = nib.Nifti1Image(output, target.affine, header)
        result.header.set_zooms((*target.header.get_zooms()[:3], image.header.get_zooms()[3]))
        result.header.set_xyzt_units(xyz=target.header.get_xyzt_units()[0],
                                    t=image.header.get_xyzt_units()[1])
        result.set_qform(target.affine, code=int(target.header["qform_code"]))
        result.set_sform(target.affine, code=int(target.header["sform_code"]))
        return result
    shape = tuple(target.shape[:3])
    output_mm = _spatial_grid(shape, _fsl_voxel_matrix(target), device)
    post_inverse = torch.as_tensor(np.linalg.inv(post), dtype=torch.float64, device=device)
    query = (post_inverse[:3, :3] @ output_mm + post_inverse[:3, 3:4]).reshape(3, *shape)
    warp_valid = torch.ones(shape, dtype=torch.bool, device=device)
    if warp is None:
        source_mm = query
    else:
        field_image = _load_nifti(warp, "warp")
        embedded = None
        if int(field_image.header["intent_code"]) == FSL_CUBIC_SPLINE_COEFFICIENTS:
            field, _, field_fsl, embedded = _expand_cubic_coefficients(field_image, device)
            convention = "relative"
        else:
            _, field_data, field_fsl, convention, _ = _load_dense_warp(
                field_image, warp_convention
            )
            field = torch.as_tensor(np.moveaxis(field_data, -1, 0).copy(),
                                    dtype=torch.float32, device=device)
        if postmat is None and (
            tuple(field.shape[1:]) != shape
            or not np.allclose(field_fsl, _fsl_voxel_matrix(target), atol=1e-4)
        ):
            raise ValueError("warp reference grid differs from reference; supply postmat")
        world_to_field = torch.as_tensor(np.linalg.inv(field_fsl), dtype=torch.float64,
                                         device=device)
        flat = query.reshape(3, -1)
        voxels = (world_to_field[:3, :3] @ flat + world_to_field[:3, 3:4]).reshape(3, *shape)
        displacement, warp_valid = _sample_linear(field, voxels)
        if embedded is not None:
            affine_inverse = torch.as_tensor(np.linalg.inv(embedded), dtype=torch.float64,
                                             device=device)
            base = (affine_inverse[:3, :3] @ flat + affine_inverse[:3, 3:4]).reshape(3, *shape)
            source_mm = base + displacement
        else:
            source_mm = displacement if convention == "absolute" else query + displacement
    source_flat = source_mm.reshape(3, -1).to(torch.float64)
    input_world_to_voxel = torch.as_tensor(np.linalg.inv(_fsl_voxel_matrix(image)),
                                           dtype=torch.float64, device=device)
    data = np.asarray(image.dataobj, dtype=np.float32)
    output = np.empty((*shape, image.shape[3]), dtype=np.float32)
    for start in range(0, image.shape[3], batch_size):
        stop = min(start + batch_size, image.shape[3])
        inverse = torch.as_tensor(np.linalg.inv(matrices[start:stop]),
                                  dtype=torch.float64, device=device)
        mm = torch.matmul(inverse[:, :3, :3], source_flat) + inverse[:, :3, 3:4]
        coords = torch.matmul(input_world_to_voxel[:3, :3], mm) + input_world_to_voxel[:3, 3:4]
        inside = torch.ones((stop - start, coords.shape[-1]), dtype=torch.bool, device=device)
        for axis in range(3):
            inside &= (coords[:, axis] >= -1e-6) & (coords[:, axis] <= image.shape[axis] - 1 + 1e-6)
        valid = inside.reshape(stop - start, *shape) & warp_valid
        if interpolation == "spline":
            sample_coords = coords.reshape(stop - start, 3, *shape).cpu().numpy()
            valid_cpu = valid.cpu().numpy()
            for offset in range(stop - start):
                sampled = map_coordinates(
                    data[..., start + offset], sample_coords[offset],
                    order=3, mode="nearest",
                )
                # MCFLIRT extends the edge slices during final interpolation.
                # For a warp, preserve the existing out-of-field zero policy.
                if warp is not None:
                    sampled *= valid_cpu[offset]
                output[..., start + offset] = sampled
        else:
            normalized = torch.stack([
                2 * coords[:, axis] / max(image.shape[axis] - 1, 1) - 1
                for axis in (2, 1, 0)
            ], -1).reshape(stop - start, *shape, 3).to(torch.float32)
            frames = torch.as_tensor(np.moveaxis(data[..., start:stop], -1, 0).copy(),
                                     dtype=torch.float32, device=device)[:, None]
            sampled = F.grid_sample(frames, normalized, mode="bilinear", padding_mode="border",
                                    align_corners=True)[:, 0]
            sampled *= valid
            output[..., start:stop] = np.moveaxis(sampled.cpu().numpy(), 0, -1)
    header = target.header.copy()
    header.set_data_dtype(np.float32)
    result = nib.Nifti1Image(output, target.affine, header)
    result.header.set_zooms((*target.header.get_zooms()[:3], image.header.get_zooms()[3]))
    result.header.set_xyzt_units(xyz=target.header.get_xyzt_units()[0],
                                 t=image.header.get_xyzt_units()[1])
    result.set_qform(target.affine, code=int(target.header["qform_code"]))
    result.set_sform(target.affine, code=int(target.header["sform_code"]))
    return result
