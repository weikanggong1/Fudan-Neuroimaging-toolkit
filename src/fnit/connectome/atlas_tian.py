"""Invert an FSL FNIRT cubic warp on the reference grid with PyTorch."""

from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from fnit.applywarp.core import (
    FSL_CUBIC_SPLINE_COEFFICIENTS,
    FSL_FNIRT_DISPLACEMENT_FIELD,
    _expand_cubic_coefficients,
)


def _vertex_positions(indices: torch.Tensor, field: torch.Tensor,
                      inverse_affine: torch.Tensor,
                      reference_spacing: torch.Tensor) -> torch.Tensor:
    """Map source lattice vertices to reference voxel coordinates."""
    shape = field.shape[1:]
    valid = torch.ones(indices.shape[:-1], dtype=torch.bool, device=indices.device)
    for axis, size in enumerate(shape):
        valid &= (indices[..., axis] >= 0) & (indices[..., axis] < size)
    clipped = torch.stack([indices[..., axis].clamp(0, size - 1)
                           for axis, size in enumerate(shape)], dim=-1)
    flat = ((clipped[..., 0] * shape[1] + clipped[..., 1]) * shape[2]
            + clipped[..., 2])
    residual = field.reshape(3, -1)[:, flat.reshape(-1)].T
    residual = residual.reshape(*indices.shape[:-1], 3).to(torch.float64)
    return (indices.to(torch.float64) @ inverse_affine[:3, :3].T
            + inverse_affine[:3, 3] +
            torch.where(valid[..., None], residual / reference_spacing, 0.0))


def _outside_face(vertices: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Return the first crossed tetrahedral face; -1 means inside."""
    result = torch.full(points.shape[:1], -1, dtype=torch.long, device=points.device)
    for index in range(4):
        others = [axis for axis in range(4) if axis != index]
        a, b, c = (vertices[:, axis] for axis in others)
        normal = torch.cross(b - a, c - a, dim=-1)
        unknown = ((points - a) * normal).sum(dim=-1)
        known = ((vertices[:, index] - a) * normal).sum(dim=-1)
        outside = ((unknown < 0) & (known >= 0)) | ((unknown > 0) & (known <= 0))
        result = torch.where((result < 0) & outside, index, result)
    return result


def _tetrahedral_point(vertices: torch.Tensor, indices: torch.Tensor,
                       points: torch.Tensor) -> torch.Tensor:
    """Invert the affine map inside each selected tetrahedron."""
    edge1 = vertices[:, 1] - vertices[:, 0]
    edge2 = vertices[:, 2] - vertices[:, 0]
    edge3 = vertices[:, 3] - vertices[:, 0]
    delta = points - vertices[:, 0]
    cross23 = torch.cross(edge2, edge3, dim=-1)
    denominator = (edge1 * cross23).sum(dim=-1)
    weights = torch.stack((
        (delta * cross23).sum(dim=-1) / denominator,
        (edge1 * torch.cross(delta, edge3, dim=-1)).sum(dim=-1) / denominator,
        (edge1 * torch.cross(edge2, delta, dim=-1)).sum(dim=-1) / denominator,
    ), dim=-1)
    return (indices[:, 0].to(torch.float64)
            + (weights[..., None] * (indices[:, 1:] - indices[:, :1])).sum(dim=1))


def _fill_undefined(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Fill failed inversions with FSL's six-neighbour mean dilation."""
    while not bool(valid.all()):
        sums = torch.zeros_like(values)
        counts = torch.zeros_like(valid, dtype=torch.int32)
        for axis in range(3):
            lower_dst = [slice(None)] * 3
            lower_src = [slice(None)] * 3
            lower_dst[axis] = slice(2, None)
            lower_src[axis] = slice(1, -1)
            dst, src = tuple(lower_dst), tuple(lower_src)
            sums[dst] += torch.where(valid[src][..., None], values[src], 0.0)
            counts[dst] += valid[src]
            upper_dst = [slice(None)] * 3
            upper_src = [slice(None)] * 3
            upper_dst[axis] = slice(None, -1)
            upper_src[axis] = slice(1, None)
            dst, src = tuple(upper_dst), tuple(upper_src)
            sums[dst] += torch.where(valid[src][..., None], values[src], 0.0)
            counts[dst] += valid[src]
        fill = ~valid & (counts > 0)
        if not bool(fill.any()):
            raise RuntimeError("undefined inverse warp voxels have no defined neighbours")
        values[fill] = sums[fill] / counts[fill][:, None].to(values.dtype)
        valid[fill] = True
    return values


@torch.no_grad()
def invert_fnirt_t1_warp(
    forward_coefficients: str | Path | nib.Nifti1Image,
    native_t1_reference: str | Path | nib.Nifti1Image,
    *,
    device: str = "cuda:0",
    max_mirror_steps: int = 1000,
) -> nib.Nifti1Image:
    """Return an intent-2006 relative MNI pull field on the native T1 grid.

    FSL ``invwarp`` expands the cubic coefficients into a float32 source
    lattice, then walks a tetrahedron across that lattice for each reference
    voxel. Rows are independent and run together on the selected PyTorch
    device. Vertices beyond the source FOV have zero nonlinear displacement.
    FSL's optional Jacobian topology correction is not implemented; the
    matched real UKB coefficient does not require a correction.
    """
    if max_mirror_steps < 1:
        raise ValueError("max_mirror_steps must be positive")
    coefficients = (nib.load(str(forward_coefficients)) if isinstance(
        forward_coefficients, (str, Path)) else forward_coefficients)
    reference = (nib.load(str(native_t1_reference)) if isinstance(
        native_t1_reference, (str, Path)) else native_t1_reference)
    if int(coefficients.header["intent_code"]) != FSL_CUBIC_SPLINE_COEFFICIENTS:
        raise ValueError("forward_coefficients must be FSL cubic FNIRT intent 2007")
    if len(reference.shape) != 3:
        raise ValueError("native_t1_reference must be a 3D image")
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    field, _, source_matrix, affine = _expand_cubic_coefficients(
        coefficients, torch_device)
    field = field.to(torch.float32)  # FSL expands coefficients into volume4D<float>.
    reference_spacing = torch.as_tensor(reference.header.get_zooms()[:3],
                                        dtype=torch.float64, device=torch_device)
    source_spacing = torch.as_tensor(np.diag(source_matrix)[:3].copy(),
                                     dtype=torch.float64, device=torch_device)
    native_sampling = np.diag((*reference.header.get_zooms()[:3], 1.0))
    source_sampling = np.diag((*np.diag(source_matrix)[:3], 1.0))
    forward = torch.as_tensor(np.linalg.inv(source_sampling) @ affine @ native_sampling,
                              dtype=torch.float64, device=torch_device)
    inverse = torch.as_tensor(np.linalg.inv(native_sampling) @
                              np.linalg.inv(affine) @ source_sampling,
                              dtype=torch.float64, device=torch_device)
    nx, ny, nz = map(int, reference.shape)
    row_index = torch.arange(ny * nz, device=torch_device)
    points = torch.stack((torch.zeros_like(row_index), row_index // nz,
                          row_index % nz), dim=-1).to(torch.float64)
    corner_offsets = torch.tensor(((0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)),
                                  dtype=torch.long, device=torch_device)
    nonlinear_grid = torch.empty((nx, ny, nz, 3), dtype=torch.float32, device=torch_device)
    affine_grid = torch.empty_like(nonlinear_grid)
    valid_grid = torch.empty((nx, ny, nz), dtype=torch.bool, device=torch_device)
    found = torch.zeros(ny * nz, dtype=torch.bool, device=torch_device)
    vertices = None
    positions = None
    row_numbers = torch.arange(ny * nz, device=torch_device)
    for x in range(nx):
        points[:, 0] = x
        affine_point = points @ forward[:3, :3].T + forward[:3, 3]
        if x == 0:
            vertices = torch.trunc(affine_point).to(torch.long)[:, None] + corner_offsets
            positions = _vertex_positions(vertices, field, inverse, reference_spacing)
        elif not bool(found.all()):
            reset = ~found
            vertices[reset] = (torch.trunc(affine_point[reset]).to(torch.long)[:, None]
                               + corner_offsets)
            positions[reset] = _vertex_positions(vertices[reset], field, inverse,
                                                  reference_spacing)
        active = torch.ones(ny * nz, dtype=torch.bool, device=torch_device)
        result = affine_point.clone()
        for _ in range(max_mirror_steps):
            face = _outside_face(positions, points)
            inside = active & (face < 0)
            if bool(inside.any()):
                result[inside] = _tetrahedral_point(positions[inside], vertices[inside],
                                                    points[inside])
                active[inside] = False
            if not bool(active.any()):
                break
            candidates = torch.stack((2 * vertices[:, 1] - vertices[:, 0],
                                      vertices[:, 0] + vertices[:, 2] - vertices[:, 1],
                                      vertices[:, 1] + vertices[:, 3] - vertices[:, 2],
                                      2 * vertices[:, 2] - vertices[:, 3]), dim=1)
            face = face.clamp_min(0)
            replacement = candidates[row_numbers, face]
            vertices[row_numbers, face] = torch.where(active[:, None], replacement,
                                                       vertices[row_numbers, face])
            positions[row_numbers, face] = _vertex_positions(
                vertices[row_numbers, face], field, inverse, reference_spacing)
        found = ~active
        # FSL stores the nonlinear component in float32 before adding affine.
        nonlinear = (result - affine_point).to(torch.float32)
        affine_mm = (points * reference_spacing) @ torch.as_tensor(
            affine[:3, :3].T, dtype=torch.float64, device=torch_device)
        affine_mm += torch.as_tensor(affine[:3, 3], dtype=torch.float64,
                                    device=torch_device)
        affine_mm -= points * reference_spacing
        nonlinear_grid[x] = nonlinear.reshape(ny, nz, 3)
        affine_grid[x] = affine_mm.to(torch.float32).reshape(ny, nz, 3)
        valid_grid[x] = found.reshape(ny, nz)
    if not bool(valid_grid.all()):
        _fill_undefined(nonlinear_grid, valid_grid)
    output = nonlinear_grid * source_spacing.to(torch.float32) + affine_grid
    # NEWIMAGE internally flips neurological reference images to radiological.
    if np.linalg.det(reference.affine[:3, :3]) > 0:
        output = output.flip(0)
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    header["intent_code"] = FSL_FNIRT_DISPLACEMENT_FIELD
    header.set_slope_inter(1, 0)
    image = nib.Nifti1Image(output.cpu().numpy(), reference.affine, header=header)
    qform, qcode = reference.get_qform(coded=True)
    sform, scode = reference.get_sform(coded=True)
    image.set_qform(qform, int(qcode))
    image.set_sform(sform, int(scode))
    return image
