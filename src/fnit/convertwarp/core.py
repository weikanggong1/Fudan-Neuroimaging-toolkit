"""Compose FSL scaled-mm affine and nonlinear pull transforms."""

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from ..applywarp.core import (
    FSL_CUBIC_SPLINE_COEFFICIENTS,
    _expand_cubic_coefficients,
    _fsl_voxel_matrix,
    _load_dense_warp,
    _load_nifti,
    _matrix,
    _sample_linear,
    _spatial_grid,
)


def _device(value):
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return device


def _inverse_transform(coordinates, matrix, device):
    """Apply a scaled-mm inverse affine, keeping CUDA's existing arithmetic.

    Omitted and explicitly identity matrices are common in coefficient-field
    conversion. A CPU identity GEMM needlessly copies three full-volume
    float64 arrays; the input already has exactly the required coordinates.
    """
    inverse = np.linalg.inv(matrix)
    if device.type == "cpu" and np.array_equal(inverse, np.eye(4)):
        return coordinates
    inverse = torch.as_tensor(inverse, dtype=torch.float64, device=device)
    flat = coordinates.reshape(3, -1)
    return (inverse[:3, :3] @ flat + inverse[:3, 3:4]).reshape(coordinates.shape)


def _diagonal_scaled_coordinates_cpu(query, matrix):
    """Apply a finite diagonal CPU transform with the original FP64 stores.

    The final positive-zero translation preserves signed-zero behaviour.
    Nonfinite queries keep the full matrix product, including its zero times
    NaN/Inf terms. This helper is used only for prepared repeated sampling.
    """
    if (query.device.type != "cpu" or matrix.device.type != "cpu"
            or query.dtype != torch.float64 or matrix.dtype != torch.float64
            or query.requires_grad or matrix.requires_grad
            or query.layout != torch.strided or matrix.layout != torch.strided
            or not query.is_contiguous() or not matrix.is_contiguous()
            or query.is_neg() or matrix.is_neg()
            or query.ndim < 2 or query.shape[0] != 3 or not query.numel()
            or tuple(matrix.shape) != (4, 4)):
        return None
    diagonal = matrix.diagonal()[:3]
    if (not torch.equal(matrix[:3, :3], torch.diag(diagonal))
            or bool(torch.count_nonzero(matrix[:3, 3].view(torch.int64)))
            or not bool(torch.isfinite(diagonal).all())):
        return None
    low, high = torch.aminmax(query)
    if not bool(torch.isfinite(torch.stack((low, high))).all()):
        return None
    output = torch.empty_like(query, device=query.device)
    layout = (3, *([1] * (query.ndim - 1)))
    torch.mul(query, diagonal.reshape(layout), out=output)
    output.add_(matrix[:3, 3].reshape(layout))
    return output


class _PullField:
    def __init__(self, warp, device, warp_convention):
        image = _load_nifti(warp, "warp")
        self.shape = tuple(int(size) for size in image.shape[:3])
        intent = int(image.header["intent_code"])
        self.embedded_inverse = None
        if intent == FSL_CUBIC_SPLINE_COEFFICIENTS:
            self.values, self.shape, scaled, embedded = _expand_cubic_coefficients(image, device)
            self.embedded_inverse = torch.as_tensor(
                np.linalg.inv(embedded), dtype=torch.float64, device=device)
            self.convention = "relative"
        else:
            image, values, scaled, self.convention, _ = _load_dense_warp(
                image, warp_convention)
            self.values = torch.as_tensor(
                np.moveaxis(values, -1, 0).copy(), device=device)
        self.scaled = scaled
        self.scaled_inverse = torch.as_tensor(
            np.linalg.inv(scaled), dtype=torch.float64, device=device)

    def sample(self, query, *, prepared_source=None, calculate_valid=True):
        flat = query.reshape(3, -1)
        voxels = None
        if (self.values.device.type == "cpu" and prepared_source is not None
                and not self.values.requires_grad):
            voxels = _diagonal_scaled_coordinates_cpu(query, self.scaled_inverse)
        if voxels is None:
            voxels = (self.scaled_inverse[:3, :3] @ flat
                      + self.scaled_inverse[:3, 3:4]).reshape(query.shape)
        if self.values.device.type == "cpu" and (prepared_source is not None or not calculate_valid):
            values, valid = _sample_linear(self.values, voxels, prepared_source=prepared_source,
                                           calculate_valid=calculate_valid)
        else:
            values, valid = _sample_linear(self.values, voxels)
        if self.embedded_inverse is not None:
            if (self.values.device.type == "cpu" and prepared_source is not None
                    and not query.requires_grad and not self.values.requires_grad
                    and not values.requires_grad and not prepared_source.requires_grad
                    and not self.embedded_inverse.requires_grad
                    and query.dtype == self.embedded_inverse.dtype == values.dtype == torch.float64
                    and not torch._C._are_functorch_transforms_active()
                    and all(torch.autograd.forward_ad.unpack_dual(tensor).tangent is None
                            for tensor in (query, self.values, values, prepared_source,
                                           self.embedded_inverse))):
                # All three operands are FP64, so both functional additions
                # would retain FP64. The matrix product is fresh for this call;
                # keep two separate stores without two full-volume allocations.
                base = self.embedded_inverse[:3, :3] @ flat
                base.add_(self.embedded_inverse[:3, 3:4])
                base = base.reshape(query.shape)
                base.add_(values.to(torch.float64))
                return base, valid
            base = (self.embedded_inverse[:3, :3] @ flat
                    + self.embedded_inverse[:3, 3:4]).reshape(query.shape)
            return base + values.to(torch.float64), valid
        if self.convention == "absolute":
            return values.to(torch.float64), valid
        return query + values.to(torch.float64), valid

    def best_fit_affine(self, premat):
        """Fit the float32 absolute prewarp used by FSL outside its grid.

        A regular Cartesian grid has independent centred coordinate columns.
        Its least-squares affine therefore needs only per-axis marginal sums,
        rather than an N-voxel design matrix. Keep the fit in float64 on the
        selected device, after the absolute field's float32 storage boundary.
        The default sampler remains unchanged for InvWarp's iterative solver.
        """
        scaled = torch.as_tensor(self.scaled, dtype=torch.float64,
                                 device=self.values.device)
        if self.embedded_inverse is not None:
            base = self.embedded_inverse @ scaled
        elif self.convention == "relative":
            base = scaled
        else:
            base = torch.zeros_like(scaled)
        axes = [torch.arange(size, dtype=torch.float64, device=self.values.device)
                for size in self.shape]
        absolute = []
        for component in range(3):
            linear = (base[component, 0] * axes[0][:, None, None]
                      + base[component, 1] * axes[1][None, :, None]
                      + base[component, 2] * axes[2][None, None, :]
                      + base[component, 3])
            absolute.append((linear + self.values[component].to(torch.float64)).float())
        absolute = torch.stack(absolute)
        # FSL stores the composed premat/warp1 field in float32 before fitting.
        pre_inverse = torch.as_tensor(np.linalg.inv(premat), dtype=torch.float64,
                                      device=self.values.device)
        flat = absolute.reshape(3, -1).to(torch.float64)
        absolute = (pre_inverse[:3, :3] @ flat
                    + pre_inverse[:3, 3:4]).float().reshape(3, *self.shape)
        count = int(np.prod(self.shape))
        mean = absolute.sum(dim=(1, 2, 3), dtype=torch.float64) / count
        centre = absolute.new_tensor([(size - 1) / 2 for size in self.shape],
                                      dtype=torch.float64)
        affine_voxel = torch.zeros((4, 4), dtype=torch.float64,
                                    device=self.values.device)
        affine_voxel[3, 3] = 1
        for axis, size in enumerate(self.shape):
            if size > 1:
                margins = absolute.sum(dim=tuple(index + 1 for index in range(3)
                                                 if index != axis), dtype=torch.float64)
                covariance = (margins * (axes[axis] - centre[axis])).sum(dim=1) / count
                affine_voxel[:3, axis] = covariance / ((size * size - 1) / 12)
        affine_voxel[:3, 3] = mean - affine_voxel[:3, :3] @ centre
        return affine_voxel @ self.scaled_inverse


def _needs_affine_extrapolation(reference_shape, reference_scaled, field, postmat):
    """Conservatively bound the static affine query box without a CUDA sync."""
    if (tuple(reference_shape) == field.shape
            and np.array_equal(reference_scaled, field.scaled)
            and np.array_equal(postmat, np.eye(4))):
        # FSL stores affine2warp coordinates in float32 before reading them
        # back as voxel queries. Even identity geometry with noninteger voxel
        # sizes can round a last voxel beyond its endpoint. Only skip when
        # all eight endpoint queries remain strictly in bounds after that
        # exact storage boundary; the common same-grid 1/2-mm case is fast.
        corners = np.array(np.meshgrid(
            *[(0, size - 1) for size in reference_shape], indexing="ij"
        )).reshape(3, -1)
        stored = (reference_scaled[:3, :3] @ corners
                  + reference_scaled[:3, 3:4]).astype(np.float32).astype(np.float64)
        inverse = np.linalg.inv(field.scaled)
        voxels = (inverse[:3, :3] @ stored + inverse[:3, 3:4]).astype(np.float32)
        return bool(np.any(voxels < 0) or np.any(voxels > np.asarray(field.shape)[:, None] - 1))
    mapping = np.linalg.inv(field.scaled) @ np.linalg.inv(postmat) @ reference_scaled
    steps = mapping[:3, :3] * (np.asarray(reference_shape) - 1)[None]
    lower = mapping[:3, 3] + np.minimum(steps, 0).sum(axis=1)
    upper = mapping[:3, 3] + np.maximum(steps, 0).sum(axis=1)
    offset_voxels = np.abs(field.scaled[:3, 3] / np.diag(field.scaled)[:3])
    margin = (8 * np.finfo(np.float32).eps
              * (np.maximum(np.abs(lower), np.abs(upper)) + offset_voxels + 1))
    # Nonidentity transforms touching a boundary must not be skipped: a
    # negative sub-voxel offset stays negative after float32 conversion.
    return bool(np.any(lower <= margin) or np.any(upper >= np.asarray(field.shape) - 1 - margin))


def _conversion_inside(field, query):
    """FSL concat_warps uses strict bounds on a float32 voxel query."""
    stored = query.float().double()
    flat = stored.reshape(3, -1)
    voxels = (field.scaled_inverse[:3, :3] @ flat
              + field.scaled_inverse[:3, 3:4]).float().reshape(query.shape)
    valid = torch.ones(query.shape[1:], dtype=torch.bool, device=query.device)
    for axis, size in enumerate(field.shape):
        valid &= (voxels[axis] >= 0) & (voxels[axis] <= size - 1)
    return stored, valid


def _field_image(reference, coordinates, *, intent_code=0):
    reference = _load_nifti(reference, "reference")
    if len(reference.shape) != 3:
        raise ValueError("reference must be a 3D NIfTI image")
    data = np.moveaxis(coordinates.detach().cpu().numpy(), 0, -1).astype(np.float32)
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    image = type(reference)(data, reference.affine, header=header)
    qform, qcode = reference.get_qform(coded=True)
    sform, scode = reference.get_sform(coded=True)
    image.set_qform(qform, int(qcode))
    image.set_sform(sform, int(scode))
    image.header["intent_code"] = intent_code
    return image


@dataclass(frozen=True)
class WarpFieldResult:
    image: nib.spatialimages.SpatialImage
    valid_fraction: float
    qc: dict

    def save(self, output):
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(self.image, str(path))
        return self


class TorchConvertWarp:
    """FSL ``convertwarp --warp1 --premat`` subset; output is a pull field."""

    def __init__(self, device="cpu"):
        self.device = _device(device)

    @torch.inference_mode()
    def __call__(self, reference, warp1, *, premat=None, postmat=None,
                 warp_convention="auto", output_convention="relative"):
        if output_convention not in ("relative", "absolute"):
            raise ValueError("output_convention must be relative or absolute")
        reference = _load_nifti(reference, "reference")
        if len(reference.shape) != 3:
            raise ValueError("reference must be a 3D NIfTI image")
        field = _PullField(warp1, self.device, warp_convention)
        reference_scaled = _fsl_voxel_matrix(reference)
        reference_mm = _spatial_grid(reference.shape, reference_scaled,
                                     self.device).reshape(3, *reference.shape)
        pre_matrix = _matrix(premat, "premat")
        post_matrix = _matrix(postmat, "postmat")
        query = _inverse_transform(
            reference_mm, post_matrix, self.device)
        source, valid = field.sample(query)
        source = _inverse_transform(source, pre_matrix, self.device)
        if _needs_affine_extrapolation(reference.shape, reference_scaled, field, post_matrix):
            stored_query, valid = _conversion_inside(field, query)
            fitted = field.best_fit_affine(pre_matrix)
            flat = stored_query.reshape(3, -1)
            extrapolated = (fitted[:3, :3] @ flat
                            + fitted[:3, 3:4]).reshape(query.shape)
            source = torch.where(valid[None], source, extrapolated)
        output = source - reference_mm if output_convention == "relative" else source
        image = _field_image(reference, output)
        return WarpFieldResult(image, float(valid.float().mean().cpu()), {
            "device": str(self.device), "output_convention": output_convention,
            "input_convention": field.convention,
            "transform_order": "premat then warp1 then postmat",
            "matrix_coordinates": "FSL scaled-mm",
            "outside_field": "FSL best-fit affine extrapolation",
        })

    def run(self, reference, warp1, output, **kwargs):
        return self(reference, warp1, **kwargs).save(output)

    @torch.inference_mode()
    def from_mmorf(self, reference, source, mmorf_warp, *, affine,
                   output_convention="relative"):
        """Convert an FNIT MMORF pull field and FA FLIRT matrix to an FSL field."""
        if output_convention not in ("relative", "absolute"):
            raise ValueError("output_convention must be relative or absolute")
        from ..mmorf.core import _mm_to_voxel_axes_rotation, _world_forward

        reference = _load_nifti(reference, "reference")
        source = _load_nifti(source, "source")
        mmorf_warp = _load_nifti(mmorf_warp, "mmorf_warp")
        if len(reference.shape) != 3 or len(source.shape) != 3:
            raise ValueError("reference and source must be 3D NIfTI images")
        if mmorf_warp.shape != (*reference.shape, 3) or not np.allclose(
                mmorf_warp.affine, reference.affine, atol=1e-5, rtol=0):
            raise ValueError("MMORF warp must share the reference grid")
        field_data = np.asarray(mmorf_warp.dataobj, dtype=np.float32)
        if not np.isfinite(field_data).all():
            raise ValueError("MMORF warp must contain only finite values")
        world_forward = _world_forward(_matrix(affine, "affine"), source, reference)
        source_scaled_pull = torch.as_tensor(
            _fsl_voxel_matrix(source) @ np.linalg.inv(source.affine)
            @ np.linalg.inv(world_forward), dtype=torch.float64,
            device=self.device)
        reference_world = _spatial_grid(reference.shape, reference.affine,
                                        self.device).reshape(3, *reference.shape)
        field = torch.as_tensor(
            np.moveaxis(field_data, -1, 0).copy(),
            device=self.device)
        rotation = _mm_to_voxel_axes_rotation(reference.affine, device=self.device)
        displacement = rotation.T.to(torch.float64) @ field.reshape(3, -1).to(torch.float64)
        query = reference_world.reshape(3, -1) + displacement
        source_scaled = (source_scaled_pull[:3, :3] @ query
                         + source_scaled_pull[:3, 3:4]).reshape(reference_world.shape)
        if self.device.type == "cpu" and output_convention == "absolute":
            output = source_scaled
        else:
            reference_scaled = _spatial_grid(
                reference.shape, _fsl_voxel_matrix(reference), self.device
            ).reshape(reference_world.shape)
            output = (source_scaled - reference_scaled if output_convention == "relative"
                      else source_scaled)
        return WarpFieldResult(_field_image(reference, output), 1.0, {
            "device": str(self.device), "output_convention": output_convention,
            "input_convention": "MMORF reference-axis mm",
            "transform_order": "FA FLIRT affine then MMORF nonlinear",
            "matrix_coordinates": "FSL scaled-mm",
        })

    def run_mmorf(self, reference, source, mmorf_warp, output, *, affine,
                  output_convention="relative"):
        return self.from_mmorf(
            reference, source, mmorf_warp, affine=affine,
            output_convention=output_convention).save(output)


def convertwarp(reference, warp1, **kwargs):
    device = kwargs.pop("device", "cpu")
    return TorchConvertWarp(device)(reference, warp1, **kwargs)
