"""CUDA-capable resampling for FSL displacement fields.

The implemented transform is the dense-field subset of FSL ``applywarp``:

``input --premat--> warp source --warp--> warp reference --postmat--> reference``

Matrices use FLIRT scaled-mm coordinates. Fields are pull fields and store
millimetres along the FSL scaled-mm axes. Dense fields and FNIRT cubic
B-spline coefficient files are supported.
"""

from __future__ import annotations

from dataclasses import dataclass
from .._world_resampling import WorldTransformChain, resample_world_image
import os
import operator
from pathlib import Path
import uuid

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from .._sampling_plan import SamplingGeometry


FSL_FNIRT_DISPLACEMENT_FIELD = 2006
FSL_CUBIC_SPLINE_COEFFICIENTS = 2007
FSL_DCT_COEFFICIENTS = 2008
FSL_QUADRATIC_SPLINE_COEFFICIENTS = 2009
FSL_COEFFICIENT_INTENTS = {
    FSL_CUBIC_SPLINE_COEFFICIENTS,
    FSL_DCT_COEFFICIENTS,
    FSL_QUADRATIC_SPLINE_COEFFICIENTS,
}
_DTYPES = {
    "char": np.dtype("uint8"),
    "short": np.dtype("int16"),
    "int": np.dtype("int32"),
    "float": np.dtype("float32"),
    "double": np.dtype("float64"),
}
_CUDA_MEMORY_LIMIT_BYTES = 20_000_000_000
_CUDA_MEMORY_HEADROOM_BYTES = 512 * 1024 * 1024
_FRAME_BUFFER_BUDGET_BYTES = 8 * 1024 * 1024 * 1024


def _check_frame_chunk_size(value):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("frame_chunk_size must be a positive integer or None")
    try:
        value = operator.index(value)
    except TypeError as error:
        raise ValueError("frame_chunk_size must be a positive integer or None") from error
    if value < 1:
        raise ValueError("frame_chunk_size must be a positive integer or None")
    return value


def _load_nifti(value, name):
    if isinstance(value, (str, os.PathLike)):
        image = nib.load(str(value))
    elif isinstance(value, (nib.Nifti1Image, nib.Nifti2Image)):
        image = value
    else:
        raise TypeError(f"{name} must be a NIfTI path or image")
    if not isinstance(image, (nib.Nifti1Image, nib.Nifti2Image)):
        raise TypeError(f"{name} must be a NIfTI image")
    if len(image.shape) < 3 or any(size < 1 for size in image.shape[:3]):
        raise ValueError(f"{name} must have three nonempty spatial dimensions")
    return image


def _input_data(image):
    data = np.asanyarray(image.dataobj)
    if data.ndim not in (3, 4):
        raise ValueError("input must be a 3D image or a 4D series")
    data = np.asarray(data, dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError("input must contain only finite values")
    if data.ndim == 3:
        data = data[..., None]
    return data


def _matrix(value, name):
    if value is None:
        return np.eye(4, dtype=np.float64)
    if isinstance(value, (str, os.PathLike)):
        value = np.loadtxt(value, dtype=np.float64)
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be one finite 4x4 matrix")
    if not np.allclose(matrix[3], (0, 0, 0, 1), atol=1e-8, rtol=0):
        raise ValueError(f"{name} must be a homogeneous affine matrix")
    if abs(float(np.linalg.det(matrix[:3, :3]))) < 1e-10:
        raise ValueError(f"{name} must be invertible")
    return matrix


def _fsl_voxel_matrix(image):
    affine = np.asarray(image.affine, dtype=np.float64)
    voxel_sizes = np.asarray(image.header.get_zooms()[:3], dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("NIfTI affine must be one finite 4x4 matrix")
    if abs(float(np.linalg.det(affine[:3, :3]))) < 1e-10:
        raise ValueError("NIfTI affine must be invertible")
    if not np.isfinite(voxel_sizes).all() or np.any(voxel_sizes <= 0):
        raise ValueError("NIfTI voxel sizes must be positive and finite")
    matrix = np.diag((*voxel_sizes, 1.0))
    if np.linalg.det(affine[:3, :3]) > 0:
        matrix[0, 0] *= -1
        matrix[0, 3] = voxel_sizes[0] * (image.shape[0] - 1)
    return matrix


def _spatial_grid(shape, matrix, device):
    # FSL coordinate matrices are double precision. Keep geometry out of TF32.
    axes = torch.meshgrid(
        *(torch.arange(size, dtype=torch.float64, device=device) for size in shape),
        indexing="ij",
    )
    voxels = torch.stack(axes).reshape(3, -1)
    transform = torch.as_tensor(matrix, dtype=torch.float64, device=device)
    return transform[:3, :3] @ voxels + transform[:3, 3:4]


def _to_grid(coordinates, shape):
    normalized = []
    for axis, size in enumerate(shape):
        if size == 1:
            normalized.append(torch.zeros_like(coordinates[axis]))
        else:
            normalized.append(2 * coordinates[axis] / (size - 1) - 1)
    output_shape = coordinates.shape[1:]
    return torch.stack(normalized[::-1], dim=-1).reshape(1, *output_shape, 3)


def _inside(coordinates, shape):
    valid = torch.ones(
        coordinates.shape[1:], dtype=torch.bool, device=coordinates.device
    )
    for axis, size in enumerate(shape):
        valid &= coordinates[axis] >= -1e-6
        valid &= coordinates[axis] <= size - 1 + 1e-6
    return valid


def _sample_linear(data, coordinates):
    shape = tuple(int(size) for size in data.shape[1:])
    sampled = F.grid_sample(
        data[None],
        _to_grid(coordinates, shape).to(dtype=data.dtype),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )[0]
    return sampled, _inside(coordinates, shape)


def _sample_nearest(data, coordinates):
    shape = tuple(int(size) for size in data.shape[1:])
    flat = _nearest_indices(coordinates, shape)
    sampled = data.reshape(data.shape[0], -1)[:, flat]
    sampled = sampled.reshape(data.shape[0], *coordinates.shape[1:])
    return sampled, _inside(coordinates, shape)


def _nearest_indices(coordinates, shape):
    # FSL's MISCMATHS::round rounds positive half-integers upward. PyTorch's
    # nearest grid sampler does not promise that tie rule, so gather directly.
    indices = torch.floor(coordinates + 0.5).to(torch.long)
    for axis, size in enumerate(shape):
        indices[axis].clamp_(0, size - 1)
    flat = indices[0] * (shape[1] * shape[2]) + indices[1] * shape[2] + indices[2]
    return flat.reshape(-1)


def _normalise_convention(value):
    aliases = {
        "auto": "auto",
        "rel": "relative",
        "relative": "relative",
        "abs": "absolute",
        "absolute": "absolute",
    }
    try:
        return aliases[str(value).lower()]
    except KeyError as error:
        raise ValueError("warp_convention must be auto, relative, or absolute") from error


def _infer_convention(field, warp_fsl):
    original_sd = sum(float(np.std(field[..., axis])) for axis in range(3))
    relative_sd = 0.0
    for axis, size in enumerate(field.shape[:3]):
        index = np.arange(size, dtype=np.float32)
        reshape = [1, 1, 1]
        reshape[axis] = size
        base = warp_fsl[axis, axis] * index.reshape(reshape) + warp_fsl[axis, 3]
        relative_sd += float(np.std(field[..., axis] - base))
    return "absolute" if not np.isfinite(relative_sd) or original_sd > relative_sd else "relative"


def _coefficient_metadata(image):
    if len(image.shape) != 4 or image.shape[3] != 3:
        raise ValueError("FNIRT coefficients must be a 4D NIfTI image with final dimension 3")
    header = image.header
    field_shape = tuple(
        int(float(header[name]))
        for name in ("qoffset_x", "qoffset_y", "qoffset_z")
    )
    voxel_sizes = tuple(
        float(header[name]) for name in ("intent_p1", "intent_p2", "intent_p3")
    )
    knot_spacing = tuple(
        int(np.floor(abs(float(value)) + 0.5)) for value in header["pixdim"][1:4]
    )
    if any(size < 1 for size in field_shape):
        raise ValueError("FNIRT coefficient qform offsets do not contain a valid field size")
    if any(not np.isfinite(size) or size <= 0 for size in voxel_sizes):
        raise ValueError("FNIRT coefficient intent parameters do not contain valid voxel sizes")
    if any(spacing < 1 for spacing in knot_spacing):
        raise ValueError("FNIRT coefficient pixdims do not contain valid knot spacings")
    expected_shape = tuple(
        size if spacing == 1 else int(np.ceil((size + 1) / spacing)) + 2
        for size, spacing in zip(field_shape, knot_spacing)
    )
    if tuple(image.shape[:3]) != expected_shape:
        raise ValueError(
            "FNIRT cubic coefficient dimensions are inconsistent with field size "
            "and knot spacing metadata"
        )
    affine = np.asarray(image.get_sform(), dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("FNIRT coefficient sform does not contain a finite embedded affine")
    if abs(float(np.linalg.det(affine[:3, :3]))) < 1e-10:
        raise ValueError("FNIRT coefficient embedded affine is not invertible")
    return field_shape, voxel_sizes, knot_spacing, affine


def _cubic_basis_matrix(size, coefficient_count, knot_spacing, device):
    # FNIRT evaluates coefficient geometry in double precision.
    voxels = torch.arange(size, dtype=torch.float64, device=device)[:, None]
    coefficients = torch.arange(
        coefficient_count, dtype=torch.float64, device=device
    )[None]
    if knot_spacing == 1:
        centres = coefficients
    else:
        centres = (coefficients - 1) * knot_spacing
    distance = torch.abs((voxels - centres) / knot_spacing)
    inner = (2.0 / 3.0) + distance.square() * (0.5 * distance - 1.0)
    outer = (1.0 / 6.0) * (2.0 - distance).clamp_min(0).pow(3)
    return torch.where(distance <= 1, inner, outer) * (distance < 2)


def _expand_cubic_coefficients(image, device):
    field_shape, voxel_sizes, knot_spacing, embedded_affine = (
        _coefficient_metadata(image)
    )
    coefficients = np.asarray(image.dataobj, dtype=np.float32)
    if not np.isfinite(coefficients).all():
        raise ValueError("FNIRT coefficients must contain only finite values")
    coefficient_tensor = torch.as_tensor(
        coefficients.copy(), dtype=torch.float64, device=device
    )
    bases = tuple(
        _cubic_basis_matrix(size, count, spacing, device)
        for size, count, spacing in zip(
            field_shape, image.shape[:3], knot_spacing
        )
    )
    expanded = torch.einsum("xi,ijkc->xjkc", bases[0], coefficient_tensor)
    expanded = torch.einsum("yj,xjkc->xykc", bases[1], expanded)
    expanded = torch.einsum("zk,xykc->xyzc", bases[2], expanded)
    field_fsl = np.diag((*voxel_sizes, 1.0))
    return (
        expanded.permute(3, 0, 1, 2).contiguous(),
        field_shape,
        field_fsl,
        embedded_affine,
    )


def _load_dense_warp(value, requested_convention):
    image = _load_nifti(value, "warp")
    intent = int(image.header["intent_code"])
    if intent in (FSL_DCT_COEFFICIENTS, FSL_QUADRATIC_SPLINE_COEFFICIENTS):
        raise NotImplementedError(
            "only FNIRT cubic coefficient intent 2007 is supported; convert DCT or "
            "quadratic coefficients to a dense field with fnirtfileutils"
        )
    if intent == FSL_CUBIC_SPLINE_COEFFICIENTS:
        raise ValueError("internal error: cubic coefficients must use the coefficient loader")
    if len(image.shape) != 4 or image.shape[3] != 3:
        raise ValueError("warp must be a dense 4D NIfTI field with final dimension 3")
    field = np.asarray(image.dataobj, dtype=np.float32)
    if not np.isfinite(field).all():
        raise ValueError("warp must contain only finite values")
    warp_fsl = _fsl_voxel_matrix(image)
    requested = _normalise_convention(requested_convention)
    if intent == FSL_FNIRT_DISPLACEMENT_FIELD:
        convention = "relative"
        convention_source = "FSL intent 2006"
    elif requested == "auto":
        convention = _infer_convention(field, warp_fsl)
        convention_source = "FSL standard-deviation heuristic"
    else:
        convention = requested
        convention_source = "explicit argument"
    return image, field, warp_fsl, convention, convention_source


def _resolve_dtype(input_image, result, output_dtype):
    if output_dtype is not None:
        if isinstance(output_dtype, str) and output_dtype in _DTYPES:
            return _DTYPES[output_dtype]
        try:
            dtype = np.dtype(output_dtype)
        except TypeError as error:
            raise ValueError("unsupported output dtype") from error
        if dtype not in set(_DTYPES.values()):
            raise ValueError("output dtype must be uint8, int16, int32, float32, or float64")
        return dtype
    source_dtype = np.dtype(input_image.header.get_data_dtype())
    if np.issubdtype(source_dtype, np.integer):
        if result.ndim == 4 and result.shape[3] > 10:
            value_range = float(result[..., 0].max() - result[..., 1].min())
        else:
            value_range = float(result.max() - result.min()) if result.size else 0.0
        return np.dtype("float32") if value_range < 100 else source_dtype
    if source_dtype == np.dtype("float64"):
        return source_dtype
    return np.dtype("float32")


def _cast_output(data, dtype):
    if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)
        data = np.clip(np.trunc(data), limits.min, limits.max)
    return data.astype(dtype, copy=False)


def _output_image(reference, input_image, data, dtype):
    header = reference.header.copy()
    header.set_data_dtype(dtype)
    header.set_slope_inter(1.0, 0.0)
    header["cal_min"] = 0
    header["cal_max"] = 0
    image_class = type(reference)
    output = image_class(_cast_output(data, dtype), reference.affine, header=header)
    qform, qcode = reference.get_qform(coded=True)
    sform, scode = reference.get_sform(coded=True)
    output.set_qform(qform, int(qcode))
    output.set_sform(sform, int(scode))
    if data.ndim == 4:
        spatial_zooms = tuple(reference.header.get_zooms()[:3])
        input_zooms = input_image.header.get_zooms()
        time_zoom = input_zooms[3] if len(input_zooms) > 3 else 1.0
        output.header.set_zooms((*spatial_zooms, time_zoom))
    return output


@dataclass(frozen=True)
class ApplyWarpResult:
    """Reference-grid image, validity mask, and transform diagnostics."""

    image: nib.spatialimages.SpatialImage
    valid_mask: np.ndarray
    qc: dict

    def save(self, output):
        """Atomically save the warped NIfTI image."""
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = ".nii.gz" if path.name.endswith(".nii.gz") else path.suffix
        temporary = path.with_name(
            f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}{suffix}"
        )
        try:
            nib.save(self.image, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return self


@dataclass(frozen=True)
class ApplyWarpPlan:
    """A reusable transform snapshot, restricted to one exact image grid."""

    device: torch.device
    input_geometry: SamplingGeometry
    reference_geometry: SamplingGeometry
    interpolation: str
    _reference: object
    _coordinates: torch.Tensor
    _grid: torch.Tensor | None
    _valid: torch.Tensor
    _valid_mask: np.ndarray
    _valid_fraction: float
    _warp_representation: str
    _intent: int | None
    _convention: str | None
    _convention_source: str | None
    frame_chunk_size: int | None = None
    _nearest_flat: torch.Tensor | None = None

    def apply(self, input, *, reference=None, output_dtype=None, frame_chunk_size=None):
        """Sample another 3D/4D image on the captured source grid.

        A positive ``frame_chunk_size`` overrides the captured frame policy.
        ``None`` uses that policy: CPU samples all frames, while CUDA limits
        working buffers to 8 GiB and a 20 GB allocation budget with 512 MiB
        reserved for other temporary allocations. All frames use one call
        when their conservative buffer estimate fits these budgets.
        """
        input_image = _load_nifti(input, "input")
        reference_image = self._reference if reference is None else _load_nifti(reference, "reference")
        self.input_geometry.require(input_image, "input")
        self.reference_geometry.require(reference_image, "reference")
        return self._apply_loaded(
            input_image, _input_data(input_image), reference_image, output_dtype,
            frame_chunk_size=frame_chunk_size,
        )

    def _frame_chunk(self, frame_count, input_shape, requested):
        requested = _check_frame_chunk_size(requested)
        requested = self.frame_chunk_size if requested is None else requested
        if self.device.type != "cuda":
            return frame_count if requested is None else min(frame_count, requested)
        # Count the reusable input and conservatively two output buffers. The
        # prepared geometry is already included in memory_allocated(). This
        # policy deliberately does not vary with other jobs' free GPU memory.
        bytes_per_frame = 4 * (
            int(np.prod(input_shape)) + 2 * int(np.prod(self.reference_geometry.shape))
        )
        allocated = torch.cuda.memory_allocated(self.device)
        available = (
            _CUDA_MEMORY_LIMIT_BYTES - allocated - _CUDA_MEMORY_HEADROOM_BYTES
        )
        if requested is None:
            chunk = min(
                frame_count,
                min(_FRAME_BUFFER_BUDGET_BYTES, available) // bytes_per_frame,
            )
        else:
            chunk = min(frame_count, requested)
        if chunk < 1 or chunk * bytes_per_frame > available:
            raise RuntimeError(
                "frame sampling exceeds the 20 GB CUDA allocation budget; "
                "use a smaller frame_chunk_size or release other tensors"
            )
        return chunk

    def _sample_frames(self, frames, valid_weights):
        if self.interpolation == "trilinear":
            sampled = F.grid_sample(
                frames[None], self._grid, mode="bilinear",
                padding_mode="border", align_corners=True,
            )[0]
        else:
            sampled = frames.reshape(frames.shape[0], -1)[:, self._nearest_flat]
            sampled = sampled.reshape(frames.shape[0], *self.reference_geometry.shape)
        # Retain multiplication rather than zero filling: negative samples
        # outside the valid mask retain the original negative-zero bits.
        sampled.mul_(valid_weights[None])
        return sampled

    def _apply_loaded(
        self, input_image, input_data, reference_image, output_dtype,
        *, frame_chunk_size=None,
    ):
        frame_count = input_data.shape[-1]
        chunk_size = self._frame_chunk(frame_count, input_data.shape[:3], frame_chunk_size)
        valid_weights = self._valid.to(torch.float32)
        if chunk_size == frame_count:
            frames = torch.as_tensor(
                np.ascontiguousarray(np.moveaxis(input_data, -1, 0)),
                dtype=torch.float32, device=self.device,
            )
            result_frames = self._sample_frames(frames, valid_weights).detach().cpu().numpy()
        else:
            result_frames = np.empty(
                (frame_count, *self.reference_geometry.shape), dtype=np.float32,
            )
            output_tensor = torch.from_numpy(result_frames)
            frame_buffer = torch.empty(
                (chunk_size, *input_data.shape[:3]),
                dtype=torch.float32, device=self.device,
            )
            for start in range(0, frame_count, chunk_size):
                end = min(start + chunk_size, frame_count)
                host_frames = np.ascontiguousarray(
                    np.moveaxis(input_data[..., start:end], -1, 0), dtype=np.float32,
                )
                frames = frame_buffer[:end - start]
                frames.copy_(torch.from_numpy(host_frames))
                sampled = self._sample_frames(frames, valid_weights)
                # Copy directly into the final CPU array instead of allocating
                # another CPU result and copying it into the output afterwards.
                output_tensor[start:end].copy_(sampled.detach())
                del sampled
        result = np.moveaxis(result_frames, 0, -1)
        if input_image.ndim == 3:
            result = result[..., 0]
        dtype = _resolve_dtype(input_image, result, output_dtype)
        output = _output_image(reference_image, input_image, result, dtype)
        qc = {
            "device": str(self.device),
            "tf32": {
                "matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                "cudnn": bool(torch.backends.cudnn.allow_tf32),
                "reduced_precision_tensor_dtype": False,
            },
            "output_grid": "reference",
            "matrix_coordinate_system": "FSL scaled-mm",
            "transform_order": "premat then warp then postmat",
            "warp_representation": self._warp_representation,
            "warp_intent_code": self._intent,
            "warp_convention": self._convention,
            "warp_convention_source": self._convention_source,
            "interpolation": self.interpolation,
            "valid_fraction": self._valid_fraction,
            "output_dtype": dtype.name,
            "supports_fnirt_cubic_coefficients": True,
            "supports_fnirt_dct_or_quadratic_coefficients": False,
            "covered_fsl_applywarp_subset": (
                "dense relative/absolute field or FNIRT cubic coefficients, "
                "one premat, one postmat, trilinear/nearest, reference-grid output"
            ),
        }
        return ApplyWarpResult(image=output, valid_mask=self._valid_mask.copy(), qc=qc)


class TorchApplyWarp:
    """Apply FSL pull fields on CPU or CUDA.

    The output always uses ``reference`` shape and NIfTI geometry. ``premat``
    maps input to the warp source; ``postmat`` maps the warp reference to the
    final reference. Both matrices follow FLIRT scaled-mm convention.

    Supported interpolation modes are ``trilinear`` and ``nearest``. Dense
    relative/absolute fields and FNIRT cubic coefficient files are supported.
    DCT/quadratic coefficients, sinc/spline image interpolation, supersampling,
    padding, and stacked per-frame matrices are never silently approximated.
    """

    def __init__(self, device=None, *, frame_chunk_size=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.frame_chunk_size = _check_frame_chunk_size(frame_chunk_size)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def __call__(
        self,
        input,
        reference,
        *,
        warp=None,
        premat=None,
        postmat=None,
        interpolation="trilinear",
        warp_convention="auto",
        output_dtype=None,
    ):
        input_image = _load_nifti(input, "input")
        input_data = _input_data(input_image)
        plan = self.prepare(
            input_image, reference, warp=warp, premat=premat, postmat=postmat,
            interpolation=interpolation, warp_convention=warp_convention,
        )
        return plan._apply_loaded(input_image, input_data, plan._reference, output_dtype)

    def prepare(
        self, input, reference, *, warp=None, premat=None, postmat=None,
        interpolation="trilinear", warp_convention="auto",
    ):
        """Snapshot one transform and its exact source/reference geometry.

        Reuse ``plan.apply(image)`` only for images on the captured native grid.
        Coordinates retain float64; each image still uses its own float32
        sampling call, dtype selection and reference-header output path.
        Transform values are captured here; changed warps require a new plan.
        """
        input_image = _load_nifti(input, "input")
        reference_image = _load_nifti(reference, "reference")
        if input_image.ndim not in (3, 4):
            raise ValueError("input must be a 3D image or a 4D series")
        if interpolation in ("nn", "nearest-neighbour", "nearest_neighbor"):
            interpolation = "nearest"
        if interpolation not in ("trilinear", "nearest"):
            raise ValueError("interpolation must be trilinear or nearest")
        premat_array = _matrix(premat, "premat")
        postmat_array = _matrix(postmat, "postmat")

        input_fsl = _fsl_voxel_matrix(input_image)
        reference_fsl = _fsl_voxel_matrix(reference_image)
        reference_shape = tuple(int(size) for size in reference_image.shape[:3])
        output_mm = _spatial_grid(reference_shape, reference_fsl, self.device)
        post_inverse = torch.as_tensor(
            np.linalg.inv(postmat_array), dtype=torch.float64, device=self.device
        )
        warp_query_mm = (
            post_inverse[:3, :3] @ output_mm + post_inverse[:3, 3:4]
        ).reshape(3, *reference_shape)

        warp_valid = torch.ones(reference_shape, dtype=torch.bool, device=self.device)
        convention = None
        convention_source = None
        intent = None
        if warp is None:
            source_mm = warp_query_mm
            warp_representation = "none"
        else:
            warp_image = _load_nifti(warp, "warp")
            intent = int(warp_image.header["intent_code"])
            embedded_affine = None
            if intent == FSL_CUBIC_SPLINE_COEFFICIENTS:
                (
                    warp_tensor,
                    _,
                    warp_fsl,
                    embedded_affine,
                ) = _expand_cubic_coefficients(warp_image, self.device)
                convention = "relative"
                convention_source = "FSL coefficient intent 2007"
                warp_representation = "FNIRT cubic spline coefficients"
            else:
                (
                    warp_image,
                    warp_data,
                    warp_fsl,
                    convention,
                    convention_source,
                ) = _load_dense_warp(warp_image, warp_convention)
                warp_tensor = torch.as_tensor(
                    np.moveaxis(warp_data, -1, 0).copy(),
                    dtype=torch.float32,
                    device=self.device,
                )
                warp_representation = "FSL dense field"
            warp_world_to_voxel = torch.as_tensor(
                np.linalg.inv(warp_fsl), dtype=torch.float64, device=self.device
            )
            flat_query = warp_query_mm.reshape(3, -1)
            warp_voxels = (
                warp_world_to_voxel[:3, :3] @ flat_query
                + warp_world_to_voxel[:3, 3:4]
            ).reshape(3, *reference_shape)
            displacement, warp_valid = _sample_linear(warp_tensor, warp_voxels)
            if embedded_affine is not None:
                affine_inverse = torch.as_tensor(
                    np.linalg.inv(embedded_affine),
                    dtype=torch.float64,
                    device=self.device,
                )
                flat_query = warp_query_mm.reshape(3, -1)
                affine_source = (
                    affine_inverse[:3, :3] @ flat_query
                    + affine_inverse[:3, 3:4]
                ).reshape(3, *reference_shape)
                source_mm = affine_source + displacement
            else:
                source_mm = (
                    displacement
                    if convention == "absolute"
                    else warp_query_mm + displacement
                )

        premat_inverse = torch.as_tensor(
            np.linalg.inv(premat_array), dtype=torch.float64, device=self.device
        )
        source_mm = source_mm.reshape(3, -1).to(torch.float64)
        input_mm = (
            premat_inverse[:3, :3] @ source_mm + premat_inverse[:3, 3:4]
        )
        input_world_to_voxel = torch.as_tensor(
            np.linalg.inv(input_fsl), dtype=torch.float64, device=self.device
        )
        input_voxels = (
            input_world_to_voxel[:3, :3] @ input_mm
            + input_world_to_voxel[:3, 3:4]
        ).reshape(3, *reference_shape)

        input_valid = _inside(input_voxels, input_image.shape[:3])
        valid = warp_valid & input_valid
        grid = (
            _to_grid(input_voxels, input_image.shape[:3]).to(torch.float32)
            if interpolation == "trilinear" else None
        )
        return ApplyWarpPlan(
            device=self.device,
            input_geometry=SamplingGeometry.capture(input_image),
            reference_geometry=SamplingGeometry.capture(reference_image, output_header=True),
            interpolation=interpolation,
            _reference=reference_image,
            _coordinates=input_voxels,
            _grid=grid,
            _valid=valid,
            _valid_mask=valid.detach().cpu().numpy(),
            _valid_fraction=float(valid.float().mean().cpu()),
            _warp_representation=warp_representation,
            _intent=intent,
            _convention=convention,
            _convention_source=convention_source,
            frame_chunk_size=self.frame_chunk_size,
            _nearest_flat=(
                _nearest_indices(input_voxels, input_image.shape[:3])
                if interpolation == "nearest" else None
            ),
        )

    def apply_world(
        self, input, transformation, *, interpolation="spline",
        boundary="grid-constant", output_mask=None, batch_size=8,
        spatial_chunk_size=262144,
    ):
        """Apply an explicit RAS world chain and return its NIfTI image.

        This world-coordinate entry supports cubic image sampling and
        per-frame motion. It preserves the mature world sampler's coordinate
        rounding, boundary rules and target/source metadata. The usual FSL
        ``__call__`` entry retains its scaled-mm and nearest/trilinear rules.
        ``batch_size`` defaults to 8 and does not change the constructor's
        separate FSL frame policy. No static FSL ``valid_mask`` is inferred
        from this per-frame chain; an optional ``output_mask`` is explicit.
        """
        if not isinstance(transformation, WorldTransformChain):
            raise TypeError("transformation must be a WorldTransformChain")
        return resample_world_image(
            input, transformation.reference,
            transformation.reference_to_source_world,
            pre_affine_pull_ras=transformation.pre_affine_pull_ras,
            motion_pull_world=transformation.motion_pull_world,
            coordinate_precision=transformation.coordinate_precision,
            output_mask=output_mask, interpolation=interpolation,
            boundary=boundary, batch_size=batch_size,
            spatial_chunk_size=spatial_chunk_size, device=self.device,
        )

    def run_world(self, input, transformation, output, **kwargs):
        """Apply a world chain, save the full image and return its Path."""
        import nibabel as nib

        image = self.apply_world(input, transformation, **kwargs)
        output_path = Path(output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(image, str(output_path))
        return output_path

    def run(self, input, reference, output, **kwargs):
        """Apply the transform and save ``output``."""
        return self(input, reference, **kwargs).save(output)


def applywarp(input, reference, **kwargs):
    """Functional interface for :class:`TorchApplyWarp`."""
    device = kwargs.pop("device", None)
    frame_chunk_size = kwargs.pop("frame_chunk_size", None)
    return TorchApplyWarp(device=device, frame_chunk_size=frame_chunk_size)(input, reference, **kwargs)


__all__ = [
    "ApplyWarpResult",
    "ApplyWarpPlan",
    "FSL_COEFFICIENT_INTENTS",
    "FSL_CUBIC_SPLINE_COEFFICIENTS",
    "FSL_DCT_COEFFICIENTS",
    "FSL_FNIRT_DISPLACEMENT_FIELD",
    "FSL_QUADRATIC_SPLINE_COEFFICIENTS",
    "TorchApplyWarp",
    "WorldTransformChain",
    "applywarp",
]
