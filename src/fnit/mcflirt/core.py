"""MCFLIRT 2111.0 的三阶段刚体估计；运行时只使用 NumPy 和 PyTorch。"""

from dataclasses import dataclass
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from ..flirt.core import (
    _BASE_TOLERANCE, _centre_of_gravity, _flip_to_radiological,
    _fsl_pull_coefficients, _manual_trilinear, fsl_affine_from_parameters,
    fsl_coordinate_optimize, fsl_parameters_from_affine,
)

MCFLIRT_SOURCE_VERSION = "2111.0"
MCFLIRT_SOURCE_URL = "https://git.fmrib.ox.ac.uk/fsl/mcflirt/-/blob/2111.0/mcflirt.cc"


def _load_image(value):
    image = nib.load(str(value)) if isinstance(value, (str, bytes, os.PathLike)) else value
    if not isinstance(image, (nib.Nifti1Image, nib.Nifti2Image)):
        raise TypeError("input and reference must be NIfTI images or paths")
    return image


def mcflirt_output_dtype(input_dtype):
    """NEWIMAGE closestTemplatedType 的 MCFLIRT 输出数据类型。"""
    dtype = np.dtype(input_dtype)
    if dtype.kind in "iu" and dtype.itemsize == 1:
        return np.dtype(np.uint8)
    if dtype == np.dtype(np.int16):
        return np.dtype(np.int16)
    if dtype in (np.dtype(np.int32), np.dtype(np.uint16)):
        return np.dtype(np.int32)
    if dtype.kind in "iu" or dtype == np.dtype(np.float32):
        return np.dtype(np.float32)
    if dtype.kind == "f" and dtype.itemsize >= 8:
        return np.dtype(np.float64)
    raise ValueError("MCFLIRT requires a supported real-valued NIfTI data type")


def _isotropic_reference(data, voxel_sizes, scale):
    """NEWIMAGE isotropic_resample：无预滤波，逐轴 float32 累加坐标。"""
    steps = np.asarray(scale / np.asarray(voxel_sizes), dtype=np.float32)
    shape = tuple(max(1, int(np.float32(data.shape[i]) / steps[i])) for i in range(3))
    axes = []
    for size, step in zip(shape, steps):
        axis = np.empty(size, dtype=np.float32)
        position = np.float32(0)
        for index in range(size):
            axis[index] = position
            position = np.float32(position + step)
        axes.append(torch.as_tensor(axis, device=data.device))
    coordinates = torch.stack(torch.meshgrid(*axes, indexing="ij")).reshape(3, -1)
    upper = data.new_tensor(data.shape)[:, None] - 1
    valid = ((coordinates >= 0) & (coordinates <= upper)).all(0)
    output = data.new_zeros(int(np.prod(shape)))
    output[valid] = _manual_trilinear(data, coordinates[:, valid])
    return output.reshape(shape)


def _serial_sum_last(values):
    """Float32 行内累加，保持 NEWIMAGE 的 x、y、z 分块次序。"""
    total = torch.zeros_like(values[..., 0])
    for index in range(values.shape[-1]):
        total = total + values[..., index]
    return total


def _normcorr_reduce(reference, values, weights):
    """NEWIMAGE p_normcorr_smoothed 的方差式，不是通常的加权 Pearson。"""
    terms = torch.stack((weights, weights * reference, (weights * reference) * reference,
                         weights * values, (weights * values) * values,
                         (weights * reference) * values))
    # The source resets intensity sums at row/plane boundaries, but leaves
    # num and numA running. Keep that observed count contract: replacing it
    # with the ordinary weight sum changes the default MCFLIRT objective.
    rows = _serial_sum_last(terms)
    planes = _serial_sum_last(rows)
    sums = _serial_sum_last(planes)
    _, sx, sx2, sy, sy2, sxy = sums.unbind()
    row_counts = rows[0].reshape(-1)
    cumulative_count = torch.cumsum(row_counts, 0)
    cumulative_count_a = torch.cumsum(cumulative_count, 0)
    count = cumulative_count_a[weights.shape[1] - 1::weights.shape[1]].sum()
    denominator = count - 1.0
    count_square = count * count
    covariance = sxy / denominator - (sx * sy) / count_square
    vx = sx2 / denominator - (sx * sx) / count_square
    vy = sy2 / denominator - (sy * sy) / count_square
    corr = covariance / torch.sqrt(vx) / torch.sqrt(vy)
    valid = (count > 2) & (vx > 0) & (vy > 0)
    return torch.where(valid, 1.0 - corr.abs(), torch.ones_like(corr))


class FSLMotionNormCorr:
    """MCFLIRT 的 1 mm 边界降权 NCC；全部 moving 体素参与，不使用脑掩膜。"""

    def __init__(self, reference, moving, reference_sizes, moving_sizes):
        self.reference = reference.permute(2, 1, 0).contiguous()
        self.moving = moving.contiguous()
        self.reference_sizes = reference_sizes
        self.moving_sizes = moving_sizes
        self.device = moving.device
        z, y = torch.meshgrid(
            torch.arange(reference.shape[2], device=self.device, dtype=torch.float32),
            torch.arange(reference.shape[1], device=self.device, dtype=torch.float32),
            indexing="ij")
        self.y, self.z = y, z
        self.xsize = reference.shape[0]
        self.upper = moving.new_tensor([size - 1.0001 for size in moving.shape])
        self.smooth = moving.new_tensor([1.0 / size for size in moving_sizes])
        self.centre = _centre_of_gravity(moving, np.diag([*moving_sizes, 1.0]))
        self.cost_evaluations = 0
        self.reducer = _normcorr_reduce
        if self.device.type == "cuda":
            # The fixed small reference grid lets Inductor fuse row-wise
            # scalar accumulation without changing float32 arithmetic.
            self.reducer = torch.compile(_normcorr_reduce, fullgraph=True)

    def __call__(self, matrix):
        coefficients = _fsl_pull_coefficients(matrix, self.moving_sizes,
                                             self.reference_sizes, device=self.device)
        # NEWIMAGE starts each row at xmin and then updates coordinates by
        # repeated float32 addition. Derive xmin first, then preserve that
        # addition order instead of a TF32 affine matrix multiplication.
        origins = torch.stack([
            self.y * row[1] + self.z * row[2] + row[3] for row in coefficients])
        directions = coefficients[:, 0]
        xmin = torch.zeros_like(self.y)
        xmax = torch.full_like(self.y, float(self.xsize - 1))
        for axis in range(3):
            direction = directions[axis]
            if abs(float(direction)) < 1e-8:
                outside = (origins[axis] < 0) | (origins[axis] > self.upper[axis])
                xmin = torch.where(outside, torch.full_like(xmin, self.xsize), xmin)
            else:
                bound0 = -origins[axis] / direction
                bound1 = (self.upper[axis] - origins[axis]) / direction
                lower = torch.minimum(bound0, bound1)
                upper = torch.maximum(bound0, bound1)
                xmin = torch.maximum(xmin, torch.ceil(lower))
                xmax = torch.minimum(xmax, torch.floor(upper))
        xmin = xmin.clamp(0, self.xsize)
        coordinate = origins + xmin[None] * directions[:, None, None]
        coordinates = []
        for _ in range(self.xsize):
            coordinates.append(coordinate)
            coordinate = coordinate + directions[:, None, None]
        coordinates = torch.stack(coordinates, -1)
        offsets = torch.arange(self.xsize, device=self.device)[None, None, :]
        actual_x = xmin[..., None] + offsets
        valid = actual_x <= xmax[..., None]
        valid &= ((coordinates >= 0) & (coordinates <= self.upper[:, None, None, None])).all(0)
        flat_coordinates = coordinates.reshape(3, -1)
        clamped = torch.minimum(flat_coordinates.clamp_min(0), self.upper[:, None])
        values = _manual_trilinear(self.moving, clamped).reshape(*self.y.shape, self.xsize)
        smooth = self.smooth[:, None, None, None]
        upper = self.upper[:, None, None, None]
        weights = torch.where(coordinates < smooth, coordinates / smooth,
                              torch.where(upper - coordinates < smooth,
                                          (upper - coordinates) / smooth, 1.0)).prod(0).clamp_min(0)
        weights = weights * valid
        x_index = actual_x.long().clamp_max(self.xsize - 1)
        reference = torch.gather(self.reference, -1, x_index)
        self.cost_evaluations += 1
        return float(self.reducer(reference, values, weights))


@dataclass
class MCFLIRTResult:
    matrices: np.ndarray
    parameters: np.ndarray
    reference: nib.Nifti1Image
    cost_evaluations: int
    corrected: nib.Nifti1Image | None = None
    output_paths: dict | None = None


class TorchMCFLIRT:
    """估计每帧 input→reference 的 FSL scaled-mm 矩阵及六列 .par 参数。"""

    def __init__(self, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def run(self, input_bold, reference=None, *, stages=3, stage_iterations=(1, 1, 1),
            resample=False, interpolation="linear", output=None, mats=False, plots=False,
            overwrite=False, rmsrel=False, rmsabs=False):
        image = _load_image(input_bold)
        if image.ndim != 4:
            raise ValueError("input_bold must be 4D")
        output_dtype = mcflirt_output_dtype(image.get_data_dtype())
        if stages not in (1, 2, 3) or len(stage_iterations) != 3 or any(v < 0 for v in stage_iterations):
            raise ValueError("stages must be 1, 2 or 3; stage_iterations requires three nonnegative values")
        output_paths = {}
        if output is not None:
            prefix = str(Path(output).expanduser())
            prefix = prefix[:-7] if prefix.endswith(".nii.gz") else (prefix[:-4] if prefix.endswith(".nii") else prefix)
            output_paths["corrected"] = prefix + ".nii.gz"
            if mats:
                output_paths["matrices"] = prefix + ".mat"
            if plots:
                output_paths["parameters"] = prefix + ".par"
            if rmsrel:
                output_paths["rms_relative"] = prefix + "_rel.rms"
                output_paths["rms_relative_mean"] = prefix + "_rel_mean.rms"
            if rmsabs:
                output_paths["rms_absolute"] = prefix + "_abs.rms"
                output_paths["rms_absolute_mean"] = prefix + "_abs_mean.rms"
            for path in output_paths.values():
                if Path(path).exists() and not overwrite:
                    raise FileExistsError(path)
        elif mats or plots or rmsrel or rmsabs:
            raise ValueError("mats, plots and RMS outputs require output")
        if interpolation not in ("linear", "spline"):
            raise ValueError("interpolation must be linear or spline")
        data = np.asarray(image.dataobj, dtype=np.float32)
        frame_count = data.shape[-1]
        if frame_count < 1 or not np.isfinite(data).all():
            raise ValueError("input_bold must contain finite values and at least one frame")
        external_reference = reference is not None
        reference_index = -1 if external_reference else frame_count // 2
        target = _load_image(reference) if external_reference else nib.Nifti1Image(
            data[..., reference_index], image.affine, image.header.copy())
        if target.ndim != 3 or target.shape != image.shape[:3] or not np.allclose(
                target.affine, image.affine, atol=1e-4):
            raise ValueError("BOLD and reference must have the same voxel grid")
        if min(target.shape) < 3 or target.shape[2] * target.header.get_zooms()[2] < 20:
            raise NotImplementedError("MCFLIRT 2D correction is not implemented")
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        reference_array = np.asarray(target.dataobj, dtype=np.float32)
        if not np.isfinite(reference_array).all():
            raise ValueError("reference must contain only finite values")
        reference_data = torch.as_tensor(_flip_to_radiological(reference_array, target.affine), device=self.device)
        sizes = tuple(float(v) for v in image.header.get_zooms()[:3])
        reference_sizes = tuple(float(v) for v in target.header.get_zooms()[:3])
        references = {scale: _isotropic_reference(reference_data, reference_sizes, scale)
                      for scale in (8.0, 4.0)}
        matrices = np.repeat(np.eye(4)[None], frame_count, axis=0)
        order = list(range(reference_index + 1, frame_count)) + list(range(reference_index - 1, -1, -1))
        cost_evaluations = 0
        for stage, (scale, tolerance_multiplier) in enumerate(((8.0, .8), (4.0, .8), (4.0, .1))):
            if stage >= stages or not stage_iterations[stage]:
                continue
            previous = matrices.copy()
            initial = matrices.copy()
            for frame in order:
                moving = torch.as_tensor(_flip_to_radiological(data[..., frame], image.affine),
                                         device=self.device)
                cost = FSLMotionNormCorr(references[scale], moving, (scale,) * 3, sizes)
                parameters = fsl_parameters_from_affine(initial[frame], cost.centre)
                def objective(values):
                    matrix = fsl_affine_from_parameters(torch.as_tensor(values, dtype=torch.float64),
                                                        cost.centre, 6).numpy()
                    return cost(matrix)
                fitted, _ = fsl_coordinate_optimize(
                    parameters, _BASE_TOLERANCE * tolerance_multiplier, objective,
                    maximum_iterations=int(stage_iterations[stage]), numopt=6, bound_guess=(10.0, 1.0))
                previous[frame] = fsl_affine_from_parameters(torch.as_tensor(fitted, dtype=torch.float64),
                                                           cost.centre, 6).numpy()
                cost_evaluations += cost.cost_evaluations
                next_frame = frame + (1 if frame > reference_index else -1)
                # Match source's strict i < no_volumes-1 condition. In the
                # reffile profile the final frame keeps its identity start.
                if stage == 0 and 0 <= next_frame < frame_count - 1:
                    initial[next_frame] = previous[frame]
            matrices = previous
        cog_data = _flip_to_radiological(np.asarray(target.dataobj, dtype=np.float32), target.affine)
        centre = _centre_of_gravity(torch.as_tensor(cog_data), np.diag([*reference_sizes, 1.0]))
        parameters = np.stack([fsl_parameters_from_affine(matrix, centre)[:6] for matrix in matrices])
        corrected = None
        if resample or output is not None:
            from ..fmri.spatial import apply_motion_warp
            corrected = apply_motion_warp(image, target, matrices, device=self.device,
                                           interpolation=interpolation, mcflirt=True)
        if corrected is not None:
            output_data = np.asarray(corrected.dataobj, dtype=np.float32)
            # NEWIMAGE casts float32 interpolation values toward zero for
            # integer output; uint16 is promoted to int32 by its type table.
            output_data = (np.trunc(output_data).astype(output_dtype)
                           if output_dtype.kind in "iu" else output_data.astype(output_dtype))
            header = corrected.header.copy()
            header.set_data_dtype(output_dtype)
            corrected = nib.Nifti1Image(output_data, corrected.affine, header)
        if output is not None:
            Path(output_paths["corrected"]).parent.mkdir(parents=True, exist_ok=True)
            nib.save(corrected, output_paths["corrected"])
            if mats:
                matrix_dir = Path(output_paths["matrices"])
                matrix_dir.mkdir(parents=True, exist_ok=True)
                for frame, matrix in enumerate(matrices):
                    np.savetxt(matrix_dir / f"MAT_{frame:04d}", matrix, fmt="%.12g")
            if plots:
                np.savetxt(output_paths["parameters"], parameters, fmt="%.9g")
            sphere_centre = (np.asarray(target.shape, dtype=np.float64) - 1) * np.asarray(reference_sizes) / 2
            def rms_deviation(first, second):
                difference = first @ np.linalg.inv(second) - np.eye(4)
                linear = difference[:3, :3]
                translation = difference[:3, 3] + linear @ sphere_centre
                return np.float32(np.sqrt(translation @ translation + 80.0**2 / 5 * np.trace(linear.T @ linear)))
            for enabled, name, values in (
                (rmsrel, "relative", [rms_deviation(matrices[t - 1], matrices[t]) for t in range(1, frame_count)]),
                (rmsabs, "absolute", [rms_deviation(np.eye(4), matrix) for matrix in matrices]),
            ):
                if enabled:
                    values = np.asarray(values, dtype=np.float32)
                    mean = np.float32(np.add.accumulate(values, dtype=np.float32)[-1] / len(values)) if len(values) else np.float32(0)
                    np.savetxt(output_paths["rms_" + name], values, fmt="%.9g")
                    np.savetxt(output_paths["rms_" + name + "_mean"], [mean], fmt="%.9g")
        return MCFLIRTResult(matrices, parameters, target, cost_evaluations, corrected, output_paths)

    __call__ = run
