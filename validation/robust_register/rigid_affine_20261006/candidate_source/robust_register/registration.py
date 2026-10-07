"""Experimental standalone robust rigid/affine registration.

Modified PyTorch/nibabel adaptation of FreeSurfer d932c45. See the MGH
notice and licenses/FreeSurfer.txt. Not an official FreeSurfer release.
This explicit API is not connected to GEMS or other FNIT pipelines.

Original author: Martin Reuter. Copyright (c) 2021 The General Hospital
Corporation (Boston, MA), "MGH". FNIT modification: PyTorch/nibabel, 2026.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from math import prod
from time import monotonic

import nibabel as nib
from nibabel.freesurfer.mghformat import MGHHeader, MGHImage
import numpy as np
import torch

from .._nib import load_image
from .._transforms import AffineTransform
from ._sampling import downsample, native_inverse, native_matmul, resample
from ._solver import analytic_system, parameters_to_matrix, robust_regression, transform_distance


@dataclass
class RobustRegistrationResult:
    transform: AffineTransform
    header_image: MGHImage
    report: dict


@contextmanager
def _precision(device, tf32):
    old = torch.backends.cuda.matmul.allow_tf32
    try:
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = tf32
        with torch.autocast(device_type=device.type, enabled=False):
            yield
    finally:
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = old


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _native_fields(image):
    """Stored Float geometry, native extract_i_to_r Float computation."""
    shape = np.asarray(image.shape, dtype=np.int64)
    if isinstance(image.header, MGHHeader):
        if not int(image.header["goodRASFlag"]):
            raise ValueError("registration requires a valid MGH RAS geometry")
        size = np.asarray(image.header["delta"], np.float32)
        rotation = np.asarray(image.header["Mdc"], np.float32).T
        center = np.asarray(image.header["Pxyz_c"], np.float32)
    else:
        units = image.header.get_xyzt_units()[0]
        if units not in ("unknown", "mm"):
            raise ValueError("registration coordinates must be millimetres")
        affine = np.asarray(image.affine, np.float64)
        size = np.linalg.norm(affine[:3, :3], axis=0).astype(np.float32)
        rotation = (affine[:3, :3] / size).astype(np.float32)
        center = (affine @ np.r_[shape / 2, 1])[:3].astype(np.float32)
    if (not np.isfinite(size).all() or np.any(size <= 0)
            or not np.isfinite(rotation).all() or not np.isfinite(center).all()):
        raise ValueError("image geometry must be finite and nonsingular")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4, rtol=0):
        raise ValueError("robust registration currently requires orthogonal voxel axes")
    return size, rotation, center


def _native_geometry(image):
    size, rotation, center = _native_fields(image)
    return _compose_native(image.shape, size, rotation, center), size


def _compose_native(shape, size, rotation, center):
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = np.float32(rotation * size)
    for axis in range(3):
        offset = np.float32(sum(float(out[axis, j]) * float(shape[j]) / 2 for j in range(3)))
        out[axis, 3] = np.float32(center[axis] - offset)
    return out


def _reordered_fields(shape, matrix, sizes):
    # MRIreorderVox2RAS -> MRIsetVox2RASFromMatrix normalizes columns
    # using Double norms, retains the permuted stored Float voxel sizes,
    # then MRIp0ToCRAS computes the new stored Float center. This differs
    # from mapmovhdr's MRIsetVoxelToRasXform scale/shear policy.
    lengths = np.linalg.norm(matrix[:3, :3].astype(float), axis=0)
    rotation = np.asarray(matrix[:3, :3].astype(float) / lengths, np.float32)
    linear = _compose_native(shape, sizes, rotation, np.zeros(3, np.float32))
    linear[:3, 3] = matrix[:3, 3]
    center = native_matmul(linear, np.r_[np.asarray(shape, np.float32) / 2, 1][:, None])[:3, 0]
    return rotation, center


def _reorient(values, source_affine, target_affine, *, source_fields=None):
    matrix = native_matmul(native_inverse(target_affine), source_affine)
    destinations = np.argmax(np.abs(matrix[:3, :3]), axis=0).tolist()
    signs = [1 if matrix[destinations[j], j] >= 0 else -1 for j in range(3)]
    if len(set(destinations)) != 3:
        if destinations[0] == 1 and destinations[1] == 0:
            destinations[2] = 2
        elif destinations[0] == 2 and destinations[2] == 0:
            destinations[1] = 1
        elif destinations[1] == 2 and destinations[2] == 1:
            destinations[0] = 0
        else:
            destinations = [0, 1, 2]
        signs = [1 if matrix[destinations[j], j] >= 0 else -1 for j in range(3)]
    permutation = np.argsort(destinations).tolist()
    pull = np.zeros((4, 4), np.float32)
    pull[3, 3] = 1
    result = values.permute(permutation)
    for old in range(3):
        new = destinations[old]
        pull[old, new] = signs[old]
        if signs[old] < 0:
            pull[old, 3] = values.shape[old] - 1
            result = result.flip(new)
    affine = native_matmul(source_affine, pull)
    changed = destinations != [0, 1, 2] or signs != [1, 1, 1]
    fields = source_fields
    if fields is not None and changed:
        size = np.asarray(fields[0][permutation], np.float32)
        rotation, center = _reordered_fields(result.shape, affine, size)
        fields = (size, rotation, center)
        affine = _compose_native(result.shape, *fields)
        # Installed Rsrc is recovered from the newly stored geometry,
        # not from the exact integer permutation matrix alone.
        pull = native_matmul(native_inverse(source_affine), affine)
    if np.linalg.det(native_matmul(native_inverse(target_affine), affine)[:3, :3]) <= 0:
        raise ValueError("reoriented source/target still have incompatible handedness")
    report = {"old_axis_destination_signed": [(destinations[j] + 1) * signs[j] for j in range(3)],
              "reordered": changed}
    if fields is not None:
        report["stored_geometry"] = {"delta": fields[0].tolist(), "directions": fields[1].tolist(),
                                     "center": fields[2].tolist()}
    return result.contiguous(), affine, pull, report


def _isotropic(values, affine, voxel_mm, shape, chunk_size, *, fields):
    size, rotation, center = fields
    # makeIsotropic copies VOL_GEOM fields; reconstructing center from an
    # already rounded affine would introduce an extra Float round trip.
    new_affine = _compose_native(np.asarray(shape), np.full(3, voxel_mm, np.float32), rotation, center)
    pull = native_matmul(native_inverse(affine), new_affine)
    if tuple(values.shape) == tuple(shape) and np.all(np.abs(size - voxel_mm) <= 1e-5):
        return values.clone(), affine.copy(), np.eye(4, dtype=np.float32)
    result = resample(values, shape, pull, cubic=True, chunk_size=chunk_size)
    return result, new_affine, pull


def _centroid(values):
    mass = values.double()
    total = mass.sum()
    if float(total) <= 0:
        raise ValueError("registration input has no positive intensity mass")
    center = []
    for axis, n in enumerate(values.shape):
        marginal = mass.sum(dim=tuple(a for a in range(3) if a != axis))
        coordinate = torch.arange(1, n + 1, dtype=torch.float64, device=values.device)
        center.append(float((marginal * coordinate).sum() / total))
    return np.asarray(center)


def _pyramid_limits(shape, minimum, maximum):
    if min(shape) < minimum:
        raise ValueError("resliced images are smaller than pyramid_min_size")
    maximum_level, temp = 0, min(shape) // 2
    while temp > minimum:
        maximum_level += 1
        temp //= 2
    first, temp = 0, max(shape)
    if maximum != -1:
        while first < maximum_level and temp >= maximum:
            first += 1
            temp //= 2
    return first, maximum_level


def _sqrt_matrix(matrix, *, rigid):
    # Native complex Schur / triangular recurrence. Only a 4x4 Double
    # matrix is transferred to CPU; image sampling/QR stays on device.
    from scipy.linalg import schur
    triangular, unitary = schur(matrix, output="complex")
    root = np.zeros_like(triangular)
    for j in range(4):
        root[j, j] = np.sqrt(triangular[j, j])
        for i in range(j - 1, -1, -1):
            denominator = root[i, i] + root[j, j]
            if denominator == 0:
                raise ValueError("halfway transform has no admissible principal square root")
            root[i, j] = (triangular[i, j] - sum(root[i, k] * root[k, j]
                                                 for k in range(i + 1, j))) / denominator
    result = unitary @ root @ unitary.conj().T
    imag = np.linalg.norm(result.imag, 1)
    if imag > 40 * np.finfo(float).eps * np.linalg.norm(result, 1):
        raise ValueError("halfway square root is materially complex")
    result = result.real
    if np.linalg.norm(result @ result - matrix) > 1e-10:
        raise ValueError("halfway square root residual exceeds source tolerance")
    singular = np.linalg.svd(result[:3, :3], compute_uv=False)
    if np.linalg.det(result[:3, :3]) <= 0 or singular.min() < .001:
        raise ValueError("halfway transform reflects or nearly projects")
    if rigid and np.linalg.norm(result[:3, :3].T @ result[:3, :3] - np.eye(3)) > 2e-6:
        raise ValueError("rigid halfway transform introduces scaling")
    return result


def _header_after_matrix(image, source_affine, ras_matrix):
    new = native_matmul(ras_matrix, source_affine)
    # Native MRIsetVoxelToRasXform preserves original voxel sizes by
    # default, even for affine scale/shear; directions need not be unit.
    # The environment opt-in FS_SetVoxToRasXform_Change_VoxSize is not used.
    if isinstance(image.header, MGHHeader):
        size = np.asarray(image.header["delta"], np.float32)
    else:
        size = np.asarray(image.header.get_zooms()[:3], np.float32)
    directions = np.float32(new[:3, :3] / size)
    center = native_matmul(new, np.r_[np.asarray(image.shape, np.float32) / 2, 1][:, None])[:3, 0]
    header = image.header.copy() if isinstance(image.header, MGHHeader) else MGHHeader()
    data = np.asarray(image.dataobj)
    if data.dtype.kind == "f":
        data = np.asarray(data, np.float32)
    header.set_data_dtype(data.dtype)
    header.set_data_shape(data.shape)
    header["goodRASFlag"], header["delta"] = 1, size
    header["Mdc"], header["Pxyz_c"] = directions.T, center
    # Use the actual stored fields to avoid nibabel rederiving them.
    result = MGHImage(np.array(data, copy=True), header.get_affine(), header)
    for key in ("delta", "Mdc", "Pxyz_c"):
        if not np.array_equal(result.header[key], header[key]):
            raise ValueError("nibabel unexpectedly changed mapped header fields")
    return result


@torch.inference_mode()
def robust_register(
    source, target, *, mode="rigid", saturation=50., iterations_per_level=5,
    stop_distance=.01, initialize_translation=True, pyramid_min_size=16,
    pyramid_max_size=-1, highres_iterations=-1, device="cuda:0", tf32=True,
    spatial_chunk_size=131072, memory_budget_gb=20.,
) -> RobustRegistrationResult:
    """Experimental symmetric robust registration of nonnegative 3D images.

    Inputs: paths or nibabel images in scanner RAS mm. Output: a geometry
    tagged source→target world transform, unresampled source data with an
    MGH mapped header, and scalar stage timings. CUDA absence is an error.
    Only the two source-default rigid/affine --sat50 command profiles are
    implemented; no intensity scaling, random subsampling or arbitrary
    initialization is silently substituted. Native equivalence is pending.
    """
    started = monotonic()
    dev = torch.device(device)
    if dev.type not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or an explicit CUDA device")
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA is unavailable; no CPU fallback")
    if mode not in ("rigid", "affine") or not isinstance(tf32, bool):
        raise ValueError("mode must be rigid/affine and tf32 must be bool")
    for name, value in (("iterations_per_level", iterations_per_level),
                        ("pyramid_min_size", pyramid_min_size),
                        ("spatial_chunk_size", spatial_chunk_size)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(name + " must be a positive integer")
    if pyramid_max_size != -1 and (not isinstance(pyramid_max_size, int) or pyramid_max_size < 1):
        raise ValueError("pyramid_max_size must be -1 or positive")
    if not isinstance(highres_iterations, int) or highres_iterations < -1:
        raise ValueError("highres_iterations must be -1,0 or positive")
    if not isinstance(initialize_translation, bool):
        raise ValueError("initialize_translation must be bool")
    for name, value in (("saturation", saturation), ("stop_distance", stop_distance),
                        ("memory_budget_gb", memory_budget_gb)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(name + " must be finite and positive")
    source_image, target_image = load_image(source), load_image(target)
    for image in (source_image, target_image):
        if len(image.shape) != 3 or min(image.shape) < 2:
            raise ValueError("only one-frame3D images with all axes≥2 are supported")
        dtype = np.dtype(image.header.get_data_dtype())
        if (dtype.kind, dtype.itemsize) not in (("u", 1), ("i", 2), ("i", 4), ("f", 4)):
            raise ValueError("this profile supports uint8, int16, int32 or float32 stored images")
    source_fields, target_fields = _native_fields(source_image), _native_fields(target_image)
    source_sizes, target_sizes = source_fields[0], target_fields[0]
    source_affine = _compose_native(source_image.shape, *source_fields)
    target_affine = _compose_native(target_image.shape, *target_fields)
    load_clock = monotonic() - started
    voxel_mm = np.float32(max(float(source_sizes.min()), float(target_sizes.min())))
    source_shape = np.ceil(source_sizes.astype(float) * source_image.shape / voxel_mm - .0001).astype(int)
    target_shape = np.ceil(target_sizes.astype(float) * target_image.shape / voxel_mm - .0001).astype(int)
    # Conservative worst orientation-independent allocation bound; includes
    # pyramid, Double prefilter, QR and compact rows, not full coordinate grids.
    bound_shape = tuple(max(int(source_shape.max()), int(target_shape.max())) for _ in range(3))
    estimated = prod(bound_shape) * 600 + min(spatial_chunk_size, prod(bound_shape)) * 1024
    if estimated > min(memory_budget_gb * 1e9, 32e9):
        raise MemoryError("conservative robust registration memory preflight exceeds declared budget")
    if dev.type == "cuda" and estimated + torch.cuda.memory_allocated(dev) > min(memory_budget_gb * 1e9, 20e9):
        raise MemoryError("robust registration would exceed20GB CUDA budget")
    with _precision(dev, tf32):
        s = torch.as_tensor(np.array(source_image.dataobj, dtype=np.float32, copy=True), device=dev)
        t = torch.as_tensor(np.array(target_image.dataobj, dtype=np.float32, copy=True), device=dev)
        for values in (s, t):
            if not bool(torch.isfinite(values).all() and (values >= 0).all()):
                raise ValueError("this profile requires finite nonnegative image intensities")
        _sync(dev)
        preparation_start = monotonic()
        s, sa, reorder, reorder_report = _reorient(s, source_affine, target_affine,
                                                  source_fields=source_fields)
        reordered_fields = reorder_report["stored_geometry"]
        source_fields = tuple(np.asarray(reordered_fields[key], np.float32)
                              for key in ("delta", "directions", "center"))
        ss = source_fields[0].astype(float)
        sd = np.ceil(ss * np.asarray(s.shape) / voxel_mm - .0001).astype(int)
        td = np.ceil(target_sizes.astype(float) * np.asarray(t.shape) / voxel_mm - .0001).astype(int)
        shape = np.maximum(sd, td)
        s, sa, source_pull = _isotropic(s, sa, voxel_mm, shape, spatial_chunk_size, fields=source_fields)
        t, ta, target_pull = _isotropic(t, target_affine, voxel_mm, shape, spatial_chunk_size, fields=target_fields)
        source_reslice = reorder.astype(float) @ source_pull.astype(float)
        # Centroids have native +1 voxel coordinates; their difference is
        # retained rather than using world/header alignment for default init.
        initial = np.eye(4)
        if initialize_translation:
            initial[:3, 3] = _centroid(t) - _centroid(s)
        else:
            initial = np.linalg.inv(ta.astype(float)) @ sa.astype(float)
        _sync(dev)
        preparation_clock = monotonic() - preparation_start
        pyramid_start = monotonic()
        first, last = _pyramid_limits(shape, pyramid_min_size, pyramid_max_size)
        sp, tp = [s], [t]
        for _ in range(last):
            sp.append(downsample(sp[-1]))
            tp.append(downsample(tp[-1]))
        _sync(dev)
        pyramid_clock = monotonic() - pyramid_start
        matrix = initial.copy()
        matrix[:3, 3] *= .5 ** last
        stages = []
        for level in range(last, first - 1, -1):
            max_iterations = highres_iterations if level == 0 and highres_iterations >= 0 else iterations_per_level
            traces = []
            for iteration in range(1, max_iterations + 1):
                _sync(dev)
                step_start = monotonic()
                half = _sqrt_matrix(matrix, rigid=mode == "rigid" and initialize_translation)
                target_half = half @ np.linalg.inv(matrix)
                sw = resample(sp[level], sp[level].shape, native_inverse(half), chunk_size=spatial_chunk_size)
                tw = resample(tp[level], sp[level].shape, native_inverse(target_half), chunk_size=spatial_chunk_size)
                _sync(dev)
                sampling_clock = monotonic() - step_start
                build_start = monotonic()
                a, b, rows = analytic_system(sw, tw, mode=mode)
                _sync(dev)
                build_clock = monotonic() - build_start
                solve_start = monotonic()
                result = robust_regression(a, b, saturation=saturation)
                _sync(dev)
                solve_clock = monotonic() - solve_start
                update = parameters_to_matrix(result.parameters, mode=mode)
                previous = matrix.copy()
                matrix = (np.linalg.inv(target_half) @ update) @ half
                distance = transform_distance(matrix, previous)
                traces.append({"iteration": iteration, **rows, "sampling_seconds": sampling_clock,
                               "design_seconds": build_clock, "IRLS_QR_seconds": solve_clock,
                               "step_seconds": monotonic() - step_start,
                               "distance": distance, "converged": distance <= stop_distance,
                               "matrix": matrix.tolist(), "regression": result.report})
                del a, b, result, sw, tw
                if distance <= stop_distance:
                    break
            stages.append({"level": level, "shape": list(sp[level].shape), "steps": traces,
                           "stop": "converged" if traces and traces[-1]["converged"] else
                                   "skipped" if max_iterations == 0 else "iteration_budget_exhausted"})
            if level > 0:
                matrix[:3, 3] *= 2
        if first > 0:
            matrix[:3, 3] *= 2 ** (first - 1)
        final_voxel = target_pull.astype(float) @ matrix @ np.linalg.inv(source_reslice.astype(float))
        # Native public LTA/mapmovhdr goes back through MATRIX_REAL.
        ras = native_matmul(target_affine, native_matmul(final_voxel, native_inverse(source_affine)))
        transform = AffineTransform(ras.astype(float), source=source_image, target=target_image, space="world")
        header_image = _header_after_matrix(source_image, source_affine, ras)
        _sync(dev)
        report = {
            "experimental_native_equivalence": "not_assessed", "mode": mode,
            "source_shape": [int(v) for v in source_image.shape],
            "target_shape": [int(v) for v in target_image.shape],
            "device": str(dev), "TF32_actual_in_context": torch.backends.cuda.matmul.allow_tf32 if dev.type == "cuda" else None,
            "image_design_QR_dtype": "torch.float32", "small_matrix_state_dtype": "numpy.float64",
            "interpolation_weights_dtype": "torch.float64", "coefficient_axis_storage": "torch.float32",
            "small_matrix_Schur_device": "cpu", "CUDA_image_CPU_fallback": False,
            "memory_estimate_bytes": estimated, "memory_budget_gb": memory_budget_gb,
            "parameters": {"saturation": saturation, "iterations_per_level": iterations_per_level,
                "stop_distance": stop_distance, "initialize_translation": initialize_translation,
                "pyramid_min_size": pyramid_min_size, "pyramid_max_size": pyramid_max_size,
                "highres_iterations": highres_iterations, "spatial_chunk_size": spatial_chunk_size, "tf32": tf32},
            "load_seconds": load_clock, "preparation_seconds": preparation_clock,
            "pyramid_seconds": pyramid_clock, "stages": stages,
            "axis_reorder": reorder_report, "isotropic_shape": [int(v) for v in shape],
            "isotropic_voxel_mm": float(voxel_mm), "initial_voxel_matrix": initial.tolist(),
            "final_voxel_matrix": final_voxel.tolist(), "RAS_matrix": ras.tolist(),
            "source_data_resampled_in_header_output": False, "API_seconds": monotonic() - started,
        }
    return RobustRegistrationResult(transform, header_image, report)


def robust_rigid_affine(source, target, *, stage_directory, **kwargs):
    """Rigid then affine with an explicit MGH save/reload header boundary.

    ``stage_directory`` must not exist. Returns a dict with both results,
    combined RAS transform and mapped source image. It runs neither GEMS
    nor an official executable. Timings include each stage API and MGH I/O.
    """
    from pathlib import Path
    if "mode" in kwargs:
        raise ValueError("the two-stage API fixes mode to rigid then affine")
    path = Path(stage_directory)
    path.mkdir(parents=True, exist_ok=False)
    start = monotonic()
    rigid = robust_register(source, target, mode="rigid", **kwargs)
    rigid_path = path / "rigid.header.mgz"
    nib.save(rigid.header_image, str(rigid_path))
    reloaded = nib.load(str(rigid_path))
    affine = robust_register(reloaded, target, mode="affine", **kwargs)
    nib.save(affine.header_image, str(path / "affine.header.mgz"))
    rigid.transform.save(path / "rigid.lta")
    affine.transform.save(path / "affine.lta")
    combined = native_matmul(affine.transform.matrix, rigid.transform.matrix).astype(float)
    return {"rigid": rigid, "affine": affine, "combined_RAS_matrix": combined,
            "header_image": affine.header_image, "seconds": monotonic() - start}
