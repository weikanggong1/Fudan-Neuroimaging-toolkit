"""FSL-style single-subject wrapper for the PyTorch FNIRT GM path."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import uuid

import nibabel as nib
import numpy as np
import torch

from .._nib import load_image, new_image
from .._transforms import AffineTransform
from ..flirt.coordinates import flirt_to_world_affine
from .io import FSL_CUBIC_SPLINE_COEFFICIENTS
from .registration import GMFNIRTConfig, TorchFNIRT


SUPPORTED_CONFIG = "GM_2_MNI152GM_2mm.cnf"
SUPPORTED_CONFIG_SHA256 = (
    "3bc82d0ff4d8f53d89a741bd853a427607a5cb108e170b44d8835c4b542a4980"
)
DEFAULT_REFERENCE_MASK = "MNI152_T1_2mm_brain_mask_dil.nii.gz"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_config(value):
    if value is None:
        return SUPPORTED_CONFIG
    text = os.fspath(value)
    path = Path(text)
    if path.name not in (SUPPORTED_CONFIG, Path(SUPPORTED_CONFIG).stem):
        raise NotImplementedError(
            f"only {SUPPORTED_CONFIG} is implemented; received {text}"
        )
    has_directory = path.is_absolute() or path.parent != Path(".")
    if has_directory and not path.is_file():
        raise FileNotFoundError(path)
    if path.is_file() and _sha256(path) != SUPPORTED_CONFIG_SHA256:
        raise NotImplementedError(
            f"{path} is not the unmodified FSL {SUPPORTED_CONFIG} configuration"
        )
    return SUPPORTED_CONFIG


def _load_volume(value, name):
    volume = load_image(value, name)
    data = np.asanyarray(volume.dataobj)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
        volume = new_image(data, volume)
    if data.ndim != 3 or any(size < 2 for size in data.shape):
        raise ValueError(f"{name} must contain one 3D image")
    if not np.isfinite(data).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return volume


def _load_affine(value):
    if value is None:
        matrix = np.eye(4, dtype=np.float64)
    elif isinstance(value, (str, os.PathLike)):
        matrix = np.loadtxt(os.fspath(value), dtype=np.float64)
    else:
        matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("aff must contain one finite 4x4 matrix")
    if not np.allclose(matrix[3], (0, 0, 0, 1), atol=1e-8, rtol=0):
        raise ValueError("aff must be a homogeneous FSL matrix")
    if abs(float(np.linalg.det(matrix[:3, :3]))) < 1e-10:
        raise ValueError("aff must be invertible")
    return matrix


def _resolve_reference_mask(value):
    if value is None:
        raise ValueError(
            "refmask is required for the GM configuration and must be provided "
            "explicitly"
        )
    return value


def _validate_reference_mask(mask, reference):
    mask = _load_volume(mask, "refmask")
    if tuple(mask.shape[:3]) != tuple(reference.shape[:3]) or not np.allclose(
        mask.affine,
        reference.affine,
        atol=1e-5,
        rtol=0,
    ):
        raise ValueError("refmask must use the reference image grid")
    values = np.asanyarray(mask.dataobj)
    if not np.all((values == 0) | (values == 1)):
        raise ValueError("refmask must be binary with values 0 and 1")
    if not np.any(values == 1):
        raise ValueError("refmask is empty")
    return mask


def _output_extension():
    output_type = os.environ.get("FSLOUTPUTTYPE", "NIFTI_GZ").upper()
    extensions = {"NIFTI": ".nii", "NIFTI_GZ": ".nii.gz"}
    try:
        return extensions[output_type]
    except KeyError as error:
        raise NotImplementedError(
            "extensionless outputs require FSLOUTPUTTYPE=NIFTI or NIFTI_GZ; "
            f"received {output_type}"
        ) from error


def _nifti_output_path(value, name):
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.name.endswith(".nii") or path.name.endswith(".nii.gz"):
        return path
    if path.suffix:
        raise ValueError(f"{name} must be an extensionless root, .nii, or .nii.gz")
    path = path.with_name(path.name + _output_extension())
    return path


def _default_coefficient_root(input):
    if not isinstance(input, (str, os.PathLike)):
        raise ValueError(
            "cout is required when input is an in-memory nibabel image"
        )
    path = Path(input).expanduser()
    name = path.name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    return path.with_name(f"{name}_warpcoef")


def _default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolved_path(value):
    return Path(value).expanduser().resolve(strict=False)


def _preflight_outputs(outputs, inputs, overwrite):
    selected = {name: path for name, path in outputs.items() if path is not None}
    if not selected:
        raise ValueError("at least one of cout, iout, or jout is required")
    resolved = [_resolved_path(path) for path in selected.values()]
    if len(set(resolved)) != len(resolved):
        raise ValueError("cout, iout, and jout must use different paths")
    protected = {
        _resolved_path(value)
        for value in inputs
        if isinstance(value, (str, os.PathLike))
    }
    if any(path in protected for path in resolved):
        raise ValueError("an output path must not replace an input, reference, affine, or mask")
    existing = [path for path in selected.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"output exists: {existing[0]}; pass overwrite=True or --overwrite"
        )
    return selected


def _temporary_path(path, token):
    suffix = ".nii.gz" if path.name.endswith(".nii.gz") else ".nii"
    stem = path.name[: -len(suffix)]
    return path.with_name(f".{stem}.tmp-{token}{suffix}")


def _same_file(left, right):
    try:
        left_stat = left.stat(follow_symlinks=False)
        right_stat = right.stat(follow_symlinks=False)
    except FileNotFoundError:
        return False
    return (left_stat.st_dev, left_stat.st_ino) == (
        right_stat.st_dev,
        right_stat.st_ino,
    )


def _write_outputs_atomic(result, outputs, overwrite):
    token = f"{os.getpid()}-{uuid.uuid4().hex}"
    staged = []
    created = []
    try:
        for name, destination in outputs.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = _temporary_path(destination, token)
            staged.append((temporary, destination))
            if name == "cout":
                coefficient_image = result.coefficient_image
                intent = int(coefficient_image.header["intent_code"])
                if intent != FSL_CUBIC_SPLINE_COEFFICIENTS:
                    raise RuntimeError(
                        "TorchFNIRT did not return an intent-2007 coefficient image"
                    )
                nib.save(coefficient_image, str(temporary))
            elif name == "iout":
                nib.save(result.moved, str(temporary))
            else:
                nib.save(result.nonlinear_jacobian, str(temporary))
        for temporary, destination in staged:
            if overwrite:
                os.replace(temporary, destination)
            else:
                try:
                    os.link(temporary, destination)
                except FileExistsError as error:
                    raise FileExistsError(
                        f"output exists: {destination}; pass overwrite=True or --overwrite"
                    ) from error
                created.append((temporary, destination))
    except Exception:
        if not overwrite:
            for temporary, destination in reversed(created):
                if _same_file(temporary, destination):
                    destination.unlink()
        raise
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)


def run_fnirt(
    input,
    reference,
    affine=None,
    *,
    cout=None,
    iout=None,
    jout=None,
    refmask=None,
    config=SUPPORTED_CONFIG,
    device=None,
    overwrite=False,
):
    """Run the supported FSL GM FNIRT path and optionally write its outputs.

    ``affine`` follows an FSL FLIRT ``.mat`` contract: input to reference in
    FSL scaled-mm coordinates.  When omitted, it is the scaled-mm identity,
    matching FSL FNIRT.  It is converted to a geometry-tagged moving-to-
    reference world-RAS affine before calling :class:`TorchFNIRT`.
    ``jout`` is the nonlinear-only determinant written by FSL FNIRT's
    ``SaveJacobian`` path; it excludes the affine determinant.
    """
    _validate_config(config)
    selected_mask = _resolve_reference_mask(refmask)
    if cout is None:
        cout = _default_coefficient_root(input)
    selected_outputs = _preflight_outputs(
        {
            "cout": _nifti_output_path(cout, "cout"),
            "iout": _nifti_output_path(iout, "iout"),
            "jout": _nifti_output_path(jout, "jout"),
        },
        (input, reference, affine, selected_mask),
        overwrite,
    )
    moving = _load_volume(input, "input")
    fixed = _load_volume(reference, "reference")
    selected_mask = _validate_reference_mask(selected_mask, fixed)
    fsl_affine = _load_affine(affine)
    forward_world = flirt_to_world_affine(
        fsl_affine,
        moving.affine,
        fixed.affine,
        moving.shape[:3],
        fixed.shape[:3],
        nib.affines.voxel_sizes(moving.affine),
        nib.affines.voxel_sizes(fixed.affine),
    )
    moving_to_fixed = AffineTransform(
        forward_world, source=moving, target=fixed, space="world"
    )
    result = TorchFNIRT(
        device=_default_device() if device is None else device,
        config=GMFNIRTConfig(),
    )(
        moving,
        fixed,
        moving_to_fixed,
        reference_mask=selected_mask,
    )
    _write_outputs_atomic(result, selected_outputs, overwrite)
    return result


__all__ = [
    "DEFAULT_REFERENCE_MASK",
    "SUPPORTED_CONFIG",
    "SUPPORTED_CONFIG_SHA256",
    "run_fnirt",
]
