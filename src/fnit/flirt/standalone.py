"""FSL-style single-subject wrapper for the PyTorch FLIRT path."""

from __future__ import annotations

import os
from pathlib import Path
import uuid

import numpy as np
import torch

from .core import TorchFLIRT


def _default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _output_extension():
    output_type = os.environ.get("FSLOUTPUTTYPE", "NIFTI_GZ").upper()
    extensions = {"NIFTI": ".nii", "NIFTI_GZ": ".nii.gz"}
    try:
        return extensions[output_type]
    except KeyError as error:
        raise NotImplementedError(
            "extensionless image output requires FSLOUTPUTTYPE=NIFTI or "
            f"NIFTI_GZ; received {output_type}"
        ) from error


def _image_output_path(value):
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.name.endswith((".nii", ".nii.gz")):
        return path
    if path.suffix:
        raise ValueError("output must be an extensionless root, .nii, or .nii.gz")
    return path.with_name(path.name + _output_extension())


def _resolved_path(value):
    return Path(value).expanduser().resolve(strict=False)


def _preflight_outputs(output, omat, inputs, overwrite):
    selected = {
        name: path
        for name, path in {"output": output, "omat": omat}.items()
        if path is not None
    }
    if not selected:
        raise ValueError("provide output and/or omat")
    resolved = [_resolved_path(path) for path in selected.values()]
    if len(set(resolved)) != len(resolved):
        raise ValueError("output and omat must use different paths")
    protected = {
        _resolved_path(value)
        for value in inputs
        if isinstance(value, (str, os.PathLike))
    }
    if any(path in protected for path in resolved):
        raise ValueError(
            "an output path must not replace the input, reference, or init matrix"
        )
    existing = [path for path in selected.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"output exists: {existing[0]}; pass overwrite=True or --overwrite"
        )
    return selected


def _temporary_path(path, token, kind):
    if kind == "output":
        suffix = ".nii.gz" if path.name.endswith(".nii.gz") else ".nii"
        stem = path.name[: -len(suffix)]
        return path.with_name(f".{stem}.tmp-{token}{suffix}")
    return path.with_name(f".{path.name}.tmp-{token}")


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
            temporary = _temporary_path(destination, token, name)
            staged.append((temporary, destination))
            if name == "output":
                result.moved.save(str(temporary))
            else:
                np.savetxt(temporary, result.matrix, fmt="%.12g")
        for temporary, destination in staged:
            if overwrite:
                os.replace(temporary, destination)
            else:
                try:
                    os.link(temporary, destination)
                except FileExistsError as error:
                    raise FileExistsError(
                        f"output exists: {destination}; pass overwrite=True or "
                        "--overwrite"
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


def run_flirt(
    input,
    reference,
    *,
    output=None,
    omat=None,
    init=None,
    inweight=None,
    refweight=None,
    dof=12,
    cost="corratio",
    device=None,
    overwrite=False,
):
    """Run the supported default FLIRT path and atomically write outputs.

    ``input`` is the moving image and ``reference`` defines the output grid.
    ``init``, ``omat`` and ``result.matrix`` use FSL scaled-mm coordinates and
    map input to reference. Supported profiles are 12-DOF/corratio and
    6-DOF/normmi.
    """
    if (dof, cost) not in ((12, "corratio"), (6, "normmi")):
        raise NotImplementedError("supported profiles are -dof 12 -cost corratio and -dof 6 -cost normmi")
    selected_outputs = _preflight_outputs(
        _image_output_path(output),
        Path(omat).expanduser() if omat is not None else None,
        (input, reference, init, inweight, refweight),
        overwrite,
    )
    result = TorchFLIRT(
        device=_default_device() if device is None else device, dof=dof, cost=cost
    )(
        input,
        reference,
        init=init,
        inweight=inweight,
        refweight=refweight,
    )
    _write_outputs_atomic(result, selected_outputs, overwrite)
    return result


__all__ = ["run_flirt"]
