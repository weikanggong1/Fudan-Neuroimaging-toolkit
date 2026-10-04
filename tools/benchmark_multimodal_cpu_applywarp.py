#!/usr/bin/env python3
"""Complete ApplyWarp operations for the coordinated real-data CPU suite.

This module never executes official software. ``reference_command`` only
describes the isolated reference commands for the parent benchmark runner.
FSL mode saves one full 3D/4D output. Plan mode prepares once and saves every
same-grid image; the official comparison performs one applywarp per image.
The world API has no universal matching FSL CLI and is deliberately excluded
from an official timing ratio unless a separately validated oracle is added.
"""
from __future__ import annotations

from pathlib import Path
import re


def _inputs(case):
    mode = case.get("mode", "fsl")
    if mode not in {"fsl", "plan", "world"}:
        raise ValueError("ApplyWarp mode must be fsl, plan or world")
    inputs = case.get("maps") if mode == "plan" else {"warped": case["input"]}
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("Plan mode needs a nonempty maps name:path mapping")
    for name in inputs:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("Output names must contain only letters, digits, '_' or '-'")
    return inputs


def _paths(case, output_dir):
    suffix = case.get("output_suffix", ".nii.gz")
    if suffix not in {".nii", ".nii.gz"}:
        raise ValueError("output_suffix must be .nii or .nii.gz")
    return {name: Path(output_dir) / (name + suffix) for name in _inputs(case)}


def _options(case):
    return {key: case[key] for key in (
        "warp", "premat", "postmat", "interpolation", "warp_convention", "output_dtype",
    ) if key in case}


def _matrix(value):
    import numpy as np
    if isinstance(value, (str, Path)):
        return np.load(value, allow_pickle=False) if str(value).endswith(".npy") else np.loadtxt(value)
    return np.asarray(value, dtype=np.float64)


def run_case(case: dict, output_dir: Path, device: str) -> dict:
    """Load, transform and save every output; no warmup or cropped input."""
    from fnit.applywarp import TorchApplyWarp, WorldTransformChain

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    warper = TorchApplyWarp(device=device, frame_chunk_size=case.get("frame_chunk_size"))
    mode = case.get("mode", "fsl")
    inputs, outputs = _inputs(case), _paths(case, output_dir)
    if mode == "world":
        specification = case["world"]
        chain = WorldTransformChain(
            reference=case["reference"],
            reference_to_source_world=_matrix(specification["reference_to_source_world"]),
            pre_affine_pull_ras=specification.get("pre_affine_pull_ras"),
            motion_pull_world=(
                _matrix(specification["motion_pull_world"])
                if "motion_pull_world" in specification else None
            ),
            coordinate_precision=specification.get("coordinate_precision", "float64"),
        )
        kwargs = {key: specification[key] for key in (
            "interpolation", "boundary", "output_mask", "batch_size", "spatial_chunk_size",
        ) if key in specification}
        warper.run_world(case["input"], chain, outputs["warped"], **kwargs)
    elif mode == "plan":
        plan = warper.prepare(
            case.get("input", next(iter(inputs.values()))), case["reference"],
            **{key: value for key, value in _options(case).items() if key != "output_dtype"},
        )
        for name, source in inputs.items():
            plan.apply(source, output_dtype=case.get("output_dtype")).save(outputs[name])
    else:
        warper.run(case["input"], case["reference"], outputs["warped"], **_options(case))
    return outputs


def _official_applywarp(resources):
    for key in ("fsl_applywarp", "applywarp"):
        if key in resources:
            return str(resources[key])
    if "fsl_bin" in resources:
        return str(Path(resources["fsl_bin"]) / "applywarp")
    if "fsl_root" in resources:
        return str(Path(resources["fsl_root"]) / "bin" / "applywarp")
    raise ValueError("resources must declare an explicit fsl_applywarp or fsl_root")


def reference_command(case: dict, output_dir: Path, resources: dict) -> list:
    """Describe equal-output FSL commands; the runner executes them in isolation."""
    if case.get("mode", "fsl") == "world":
        raise NotImplementedError(
            "The world chain requires a separately validated coordinate/interpolation "
            "oracle; it has no universal FSL applywarp command."
        )
    interpolation = case.get("interpolation", "trilinear")
    interpolation = "nn" if interpolation in {"nn", "nearest", "nearest-neighbour", "nearest_neighbor"} else interpolation
    if interpolation not in {"trilinear", "nn"}:
        raise ValueError("Only the supported nearest/trilinear FSL subset can be benchmarked")
    outputs = _paths(case, output_dir)
    commands = []
    for name, source in _inputs(case).items():
        command = [_official_applywarp(resources), f"--in={source}",
                   f"--ref={case['reference']}", f"--out={outputs[name]}",
                   f"--interp={interpolation}"]
        for key in ("warp", "premat", "postmat"):
            if case.get(key) is not None:
                command.append(f"--{key}={case[key]}")
        convention = case.get("warp_convention", "auto")
        if convention in {"relative", "rel"}:
            command.append("--rel")
        elif convention in {"absolute", "abs"}:
            command.append("--abs")
        elif convention != "auto":
            raise ValueError("Invalid warp convention")
        if case.get("output_dtype") is not None:
            datatype = case["output_dtype"]
            aliases = {"uint8": "char", "int16": "short", "int32": "int",
                       "float32": "float", "float64": "double"}
            datatype = aliases.get(str(datatype), str(datatype))
            if datatype not in {"char", "short", "int", "float", "double"}:
                raise ValueError("No supported FSL datatype correspondence")
            command.append(f"--datatype={datatype}")
        commands.append(command)
    return commands[0] if len(commands) == 1 else commands


def reference_outputs(case: dict, output_dir: Path, resources: dict) -> dict:
    return _paths(case, output_dir)


def _image_contract(image):
    import numpy as np
    return {
        "shape": list(image.shape), "dtype": np.dtype(image.get_data_dtype()).name,
        "zooms": [float(value) for value in image.header.get_zooms()],
        "xyzt_units": list(image.header.get_xyzt_units()),
        "qform_code": int(image.header["qform_code"]),
        "sform_code": int(image.header["sform_code"]),
        # Nibabel consumes slope/intercept into the proxy on load. Report the
        # values actually used to decode saved values, not reset header NaNs.
        "storage_slope": float(getattr(image.dataobj, "slope", 1.0)),
        "storage_intercept": float(getattr(image.dataobj, "inter", 0.0)),
    }


def _region_metrics(left, right, mask):
    """Bound comparison scratch space; count every spatial voxel/time frame."""
    import numpy as np
    total = differing = 0
    absolute_sum = squared_sum = reference_squared_sum = maximum = 0.0
    frames = 1 if left.ndim == 3 else left.shape[-1]
    selected = mask.reshape(-1)
    for frame in range(frames):
        a = (left if left.ndim == 3 else left[..., frame]).reshape(-1)
        b = (right if right.ndim == 3 else right[..., frame]).reshape(-1)
        for start in range(0, a.size, 262144):
            stop = min(start + 262144, a.size)
            selection = selected[start:stop]
            av = a[start:stop][selection].astype(np.float64)
            bv = b[start:stop][selection].astype(np.float64)
            delta = av - bv
            total += delta.size
            differing += int(np.count_nonzero(delta))
            if delta.size:
                absolute_sum += float(np.abs(delta).sum())
                squared_sum += float((delta * delta).sum())
                reference_squared_sum += float((bv * bv).sum())
                maximum = max(maximum, float(np.abs(delta).max()))
    return {
        "values": total, "different_values": differing,
        "different_fraction": differing / total if total else None,
        "max_absolute_error": maximum if total else None,
        "mae": absolute_sum / total if total else None,
        "rmse": (squared_sum / total) ** 0.5 if total else None,
        "relative_l2": (squared_sum / max(reference_squared_sum, 1e-60)) ** 0.5 if total else None,
    }


def compare_case(case: dict, output_sets: dict, resources: dict) -> dict:
    """Saved-image regional precision and dtype/scaling, outside timed calls.

    Use ``precision_mask`` for a declared brain mask. Otherwise report only
    reference nonzero support, which must not be silently called a brain mask.
    Label Dice is enabled only by an explicit ``label_image`` declaration.
    """
    import nibabel as nib
    import numpy as np

    target = nib.load(case.get("precision_mask", case["reference"]))
    support = np.asanyarray(target.dataobj) > 0
    if support.ndim != 3:
        raise ValueError("precision_mask/reference must provide one 3D support grid")
    support_name = "declared_brain_mask" if "precision_mask" in case else "reference_nonzero_support"
    report = {"support": support_name, "support_spatial_voxels": int(support.sum()), "comparisons": {}}
    for backend in ("official", "baseline"):
        if backend not in output_sets:
            continue
        pair = {}
        for name, candidate_path in output_sets["candidate"].items():
            if name not in output_sets[backend]:
                pair[name] = {"status": "no_corresponding_output"}
                continue
            candidate = nib.load(candidate_path)
            reference = nib.load(output_sets[backend][name])
            a, b = np.asanyarray(candidate.dataobj), np.asanyarray(reference.dataobj)
            if a.shape != b.shape or a.shape[:3] != support.shape:
                pair[name] = {"status": "shape_mismatch"}
                continue
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                pair[name] = {"status": "nonfinite_output"}
                continue
            values = {
                "status": "compared", "candidate_contract": _image_contract(candidate),
                "reference_contract": _image_contract(reference),
                "same_dtype": candidate.get_data_dtype() == reference.get_data_dtype(),
                "affine_max_error": float(np.abs(candidate.affine - reference.affine).max()),
                "whole_grid": _region_metrics(a, b, np.ones(support.shape, dtype=bool)),
                support_name: _region_metrics(a, b, support),
                "outside_support": _region_metrics(a, b, ~support),
            }
            if case.get("label_image", False):
                if a.ndim != 3:
                    raise ValueError("label_image must be one 3D label map")
                values["label_dice"] = {}
                for label in np.union1d(np.unique(a), np.unique(b)):
                    ca, cb = a == label, b == label
                    denominator = int(ca.sum()) + int(cb.sum())
                    values["label_dice"][str(label)] = (
                        2 * int((ca & cb).sum()) / denominator if denominator else 1.0
                    )
            pair[name] = values
        report["comparisons"]["candidate_vs_" + backend] = pair
    return report


CASE_SCHEMA = {
    "required": ["id", "adapter", "input", "reference"],
    "adapter": "tools/benchmark_multimodal_cpu_applywarp.py",
    "mode": "fsl (default), plan, or world (no general official match)",
    "warp": "dense XYZ3 relative/absolute or intent-2007 cubic coefficient NIfTI",
    "premat": "optional FLIRT forward input-to-warp-source 4x4 text matrix",
    "postmat": "optional FLIRT forward warp-reference-to-output-reference 4x4 text matrix",
    "interpolation": "trilinear (default) or nearest/nn",
    "warp_convention": "auto (default), relative or absolute",
    "output_dtype": "omit for automatic, or char/short/int/float/double",
    "frame_chunk_size": "omit for default or positive integer; every frame is processed",
    "maps": "plan mode only: nonempty output_name:full_3D_or_4D_source mapping",
    "output_suffix": ".nii.gz (default) or .nii; equal on all backends",
    "precision_mask": "optional declared 3D brain mask for post-timing regional metrics",
    "label_image": "explicit bool: 3D discrete atlas/segmentation output; enables per-label Dice",
    "world": "explicit world chain; isolated oracle must establish semantics before official timing",
}

