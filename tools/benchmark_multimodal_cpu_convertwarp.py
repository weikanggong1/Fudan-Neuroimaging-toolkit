#!/usr/bin/env python3
"""Full load/compose/save adapter for real ConvertWarp CPU comparisons.

The runner, rather than this module, executes native FSL reference programs.
All case paths must point to frozen real image/registration inputs. Example::

    {"id": "convertwarp_tbss_coeff", "adapter":
     "tools/benchmark_multimodal_cpu_convertwarp.py", "mode": "fsl",
     "reference": "/data/standard_FA.nii.gz", "warp1": "/data/FA_to_MNI_coef.nii.gz",
     "warp_convention": "auto", "output_convention": "relative"}

MMORF reference-axis fields have no direct FSL ``convertwarp`` CLI. Those
cases support candidate/baseline comparisons; a native reference command is
deliberately unavailable, rather than interpreting an MMORF field as FSL.
"""

from pathlib import Path


def _output(case, output_dir):
    suffix = case.get("output_suffix", ".nii.gz")
    if suffix not in {".nii", ".nii.gz"}:
        raise ValueError("output_suffix must be .nii or .nii.gz")
    return Path(output_dir) / ("warp" + suffix)


def _mode(case):
    mode = case.get("mode", "fsl")
    if mode not in {"fsl", "mmorf"}:
        raise ValueError("ConvertWarp mode must be fsl or mmorf")
    return mode


def run_case(case: dict, output_dir: Path, device: str) -> dict:
    """Read every input, perform the public operation, and save its full field."""
    from fnit.convertwarp import TorchConvertWarp

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = _output(case, output_dir)
    model = TorchConvertWarp(device=device)
    convention = case.get("output_convention", "relative")
    if _mode(case) == "mmorf":
        model.run_mmorf(
            reference=case["reference"], source=case["source"],
            mmorf_warp=case["mmorf_warp"], affine=case["affine"], output=output,
            output_convention=convention,
        )
    else:
        model.run(
            reference=case["reference"], warp1=case["warp1"],
            premat=case.get("premat"), postmat=case.get("postmat"), output=output,
            warp_convention=case.get("warp_convention", "auto"),
            output_convention=convention,
        )
    return {"warp": str(output)}


def _binary(resources):
    for key in ("fsl_convertwarp", "convertwarp"):
        if key in resources:
            return str(resources[key])
    if "fsl_bin" in resources:
        return str(Path(resources["fsl_bin"]) / "convertwarp")
    for key in ("fsl_dir", "fsl_root"):
        if key in resources:
            return str(Path(resources[key]) / "bin" / "convertwarp")
    raise ValueError("resources must declare fsl_convertwarp, fsl_bin or fsl_dir")


def reference_command(case: dict, output_dir: Path, resources: dict) -> list:
    if _mode(case) == "mmorf":
        raise ValueError("FNIT MMORF reference-axis input has no direct FSL convertwarp CLI")
    convention = case.get("warp_convention", "auto")
    aliases = {"auto": None, "rel": "--rel", "relative": "--rel",
               "abs": "--abs", "absolute": "--abs"}
    try:
        input_flag = aliases[str(convention).lower()]
    except KeyError as error:
        raise ValueError("warp_convention must be auto, relative or absolute") from error
    output_convention = case.get("output_convention", "relative")
    if output_convention not in {"relative", "absolute"}:
        raise ValueError("output_convention must be relative or absolute")
    command = [_binary(resources), f"--ref={case['reference']}", f"--warp1={case['warp1']}",
               f"--out={_output(case, output_dir)}",
               "--relout" if output_convention == "relative" else "--absout"]
    if input_flag is not None:
        command.append(input_flag)
    for key in ("premat", "postmat"):
        if case.get(key) is not None:
            if not isinstance(case[key], (str, Path)):
                raise ValueError("official benchmark affine inputs must be frozen matrix files")
            command.append(f"--{key}={case[key]}")
    return command


def reference_outputs(case: dict, output_dir: Path, resources: dict) -> dict:
    if _mode(case) == "mmorf":
        raise ValueError("MMORF conversion has no matching FSL reference output")
    return {"warp": str(_output(case, output_dir))}


def compare_case(case: dict, output_sets: dict, resources: dict) -> dict:
    """Full/inside/outside/brain metrics and spatial headers; no image export."""
    import nibabel as nib
    import numpy as np
    from fnit.applywarp.core import _coefficient_metadata, _fsl_voxel_matrix, _matrix

    reference = nib.load(case["reference"])
    foreground = np.asarray(reference.dataobj) > 0
    if _mode(case) == "mmorf":
        valid = np.ones(reference.shape, dtype=bool)
    else:
        warp = nib.load(case["warp1"])
        if int(warp.header["intent_code"]) == 2007:
            shape, zooms, _, _ = _coefficient_metadata(warp)
            scaled = np.diag((*zooms, 1.0))
        else:
            shape, scaled = warp.shape[:3], _fsl_voxel_matrix(warp)
        post_pull = np.linalg.inv(_matrix(case.get("postmat"), "postmat")) @ _fsl_voxel_matrix(reference)
        inverse_scaled = np.linalg.inv(scaled)
        axes = [np.arange(size, dtype=np.float64) for size in reference.shape]
        valid = np.ones(reference.shape, dtype=bool)
        for axis in range(3):
            stored = (post_pull[axis, 0] * axes[0][:, None, None]
                      + post_pull[axis, 1] * axes[1][None, :, None]
                      + post_pull[axis, 2] * axes[2][None, None, :] + post_pull[axis, 3]).astype(np.float32)
            # FSL scaled matrices are diagonal; reproduce both float32
            # storage boundaries and strict concat_warps in_bounds.
            query = (inverse_scaled[axis, axis] * stored.astype(np.float64)
                     + inverse_scaled[axis, 3]).astype(np.float32)
            valid &= (query >= 0) & (query <= shape[axis] - 1)
    regions = {"all": np.ones(reference.shape, dtype=bool), "inside_warp_grid": valid,
               "outside_warp_grid": ~valid, "positive_reference": foreground,
               "positive_reference_inside_warp": foreground & valid,
               "positive_reference_outside_warp": foreground & ~valid}
    report = {"region_definition": "positive_reference uses reference intensities > 0; grid validity is strict FSL float32 postmat/voxel-query bounds",
              "region_voxels": {name: int(mask.sum()) for name, mask in regions.items()},
              "metadata": {}, "pairs": {}}
    images = {}
    reference_q, reference_qcode = reference.get_qform(coded=True)
    reference_s, reference_scode = reference.get_sform(coded=True)
    for backend, outputs in output_sets.items():
        image = nib.load(outputs["warp"])
        images[backend] = np.asarray(image.dataobj, dtype=np.float64)
        qform, qcode = image.get_qform(coded=True)
        sform, scode = image.get_sform(coded=True)
        report["metadata"][backend] = {
            "shape": list(image.shape), "dtype": str(image.get_data_dtype()),
            "intent_code": int(image.header["intent_code"]),
            "zooms": [float(value) for value in image.header.get_zooms()],
            "qform_code": int(qcode), "sform_code": int(scode),
            "qform_code_matches_reference": bool(qcode == reference_qcode),
            "sform_code_matches_reference": bool(scode == reference_scode),
            "qform_max_difference_from_reference": float(np.max(np.abs(qform - reference_q))) if qform is not None and reference_q is not None else None,
            "sform_max_difference_from_reference": float(np.max(np.abs(sform - reference_s))) if sform is not None and reference_s is not None else None,
        }
    for left, right in (("candidate", "official"), ("candidate", "baseline"), ("baseline", "official")):
        if left not in images or right not in images:
            continue
        if images[left].shape != (*reference.shape, 3) or images[right].shape != (*reference.shape, 3):
            report["pairs"][f"{left}_vs_{right}"] = {"status": "shape_mismatch"}
            continue
        delta = images[left] - images[right]
        pair = {}
        for name, mask in regions.items():
            differences = delta[mask]
            finite = np.isfinite(differences).all(axis=1)
            values = np.abs(differences[finite])
            distance = np.linalg.norm(differences[finite], axis=1)
            pair[name] = {"voxels": int(mask.sum()), "all_finite": bool(finite.all()),
                "max_component_error_mm": float(values.max()) if values.size else None,
                "mae_component_mm": float(values.mean()) if values.size else None,
                "p95_component_error_mm": float(np.percentile(values, 95)) if values.size else None,
                "p99_component_error_mm": float(np.percentile(values, 99)) if values.size else None,
                "max_vector_error_mm": float(distance.max()) if distance.size else None,
                "voxels_component_error_gt_1e_3_mm": int(np.sum(np.max(values, axis=1) > 1e-3)) if values.size else 0}
        report["pairs"][f"{left}_vs_{right}"] = pair
    return report


CASE_SCHEMA = {
    "required": ["id", "adapter", "reference"],
    "fsl_required": ["warp1"],
    "mmorf_required": ["source", "mmorf_warp", "affine"],
    "defaults": {"mode": "fsl", "warp_convention": "auto",
                 "output_convention": "relative", "output_suffix": ".nii.gz"},
    "optional": ["premat", "postmat", "accuracy_mask"],
}

FUNCTION_COVERAGE = {
    "supported_official": [
        "dense_intent2006", "dense_intent0_relative", "dense_intent0_absolute",
        "auto_convention", "explicit_relative_convention", "explicit_absolute_convention",
        "cubic_coefficients_intent2007", "embedded_coefficient_affine",
        "premat", "postmat", "premat_and_postmat", "identity_matrices",
        "relative_output", "absolute_output", "reference_grid", "qform_sform_preservation",
        "float32_output", "inside_field_trilinear", "outside_field_best_fit_affine",
        "valid_fraction", "save", "run", "CLI",
    ],
    "fnit_extra": ["MMORF_reference_axis_mm", "MMORF_affine_composition",
                   "MMORF_relative_output", "MMORF_absolute_output", "run_mmorf"],
    "unsupported": ["warp2", "shiftmap", "shift_direction", "jout", "jacobian_constraint",
                    "midmat", "DCT_intent2008", "quadratic_coefficients_intent2009",
                    "direct_original_MMORF_cli"],
}
