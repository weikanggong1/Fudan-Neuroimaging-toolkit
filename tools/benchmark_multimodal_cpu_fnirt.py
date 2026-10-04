"""FNIRT adapter for the real-data, same-thread paired benchmark runner.

Sample case (paths must point to frozen real inputs)::

    {"id": "fnirt_t1_sub01", "adapter": "tools/benchmark_multimodal_cpu_fnirt.py",
     "input": "/data/T1_brain.nii.gz", "reference": "/data/MNI152_T1_2mm_brain.nii.gz",
     "affine": "/data/T1_to_MNI.mat", "refmask": "/data/MNI_mask.nii.gz",
     "preset": "t1", "overrides": {}, "full_pull_jacobian": false}

``preset`` is default/gm/t1/tbss. Overrides are the existing FNIRTConfig
fields; schedules and iteration counts are never shortened by this adapter.
The official TBSS chain includes required coefficient/intensity handoffs.
Multi-process overrides may estimate intensity only in their first process;
later intensity re-estimation requires a distinct native handoff adapter.
Official programs are launched only by the benchmark runner, never FNIT.
"""

from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from fnit.fnirt import resolve_fnirt_config
from fnit.fnirt.standalone import run_fnirt


def _config(case):
    config = resolve_fnirt_config(case.get("preset", "default"))
    overrides = dict(case.get("overrides", {}))
    known = {field.name for field in fields(config)}
    if set(overrides) - known:
        raise ValueError(f"unsupported FNIRTConfig fields: {sorted(set(overrides) - known)}")
    for key, value in overrides.items():
        if isinstance(value, list):
            overrides[key] = tuple(tuple(item) if isinstance(item, list) else item for item in value)
    config = replace(config, **overrides)
    # The native parser accepts one minimizer per invocation. A mixed
    # minimizer case must declare the same process boundary in both solvers;
    # silently splitting only the native command changes coefficient rounding.
    stages = config.process_stages or (1,) * len(config.subsampling)
    if config.minimization_methods is not None:
        for position in range(1, len(stages)):
            if (stages[position] == stages[position - 1]
                    and config.minimization_methods[position]
                    != config.minimization_methods[position - 1]):
                raise ValueError("mixed minimization_methods require explicit process_stages at each method change")
    return config


def _outputs(case, output_dir):
    output_dir = Path(output_dir)
    suffix = case.get("output_suffix", ".nii.gz")
    if suffix not in (".nii", ".nii.gz"):
        raise ValueError("output_suffix must be .nii or .nii.gz")
    outputs = {name: str(output_dir / f"{name}{suffix}") for name in ("cout", "iout", "jout")}
    if case.get("full_pull_jacobian", False):
        outputs["full_pull_jacobian"] = str(output_dir / f"full_pull_jacobian{suffix}")
    return outputs


def run_case(case, output_dir, device):
    outputs = _outputs(case, output_dir)
    result = run_fnirt(
        input=case["input"], reference=case["reference"], affine=case.get("affine"),
        refmask=case.get("refmask"), config=_config(case), device=device,
        execution=case.get("execution", "optimized"),
        cout=outputs["cout"], iout=outputs["iout"], jout=outputs["jout"], overwrite=False,
    )
    if "full_pull_jacobian" in outputs:
        nib.save(result.full_pull_jacobian, outputs["full_pull_jacobian"])
    return outputs


def _binary(resources, name):
    configured = resources.get(name)
    return str(configured if configured else Path(resources["fsl_dir"]) / "bin" / name)


def _csv(values):
    return ",".join(str(int(value)) if isinstance(value, bool) else str(value) for value in values)


def _stage_options(config, start, stop):
    options = []
    for flag, field in (
        ("subsamp", "subsampling"), ("miter", "maximum_iterations"),
        ("infwhm", "input_fwhm_mm"), ("reffwhm", "reference_fwhm_mm"),
        ("lambda", "regularization"), ("estint", "estimate_intensity"),
        ("applyrefmask", "apply_reference_mask"),
    ):
        options.append(f"--{flag}={_csv(getattr(config, field)[start:stop])}")
    # FSL 6.0.7.4 retains a four-entry default applyinmask schedule and
    # rejects a six-level or one-level process unless this is explicit.
    # No explicit input mask is supplied: enabled is the unchanged default.
    options.append(f"--applyinmask={_csv((True,) * (stop - start))}")
    if config.minimization_methods is not None:
        methods = config.minimization_methods[start:stop]
        if len(set(methods)) != 1:
            raise ValueError("official FNIRT has one --minmet per process; mixed methods require explicit process_stages")
        options.append(f"--minmet={methods[0]}")
    resolutions = config.warp_resolution_schedule_mm
    resolution = config.warp_resolution_mm if resolutions is None else resolutions[start]
    if resolutions is not None and any(value != resolution for value in resolutions[start:stop]):
        raise ValueError("official FNIRT has one --warpres per process; split process_stages")
    options.extend([
        f"--warpres={_csv(resolution)}", f"--jacrange={_csv(config.jacobian_range)}",
        f"--intmod={config.intensity_model}", f"--intorder={config.intensity_order}",
        f"--biasres={_csv(config.bias_resolution_mm)}", f"--biaslambda={config.bias_regularization}",
        f"--ssqlambda={int(config.ssd_weighted_lambda)}",
        f"--imprefm={int(config.implicit_reference_mask)}", f"--impinm={int(config.implicit_input_mask)}",
        # FNIT computes trilinear image samples; FNIRT names that mode
        # "linear". Its parser rejects the applywarp-style "trilinear" name.
        "--regmod=bending_energy", "--splineorder=3", "--interp=linear",
    ])
    return options


def reference_command(case, output_dir, resources):
    """Build the native FSL chain with exactly the resolved FNIT schedule."""
    output_dir = Path(output_dir)
    outputs = _outputs(case, output_dir)
    config = _config(case)
    stages = config.process_stages or (1,) * len(config.subsampling)
    bounds = []
    start = 0
    for position in range(1, len(stages) + 1):
        if position == len(stages) or stages[position] != stages[start]:
            bounds.append((start, position))
            start = position
    if any(any(config.estimate_intensity[start:stop]) for start, stop in bounds[1:]):
        raise ValueError("official multi-process adapter supports fixed intensity after the first process; later estimate_intensity requires separate intensity output/input handoffs")
    commands = []
    previous = None
    intensity = output_dir / "handoff_intensity.txt"
    for index, (start, stop) in enumerate(bounds):
        final = index == len(bounds) - 1
        coefficients = outputs["cout"] if final else str(output_dir / f"handoff_stage{index + 1}{case.get('output_suffix', '.nii.gz')}")
        command = [_binary(resources, "fnirt"), f"--in={case['input']}", f"--ref={case['reference']}",
                   f"--cout={coefficients}", *_stage_options(config, start, stop)]
        if case.get("refmask"):
            command.append(f"--refmask={case['refmask']}")
        if previous is None:
            if case.get("affine"):
                command.append(f"--aff={case['affine']}")
            if len(bounds) > 1:
                command.append(f"--intout={intensity.with_suffix('')}")
        else:
            command.extend([f"--inwarp={previous}", f"--intin={intensity}"])
        if final:
            command.extend([f"--iout={outputs['iout']}", f"--jout={outputs['jout']}"])
        commands.append(command)
        previous = coefficients
    if "full_pull_jacobian" in outputs:
        commands.append([_binary(resources, "fnirtfileutils"), f"--in={outputs['cout']}",
                         f"--ref={case['reference']}", "--withaff", f"--jac={outputs['full_pull_jacobian']}"])
    return commands[0] if len(commands) == 1 else commands


def reference_outputs(case, output_dir, resources):
    return _outputs(case, output_dir)


def compare_case(case, output_sets, resources):
    """Region metrics complement full-array comparisons; no image is exported."""
    output_sets = {backend: outputs for backend, outputs in output_sets.items()
                   if isinstance(outputs, dict) and outputs}
    mask_path = case.get("accuracy_mask") or case.get("refmask")
    mask = np.asarray(nib.load(mask_path or case["reference"]).dataobj) > 0
    region = "explicit_accuracy_mask" if case.get("accuracy_mask") else "reference_mask" if mask_path else "positive_reference_foreground"
    report = {"region": region, "region_voxels": int(mask.sum()),
              "available_backends": sorted(output_sets), "pairs": {}}
    if "official" in output_sets and output_sets["official"].get("cout"):
        directory = Path(output_sets["official"]["cout"]).parent
        handoffs = sorted(directory.glob("handoff_stage*")) + sorted(directory.glob("handoff_intensity*"))
        report["native_process_handoffs"] = {
            path.name: {"bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in handoffs if path.is_file()
        }
        report["native_process_clocks"] = [json.loads(path.read_text())
            for path in sorted(directory.glob("command_*/timing.private.json"))]
    for left, right in (("candidate", "baseline"), ("candidate", "official"), ("baseline", "official")):
        if left not in output_sets or right not in output_sets:
            continue
        pair = {}
        for name in ("iout", "jout", "full_pull_jacobian"):
            if name not in output_sets[left] or name not in output_sets[right]:
                continue
            a = np.asarray(nib.load(output_sets[left][name]).dataobj, dtype=np.float64)
            b = np.asarray(nib.load(output_sets[right][name]).dataobj, dtype=np.float64)
            if a.shape != mask.shape or b.shape != mask.shape:
                pair[name] = {"status": "mask_or_output_grid_mismatch"}
                continue
            x, y = a[mask], b[mask]
            finite = np.isfinite(x) & np.isfinite(y)
            delta = np.abs(x[finite] - y[finite])
            xc, yc = x[finite], y[finite]
            if xc.size:
                xc, yc = xc - xc.mean(), yc - yc.mean()
            denominator = float(np.linalg.norm(xc) * np.linalg.norm(yc))
            metrics = {"all_finite": bool(finite.all()), "mean_absolute_error": float(delta.mean()) if delta.size else None,
                       "p95_absolute_error": float(np.percentile(delta, 95)) if delta.size else None,
                       "pearson_r": float(np.dot(xc, yc) / denominator) if denominator else None}
            if name == "iout":
                a_support, b_support = x != 0, y != 0
                total = int(a_support.sum() + b_support.sum())
                metrics["nonzero_support_dice"] = 2 * int((a_support & b_support).sum()) / total if total else 1.0
            else:
                metrics["candidate_nonpositive_fraction"] = float(np.mean(x <= 0)) if x.size else None
                metrics["reference_nonpositive_fraction"] = float(np.mean(y <= 0)) if y.size else None
                metrics["candidate_range"] = [float(x[finite].min()), float(x[finite].max())] if finite.any() else None
                metrics["reference_range"] = [float(y[finite].min()), float(y[finite].max())] if finite.any() else None
            pair[name] = metrics
        report["pairs"][f"{left}_vs_{right}"] = pair
    return report


FUNCTION_COVERAGE = {
    "supported": ["default", "gm", "t1", "tbss", "global_linear", "global_non_linear_with_bias",
                  "lm", "scg", "reference_mask", "implicit_masks", "FLIRT_affine", "identity_affine",
                  "coefficient_output", "warped_input_output", "nonlinear_jacobian", "full_pull_jacobian",
                  "TBSS_fixed_intensity_process_handoff", "supported_schedule_overrides"],
    "unsupported": ["arbitrary_cnf", "inmask", "general_inwarp", "general_intin", "DCT", "quadratic_spline",
                    "local_intensity_model", "refout", "intout", "CLI_fout", "membrane_energy", "refderiv",
                    "multi_process_later_intensity_estimation_reference_adapter"],
}
