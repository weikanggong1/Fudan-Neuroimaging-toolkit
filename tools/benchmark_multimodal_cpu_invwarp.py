"""InvWarp adapter: dense/coefficient fields, relative/absolute, full real grids.

Sample case::

    {"id": "invwarp_tbss_dense_relative", "adapter": "tools/benchmark_multimodal_cpu_invwarp.py",
     "reference": "/data/native_ref.nii.gz", "warp": "/data/forward.nii.gz",
     "warp_convention": "relative", "output_convention": "relative",
     "iterations": 30, "tolerance_mm": 0.01}

FNIT fixed-point and native FSL inversion are different solvers and stopping
rules. The tested FSL 6.0.7.4 binary does not expose an ``--niter`` option;
that flag is used only when an independently probed resource declares it.
In the tested FSL 6.0.7.4 implementation, ``--abs``/``--rel`` select the
input convention; the inverse is saved as a relative FNIRT field. Absolute
output requires a separately timed native convertwarp step. This differs
from wording in some upstream documentation.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from fnit.applywarp.core import _fsl_voxel_matrix, _spatial_grid
from fnit.convertwarp.core import _PullField
from fnit.invwarp import TorchInvWarp


def _outputs(case, output_dir):
    suffix = case.get("output_suffix", ".nii.gz")
    if suffix not in (".nii", ".nii.gz"):
        raise ValueError("output_suffix must be .nii or .nii.gz")
    return {"inverse_warp": str(Path(output_dir) / f"inverse_warp{suffix}")}


def run_case(case, output_dir, device):
    outputs = _outputs(case, output_dir)
    TorchInvWarp(device).run(
        reference=case["reference"], warp=case["warp"], output=outputs["inverse_warp"],
        warp_convention=case.get("warp_convention", "auto"),
        output_convention=case.get("output_convention", "relative"),
        iterations=case.get("iterations", 30), tolerance_mm=case.get("tolerance_mm", 0.01),
    )
    return outputs


def _binary(resources, name):
    configured = resources.get(name)
    return str(configured if configured else Path(resources["fsl_dir"]) / "bin" / name)


def reference_command(case, output_dir, resources):
    outputs = _outputs(case, output_dir)
    input_convention = case.get("warp_convention", "auto")
    output_convention = case.get("output_convention", "relative")
    if input_convention not in ("auto", "relative", "absolute") or output_convention not in ("relative", "absolute"):
        raise ValueError("invalid warp convention")
    coefficient_input = int(nib.load(case["warp"]).header["intent_code"]) == 2007
    if input_convention == "auto":
        if coefficient_input or int(nib.load(case["warp"]).header["intent_code"]) == 2006:
            input_convention = "relative"
        elif case.get("official_input_convention") in ("relative", "absolute"):
            # Known from independently recorded conversion/provenance. Do not
            # reuse FNIT's heuristic to define its own reference convention.
            input_convention = case["official_input_convention"]
        else:
            raise ValueError("auto dense convention requires official_input_convention provenance or FSL intent 2006")
    # FnirtFileReader ignores the dense convention for coefficient files.
    # Its coefficients are relative, irrespective of an explicit --abs.
    convention = "relative" if coefficient_input else input_convention
    absolute_output = output_convention == "absolute"
    native_output = str(Path(output_dir) / f"native_inverse{case.get('output_suffix', '.nii.gz')}") if absolute_output else outputs["inverse_warp"]
    command = [_binary(resources, "invwarp"), f"--ref={case['reference']}", f"--warp={case['warp']}",
               f"--out={native_output}", "--abs" if convention == "absolute" else "--rel"]
    if resources.get("invwarp_supports_niter", False):
        command.append(f"--niter={case.get('iterations', 30)}")
    if not absolute_output:
        return command
    conversion = [_binary(resources, "convertwarp"), f"--ref={case['reference']}",
                  f"--warp1={native_output}", f"--out={outputs['inverse_warp']}",
                  "--rel", "--absout"]
    return [command, conversion]


def reference_outputs(case, output_dir, resources):
    return _outputs(case, output_dir)


def compare_case(case, output_sets, resources):
    """Same-grid vector errors plus forward/inverse composition residuals."""
    output_sets = {backend: outputs for backend, outputs in output_sets.items()
                   if isinstance(outputs, dict) and outputs.get("inverse_warp")}
    reference_image = nib.load(case["reference"])
    mask_path = case.get("accuracy_mask")
    mask = np.asarray(nib.load(mask_path).dataobj if mask_path else reference_image.dataobj) > 0
    report = {"region": "explicit_accuracy_mask" if mask_path else "positive_reference_foreground",
              "region_voxels": int(mask.sum()), "available_backends": sorted(output_sets),
              "solver_contract": {"fnit_solver": "global-affine-initialised fixed-point",
                                  "fnit_maximum_iterations": case.get("iterations", 30),
                                  "fnit_tolerance_mm": case.get("tolerance_mm", 0.01),
                                  "official_niter_option_available": bool(resources.get("invwarp_supports_niter", False)),
                                  "official_maximum_iterations": case.get("iterations", 30) if resources.get("invwarp_supports_niter", False) else None,
                                  "official_solver_stopping": "native FSL program defaults unless supported niter is explicitly requested",
                                  "same_stopping_rule": False},
              "pairs": {}, "composition": {}}
    arrays = {name: np.asarray(nib.load(outputs["inverse_warp"]).dataobj, dtype=np.float64)
              for name, outputs in output_sets.items()}
    for left, right in (("candidate", "baseline"), ("candidate", "official"), ("baseline", "official")):
        if left not in arrays or right not in arrays:
            continue
        a, b = arrays[left], arrays[right]
        if a.shape != (*mask.shape, 3) or b.shape != (*mask.shape, 3):
            report["pairs"][f"{left}_vs_{right}"] = {"status": "mask_or_output_grid_mismatch"}
            continue
        norms = np.linalg.norm(a[mask] - b[mask], axis=-1)
        finite = np.isfinite(norms)
        report["pairs"][f"{left}_vs_{right}"] = {
            "all_finite": bool(finite.all()),
            "mean_vector_difference_mm": float(norms[finite].mean()) if finite.any() else None,
            "median_vector_difference_mm": float(np.median(norms[finite])) if finite.any() else None,
            "p95_vector_difference_mm": float(np.percentile(norms[finite], 95)) if finite.any() else None,
            "max_vector_difference_mm": float(norms[finite].max()) if finite.any() else None,
        }
    # This diagnostic runs after timing; the forward field is identical for all
    # three backends. Boundary and valid-field residuals are reported separately.
    with torch.inference_mode():
        field = _PullField(case["warp"], torch.device("cpu"), case.get("warp_convention", "auto"))
        target = _spatial_grid(reference_image.shape, _fsl_voxel_matrix(reference_image), torch.device("cpu")).reshape(3, *reference_image.shape)
        region = torch.as_tensor(mask)
        for backend, values in arrays.items():
            if values.shape != (*mask.shape, 3):
                continue
            query = torch.from_numpy(np.moveaxis(values, -1, 0).copy())
            if case.get("output_convention", "relative") == "relative":
                query += target
            mapped, valid = field.sample(query)
            residual = (mapped - target).square().sum(dim=0).sqrt()
            selected = valid & region & torch.isfinite(residual)
            residual_values = residual[selected].numpy()
            report["composition"][backend] = {
                "valid_field_fraction_in_region": float(valid[region].float().mean()) if region.any() else None,
                "median_residual_mm_valid_region": float(np.median(residual_values)) if residual_values.size else None,
                "p95_residual_mm_valid_region": float(np.percentile(residual_values, 95)) if residual_values.size else None,
            }
    return report


FUNCTION_COVERAGE = {
    "supported": ["dense_relative", "dense_absolute", "coefficient_intent2007", "embedded_affine",
                  "auto_intent", "relative_output", "absolute_output", "iterations", "tolerance_mm"],
    "unsupported": ["FSL_regularise", "FSL_jmin", "FSL_jmax", "FSL_noconstraint"],
    "non_equivalent": ["FSL_gradient_descent_and_regularisation", "FSL_Jacobian_constraint", "FSL_iteration_semantics"],
}
