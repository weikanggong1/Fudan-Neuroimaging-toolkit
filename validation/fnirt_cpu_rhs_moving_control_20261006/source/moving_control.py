"""One saved-moving RHS control; preparation alone never starts this worker."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import socket
import struct
import sys
import time
from types import SimpleNamespace

from moving_io import bound, check_bindings, check_freeze, input_paths, write_json
from gradient_body import bindings, compiled


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "workspace", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--approved-moving-control", action="store_true", required=True)
    args = parser.parse_args()
    if sorted(os.sched_getaffinity(0)) != [32, 36, 40, 44, 48, 52, 56, 60]:
        raise RuntimeError("physical CPU8 affinity required")
    if any(os.environ.get(name) != "8" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")):
        raise RuntimeError("eight-thread environment required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or any(name in os.environ for name in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH", "OPENBLAS_CORETYPE")):
        raise RuntimeError("CPU-only without loader overrides required")
    resource.setrlimit(resource.RLIMIT_AS, (20_000_000_000, 20_000_000_000))
    started = time.monotonic()
    expected = json.loads((args.workspace / "expected.public.json").read_text())
    before, failures = check_bindings(args.root, expected)
    harness, freeze_failures = check_freeze(args.workspace)
    if failures or freeze_failures:
        raise RuntimeError("pre-import source/input/freeze changed")
    paths = input_paths(args.root, expected)
    stage1 = json.loads(paths["stage1_summary"].read_text())
    native = json.loads(paths["native_state"].read_text())
    native_state = next(row for row in native["states"] if row["solve"] == 3)
    if not stage1["accepted_checkpoint"] or bound(paths["checkpoint"]) != {
            key: stage1["checkpoint"][key] for key in ("bytes", "sha256")}:
        raise RuntimeError("accepted checkpoint identity missing")
    if native_state != expected["native_state"]:
        raise RuntimeError("saved native solve3 state changed")
    target_lambda = expected["fixed_effective_lambda"]
    if native_state["regularization_lambda"] != target_lambda:
        raise RuntimeError("samepoint native effective lambda does not match frozen control")
    registration_source = args.root / "repo/src/fnit/fnirt/registration.py"
    assembly_source = args.root / "repo/validation/fnirt_shared_followup_20261006/source/assembly_shared_v2.py"
    ast_binding = bindings(registration_source, assembly_source)
    if ast_binding != expected["gradient_ast"]:
        raise RuntimeError("bound gradient/scalar/projection AST changed")
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    sys.path.insert(0, str(args.root / "repo/src"))
    import nibabel as nib
    import numpy as np
    import torch
    import numba
    from fnit.fnirt import registration, optimizer
    from fnit.fnirt.spline import BendingOperator
    if Path(registration.__file__).resolve() != registration_source.resolve():
        raise RuntimeError("wrong actual FNIT registration import")
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    canonical_env = args.root / "envs/default"
    if Path(sys.prefix).resolve() != canonical_env.resolve() or Path(sys.executable).resolve() != (canonical_env / "bin/python").resolve():
        raise RuntimeError("actual Python prefix/interpreter differs from declared canonical environment")
    runtime = {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
               "nibabel": nib.__version__, "numba": numba.__version__,
               "canonical_environment_alias": str(canonical_env),
               "actual_prefix_basename": Path(sys.prefix).name,
               "actual_prefix_path_sha256": hashlib.sha256(str(Path(sys.prefix).resolve()).encode()).hexdigest(),
               "actual_interpreter_basename": Path(sys.executable).resolve().name,
               "actual_interpreter_identity": bound(Path(sys.executable).resolve()),
               "actual_prefix_matches_declared_alias": True}

    def require(condition, message):
        if not condition:
            raise RuntimeError(message)

    def flags():
        return {"cuda_initialized": torch.cuda.is_initialized(), "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                "matmul_tf32": torch.backends.cuda.matmul.allow_tf32, "grad_enabled": torch.is_grad_enabled(),
                "default_dtype": str(torch.get_default_dtype())}

    def mapped_fsl():
        # Before/after identity snapshots, not a full-time loader/syscall audit.
        return sorted({Path(line.split()[-1]).name for line in Path("/proc/self/maps").read_text().splitlines()
                       if len(line.split()) >= 6 and ("/apps/FSL/" in line.split()[-1]
                           or Path(line.split()[-1]).name.startswith("libfsl"))})

    def array(value):
        if isinstance(value, torch.Tensor):
            require(value.device.type == "cpu" and not value.requires_grad, "CPU/no-grad operand required")
            value = value.detach().numpy()
        require(isinstance(value, np.ndarray) and bool(np.isfinite(value).all()), "finite array required")
        return np.ascontiguousarray(value)

    def value_hash(value):
        return hashlib.sha256(array(value).tobytes()).hexdigest()

    def hashes(values):
        return {name: value_hash(value) for name, value in values.items()}

    def scalar(value):
        number = float(value)
        require(math.isfinite(number), "nonfinite scalar")
        return {"value": number, "FP64_little_endian_hex": struct.pack("<d", number).hex()}

    def metrics(reference, actual):
        left, right = array(reference), array(actual)
        require(left.shape == right.shape and left.dtype == right.dtype, "metric shape/dtype mismatch")
        difference = right.astype(np.float64) - left.astype(np.float64)
        denominator = float(np.linalg.norm(left.astype(np.float64).ravel()))
        bits = (left.view(np.dtype("u" + str(left.dtype.itemsize)))
                != right.view(np.dtype("u" + str(right.dtype.itemsize))))
        return {"shape": list(left.shape), "dtype": str(left.dtype), "different_bits": int(np.count_nonzero(bits)),
                "max_abs": float(np.max(np.abs(difference), initial=0)),
                "rmse": float(np.sqrt(np.mean(difference * difference))),
                "relative_l2": float(np.linalg.norm(difference.ravel()) / denominator) if denominator else None,
                "reference_value_sha256": value_hash(left), "actual_value_sha256": value_hash(right),
                "bitexact": left.tobytes() == right.tobytes()}

    def gate(name, reference, actual):
        row = metrics(reference, actual)
        report["gates"][name] = row
        require(row["bitexact"], "first mismatch: " + name)

    def scalar_gate(name, reference, actual):
        row = {"reference": scalar(reference), "actual": scalar(actual)}
        row["bitexact"] = row["reference"]["FP64_little_endian_hex"] == row["actual"]["FP64_little_endian_hex"]
        report["gates"][name] = row
        require(row["bitexact"], "first mismatch: " + name)

    report = {"scope": expected["scope"], "status": "started_not_accepted", "control_completed": False,
              "production_integration_accepted": False, "host": socket.gethostname(),
              "affinity": sorted(os.sched_getaffinity(0)), "torch_threads": torch.get_num_threads(),
              "interop_threads": torch.get_num_interop_threads(), "address_space_cap_bytes": 20_000_000_000,
              "runtime": runtime,
              "bindings_before": before, "harness_bindings": harness, "gradient_ast": ast_binding,
              "flags_before": flags(), "fsl_dso_snapshot_before": mapped_fsl(), "gates": {},
              "calls": {"sampler": 0, "coordinates": 0, "projection": 0, "state_scalar_prefix": 0,
                        "LM_gradient_prefix": 0, "FSL_order_prefix": 0, "actual_bending_normal": 0,
                        "cached_gradient_bending_reads": 0, "evaluate": 0, "linearize": 0,
                        "gradient_method": 0, "dense_field_expansion": 0, "design_diagonal": 0,
                        "PCG": 0, "SCG": 0, "H_callback": 0, "native": 0},
              "new_H": False, "diag_read_or_created": False, "normal_cache_created": False,
              "raw_MRI_read": False, "checkpoint_arrays_read": [], "images_read": [],
              "native_state": native_state, "lambda_policy": expected["lambda_policy"],
              "limitations": expected["limitations"], "clock": {"imports_and_binding_seconds": time.monotonic() - started}}
    require(not report["flags_before"]["cuda_initialized"] and not report["fsl_dso_snapshot_before"], "unexpected CUDA/FSL state")
    saved, operands, saved_before, operands_before, originals = {}, {}, {}, {}, []
    summary_path = args.output / "summary.public.json"

    def forbid(owner, name, counter):
        original = getattr(owner, name)
        def reject(*values, **kwargs):
            report["calls"][counter] += 1
            raise RuntimeError("out of scope call: " + counter)
        originals.append((owner, name, original))
        setattr(owner, name, reject)

    try:
        for name, counter in (("evaluate", "evaluate"), ("linearize", "linearize"), ("gradient", "gradient_method")):
            forbid(registration._LevelSystem, name, counter)
        forbid(registration, "expand_coefficients", "dense_field_expansion")
        forbid(registration, "design_diagonal", "design_diagonal")
        forbid(optimizer, "preconditioned_conjugate_gradient", "PCG")
        forbid(optimizer, "scaled_conjugate_gradient", "SCG")
        with torch.no_grad():
            restore_started = time.monotonic()
            with np.load(paths["checkpoint"], allow_pickle=False) as packed:
                require(set(packed.files) == set(stage1["checkpoint"]["arrays"]), "68-array checkpoint schema")
                for name in expected["checkpoint_arrays_read"]:
                    value, record = packed[name], stage1["checkpoint"]["arrays"][name]
                    require(list(value.shape) == record["shape"] and str(value.dtype) == record["dtype"]
                            and bool(np.isfinite(value).all()), "checkpoint array schema: " + name)
                    require(all(item >= 0 and item % value.itemsize == 0 for item in record["strides"]), "invalid saved stride")
                    source = torch.from_numpy(value.copy(order="K"))
                    tensor = torch.empty_strided(tuple(value.shape), tuple(item // value.itemsize for item in record["strides"]),
                                                dtype=source.dtype, device="cpu")
                    tensor.copy_(source)
                    require(list(item * value.itemsize for item in tensor.stride()) == record["strides"]
                            and value_hash(tensor) == value_hash(value), "values/stride restore: " + name)
                    saved[name] = tensor
                    report["checkpoint_arrays_read"].append(name)
                    report.setdefault("restored_layouts", {})[name] = {"shape": list(value.shape), "dtype": str(value.dtype),
                        "npz_byte_strides": list(value.strides), "restored_byte_strides": record["strides"],
                        "logical_value_sha256": value_hash(tensor)}
            saved_before = hashes(saved)
            report["checkpoint_values_before"] = saved_before
            pointers = [tensor.untyped_storage().data_ptr() for tensor in saved.values() if tensor.numel()]
            require(len(set(pointers)) == len(pointers), "restored operands alias")
            coefficients, scale = saved["coefficients"], saved["scale"]
            require(coefficients.dtype == scale.dtype == torch.float64 and saved["fixed"].dtype == torch.float32,
                    "F64 coefficient/scale and F32 image policy")
            packed_point = registration._pack(coefficients, scale)
            require(value_hash(packed_point) == expected["files"]["parameters"]["sha256"], "solve3 point bridge")
            operands["packed_point"] = packed_point
            operands_before["packed_point"] = value_hash(packed_point)
            geometry, state_info = stage1["geometry"], stage1["checkpoint"]["state_scalars"]
            require(not geometry["reference_mask_active"] and not geometry["moving_mask_active"], "unexpected active explicit masks")
            require(state_info == expected["state_scalars"], "accepted state scalars changed")

            current_fixed = np.load(paths["current_fixed"], allow_pickle=False)
            report["images_read"].append("current_fixed_level_npy")
            gate("current_fixed_vs_checkpoint_raw_fixed", current_fixed, saved["fixed"])
            operands["current_fixed_level"] = torch.from_numpy(current_fixed)
            operands_before["current_fixed_level"] = value_hash(current_fixed)

            def read_official(label):
                image = nib.load(paths[label])
                contract = expected["official_images"][label]
                require(list(image.shape) == contract["shape"] and str(image.get_data_dtype()) == "float32", "official image schema: " + label)
                require(list(float(item) for item in image.header.get_zooms()[:3]) == contract["pixdim"], "official header voxels: " + label)
                require(image.dataobj.slope == 1 and image.dataobj.inter == 0, "unexpected official intensity scaling")
                value = np.array(image.dataobj, dtype=np.float32, order="C", copy=True)
                require(bool(np.isfinite(value).all()), "official image nonfinite")
                report["images_read"].append(label)
                tensor = torch.from_numpy(value)
                operands[label] = tensor
                operands_before[label] = value_hash(tensor)
                return tensor

            official_fixed = read_official("official_fixed")
            gate("official_unscaled_Ref_vs_checkpoint_raw_fixed", saved["fixed"], official_fixed)
            report["fixed_phase"] = "Ref() before global scaling; same raw F32 values, no phase conversion"

            bending = BendingOperator.__new__(BendingOperator)
            bending.execution = "optimized"
            bending._normal_grams = tuple((tuple(saved[f"bending_gram_{term}_{axis}"] for axis in range(3)), multiplier)
                for term, multiplier in enumerate(stage1["checkpoint"]["bending_multipliers"]))
            require(len(bending._normal_grams) == 6, "bending Gram schema")
            report["calls"]["actual_bending_normal"] += 1
            constant_bending_gradient = bending.normal(coefficients)
            operands["constant_bending_gradient"] = constant_bending_gradient
            operands_before["constant_bending_gradient"] = value_hash(constant_bending_gradient)

            class CachedGradientBending:
                def normal(self, candidate):
                    report["calls"]["cached_gradient_bending_reads"] += 1
                    require(report["calls"]["cached_gradient_bending_reads"] <= 4, "gradient bending read cap")
                    require(candidate.dtype == coefficients.dtype and candidate.shape == coefficients.shape
                            and candidate.stride() == coefficients.stride()
                            and value_hash(candidate) == saved_before["coefficients"], "bending gradient changed coefficient point/layout")
                    return constant_bending_gradient

            system = SimpleNamespace(fixed=saved["fixed"], moving_fsl2vox=saved["moving_fsl2vox"],
                                     bases=tuple(saved[f"basis_{axis}"] for axis in range(3)),
                                     estimate_scale=True, bending=CachedGradientBending())
            functions = compiled(registration_source, assembly_source, {
                **registration.__dict__, "registration": registration})
            bend_energy = torch.tensor(state_info["bending_energy"], dtype=torch.float64)
            operands["restored_bending_energy_scalar"] = bend_energy
            operands_before["restored_bending_energy_scalar"] = value_hash(bend_energy)

            def state_from(warped, mask, gradient_fsl):
                report["calls"]["state_scalar_prefix"] += 1
                require(report["calls"]["state_scalar_prefix"] <= 2, "scalar prefix cap")
                count, scaled_fixed, residual, ssd = functions["state_scalars"](system, scale, warped, mask)
                state = {"count": count, "residual": residual, "mask": mask, "gradient_fsl": gradient_fsl,
                         "ssd": ssd, "bending_energy": bend_energy, "effective_lambda": target_lambda}
                cost = functions["fixed_lambda_cost"](state, target_lambda)
                require(bool(torch.isfinite(residual).all()) and bool(torch.isfinite(gradient_fsl).all()), "state nonfinite")
                return state, cost, scaled_fixed

            baseline, baseline_cost, scaled_fixed = state_from(saved["state_warped"], saved["state_mask"], saved["state_gradient_fsl"])
            gate("baseline_residual_vs_saved_state", saved["state_residual"], baseline["residual"])
            report["gates"]["baseline_count"] = {"reference": state_info["count"], "actual": baseline["count"],
                "exact": baseline["count"] == state_info["count"]}
            require(report["gates"]["baseline_count"]["exact"], "first mismatch: baseline count")
            scalar_gate("baseline_SSD", state_info["ssd"], baseline["ssd"])
            scalar_gate("baseline_fixed_lambda_cost", state_info["cost"], baseline_cost)
            scalar_gate("baseline_fixed_effective_lambda", state_info["effective_lambda"], target_lambda)

            def gradients(label, state):
                report["calls"]["LM_gradient_prefix"] += 1
                require(report["calls"]["LM_gradient_prefix"] <= 2, "LM prefix cap")
                lm = 2 * functions["lm_gradient"](system, coefficients, scale, state)
                report["calls"]["FSL_order_prefix"] += 1
                require(report["calls"]["FSL_order_prefix"] <= 2, "FSL-order prefix cap")
                fsl, product, scale_gradient = functions["fsl_order_gradient"](system, coefficients, state, target_lambda)
                report.setdefault("arms", {})[label] = {"full_g_LM_sha256": value_hash(lm), "full_g_FSL_order_sha256": value_hash(fsl),
                    "actual_dtypes": {"gradient_fsl": str(state["gradient_fsl"].dtype), "residual": str(state["residual"].dtype),
                        "mask_before_F32_conversion": str(state["mask"].dtype), "FSL_Jte_product_before_cast": str(product.dtype),
                        "adjoint_input": "torch.float64", "full_g": str(lm.dtype), "scale_gradient": str(scale_gradient.dtype)},
                    "FSL_order_scale_full_g": scalar(2 * scale_gradient)}
                require(lm.dtype == fsl.dtype == torch.float64 and lm.numel() == fsl.numel() == expected["dimension"], "full gradient convention")
                return lm, fsl

            baseline_lm, baseline_fsl = gradients("current_saved_moving", baseline)
            current_lm = np.fromfile(paths["current_g"], dtype="<f8")
            current_fsl = np.fromfile(paths["current_fsl_g"], dtype="<f8")
            gate("baseline_full_LM_g_vs_current_assembly", current_lm, baseline_lm)
            gate("baseline_full_FSL_order_g_vs_current_assembly", current_fsl, baseline_fsl)
            report["baseline_passed_before_moving_substitution"] = True
            report["clock"]["restore_baseline_and_fixed_gate_seconds"] = time.monotonic() - restore_started

            official_moving = read_official("official_moving")
            report["calls"]["coordinates"] += 1
            coordinates = registration._fsl_displacement_coordinates(saved["state_field"], saved["coordinate_affine"],
                saved["moving_fsl2vox"], affine_grid=saved["affine_grid"])
            require(coordinates.dtype == torch.float32 and bool(torch.isfinite(coordinates).all()), "unchanged coordinate policy")
            operands["same_saved_field_coordinates"] = coordinates
            operands_before["same_saved_field_coordinates"] = value_hash(coordinates)
            report["calls"]["sampler"] += 1
            require(report["calls"]["sampler"] == 1, "single sampler cap")
            warped, valid, gradient_voxels = registration._trilinear_sample(official_moving, coordinates, derivatives=True)
            level_shape = tuple(geometry["level_shape"])
            require(warped.dtype == gradient_voxels.dtype == torch.float32 and valid.dtype == torch.bool
                    and tuple(warped.shape) == tuple(valid.shape) == level_shape
                    and tuple(gradient_voxels.shape) == (3,) + level_shape
                    and all(value.device.type == "cpu" and not value.requires_grad
                            for value in (warped, valid, gradient_voxels)), "sampler F32/bool CPU output contract")
            report["sampler_output_contract"] = {
                name: {"dtype": str(value.dtype), "shape": list(value.shape), "stride": list(value.stride()),
                       "device": str(value.device), "logical_value_sha256": value_hash(value)}
                for name, value in (("coordinates", coordinates), ("warped", warped),
                                   ("valid_mask", valid), ("gradient_voxels", gradient_voxels))}
            gate("single_replacement_valid_mask_vs_saved_mask", saved["state_mask"], valid)
            require(int(valid.sum()) == baseline["count"], "first mismatch: replacement count")
            report["gates"]["replacement_count"] = {"reference": baseline["count"], "actual": int(valid.sum()), "exact": True}
            report["calls"]["projection"] += 1
            projected = functions["project_derivatives"]({}, system, gradient_voxels)
            require(projected.dtype == torch.float32 and tuple(projected.shape) == (3,) + level_shape,
                    "unchanged F32 FNIT derivative projection")
            alternative, alternative_cost, alternative_scaled_fixed = state_from(warped, valid, projected)
            gate("replacement_scaled_fixed_unchanged", scaled_fixed, alternative_scaled_fixed)
            alternative_lm, alternative_fsl = gradients("official_saved_moving_only", alternative)
            report["arms"]["current_saved_moving"].update({"count": baseline["count"], "ssd": scalar(baseline["ssd"]), "cost": scalar(baseline_cost)})
            report["arms"]["official_saved_moving_only"].update({"count": alternative["count"], "ssd": scalar(alternative["ssd"]), "cost": scalar(alternative_cost)})
            report["same_FNIT_projection_state_changes"] = {"warped": metrics(saved["state_warped"], warped),
                "residual": metrics(saved["state_residual"], alternative["residual"]),
                "gradient_fsl": metrics(saved["state_gradient_fsl"], projected)}

            # Saved official total g enters only after all controlled gradients
            # are computed. It is an after-the-fact comparison, never an input.
            official_g = np.fromfile(paths["native_g"], dtype="<f8")
            require(official_g.size == expected["dimension"], "saved native g dimension")
            width = math.prod(geometry["coefficient_shape"])
            require(expected["dimension"] == 3 * width + 1, "mixed coefficient/scale block schema")
            def block_metrics(reference, actual):
                actual = array(actual)
                return {"full": metrics(reference, actual), "blocks": {
                    **{axis: metrics(reference[i * width:(i + 1) * width], actual[i * width:(i + 1) * width])
                       for i, axis in enumerate(("coefficient_x", "coefficient_y", "coefficient_z"))},
                    "global_scale": metrics(reference[-1:], actual[-1:])}}
            report["full_g_samepoint_comparison"] = {
                "current_LM_vs_saved_native": block_metrics(official_g, baseline_lm),
                "current_FSL_order_vs_saved_native": block_metrics(official_g, baseline_fsl),
                "moving_only_LM_vs_saved_native": block_metrics(official_g, alternative_lm),
                "moving_only_FSL_order_vs_saved_native": block_metrics(official_g, alternative_fsl),
                "moving_only_LM_vs_current_LM": block_metrics(array(baseline_lm), alternative_lm),
                "moving_only_FSL_order_vs_current_FSL_order": block_metrics(array(baseline_fsl), alternative_fsl)}
            report["full_g_units"] = "positive full g=2*FNIT half-gradient; three 392 coefficient blocks plus one dimensionless global scale, not voxel errors"
            require(all(count == expected["limits"][name] for name, count in report["calls"].items()),
                    "actual operation counts differ from frozen limits")
            require(len(report["images_read"]) == expected["limits"]["image_payloads_read"], "image payload read count")
            report["status"] = "scientific_control_finished_postchecks_pending"
    except BaseException as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        report["status"] = "failed_first_mismatch_no_retry"
        raise
    finally:
        for owner, name, original in reversed(originals):
            setattr(owner, name, original)
        report["flags_after"] = flags()
        report["flags_unchanged"] = report["flags_before"] == report["flags_after"]
        report["postcheck_errors"] = []
        def after_hashes(values):
            actual = {}
            for name, value in values.items():
                try:
                    actual[name] = value_hash(value)
                except BaseException as error:
                    report["postcheck_errors"].append({"operand": name, "type": type(error).__name__, "message": str(error)})
            return actual
        report["checkpoint_values_after"] = after_hashes(saved)
        report["checkpoint_operands_unchanged"] = saved_before == report["checkpoint_values_after"]
        if set(operands) != set(operands_before):
            report["postcheck_errors"].append({"message": "immutable operand before-hash missing",
                "operands": sorted(set(operands) - set(operands_before))})
        report["derived_operand_values_before"] = operands_before
        report["derived_operand_values_after"] = after_hashes(operands)
        report["derived_operands_unchanged"] = operands_before == report["derived_operand_values_after"]
        try:
            report["bindings_after"], after_failures = check_bindings(args.root, expected)
            report["harness_bindings_after"], harness_after_failures = check_freeze(args.workspace)
        except BaseException as error:
            report["postcheck_errors"].append({"type": type(error).__name__, "message": str(error)})
            report["bindings_after"], after_failures = {}, ["postcheck binding unavailable"]
            report["harness_bindings_after"], harness_after_failures = {}, ["postcheck freeze unavailable"]
        report["binding_failures_after"] = after_failures + ["harness/" + name for name in harness_after_failures]
        report["all_source_inputs_harness_unchanged"] = (before == report["bindings_after"]
            and harness == report["harness_bindings_after"] and not report["binding_failures_after"])
        report["fsl_dso_snapshot_after"] = mapped_fsl()
        report["clock"]["worker_until_summary_seconds"] = time.monotonic() - started
        report["process_maxrss_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        clean = (report["flags_unchanged"] and report["checkpoint_operands_unchanged"]
                 and report["derived_operands_unchanged"] and report["all_source_inputs_harness_unchanged"]
                 and not report["fsl_dso_snapshot_after"] and not report["postcheck_errors"])
        if report["status"] == "scientific_control_finished_postchecks_pending" and clean:
            report["status"] = "moving_input_control_completed_no_solver"
            report["control_completed"] = True
        write_json(summary_path, report)
        if "error" not in report:
            require(report["control_completed"], "postcondition failed; control not accepted")
    print(json.dumps({"status": report["status"], "control_completed": report["control_completed"], "calls": report["calls"]}))


if __name__ == "__main__":
    main()
