"""One current FNIRT linearization from saved preprocessing; no callback/solve."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import resource
import socket
import struct
import sys
import time

from stage1_io import bound, check_bindings, check_freeze, input_paths, write_json


class BridgeFailure(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--approved-stage1", action="store_true", required=True)
    return parser.parse_args()


def closure_values(function):
    return dict(zip(function.__code__.co_freevars, (cell.cell_contents for cell in function.__closure__ or ())))


def main():
    os.umask(0o077)
    args = parse_args()
    if sorted(os.sched_getaffinity(0)) != [32, 36, 40, 44, 48, 52, 56, 60]:
        raise RuntimeError("stage1 requires the declared eight physical CPU cores")
    if any(os.environ.get(name) != "8" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")):
        raise RuntimeError("stage1 requires the declared eight-thread environment")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or any(name in os.environ for name in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH", "OPENBLAS_CORETYPE")):
        raise RuntimeError("stage1 environment must be CPU-only and free of undeclared loader overrides")
    # The scientific process has a finite address-space cap before large imports.
    cap = 20_000_000_000
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    import_started = time.monotonic()
    expected = json.loads((args.workspace / "expected.public.json").read_text())
    bindings, failures = check_bindings(args.root, expected)
    frozen, frozen_failures = check_freeze(args.workspace)
    if failures or frozen_failures:
        raise RuntimeError("binding failed before numerical imports")
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    paths = input_paths(args.root, expected)
    sys.path.insert(0, str(args.root / "repo/src"))
    import nibabel as nib
    import numpy as np
    import torch
    from fnit.flirt.coordinates import flirt_to_world_affine, voxel_to_fsl_scaled_mm, world_to_flirt_affine
    from fnit.fnirt import GMFNIRTConfig, registration
    from fnit.fnirt.spline import BendingOperator, fsl_control_shape, spline_bases

    if Path(registration.__file__).resolve() != (args.root / "repo/src/fnit/fnirt/registration.py").resolve():
        raise RuntimeError("wrong imported FNIRT source")
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)

    def flags():
        return {"cuda_initialized": torch.cuda.is_initialized(),
                "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                "grad_enabled": torch.is_grad_enabled(),
                "default_dtype": str(torch.get_default_dtype())}

    before_flags = flags()
    if before_flags["cuda_initialized"]:
        raise RuntimeError("CUDA was already initialized")
    started = time.monotonic()
    report = {"scope": expected["scope"], "host": socket.gethostname(),
              "affinity": sorted(os.sched_getaffinity(0)), "torch_threads": torch.get_num_threads(),
              "interop_threads": torch.get_num_interop_threads(), "address_space_cap_bytes": cap,
              "flags_before": before_flags, "bindings_before": bindings,
              "harness_bindings": frozen, "evaluate_calls": 0, "linearize_calls": 0,
              "callback_calls": 0, "solver_calls": 0, "native_process_calls": 0,
              "full_H_materialized": False, "checkpoint_written": False,
              "accepted_checkpoint": False,
              "source_point": expected["scientific_point"], "stage1_passed": False,
              "clock": {"imports_and_binding_seconds": started - import_started},
              "limitations": ["Saved old preprocessing is a candidate input, not a restored current cache.",
                              "No matrix-free action or PCG is evaluated.",
                              "A saved checkpoint contains arrays and metadata, not original Python object identities.",
                              "A later callback would initialize its own packed layout and scratch."]}
    def fsl_mappings():
        # Identity snapshots only, not a complete dynamic-loader/syscall audit.
        return sorted({Path(line.split()[-1]).name for line in Path("/proc/self/maps").read_text().splitlines()
                       if len(line.split()) >= 6 and ("/apps/FSL/" in line.split()[-1]
                                                    or Path(line.split()[-1]).name.startswith("libfsl"))})
    report["fsl_dso_snapshot_before"] = fsl_mappings()
    if report["fsl_dso_snapshot_before"]:
        raise RuntimeError("installed FSL DSO was mapped before stage1")
    evaluation_times = {}
    original_evaluate = None
    system = None

    def tensor_metadata(value):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "stride": list(value.stride()),
                "device": str(value.device), "requires_grad": value.requires_grad}

    def require(condition, message):
        if not condition:
            raise BridgeFailure(message)

    def tensor_array(value):
        require(value.device.type == "cpu" and not value.requires_grad, "checkpoint tensor policy")
        return value.detach().numpy()

    def vector_gate(name, tensor):
        require(tensor.device.type == "cpu" and tensor.dtype == torch.float64 and not tensor.requires_grad,
                name + " policy")
        reference = np.fromfile(paths[name], dtype="<f8")
        actual = np.ascontiguousarray(tensor_array(tensor), dtype="<f8")
        require(actual.shape == reference.shape, name + " shape")
        require(bool(np.isfinite(actual).all()) and bool(np.isfinite(reference).all()), name + " nonfinite")
        delta = actual - reference
        reference_norm = float(np.linalg.norm(reference))
        actual_bytes = actual.tobytes()
        import hashlib
        row = {"tensor": tensor_metadata(tensor), "different_bits": int(np.count_nonzero(actual.view("<u8") != reference.view("<u8"))),
               "max_abs": float(np.max(np.abs(delta), initial=0)), "reference_l2": reference_norm,
               "relative_l2": float(np.linalg.norm(delta) / reference_norm) if reference_norm else None,
               "actual_value_sha256": hashlib.sha256(actual_bytes).hexdigest(),
               "reference_file_sha256": expected["files"][name]["sha256"],
               "bitexact": actual_bytes == reference.tobytes()}
        report.setdefault("vector_gates", {})[name] = row
        require(row["bitexact"], name + " differs from saved current bytes")

    try:
        setup_started = time.monotonic()
        # GM/template are loaded for original shapes/affines/pixdim only. No
        # rescaling, Gaussian smoothing, FAST, FLIRT or registration is called.
        gm, fixed_image, mask_image = (nib.load(paths[key]) for key in ("gm", "template", "mask"))
        moving_shape, fixed_shape = tuple(gm.shape[:3]), tuple(fixed_image.shape[:3])
        require(len(gm.shape) == len(fixed_image.shape) == 3, "one 3D input frame required")
        moving_sizes = tuple(float(x) for x in gm.header.get_zooms()[:3])
        fixed_sizes = tuple(float(x) for x in fixed_image.header.get_zooms()[:3])
        config = GMFNIRTConfig()
        require(not config.implicit_input_mask and not config.implicit_reference_mask, "GM implicit-mask policy")
        require(config.intensity_model == "global_linear" and config.estimate_intensity[0], "GM scale policy")
        require(not config.apply_reference_mask[0], "first-level explicit mask unexpectedly enabled")
        require(mask_image.shape[:3] == fixed_shape and np.allclose(mask_image.affine, fixed_image.affine, atol=1e-5, rtol=0),
                "bound mask geometry")
        moving_level = torch.from_numpy(np.load(paths["moving_level"], allow_pickle=False).copy())
        fixed_level = torch.from_numpy(np.load(paths["fixed_level"], allow_pickle=False).copy())
        require(moving_level.dtype == fixed_level.dtype == torch.float32, "saved images must remain F32")
        require(tuple(moving_level.shape) == moving_shape, "saved moving geometry")
        require(bool(torch.isfinite(moving_level).all()) and bool(torch.isfinite(fixed_level).all()), "saved image nonfinite")
        full_positions = registration._level_positions(fixed_shape, config.subsampling[0], device="cpu", dtype=torch.float32)
        level_shape = tuple(axis.numel() for axis in full_positions)
        require(tuple(fixed_level.shape) == level_shape, "saved fixed level geometry")
        stride = config.subsampling[0]
        level_sizes = tuple(size * stride for size in fixed_sizes)
        stages = config.process_stages or (1,) * len(config.subsampling)
        resolutions = config.warp_resolution_schedule_mm or (config.warp_resolution_mm,) * len(config.subsampling)
        knot_spacing = registration._process_knot_spacing_schedule(resolutions, stages, config.subsampling, fixed_sizes)[0]
        control_shape = fsl_control_shape(level_shape, knot_spacing)
        require(list(control_shape) == expected["coefficient_shape"], "saved coefficient shape")
        level_positions = tuple(torch.arange(size, dtype=torch.float64) for size in level_shape)
        bases = spline_bases(level_shape, knot_spacing, level_sizes, device="cpu", dtype=torch.float64, positions=level_positions)
        bending = BendingOperator(level_shape, knot_spacing, level_sizes, device="cpu", dtype=torch.float64, execution="optimized")
        initial = flirt_to_world_affine(np.loadtxt(paths["affine"]), gm.affine, fixed_image.affine,
                                       moving_shape, fixed_shape, moving_sizes, fixed_sizes)
        moving_fsl = voxel_to_fsl_scaled_mm(gm.affine, moving_shape, moving_sizes)
        fixed_fsl = voxel_to_fsl_scaled_mm(fixed_image.affine, fixed_shape, fixed_sizes)
        forward = world_to_flirt_affine(initial, gm.affine, fixed_image.affine, moving_shape, fixed_shape, moving_sizes, fixed_sizes)
        fixed_exact = torch.as_tensor(fixed_fsl, dtype=torch.float64)
        moving_fsl2vox = torch.linalg.inv(torch.as_tensor(moving_fsl, dtype=torch.float64)).float()
        pull_exact = torch.linalg.inv(torch.as_tensor(forward, dtype=torch.float64))
        affine_pull = pull_exact.float()
        target_fsl = registration._coordinate_grid(fixed_exact.float(), full_positions)
        level_to_full = torch.diag(torch.tensor([stride, stride, stride, 1.0], dtype=torch.float64))
        coordinate_affine = pull_exact @ fixed_exact @ level_to_full
        parameters = np.fromfile(paths["parameters"], dtype="<f8")
        require(parameters.size == expected["dimension"], "point dimension")
        require(bool(np.isfinite(parameters).all()), "point nonfinite")
        coefficients, scale = registration._unpack(torch.from_numpy(parameters.copy()), control_shape, True)
        system = registration._LevelSystem(moving_level, fixed_level, None, None, moving_fsl2vox,
                                           target_fsl, affine_pull, bases, bending, config.regularization[0],
                                           config.ssd_weighted_lambda, True, coordinate_affine)
        report["config"] = asdict(config)
        report["geometry"] = {"moving_shape": moving_shape, "fixed_shape": fixed_shape,
                              "level_shape": level_shape, "moving_voxel_sizes": moving_sizes,
                              "fixed_voxel_sizes": fixed_sizes, "level_voxel_sizes": level_sizes,
                              "knot_spacing": knot_spacing, "coefficient_shape": control_shape,
                              "reference_mask_active": False, "moving_mask_active": False,
                              "declared_mask_resource_used_for_geometry_only_at_level1": True}
        report["input_tensor_metadata"] = {"moving": tensor_metadata(moving_level), "fixed": tensor_metadata(fixed_level),
                                           "coefficients": tensor_metadata(coefficients), "scale": tensor_metadata(scale)}
        report["clock"]["input_restore_geometry_seconds"] = time.monotonic() - setup_started
        original_evaluate = system.evaluate

        def fixed_evaluate(*values, **kwargs):
            report["evaluate_calls"] += 1
            require(report["evaluate_calls"] == 1, "more than one evaluation requested")
            tick = time.monotonic()
            state = original_evaluate(*values, **kwargs)
            evaluation_times["evaluate_seconds"] = time.monotonic() - tick
            fields = ("count", "ssd", "cost", "effective_lambda", "bending_energy")
            report["state_before_lambda_override"] = {name: float(state[name]) if torch.is_tensor(state[name]) else state[name]
                                                       for name in fields}
            state["effective_lambda"] = expected["fixed_effective_lambda"]
            state["cost"] = state["ssd"] + state["effective_lambda"] * state["bending_energy"] / state["count"]
            return state

        system.evaluate = fixed_evaluate
        linearize_started = time.monotonic()
        report["linearize_calls"] += 1
        state, gradient, matvec, diagonal = system.linearize(coefficients, scale)
        report["clock"]["linearize_including_evaluate_seconds"] = time.monotonic() - linearize_started
        report["clock"].update(evaluation_times)
        require(report["evaluate_calls"] == 1, "one internal evaluate required")
        for name in ("count", "ssd", "bending_energy", "effective_lambda", "cost"):
            actual = report["state_before_lambda_override"][name]
            reference = expected["expected_state_before_lambda_override"][name]
            exact = actual == reference if name == "count" else struct.pack("<d", actual) == struct.pack("<d", reference)
            report.setdefault("scalar_gates", {})[name] = {"actual": actual, "reference": reference, "bitexact": exact}
            require(exact, name + " differs from saved current scalar")
        gate_started = time.monotonic()
        # Same full-g/full-diagonal units used by the existing assembly report.
        vector_gate("gradient", 2 * gradient)
        vector_gate("diagonal", 2 * diagonal)
        report["clock"]["scalar_vector_bridge_seconds"] = time.monotonic() - gate_started
        report["stage1_passed"] = True

        # Only after the exact bridge, preserve actual state/weights. No
        # matvec invocation: the CPU operator's packed layout/scratch is empty.
        checkpoint_started = time.monotonic()
        cells = closure_values(matvec)
        normal_cells = closure_values(cells["data_normal"])
        cpu_normal = normal_cells["cpu_normal"]
        require(cpu_normal is not None, "optimized CPU normal was not constructed")
        require(cpu_normal.scratch is None and cpu_normal._layout is None, "unexpected pre-evaluated callback cache")
        arrays = {"coefficients": tensor_array(coefficients), "scale": tensor_array(scale),
                  "fixed": tensor_array(fixed_level), "moving_fsl2vox": tensor_array(moving_fsl2vox),
                  "target_fsl": tensor_array(target_fsl), "affine_pull": tensor_array(affine_pull),
                  "coordinate_affine": tensor_array(coordinate_affine), "affine_grid": tensor_array(system.affine_grid),
                  "gradient_half": tensor_array(gradient), "diagonal_half": tensor_array(diagonal),
                  "bending_diagonal": tensor_array(bending._diagonal),
                  "scale_weight": tensor_array(normal_cells["scale_weight"])}
        for name in ("field", "warped", "mask", "residual", "gradient_fsl"):
            arrays["state_" + name] = tensor_array(state[name])
        for axis, basis in enumerate(bases):
            arrays["basis_" + str(axis)] = tensor_array(basis)
        for row, values in enumerate(normal_cells["spatial_weights"]):
            for col, value in enumerate(values):
                arrays[f"spatial_weight_{row}_{col}"] = tensor_array(value)
        for axis, value in enumerate(normal_cells["cross_weights"]):
            arrays["cross_weight_" + str(axis)] = tensor_array(value)
        for term, (derivative_bases, multiplier) in enumerate(bending.operators):
            for axis, value in enumerate(derivative_bases):
                arrays[f"bending_basis_{term}_{axis}"] = tensor_array(value)
        for term, (grams, multiplier) in enumerate(bending._normal_grams):
            for axis, value in enumerate(grams):
                arrays[f"bending_gram_{term}_{axis}"] = tensor_array(value)
        checkpoint = args.output / "level_state.private.npz"
        np.savez(checkpoint, **arrays)
        report["checkpoint"] = {**bound(checkpoint), "array_count": len(arrays),
                                "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype), "strides": list(value.strides)}
                                           for name, value in arrays.items()},
                                "moving_array_reference": expected["files"]["moving_level"],
                                "state_scalars": {name: float(state[name]) if torch.is_tensor(state[name]) else state[name]
                                                  for name in ("count", "ssd", "cost", "effective_lambda", "bending_energy")},
                                "bending_multipliers": [float(multiplier) for _, multiplier in bending.operators],
                                "normal_cache": {"packed_layout": None, "scratch": None, "layout_copy_bytes": cpu_normal.layout_copy_bytes},
                                "closure_identity_preserved": False,
                                "meaning": "Newly reconstructed state and unpacked weights after an exact stage1 bridge; no original Python cache objects."}
        report["checkpoint_written"] = True
        report["clock"]["private_checkpoint_write_seconds"] = time.monotonic() - checkpoint_started
        report["status"] = "stage1_passed_no_callback"
    except BridgeFailure as error:
        report["status"] = "stage1_bridge_failed_stopped"
        report["error"] = str(error)
        raise
    except Exception as error:
        report["status"] = "stage1_setup_or_capture_failed"
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
        raise
    finally:
        if original_evaluate is not None:
            system.evaluate = original_evaluate
        report["flags_after"] = flags()
        report["flags_unchanged"] = report["flags_before"] == report["flags_after"]
        report["clock"]["numerical_and_private_checkpoint_before_post_bind_seconds"] = time.monotonic() - started
        report["process_maxrss_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        after, changed = check_bindings(args.root, expected)
        report["bindings_after"] = after
        report["all_source_and_inputs_unchanged"] = not changed and after == bindings
        report["binding_failures_after"] = changed
        report["fsl_dso_snapshot_after"] = fsl_mappings()
        report["accepted_checkpoint"] = bool(report["stage1_passed"] and report["checkpoint_written"]
                                             and report["flags_unchanged"] and report["all_source_and_inputs_unchanged"]
                                             and not report["fsl_dso_snapshot_after"])
        if report["stage1_passed"] and not report["accepted_checkpoint"]:
            report["status"] = "stage1_checkpoint_or_postcondition_failed"
        report["clock"]["worker_through_post_bind_before_public_summary_seconds"] = time.monotonic() - import_started
        write_json(args.output / "summary.public.json", report)
    require(report["accepted_checkpoint"], "checkpoint or postcondition was not accepted")
    print(json.dumps({key: report[key] for key in ("status", "evaluate_calls", "linearize_calls", "stage1_passed", "checkpoint_written", "accepted_checkpoint")}))


if __name__ == "__main__":
    main()
