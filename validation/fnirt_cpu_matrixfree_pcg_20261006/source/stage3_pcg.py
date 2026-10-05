"""Prepared bounded PCG comparison on accepted restored mature CPU callbacks.

No evaluation, linearization, new assembly, native process or MRI processing.
The two numerical solves require separately reviewed explicit approval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import socket
import sys
import time

from stage3_io import bound, check_bindings, check_freeze, input_paths, write_json
from restore_callback import (restored_arm, restore_tensors, source_ast_binding,
                              tensor_value_hash)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "workspace", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--approved-stage3", action="store_true", required=True)
    return parser.parse_args()


def main():
    os.umask(0o077)
    args = parse_args()
    if sorted(os.sched_getaffinity(0)) != [32, 36, 40, 44, 48, 52, 56, 60]:
        raise RuntimeError("declared physical CPU affinity required")
    if any(os.environ.get(name) != "8" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")):
        raise RuntimeError("declared eight-thread environment required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or any(name in os.environ for name in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH", "OPENBLAS_CORETYPE")):
        raise RuntimeError("CPU-only environment without loader overrides required")
    if platform.machine() != "x86_64" or "fma" not in Path("/proc/cpuinfo").read_text().split():
        raise RuntimeError("this bounded owned-FMA comparison requires x86_64 FMA")
    resource.setrlimit(resource.RLIMIT_AS, (20_000_000_000, 20_000_000_000))
    started = time.monotonic()
    expected = json.loads((args.workspace / "expected.public.json").read_text())
    bindings_before, failures = check_bindings(args.root, expected)
    frozen, freeze_failures = check_freeze(args.workspace)
    if failures or freeze_failures:
        raise RuntimeError("pre-import source/input/freeze binding failed")
    for name, identity in expected["candidate_source_provenance"].items():
        if bound(args.workspace / name) != identity:
            raise RuntimeError("published CPU candidate copy changed: " + name)
    paths = input_paths(args.root, expected)
    stage1 = json.loads(paths["stage1_summary"].read_text())
    stage2 = json.loads(paths["stage2_summary"].read_text())
    if not stage1["accepted_checkpoint"] or not stage1["stage1_passed"]:
        raise RuntimeError("stage1 did not accept the bound checkpoint")
    if (stage1["evaluate_calls"], stage1["linearize_calls"], stage1["callback_calls"]) != (1, 1, 0):
        raise RuntimeError("unexpected prior stage1 scope")
    if not stage2["valid_bounded_diagnostic"] or stage2["callback_calls"] != {"optimized": 7, "reference": 7} or stage2["strict_csc_calls"] != 7:
        raise RuntimeError("bound stage2 callback diagnostic did not pass")
    if bound(paths["checkpoint"]) != {key: stage1["checkpoint"][key] for key in ("bytes", "sha256")}:
        raise RuntimeError("checkpoint differs from accepted producer")
    ast_binding = source_ast_binding(args.root / "repo/src/fnit/fnirt/registration.py",
                                     args.root / "repo/validation/fnirt_cpu_current_replay_20261006/source/replay_current.py")
    if ast_binding != expected["callback_ast"]:
        raise RuntimeError("mature callback AST binding changed")
    if expected["lm_tau"]["value"] != 0.001 or expected["pcg"]["tolerance"] != 0.001 or expected["pcg"]["max_iterations"] != 500:
        raise RuntimeError("declared comparison policy changed")
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    sys.path.insert(0, str(args.root / "repo/src"))
    import numpy as np
    import torch
    from fnit.fnirt import registration, optimizer
    from fnit.fnirt.spline import BendingOperator
    import pcg_cpu_candidate as candidate
    if Path(registration.__file__).resolve() != (args.root / "repo/src/fnit/fnirt/registration.py").resolve():
        raise RuntimeError("wrong current registration module")
    if Path(optimizer.__file__).resolve() != (args.root / "repo/src/fnit/fnirt/optimizer.py").resolve():
        raise RuntimeError("wrong current optimizer module")
    if Path(candidate.__file__).resolve() != (args.workspace / "pcg_cpu_candidate.py").resolve():
        raise RuntimeError("wrong private candidate module")
    if Path(candidate.arithmetic.__file__).resolve() != (args.workspace / "own_reductions.py").resolve():
        raise RuntimeError("wrong owned arithmetic module")
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)

    def flags():
        return {"cuda_initialized": torch.cuda.is_initialized(), "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                "matmul_tf32": torch.backends.cuda.matmul.allow_tf32, "grad_enabled": torch.is_grad_enabled(),
                "default_dtype": str(torch.get_default_dtype())}

    def fsl_mappings():
        # File identity snapshots only; not an all-time syscall audit.
        return sorted({Path(line.split()[-1]).name for line in Path("/proc/self/maps").read_text().splitlines()
                       if len(line.split()) >= 6 and ("/apps/FSL/" in line.split()[-1]
                                                    or Path(line.split()[-1]).name.startswith("libfsl"))})

    report = {"scope": expected["scope"], "status": "started_not_accepted", "host": socket.gethostname(),
              "affinity": sorted(os.sched_getaffinity(0)), "torch_threads": torch.get_num_threads(),
              "interop_threads": torch.get_num_interop_threads(), "address_space_cap_bytes": 20_000_000_000,
              "flags_before": flags(), "bindings_before": bindings_before, "harness_bindings": frozen,
              "callback_ast": ast_binding, "stage1_checkpoint_identity": bound(paths["checkpoint"]),
              "stage2_summary_identity": bound(paths["stage2_summary"]), "lm_tau": expected["lm_tau"],
              "evaluate_calls": 0, "linearize_calls": 0, "gradient_calls": 0,
              "bending_constructor_calls": 0, "unexpected_solver_calls": 0, "native_process_calls": 0,
              "solver_calls": {"current_torch": 0, "owned_cpu": 0},
              "callback_calls": {"current_torch": 0, "owned_cpu": 0},
              "strict_csc_calls": 0, "full_H_materialized": False,
              "existing_H_loaded_only_for_unit_restore_gate": False,
              "runtime_integration_accepted": False, "numerical_improvement_established": False,
              "probe_completed": False, "arms": {},
              "limitations": ["Tau .001 is a declared comparison value, not established historical solve3 LM damping.",
                              "Restored values/layouts are new objects; the original Python alias graph is not persisted.",
                              "The selected unit column gates do not prove arbitrary-direction equivalence to materialized H.",
                              "Current Torch PCG and the division/owned-reduction candidate differ as complete algorithms.",
                              "There is no native same-system oracle, full nonlinear registration, speed claim or default integration.",
                              "The owned reduction candidate was previously validated only within its documented CPU/size scope."],
              "clock": {"imports_and_binding_seconds": time.monotonic() - started},
              "fsl_dso_snapshot_before": fsl_mappings()}
    if report["flags_before"]["cuda_initialized"] or report["fsl_dso_snapshot_before"]:
        raise RuntimeError("unexpected CUDA/FSL runtime state")

    def require(condition, message):
        if not condition:
            raise RuntimeError(message)

    def policy(value, n, label):
        require(isinstance(value, torch.Tensor) and value.device.type == "cpu" and value.dtype == torch.float64
                and not value.requires_grad and value.ndim == 1 and value.numel() == n,
                "CPU FP64 no-grad vector policy: " + label)
        require(bool(torch.isfinite(value).all()), "nonfinite: " + label)

    def hash_values(tensors):
        return {name: tensor_value_hash(value, np) for name, value in tensors.items()}

    def reject(counter):
        def forbidden(*values, **kwargs):
            report[counter] += 1
            raise RuntimeError("out-of-scope call rejected: " + counter)
        return forbidden

    original_pcg = optimizer.preconditioned_conjugate_gradient
    originals = [(registration._LevelSystem, "evaluate", "evaluate_calls"),
                 (registration._LevelSystem, "linearize", "linearize_calls"),
                 (registration._LevelSystem, "gradient", "gradient_calls"),
                 (BendingOperator, "__init__", "bending_constructor_calls"),
                 (optimizer, "preconditioned_conjugate_gradient", "unexpected_solver_calls"),
                 (registration, "preconditioned_conjugate_gradient", "unexpected_solver_calls")]
    saved_methods = [(owner, name, getattr(owner, name)) for owner, name, _ in originals]
    for owner, name, counter in originals:
        setattr(owner, name, reject(counter))
    tensors_by_arm, value_before, systems, system_before, callbacks, normals = {}, {}, {}, {}, {}, {}
    solutions = {}
    try:
        restore_started = time.monotonic()
        full_gradient = np.fromfile(paths["gradient"], dtype="<f8")
        full_diagonal = np.fromfile(paths["diagonal"], dtype="<f8")
        n = full_gradient.size
        require(n > 0 and full_diagonal.shape == (n,) and bool(np.isfinite(full_gradient).all())
                and bool(np.isfinite(full_diagonal).all()), "saved g/diag invalid")
        raw = np.fromfile(paths["H"], dtype="<f8")
        require(raw.size == n * n and bool(np.isfinite(raw).all()), "bound existing H invalid")
        matrix = raw.reshape(n, n)
        report["existing_H_loaded_only_for_unit_restore_gate"] = True
        tau, tolerance, maximum = expected["lm_tau"]["value"], expected["pcg"]["tolerance"], expected["pcg"]["max_iterations"]
        report["dimension_from_checkpoint_gradient"] = n
        report["restored_layouts"], report["unit_restore_gates"] = {}, {}
        with torch.no_grad():
            for kind in expected["pcg"]["arms"]:
                tensors, layout = restore_tensors(paths["checkpoint"], stage1["checkpoint"]["arrays"], np, torch)
                tensors_by_arm[kind] = tensors
                value_before[kind] = hash_values(tensors)
                report["restored_layouts"][kind] = layout
                gradient, diagonal = tensors["gradient_half"], tensors["diagonal_half"]
                for label, half, target in (("gradient", gradient, full_gradient), ("diagonal", diagonal, full_diagonal)):
                    policy(half, n, "half " + label)
                    require(np.ascontiguousarray((2 * half).numpy(), dtype="<f8").tobytes() == target.tobytes(),
                            "restored full " + label + " bridge failed")
                floor = torch.finfo(diagonal.dtype).eps * diagonal.abs().mean().clamp_min(1)
                damping_diagonal = diagonal.clamp_min(floor)
                rhs = -gradient
                preconditioner = (1 + tau) * damping_diagonal
                systems[kind] = {"rhs": rhs, "damping_diagonal": damping_diagonal, "preconditioner": preconditioner}
                for label, value in systems[kind].items():
                    policy(value, n, label)
                require(bool((damping_diagonal > 0).all()) and bool((preconditioner > 0).all()), "nonpositive LM diagonal")
                system_before[kind] = hash_values(systems[kind])
                callback, normal, _ = restored_arm("optimized", tensors, stage1, registration)
                callbacks[kind], normals[kind] = callback, normal
                report["arms"][kind] = {"floor": float(floor), "input_hashes": system_before[kind],
                                        "rhs_is_negative_gradient_half": True,
                                        "callback_precision": "CPU FP64, no-grad, mature optimized half action",
                                        "callback_ledger": []}
            pointers = [{value.untyped_storage().data_ptr() for value in tensors.values() if value.numel()}
                        for tensors in tensors_by_arm.values()]
            require(not pointers[0].intersection(pointers[1]), "two arms share checkpoint storage")
            require(system_before["current_torch"] == system_before["owned_cpu"], "two controlled systems differ")
            report["two_arm_storage_disjoint"] = True
            report["restored_full_g_diag_bitexact"] = True
            report["two_arm_system_values_bitexact"] = True
            report["clock"]["checkpoint_restore_and_factory_seconds"] = time.monotonic() - restore_started

            def action(kind, direction, phase, damped=True):
                policy(direction, n, "callback direction")
                require(report["callback_calls"][kind] < expected["pcg"]["callback_max_per_arm"], "callback cap reached")
                tick = time.monotonic()
                direction_hash = tensor_value_hash(direction, np)
                report["callback_calls"][kind] += 1
                value = callbacks[kind](direction)
                if damped:
                    value = value + tau * systems[kind]["damping_diagonal"] * direction
                policy(value, n, "callback output")
                require(tensor_value_hash(direction, np) == direction_hash, "callback changed direction")
                report["arms"][kind]["callback_ledger"].append({"call": report["callback_calls"][kind], "phase": phase,
                    "damped": damped, "direction_sha256": direction_hash,
                    "action_sha256": tensor_value_hash(value, np), "seconds": time.monotonic() - tick})
                return value

            # Both gates precede either solve. The column index is generic zero;
            # the vector size comes from the accepted checkpoint, not a constant.
            for kind in expected["pcg"]["arms"]:
                unit = torch.zeros_like(systems[kind]["rhs"])
                unit[0] = 1
                full = np.ascontiguousarray((2 * action(kind, unit, "unit_restore_gate", damped=False)).numpy(), dtype="<f8")
                column = np.ascontiguousarray(matrix[:, 0], dtype="<f8")
                equal = full.tobytes() == column.tobytes()
                report["unit_restore_gates"][kind] = {"column": 0, "different_bits": int(np.count_nonzero(full.view("<u8") != column.view("<u8"))),
                                                       "bitexact": equal}
                require(equal, "first unit restore mismatch; no solver allowed")

            for kind in expected["pcg"]["arms"]:
                def damped(direction):
                    return action(kind, direction, "pcg_iteration")
                tick = time.monotonic()
                require(report["solver_calls"][kind] == 0, "solver already called")
                report["solver_calls"][kind] += 1
                callbacks_before_solver = report["callback_calls"][kind]
                solver = original_pcg if kind == "current_torch" else candidate.preconditioned_conjugate_gradient_cpu
                kwargs = {"diagonal": systems[kind]["preconditioner"], "tolerance": tolerance, "max_iterations": maximum}
                if kind == "current_torch":
                    kwargs["execution"] = "optimized"
                solution, result = solver(damped, systems[kind]["rhs"], **kwargs)
                elapsed = time.monotonic() - tick
                policy(solution, n, "solution")
                require(0 <= result.iterations <= maximum and isinstance(result.converged, bool)
                        and np.isfinite(result.relative_residual), "invalid PCG report")
                solver_callbacks = report["callback_calls"][kind] - callbacks_before_solver
                # A denominator rejection uses one callback without accepting
                # that iteration. Both unchanged solver bodies return the
                # previous accepted iteration count in that case.
                valid_counts = (solver_callbacks == result.iterations or
                                (not result.converged and result.iterations < maximum
                                 and solver_callbacks == result.iterations + 1))
                require(valid_counts and solver_callbacks <= maximum, "unexpected PCG callback/iteration relation")
                recursive = {"iterations": result.iterations, "converged": result.converged,
                             "relative_residual": result.relative_residual,
                             "solver_callback_count": solver_callbacks,
                             "possible_denominator_stop_from_bound_source": solver_callbacks == result.iterations + 1}
                # Same callback and common Torch norm check actual residual;
                # the solver's recursive norm implementation stays unchanged.
                residual = systems[kind]["rhs"] - action(kind, solution, "true_residual")
                rhs_norm = torch.linalg.vector_norm(systems[kind]["rhs"])
                true_norm = torch.linalg.vector_norm(residual)
                require(float(rhs_norm) > 0, "unexpected zero real RHS")
                true_relative = float(true_norm / rhs_norm)
                require(np.isfinite(true_relative), "nonfinite true residual")
                private_path = args.output / (kind + "_step.private.f64")
                np.ascontiguousarray(solution.numpy(), dtype="<f8").tofile(private_path)
                solutions[kind] = solution
                report["arms"][kind].update({"PCG": recursive, "solver_seconds_order_observation_only": elapsed,
                    "true_residual": {"absolute_l2": float(true_norm), "relative_l2": true_relative,
                                      "rhs_l2": float(rhs_norm), "common_norm": "torch.linalg.vector_norm",
                                      "relative_le_declared_tolerance": true_relative <= tolerance,
                                      "residual_value_sha256": tensor_value_hash(residual, np)},
                    "private_solution": {**bound(private_path), "mode": oct(private_path.stat().st_mode & 0o777),
                                         "dtype": "little-endian FP64", "shape": [n]},
                    "coordinate_units": "mixed displacement-coefficient/intensity step coordinates"})
            old, new = solutions["current_torch"], solutions["owned_cpu"]
            delta = new - old
            old_array, new_array = np.ascontiguousarray(old.numpy()), np.ascontiguousarray(new.numpy())
            old_norm = float(torch.linalg.vector_norm(old))
            report["two_step_comparison"] = {"different_bits": int(np.count_nonzero(old_array.view("<u8") != new_array.view("<u8"))),
                "max_abs": float(delta.abs().max()), "relative_l2_to_current_torch": float(torch.linalg.vector_norm(delta)) / old_norm if old_norm else None,
                "reference_is_current_torch_not_native": True, "no_numerical_improvement_claim": True}
            report["normal_cache_after"] = {kind: {"packed_layout": list(normals[kind]._layout),
                "layout_copy_bytes": normals[kind].layout_copy_bytes, "scratch_shape": list(normals[kind].scratch.shape),
                "scratch_stride": list(normals[kind].scratch.stride())} for kind in expected["pcg"]["arms"]}
        report["probe_completed"] = True
        report["status"] = "bounded_actual_matrixfree_two_PCG_completed"
    except Exception as error:
        report["status"] = "matrixfree_PCG_failed_stopped_no_retry"
        report["error_type"], report["error"] = type(error).__name__, str(error)
        raise
    finally:
        for owner, name, method in saved_methods:
            setattr(owner, name, method)
        report["checkpoint_values_before"] = value_before
        report["system_values_before"] = system_before
        report["checkpoint_values_after"], report["system_values_after"] = {}, {}
        report["postcondition_bookkeeping_failures"] = {}
        for label, source, target in (("checkpoint", tensors_by_arm, report["checkpoint_values_after"]),
                                      ("system", systems, report["system_values_after"])):
            for kind, tensors in source.items():
                try:
                    target[kind] = hash_values(tensors)
                except Exception as error:
                    report["postcondition_bookkeeping_failures"][label + "/" + kind] = {"error_type": type(error).__name__, "error": str(error)}
        report["checkpoint_operand_values_unchanged"] = report["checkpoint_values_after"] == value_before
        report["system_operand_values_unchanged"] = report["system_values_after"] == system_before
        try:
            report["flags_after"] = flags()
            report["flags_unchanged"] = report["flags_before"] == report["flags_after"]
        except Exception as error:
            report["flags_unchanged"] = False
            report["postcondition_bookkeeping_failures"]["flags"] = {"error_type": type(error).__name__, "error": str(error)}
        try:
            after, changed = check_bindings(args.root, expected)
            report["bindings_after"], report["binding_failures_after"] = after, changed
            report["all_source_and_inputs_unchanged"] = not changed and after == bindings_before
        except Exception as error:
            report["all_source_and_inputs_unchanged"] = False
            report["postcondition_bookkeeping_failures"]["bindings"] = {"error_type": type(error).__name__, "error": str(error)}
        try:
            report["fsl_dso_snapshot_after"] = fsl_mappings()
        except Exception as error:
            report["fsl_dso_snapshot_after"] = None
            report["postcondition_bookkeeping_failures"]["mapped_DSO"] = {"error_type": type(error).__name__, "error": str(error)}
        report["process_maxrss_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["clock"]["worker_through_post_bind_before_summary_seconds"] = time.monotonic() - started
        report["valid_bounded_diagnostic"] = bool(report["probe_completed"] and report["flags_unchanged"]
            and report["all_source_and_inputs_unchanged"] and report["checkpoint_operand_values_unchanged"]
            and report["system_operand_values_unchanged"] and not report["postcondition_bookkeeping_failures"]
            and not report["fsl_dso_snapshot_after"] and report["solver_calls"] == {"current_torch": 1, "owned_cpu": 1}
            and sum(report["callback_calls"].values()) <= expected["pcg"]["callback_max_total"]
            and all(report[name] == 0 for name in ("evaluate_calls", "linearize_calls", "gradient_calls",
                    "bending_constructor_calls", "unexpected_solver_calls", "native_process_calls", "strict_csc_calls")))
        write_json(args.output / "summary.public.json", report)
    require(report["valid_bounded_diagnostic"], "matrix-free PCG diagnostic postcondition failed")
    print(json.dumps({name: report[name] for name in ("status", "solver_calls", "callback_calls", "valid_bounded_diagnostic")}))


if __name__ == "__main__":
    main()
