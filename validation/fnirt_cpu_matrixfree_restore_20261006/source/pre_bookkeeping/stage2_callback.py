"""Prepared bounded callback probe; no evaluation, linearization or solver."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import sys
import time

from stage2_io import bound, check_bindings, check_freeze, input_paths, write_json
from restore_callback import (compile_csc, restored_arm, restore_tensors,
                              source_ast_binding, tensor_value_hash)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "workspace", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--approved-stage2", action="store_true", required=True)
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
    resource.setrlimit(resource.RLIMIT_AS, (20_000_000_000, 20_000_000_000))
    started = time.monotonic()
    expected = json.loads((args.workspace / "expected.public.json").read_text())
    bindings_before, failures = check_bindings(args.root, expected)
    frozen, freeze_failures = check_freeze(args.workspace)
    if failures or freeze_failures:
        raise RuntimeError("pre-import source/input/freeze binding failed")
    paths = input_paths(args.root, expected)
    stage1 = json.loads(paths["stage1_summary"].read_text())
    if not stage1["accepted_checkpoint"] or not stage1["stage1_passed"]:
        raise RuntimeError("stage1 did not accept the bound checkpoint")
    if stage1["evaluate_calls"] != 1 or stage1["linearize_calls"] != 1 or stage1["callback_calls"] != 0:
        raise RuntimeError("unexpected prior stage1 scope")
    if bound(paths["checkpoint"]) != {key: stage1["checkpoint"][key] for key in ("bytes", "sha256")}:
        raise RuntimeError("checkpoint differs from the accepted producer")
    ast_binding = source_ast_binding(args.root / "repo/src/fnit/fnirt/registration.py",
                                     args.root / "repo/validation/fnirt_cpu_current_replay_20261006/source/replay_current.py")
    if ast_binding != expected["callback_ast"]:
        raise RuntimeError("callback/CSC AST binding changed")
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    sys.path.insert(0, str(args.root / "repo/src"))
    import numpy as np
    import torch
    from numba import njit
    from scipy.sparse import csc_matrix
    from fnit.fnirt import registration, optimizer
    from fnit.fnirt.spline import BendingOperator
    if Path(registration.__file__).resolve() != (args.root / "repo/src/fnit/fnirt/registration.py").resolve():
        raise RuntimeError("wrong current registration module")
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)

    def flags():
        return {"cuda_initialized": torch.cuda.is_initialized(), "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                "matmul_tf32": torch.backends.cuda.matmul.allow_tf32, "grad_enabled": torch.is_grad_enabled(),
                "default_dtype": str(torch.get_default_dtype())}

    def fsl_mappings():
        # Before/after identity snapshots, not an all-time syscall audit.
        return sorted({Path(line.split()[-1]).name for line in Path("/proc/self/maps").read_text().splitlines()
                       if len(line.split()) >= 6 and ("/apps/FSL/" in line.split()[-1]
                                                    or Path(line.split()[-1]).name.startswith("libfsl"))})

    report = {"scope": expected["scope"], "status": "started_not_accepted", "host": socket.gethostname(),
              "affinity": sorted(os.sched_getaffinity(0)), "torch_threads": torch.get_num_threads(),
              "interop_threads": torch.get_num_interop_threads(), "address_space_cap_bytes": 20_000_000_000,
              "flags_before": flags(), "bindings_before": bindings_before, "harness_bindings": frozen,
              "callback_ast": ast_binding, "stage1_checkpoint_identity": bound(paths["checkpoint"]),
              "evaluate_calls": 0, "linearize_calls": 0, "gradient_calls": 0,
              "bending_constructor_calls": 0, "solver_calls": 0, "native_process_calls": 0,
              "full_H_materialized": False, "existing_H_loaded_for_third_comparison": False,
              "callback_calls": {"optimized": 0, "reference": 0}, "strict_csc_calls": 0,
              "runtime_integration_accepted": False, "probe_completed": False,
              "limitations": ["Restored values/layouts are new objects; the original Python alias graph is not persisted.",
                              "Three original unit columns do not prove arbitrary-direction floating-point equality.",
                              "CSC action and a composed spline callback use different rounding paths; a difference is not alone an algorithm bug.",
                              "This is no linear solve, convergence, full nonlinear registration or speed comparison."],
              "clock": {"imports_and_binding_seconds": time.monotonic() - started},
              "fsl_dso_snapshot_before": fsl_mappings()}
    if report["flags_before"]["cuda_initialized"] or report["fsl_dso_snapshot_before"]:
        raise RuntimeError("unexpected CUDA/FSL runtime state")

    def require(condition, message):
        if not condition:
            raise RuntimeError(message)

    def saved_vector(label):
        array = np.fromfile(paths[label], dtype="<f8")
        require(array.ndim == 1 and array.size and bool(np.isfinite(array).all()), "invalid bound vector: " + label)
        return array

    def metrics(reference, actual):
        require(reference.shape == actual.shape, "action shape mismatch")
        require(bool(np.isfinite(reference).all()) and bool(np.isfinite(actual).all()), "action nonfinite")
        delta = actual - reference
        reference_norm = float(np.linalg.norm(reference))
        return {"different_bits": int(np.count_nonzero(reference.view("<u8") != actual.view("<u8"))),
                "max_abs": float(np.max(np.abs(delta), initial=0)),
                "relative_l2": float(np.linalg.norm(delta) / reference_norm) if reference_norm else None,
                "all_finite": True}

    def hash_values(tensors):
        return {name: tensor_value_hash(value, np) for name, value in tensors.items()}

    def reject(counter):
        def forbidden(*values, **kwargs):
            report[counter] += 1
            raise RuntimeError("out-of-scope call rejected: " + counter)
        return forbidden

    originals = [(registration._LevelSystem, "evaluate", "evaluate_calls"),
                 (registration._LevelSystem, "linearize", "linearize_calls"),
                 (registration._LevelSystem, "gradient", "gradient_calls"),
                 (BendingOperator, "__init__", "bending_constructor_calls"),
                 (optimizer, "preconditioned_conjugate_gradient", "solver_calls")]
    saved_methods = [(owner, name, getattr(owner, name)) for owner, name, _ in originals]
    for owner, name, counter in originals:
        setattr(owner, name, reject(counter))
    tensors_by_arm = {}
    try:
        restore_started = time.monotonic()
        gradient, diagonal = saved_vector("gradient"), saved_vector("diagonal")
        n = gradient.size
        require(diagonal.size == n, "bound gradient/diagonal shape mismatch")
        arms, normals = {}, {}
        value_before, layouts = {}, {}
        for kind in ("optimized", "reference"):
            tensors, layout = restore_tensors(paths["checkpoint"], stage1["checkpoint"]["arrays"], np, torch)
            tensors_by_arm[kind] = tensors
            value_before[kind], layouts[kind] = hash_values(tensors), layout
            for label, target in (("gradient", gradient), ("diagonal", diagonal)):
                full = np.ascontiguousarray((2 * tensors[label + "_half"]).numpy(), dtype="<f8")
                require(full.tobytes() == target.tobytes(), "restored full " + label + " bridge failed")
            arms[kind], normals[kind], _ = restored_arm(kind, tensors, stage1, registration)
        pointers = [{value.untyped_storage().data_ptr() for value in tensors.values() if value.numel()}
                    for tensors in tensors_by_arm.values()]
        require(not pointers[0].intersection(pointers[1]), "two arms share mutable input storage")
        report["restored_layouts"] = layouts
        report["checkpoint_values_before"] = value_before
        report["two_arm_storage_disjoint"] = True
        report["restored_full_g_diag_bitexact"] = True
        report["clock"]["checkpoint_restore_and_factory_seconds"] = time.monotonic() - restore_started

        raw = np.fromfile(paths["H"], dtype="<f8")
        require(raw.size == n * n and bool(np.isfinite(raw).all()), "bound existing H shape/nonfinite")
        matrix = raw.reshape(n, n)
        report["existing_H_loaded_for_third_comparison"] = True
        sparse = csc_matrix(matrix)
        csc_source = args.root / "repo/validation/fnirt_cpu_current_replay_20261006/source/replay_current.py"
        csc_action = compile_csc(csc_source, np, njit)
        report["CSC"] = {"shape": [n, n], "nnz": int(sparse.nnz), "compiled_fastmath": False, "compiled_cache": False,
                         "data_sha256": hashlib.sha256(sparse.data.tobytes()).hexdigest(),
                         "indices_sha256": hashlib.sha256(sparse.indices.tobytes()).hexdigest(),
                         "indptr_sha256": hashlib.sha256(sparse.indptr.tobytes()).hexdigest()}

        def callback(kind, value):
            direction = torch.from_numpy(value.copy())
            require(direction.device.type == "cpu" and direction.dtype == torch.float64 and not direction.requires_grad,
                    "callback direction policy")
            require(direction.ndim == 1 and direction.numel() == n, "callback direction shape")
            tick = time.monotonic()
            report["callback_calls"][kind] += 1
            result = 2 * arms[kind](direction)
            require(result.device.type == "cpu" and result.dtype == torch.float64 and not result.requires_grad,
                    "callback result policy")
            array = np.ascontiguousarray(result.detach().numpy(), dtype="<f8").copy()
            require(array.shape == value.shape and bool(np.isfinite(array).all()), "callback result shape/nonfinite")
            return array, time.monotonic() - tick

        def one_direction(label, value, unit_column=None):
            require(value.shape == (n,) and bool(np.isfinite(value).all()), "direction invalid")
            row = {"label": label, "coordinate_units": "mixed displacement-coefficient/intensity coordinates",
                   "direction_value_sha256": hashlib.sha256(value.tobytes()).hexdigest()}
            # Keep this row even when the first original unit bridge fails.
            report.setdefault("directions", []).append(row)
            original, elapsed_original = callback("optimized", value)
            row["optimized_seconds_including_cold_JIT_if_any"] = elapsed_original
            row["optimized_value_sha256"] = hashlib.sha256(original.tobytes()).hexdigest()
            if unit_column is not None:
                reference_column = np.ascontiguousarray(matrix[:, unit_column])
                row["stored_H_unit_column"] = unit_column
                row["optimized_vs_stored_dense_column"] = metrics(reference_column, original)
                row["original_unit_bridge_bitexact"] = original.tobytes() == reference_column.tobytes()
                require(row["original_unit_bridge_bitexact"], "first restored optimized unit column mismatch; stop")
            controlled, elapsed_reference = callback("reference", value)
            row["reference_seconds"] = elapsed_reference
            row["reference_value_sha256"] = hashlib.sha256(controlled.tobytes()).hexdigest()
            row["optimized_vs_reference"] = metrics(controlled, original)
            tick = time.monotonic()
            third = csc_action(sparse.indptr, sparse.indices, sparse.data, value)
            report["strict_csc_calls"] += 1
            row["CSC_seconds_including_cold_JIT_if_any"] = time.monotonic() - tick
            row["CSC_value_sha256"] = hashlib.sha256(third.tobytes()).hexdigest()
            row["optimized_vs_existing_CSC"] = metrics(third, original)
            row["reference_vs_existing_CSC"] = metrics(third, controlled)
            row["third_comparison_is_not_a_new_native_or_universal_callback_oracle"] = True

        unit_indices = list(dict.fromkeys((0, n // 2, n - 1)))
        for label, column in zip(("unit_first", "unit_middle", "unit_intensity"), unit_indices):
            direction = np.zeros(n, dtype="<f8")
            direction[column] = 1.0
            one_direction(label, direction, column)
        old, new = saved_vector("old_solution"), saved_vector("new_solution")
        require(old.size == new.size == n, "saved solution direction geometry")
        for label, direction in (("saved_current_full_g", gradient), ("saved_old_solution", old),
                                 ("saved_new_solution", new), ("saved_new_minus_old", new - old)):
            one_direction(label, np.ascontiguousarray(direction, dtype="<f8"))
        report["checkpoint_values_after"] = {kind: hash_values(tensors) for kind, tensors in tensors_by_arm.items()}
        report["checkpoint_operand_values_unchanged"] = report["checkpoint_values_after"] == value_before
        require(report["checkpoint_operand_values_unchanged"], "callback changed immutable checkpoint operand")
        report["normal_cache_after"] = {"optimized": {"packed_layout": list(normals["optimized"]._layout),
                                                            "layout_copy_bytes": normals["optimized"].layout_copy_bytes,
                                                            "scratch_shape": list(normals["optimized"].scratch.shape),
                                                            "scratch_stride": list(normals["optimized"].scratch.stride())},
                                         "reference": None}
        report["probe_completed"] = True
        report["status"] = "bounded_restored_callback_comparison_completed"
    except Exception as error:
        report["status"] = "restored_callback_probe_failed_stopped"
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
        raise
    finally:
        for owner, name, method in saved_methods:
            setattr(owner, name, method)
        report["flags_after"] = flags()
        report["flags_unchanged"] = report["flags_before"] == report["flags_after"]
        after, changed = check_bindings(args.root, expected)
        report["bindings_after"] = after
        report["all_source_and_inputs_unchanged"] = not changed and after == bindings_before
        report["fsl_dso_snapshot_after"] = fsl_mappings()
        report["process_maxrss_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["clock"]["worker_through_post_bind_before_summary_seconds"] = time.monotonic() - started
        report["valid_bounded_diagnostic"] = bool(report["probe_completed"] and report["flags_unchanged"]
                                                   and report["all_source_and_inputs_unchanged"]
                                                   and not report["fsl_dso_snapshot_after"]
                                                   and all(report[name] == 0 for name in
                                                           ("evaluate_calls", "linearize_calls", "gradient_calls",
                                                            "bending_constructor_calls", "solver_calls", "native_process_calls")))
        write_json(args.output / "summary.public.json", report)
    require(report["valid_bounded_diagnostic"], "callback diagnostic postcondition failed")
    print(json.dumps({name: report[name] for name in ("status", "callback_calls", "strict_csc_calls", "valid_bounded_diagnostic")}))


if __name__ == "__main__":
    main()
