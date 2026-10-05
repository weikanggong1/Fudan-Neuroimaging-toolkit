"""Prepared initial z/rho/norm probe; only two saved arrays, no linear solve."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import socket
import struct
import sys
import time
from types import SimpleNamespace

from initial_io import bound, check_bindings, check_freeze, input_paths, write_json
from prefixes import ast_bindings, compile_prefix


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "workspace", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--approved-initial-probe", action="store_true", required=True)
    args = parser.parse_args()
    if sorted(os.sched_getaffinity(0)) != [32, 36, 40, 44, 48, 52, 56, 60]:
        raise RuntimeError("physical CPU8 affinity required")
    if any(os.environ.get(name) != "8" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")):
        raise RuntimeError("eight-thread environment required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or any(name in os.environ for name in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH", "OPENBLAS_CORETYPE")):
        raise RuntimeError("CPU-only without loader overrides required")
    if platform.machine() != "x86_64" or "fma" not in Path("/proc/cpuinfo").read_text().split():
        raise RuntimeError("bounded owned-FMA x86_64 scope required")
    resource.setrlimit(resource.RLIMIT_AS, (20_000_000_000, 20_000_000_000))
    started = time.monotonic()
    expected = json.loads((args.workspace / "expected.public.json").read_text())
    before, failures = check_bindings(args.root, expected)
    frozen, freeze_failures = check_freeze(args.workspace)
    if failures or freeze_failures:
        raise RuntimeError("pre-import source/input/freeze changed")
    for name, identity in expected["candidate_source_provenance"].items():
        if bound(args.workspace / name) != identity:
            raise RuntimeError("candidate source copy changed")
    paths = input_paths(args.root, expected)
    stage1 = json.loads(paths["stage1_summary"].read_text())
    stage3 = json.loads(paths["stage3_summary"].read_text())
    if not stage1["accepted_checkpoint"] or not stage3["valid_bounded_diagnostic"]:
        raise RuntimeError("prior accepted checkpoint/pair missing")
    if bound(paths["checkpoint"]) != {key: stage1["checkpoint"][key] for key in ("bytes", "sha256")}:
        raise RuntimeError("checkpoint differs from accepted producer")
    if stage3["solver_calls"] != {"current_torch": 1, "owned_cpu": 1} or stage3["lm_tau"] != expected["lm_tau"]:
        raise RuntimeError("wrong prior controlled comparison")
    for kind, direction_sha in expected["expected_initial_direction_sha256"].items():
        first = stage3["arms"][kind]["callback_ledger"][1]
        if first["phase"] != "pcg_iteration" or first["direction_sha256"] != direction_sha:
            raise RuntimeError("bound first direction receipt differs")
    optimizer_path = args.root / "repo/src/fnit/fnirt/optimizer.py"
    candidate_path = args.workspace / "pcg_cpu_candidate.py"
    prefix_ast = ast_bindings(optimizer_path, candidate_path)
    if prefix_ast != expected["prefix_ast"]:
        raise RuntimeError("source initialization prefix AST changed")
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    import numpy as np
    import torch
    import own_reductions as arithmetic
    if Path(arithmetic.__file__).resolve() != (args.workspace / "own_reductions.py").resolve():
        raise RuntimeError("wrong owned arithmetic module")
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)

    def require(condition, message):
        if not condition:
            raise RuntimeError(message)

    def flags():
        return {"cuda_initialized": torch.cuda.is_initialized(), "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                "matmul_tf32": torch.backends.cuda.matmul.allow_tf32, "grad_enabled": torch.is_grad_enabled(),
                "default_dtype": str(torch.get_default_dtype())}

    def mapped_fsl():
        # Only before/after file identity snapshots, not syscall auditing.
        return sorted({Path(line.split()[-1]).name for line in Path("/proc/self/maps").read_text().splitlines()
                       if len(line.split()) >= 6 and ("/apps/FSL/" in line.split()[-1]
                                                    or Path(line.split()[-1]).name.startswith("libfsl"))})

    def value_array(value):
        if isinstance(value, torch.Tensor):
            require(value.device.type == "cpu" and value.dtype == torch.float64 and not value.requires_grad,
                    "CPU FP64/no-grad tensor required")
            array = value.detach().numpy()
        else:
            array = value
        require(isinstance(array, np.ndarray) and array.dtype == np.float64 and array.ndim == 1
                and bool(np.isfinite(array).all()), "finite 1D FP64 required")
        return np.ascontiguousarray(array, dtype="<f8")

    def value_hash(value):
        return hashlib.sha256(value_array(value).tobytes()).hexdigest()

    def hash_values(values):
        return {name: value_hash(value) for name, value in values.items()}

    report = {"scope": expected["scope"], "status": "started_not_accepted", "host": socket.gethostname(),
              "bindings_before": before, "harness_bindings": frozen, "prefix_ast": prefix_ast,
              "flags_before": flags(), "fsl_dso_snapshot_before": mapped_fsl(),
              "affinity": sorted(os.sched_getaffinity(0)), "torch_threads": torch.get_num_threads(),
              "interop_threads": torch.get_num_interop_threads(), "address_space_cap_bytes": 20_000_000_000,
              "prefix_calls": {"current_torch": 0, "owned_cpu": 0},
              "dot_calls": {"current_torch": 0, "owned_cpu": 0},
              "norm_calls": {"current_torch": 0, "owned_cpu": 0},
              "callback_calls": 0, "PCG_calls": 0, "native_calls": 0,
              "evaluate_calls": 0, "linearize_calls": 0, "new_H": False, "images_read": 0,
              "checkpoint_arrays_read": [], "unexpected_zero_rhs_return": 0,
              "runtime_integration_accepted": False, "probe_completed": False, "scalar_ledger": [],
              "lm_tau": expected["lm_tau"],
              "limitations": ["Initial precondition/scalar control only; no conclusion that the first fork uniquely causes the final step difference.",
                              "No callback, solve, native same-system oracle, full registration or tighter-tolerance assessment.",
                              "The current RHS difference from official assembly remains unresolved; this control does not explain its 7.4e-7 source."],
              "clock": {"imports_and_binding_seconds": time.monotonic() - started}}
    require(not report["flags_before"]["cuda_initialized"] and not report["fsl_dso_snapshot_before"], "unexpected CUDA/FSL state")

    def scalar_record(label, kind, value, inputs):
        scalar = float(value)
        require(math.isfinite(scalar), "nonfinite reduction: " + label)
        row = {"label": label, "implementation": kind, "value": scalar,
               "FP64_little_endian_hex": struct.pack("<d", scalar).hex(),
               "input_value_sha256": [value_hash(item) for item in inputs]}
        report["scalar_ledger"].append(row)
        return value

    def old_dot(left, right):
        report["dot_calls"]["current_torch"] += 1
        require(report["dot_calls"]["current_torch"] <= 2, "current dot cap")
        return scalar_record("rho", "current_torch", torch.dot(left, right), (left, right))

    def new_dot(left, right):
        report["dot_calls"]["owned_cpu"] += 1
        require(report["dot_calls"]["owned_cpu"] <= 2, "owned dot cap")
        return scalar_record("rho", "owned_cpu", arithmetic.dot(left, right), (left, right))

    def old_norm(value):
        report["norm_calls"]["current_torch"] += 1
        require(report["norm_calls"]["current_torch"] <= 1, "current norm cap")
        return scalar_record("rhs_norm", "current_torch", torch.linalg.vector_norm(value), (value,))

    def new_norm(value):
        report["norm_calls"]["owned_cpu"] += 1
        require(report["norm_calls"]["owned_cpu"] <= 1, "owned norm cap")
        return scalar_record("rhs_norm", "owned_cpu", arithmetic.norm(value), (value,))

    class TorchProxy:
        # Proxy the prefix namespace; never monkeypatch global Torch methods.
        linalg = SimpleNamespace(vector_norm=old_norm)
        dot = staticmethod(old_dot)

        def __getattr__(self, name):
            return getattr(torch, name)

    def forbidden_callback(*values, **kwargs):
        report["callback_calls"] += 1
        raise RuntimeError("callback out of scope")

    def forbidden_zero_return(*values, **kwargs):
        report["unexpected_zero_rhs_return"] += 1
        raise RuntimeError("unexpected zero RHS initialization return")

    saved, derived, before_saved, before_derived = {}, {}, {}, {}
    try:
        with torch.no_grad():
            # np.load opens the archive index; only these two array payloads
            # are materialized. No other saved image/state arrays are read.
            with np.load(paths["checkpoint"], allow_pickle=False) as packed:
                for name in expected["limits"]["checkpoint_arrays_read"]:
                    array = packed[name]
                    record = stage1["checkpoint"]["arrays"][name]
                    require(list(array.shape) == record["shape"] and str(array.dtype) == record["dtype"] == "float64"
                            and array.ndim == 1 and bool(np.isfinite(array).all()), "saved g/diag schema")
                    strides = tuple(value // array.itemsize for value in record["strides"])
                    require(all(value >= 0 and value % array.itemsize == 0 for value in record["strides"]), "invalid saved stride")
                    tensor = torch.empty_strided(tuple(array.shape), strides, dtype=torch.float64, device="cpu")
                    tensor.copy_(torch.from_numpy(array.copy(order="K")))
                    require(list(value * array.itemsize for value in tensor.stride()) == record["strides"], "stride restoration failed")
                    require(value_hash(tensor) == stage3["checkpoint_values_before"]["current_torch"][name], "g/diag value bridge")
                    saved[name] = tensor
                    report["checkpoint_arrays_read"].append(name)
            before_saved = hash_values(saved)
            gradient, diagonal = saved["gradient_half"], saved["diagonal_half"]
            require(gradient.numel() and diagonal.shape == gradient.shape, "saved vector geometry")
            require(bool(gradient.ne(0).any()), "nonzero real RHS required")
            tau = expected["lm_tau"]["value"]
            require(tau == .001, "controlled tau changed")
            floor = torch.finfo(diagonal.dtype).eps * diagonal.abs().mean().clamp_min(1)
            damping_diagonal = diagonal.clamp_min(floor)
            rhs = -gradient
            preconditioner = (1 + tau) * damping_diagonal
            require(bool((preconditioner > 0).all()), "positive preconditioner required")
            derived = {"rhs": rhs, "damping_diagonal": damping_diagonal, "preconditioner": preconditioner}
            before_derived = hash_values(derived)
            require(before_derived == stage3["system_values_before"]["current_torch"]
                    == stage3["system_values_before"]["owned_cpu"], "same initial system input bridge")
            report["input_system_bitexact_to_stage3"] = True
            namespace = {"torch": TorchProxy(), "np": np, "math": math,
                         "PCGReport": forbidden_zero_return,
                         "arithmetic": SimpleNamespace(dot=new_dot, norm=new_norm)}
            functions = {"current_torch": compile_prefix(optimizer_path, "current_torch", namespace),
                         "owned_cpu": compile_prefix(candidate_path, "owned_cpu", namespace)}
            initialized = {}
            report["initial_direction_gates"] = {}
            for kind in ("current_torch", "owned_cpu"):
                report["prefix_calls"][kind] += 1
                kwargs = {"diagonal": preconditioner, "tolerance": .001, "max_iterations": 500}
                if kind == "current_torch":
                    kwargs["execution"] = "optimized"
                initialized[kind] = functions[kind](forbidden_callback, rhs, **kwargs)
                value = initialized[kind]["direction"]
                expected_sha = expected["expected_initial_direction_sha256"][kind]
                actual_sha = value_hash(value)
                report["initial_direction_gates"][kind] = {"expected_sha256": expected_sha,
                                                            "actual_sha256": actual_sha, "bitexact": actual_sha == expected_sha}
                require(actual_sha == expected_sha, "first direction SHA mismatch; stop before cross reductions")

            old, new = initialized["current_torch"], initialized["owned_cpu"]
            effective_weights = preconditioner.clamp_min(old["floor"])
            report["inner_floor"] = {"value": float(old["floor"]),
                "value_changed_count": int(torch.count_nonzero(effective_weights != preconditioner)),
                "bits_changed_count": int(np.count_nonzero(value_array(effective_weights).view("<u8") != value_array(preconditioner).view("<u8"))),
                "supplied_weights_sha256": value_hash(preconditioner), "effective_weights_sha256": value_hash(effective_weights)}
            old_array = value_array(old["direction"])
            new_array = value_array(new["direction"])
            # Separate direct division using the exact old effective diagonal;
            # this does not run a prefix, reduction, callback or PCG again.
            direct_same_weights = value_array(rhs) / value_array(effective_weights)

            def vector_difference(left, right):
                require(left.shape == right.shape and bool(np.isfinite(left).all()) and bool(np.isfinite(right).all()), "comparison vector invalid")
                left_bits, right_bits = left.view("<u8"), right.view("<u8")
                sign = np.uint64(1 << 63)
                # Monotone integer encoding of finite doubles; signed zero
                # remains distinct in both raw bits and this ULP distance.
                left_ordered = np.where(left_bits & sign, ~left_bits, left_bits ^ sign)
                right_ordered = np.where(right_bits & sign, ~right_bits, right_bits ^ sign)
                ulp = np.maximum(left_ordered, right_ordered) - np.minimum(left_ordered, right_ordered)
                return {"different_bits": int(np.count_nonzero(left_bits != right_bits)),
                        "max_abs": float(np.max(np.abs(left - right), initial=0)),
                        "max_ordered_ULP": int(np.max(ulp, initial=0)),
                        "left_sha256": hashlib.sha256(left.tobytes()).hexdigest(),
                        "right_sha256": hashlib.sha256(right.tobytes()).hexdigest()}

            report["precondition_controls"] = {
                "old_reciprocal_multiply_vs_candidate_division": vector_difference(old_array, new_array),
                "old_reciprocal_multiply_vs_division_same_effective_weights": vector_difference(old_array, direct_same_weights),
                "candidate_division_vs_division_same_effective_weights": vector_difference(new_array, direct_same_weights),
                "ordered_ULP_is_not_voxel_error": True}
            # Each source prefix has already performed its own norm/rho.
            # Add only the complementary rho for each identical frozen z.
            cross_old_z = new_dot(value_array(old["residual"]), old_array)
            cross_new_z = old_dot(torch.from_numpy(value_array(new["residual"]).copy()), torch.from_numpy(new_array.copy()))
            report["rho_cross_control"] = {
                "old_z": {"current_torch": float(old["rz"]), "owned_cpu": float(cross_old_z),
                          "same_z_sha256": value_hash(old["direction"])},
                "new_z": {"current_torch": float(cross_new_z), "owned_cpu": float(new["rho"]),
                          "same_z_sha256": value_hash(new["direction"])} }
            report["rhs_norm_control"] = {"current_torch": float(old["rhs_norm"]), "owned_cpu": float(new["right_norm"]),
                                          "same_rhs_sha256": value_hash(rhs)}
            require(sum(report["dot_calls"].values()) == 4 and sum(report["norm_calls"].values()) == 2, "reduction counts")
        report["probe_completed"] = True
        report["status"] = "bounded_initial_precondition_scalar_control_completed"
    except Exception as error:
        report["status"] = "initial_control_failed_stopped_no_retry"
        report["error_type"], report["error"] = type(error).__name__, str(error)
        raise
    finally:
        report["immutable_values_before"] = {"saved": before_saved, "derived": before_derived}
        report["immutable_values_after"] = {}
        report["postcondition_bookkeeping_failures"] = {}
        for label, values in (("saved", saved), ("derived", derived)):
            try:
                report["immutable_values_after"][label] = hash_values(values)
            except Exception as error:
                report["postcondition_bookkeeping_failures"][label] = {"error_type": type(error).__name__, "error": str(error)}
        report["operand_values_unchanged"] = report["immutable_values_before"] == report["immutable_values_after"]
        try:
            report["flags_after"] = flags()
            report["flags_unchanged"] = report["flags_before"] == report["flags_after"]
        except Exception as error:
            report["flags_unchanged"] = False
            report["postcondition_bookkeeping_failures"]["flags"] = {"error_type": type(error).__name__, "error": str(error)}
        try:
            after, changed = check_bindings(args.root, expected)
            report["bindings_after"], report["binding_failures_after"] = after, changed
            report["all_source_and_inputs_unchanged"] = not changed and after == before
        except Exception as error:
            report["all_source_and_inputs_unchanged"] = False
            report["postcondition_bookkeeping_failures"]["bindings"] = {"error_type": type(error).__name__, "error": str(error)}
        try:
            report["fsl_dso_snapshot_after"] = mapped_fsl()
        except Exception as error:
            report["fsl_dso_snapshot_after"] = None
            report["postcondition_bookkeeping_failures"]["mapped_DSO"] = {"error_type": type(error).__name__, "error": str(error)}
        report["process_maxrss_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["clock"]["worker_through_post_bind_before_summary_seconds"] = time.monotonic() - started
        report["valid_bounded_diagnostic"] = bool(report["probe_completed"] and report["flags_unchanged"]
            and report["all_source_and_inputs_unchanged"] and report["operand_values_unchanged"]
            and not report["postcondition_bookkeeping_failures"] and not report["fsl_dso_snapshot_after"]
            and report["prefix_calls"] == {"current_torch": 1, "owned_cpu": 1}
            and sum(report["dot_calls"].values()) == 4 and sum(report["norm_calls"].values()) == 2
            and report["checkpoint_arrays_read"] == expected["limits"]["checkpoint_arrays_read"]
            and all(report[name] == 0 for name in ("callback_calls", "PCG_calls", "native_calls", "evaluate_calls",
                                                 "linearize_calls", "images_read", "unexpected_zero_rhs_return")))
        write_json(args.output / "summary.public.json", report)
    require(report["valid_bounded_diagnostic"], "initial-control postcondition failed")
    print(json.dumps({name: report[name] for name in ("status", "prefix_calls", "dot_calls", "norm_calls", "valid_bounded_diagnostic")}))


if __name__ == "__main__":
    main()
