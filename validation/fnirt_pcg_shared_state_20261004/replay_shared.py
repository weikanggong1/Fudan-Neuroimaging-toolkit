"""Replay unchanged FNIT PCG on two real, canonical native linear systems.

All matrices and vectors remain in the private server run directory. Public
output contains only anonymous metrics and hashes. No production code changes.
"""
import argparse
import csv
import hashlib
import inspect
import json
import os
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from numba import njit
from scipy.sparse import csc_matrix

from fnit.fnirt.optimizer import preconditioned_conjugate_gradient


@njit(fastmath=False, cache=True)
def column_matvec(indptr, indices, values, direction):
    """Own strict FP64 column accumulation; verified against saved native q."""
    result = np.zeros(direction.size, dtype=np.float64)
    for column in range(direction.size):
        weight = direction[column]
        for index in range(indptr[column], indptr[column + 1]):
            result[indices[index]] += weight * values[index]
    return result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(first, second):
    difference = first - second
    return {"different": int(np.count_nonzero(difference)),
            "max_abs": float(np.max(np.abs(difference))),
            "rmse": float(np.sqrt(np.mean(difference * difference))),
            "relative_l2": float(np.linalg.norm(difference) / max(np.linalg.norm(first), np.finfo(float).tiny))}


def observe_fnit(matvec, rhs, diagonal):
    """Observe via the matvec call, without replacing the FNIT PCG loop."""
    b = torch.from_numpy(rhs.copy())
    d = torch.from_numpy(diagonal.copy())
    floor = torch.finfo(b.dtype).eps * d.abs().mean().clamp_min(1)
    inverse = d.clamp_min(floor).reciprocal()
    residual = b.clone()
    norm = torch.linalg.vector_norm(b)
    records, private = [], []

    def product(direction):
        nonlocal residual
        result = matvec(direction)
        z = inverse * residual
        rho = torch.dot(residual, z)
        denominator = torch.dot(direction, result)
        alpha = rho / denominator
        after = residual - alpha * result
        records.append({"iteration": len(records) + 1,
                        "relative_before": float(torch.linalg.vector_norm(residual) / norm),
                        "relative_after": float(torch.linalg.vector_norm(after) / norm),
                        "rho": float(rho), "denominator": float(denominator), "alpha": float(alpha)})
        private.append({"r": residual.numpy().copy(), "z": z.numpy().copy(),
                        "p": direction.numpy().copy(), "q": result.numpy().copy(),
                        "r_after": after.numpy().copy()})
        residual = after
        return result

    solution, report = preconditioned_conjugate_gradient(product, b, diagonal=d,
                                                       tolerance=1e-3, max_iterations=500)
    if records and report.relative_residual != records[-1]["relative_after"]:
        raise RuntimeError("FNIT observer changed or failed to reproduce its residual")
    return solution.numpy(), asdict(report), records, private


def controlled_loop(matvec, rhs, diagonal, divide=False):
    """Validation-only same Torch reductions, with one preconditioner change."""
    b, d = torch.from_numpy(rhs.copy()), torch.from_numpy(diagonal.copy())
    x, r = torch.zeros_like(b), b.clone()
    inverse = d.reciprocal()
    apply = (lambda value: value / d) if divide else (lambda value: value * inverse)
    z = apply(r); p = z.clone(); rho = torch.dot(r, z)
    norm = torch.linalg.vector_norm(b)
    records = []
    for iteration in range(1, 501):
        q = matvec(p); denominator = torch.dot(p, q); alpha = rho / denominator
        x = x + alpha * p; r = r - alpha * q
        relative = float(torch.linalg.vector_norm(r) / norm)
        records.append({"iteration": iteration, "relative_after": relative})
        if relative <= 1e-3:
            break
        z = apply(r); new_rho = torch.dot(r, z)
        p = z + (new_rho / rho) * p; rho = new_rho
    return x.numpy(), records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8); torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.oracle / "state.json").read_text())
    dimension = metadata["dimension"]
    result = {"scope": "two bounded real shared-state linear systems; no full registration or speed benchmark",
              "host": platform.node(), "affinity": sorted(os.sched_getaffinity(0)),
              "torch": torch.__version__, "threads": torch.get_num_threads(),
              "tolerance": 1e-3, "max_iterations": 500, "x0": "all FP64 zeros",
              "rhs_convention": "native +gradient, native LM step is minus solution; FNIT production -gradient/half-H is not reassembled here",
              "solver_source_sha256": sha(Path(inspect.getfile(preconditioned_conjugate_gradient))),
              "states": []}
    for state in metadata["states"]:
        solve = state["solve"]; prefix = f"solve{solve}"
        load = lambda name: np.fromfile(args.oracle / f"{prefix}_{name}.f64", dtype=np.float64)
        matrix, rhs, diagonal, x0 = load("A").reshape(dimension, dimension), load("rhs"), load("diagonal"), load("x0")
        assert not np.any(x0)
        assert np.array_equal(diagonal, np.diag(matrix))
        sparse = csc_matrix(matrix)
        native_solution = load("native_solution")
        with (args.oracle / f"{prefix}_trace.csv").open() as stream:
            native_trace = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
        inputs = {path.name: {"bytes": path.stat().st_size, "sha256": sha(path)}
                  for path in sorted(args.oracle.glob(f"{prefix}_*.f64"))}
        dense = torch.from_numpy(matrix)
        dense_product = lambda value: dense @ value
        column_product = lambda value: torch.from_numpy(column_matvec(sparse.indptr, sparse.indices, sparse.data, value.numpy()))
        q_checks = []
        for record in native_trace:
            iteration = int(record["iteration"])
            p = load(f"p_{iteration}"); q = load(f"q_{iteration}")
            q_checks.append({"iteration": iteration,
                             "column_vs_native_q": metrics(q, column_product(torch.from_numpy(p)).numpy()),
                             "dense_vs_native_q": metrics(q, dense_product(torch.from_numpy(p)).numpy())})
        floor = np.finfo(float).eps * max(float(np.mean(np.abs(diagonal))), 1)
        report = {"native_state": state, "input_bindings": inputs,
                  "matrix_max_asymmetry": float(np.max(np.abs(matrix - matrix.T))),
                  "diagonal_min": float(diagonal.min()), "diagonal_floor": floor,
                  "diagonal_floor_trigger_count": int(np.count_nonzero(diagonal < floor)),
                  "native_trace": native_trace, "same_native_direction_products": q_checks,
                  "arms": {}}
        for name, matvec in [("fnit_dense", dense_product), ("fnit_column_order", column_product)]:
            start = time.perf_counter()
            solution, pcg_report, trace, private = observe_fnit(matvec, rhs, diagonal)
            item = {"pcg_report": pcg_report, "trace": trace,
                    "solution_vs_native": metrics(native_solution, solution),
                    "true_relative_residual": float(np.linalg.norm(rhs - matrix @ solution) / np.linalg.norm(rhs)),
                    "observed_seconds_not_benchmark": time.perf_counter() - start,
                    "same_iteration_vectors": []}
            for iteration, vectors in enumerate(private, 1):
                if iteration > len(native_trace):
                    break
                item["same_iteration_vectors"].append({"iteration": iteration,
                    **{key: metrics(load(f"{key}_{iteration}"), value) for key, value in vectors.items()}})
            np.save(args.output / f"{prefix}_{name}_solution.npy", solution)
            report["arms"][name] = item
        for name, divide in [("control_column_reciprocal", False), ("control_column_division", True)]:
            solution, trace = controlled_loop(column_product, rhs, diagonal, divide)
            report["arms"][name] = {"iterations": len(trace), "trace": trace,
                                   "solution_vs_native": metrics(native_solution, solution)}
        result["states"].append(report)
    (args.output / "comparison.public.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"states": [{"solve": state["native_state"]["solve"],
                                 "native_iterations": state["native_state"]["native_cg_iterations"],
                                 "arms": {name: arm.get("pcg_report", {"iterations": arm.get("iterations")})
                                          for name, arm in state["arms"].items()}}
                                for state in result["states"]]}))


if __name__ == "__main__":
    main()
