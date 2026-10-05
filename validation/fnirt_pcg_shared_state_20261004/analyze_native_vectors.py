"""Finite same-vector arithmetic/stop checks of the saved real PCG systems."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import torch
from scipy.linalg import eigvalsh, cho_factor, cho_solve


def difference(first, second):
    delta = np.asarray(first) - np.asarray(second)
    return {"different": int(np.count_nonzero(delta)), "max_abs": float(np.max(np.abs(delta)))}


def scalar(first, second):
    # All scalars below are positive; adjacent FP64 bit patterns are monotonic.
    return {"native": float(first), "torch": float(second), "abs_difference": abs(float(first) - float(second)),
            "ulp_difference": abs(int(np.float64(first).view(np.uint64)) - int(np.float64(second).view(np.uint64)))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8); torch.set_num_interop_threads(1)
    comparison = json.loads((args.run / "replay/comparison.public.json").read_text())
    result = {"scope": "same saved native vectors; no nonlinear continuation and no production changes",
              "host": platform.node(), "states": []}
    for state in comparison["states"]:
        solve = state["native_state"]["solve"]; prefix = f"solve{solve}"
        load = lambda name: np.fromfile(args.run / f"oracle/{prefix}_{name}.f64", dtype=np.float64)
        n = comparison["states"][0]["input_bindings"]["solve2_rhs.f64"]["bytes"] // 8
        matrix, rhs, diagonal = load("A").reshape(n, n), load("rhs"), load("diagonal")
        d = torch.from_numpy(diagonal); b = torch.from_numpy(rhs)
        norm = torch.linalg.vector_norm(b)
        entries = []
        for record in state["native_trace"]:
            iteration = int(record["iteration"])
            r, z, p, q, after = [torch.from_numpy(load(f"{key}_{iteration}")) for key in ["r", "z", "p", "q", "r_after"]]
            rho, denominator = torch.dot(r, z), torch.dot(p, q)
            entries.append({"iteration": iteration,
                            "division_vs_native_z": difference(z.numpy(), (r / d).numpy()),
                            "reciprocal_vs_native_z": difference(z.numpy(), (r * d.reciprocal()).numpy()),
                            "rho": scalar(record["rho"], rho),
                            "denominator": scalar(record["denominator"], denominator),
                            "alpha": scalar(record["alpha"], rho / denominator),
                            "update_with_native_alpha_vs_native_r_after": difference(after.numpy(), (r - record["alpha"] * q).numpy()),
                            "relative_after": scalar(record["relative_after"], torch.linalg.vector_norm(after) / norm)})
        eigenvalues = eigvalsh(matrix)
        accurate = cho_solve(cho_factor(matrix), rhs)
        reference = load("native_solution")
        floor = np.finfo(float).tiny
        report = {"solve": solve, "same_vector_checks": entries,
                  "spectral_min": float(eigenvalues[0]), "spectral_max": float(eigenvalues[-1]),
                  "spectral_condition_number": float(eigenvalues[-1] / eigenvalues[0]),
                  "high_accuracy_cholesky_relative_residual": float(np.linalg.norm(rhs - matrix @ accurate) / np.linalg.norm(rhs)),
                  "native_true_relative_residual": float(np.linalg.norm(rhs - matrix @ reference) / np.linalg.norm(rhs)),
                  "solution_relative_error_vs_cholesky": {}, "stopping_checks": {}}
        solutions = {"native": reference}
        solutions.update({name: np.load(args.run / f"replay/{prefix}_{name}_solution.npy")
                          for name in ["fnit_dense", "fnit_column_order"]})
        for name, solution in solutions.items():
            report["solution_relative_error_vs_cholesky"][name] = float(np.linalg.norm(solution - accurate) / max(np.linalg.norm(accurate), floor))
            trace = state["native_trace"] if name == "native" else state["arms"][name]["trace"]
            report["stopping_checks"][name] = {"all_before_stop_above_tolerance": all(row["relative_after"] > 1e-3 for row in trace[:-1]),
                                               "stop_at_or_below_tolerance": trace[-1]["relative_after"] <= 1e-3,
                                               "all_denominators_finite_positive": all(np.isfinite(row["denominator"]) and row["denominator"] > 0 for row in trace),
                                               "all_rho_above_tiny": all(row["rho"] > floor for row in trace)}
        result["states"].append(report)
    (args.run / "replay/native_vectors.public.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps([{key: value for key, value in state.items() if key != "same_vector_checks"} for state in result["states"]]))


if __name__ == "__main__":
    main()
