"""Compare saved first-step states; publish metrics and hashes, not MRI arrays."""
import argparse
import hashlib
import json
from pathlib import Path
import socket

import nibabel as nib
import numpy as np
import torch

from fnit.fnirt import registration
from fnit.fnirt.spline import adjoint_field, spline_bases


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(reference, candidate):
    if reference.shape != candidate.shape:
        return {"shape_match": False, "reference_shape": list(reference.shape),
                "candidate_shape": list(candidate.shape)}
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    norm = np.linalg.norm(reference.astype(np.float64).ravel())
    return {"shape": list(reference.shape), "exact": bool(np.array_equal(reference, candidate)),
            "different_values": int(np.count_nonzero(difference)),
            "max_abs": float(np.max(np.abs(difference))),
            "rmse": float(np.sqrt(np.mean(difference ** 2))),
            "relative_l2": float(np.linalg.norm(difference.ravel()) / norm) if norm else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    native, stock = args.run / "oracle", args.run / "stock-debug"
    optimized, reference = args.run / "fnit-optimized", args.run / "fnit-reference"

    def image(directory, name):
        return np.asanyarray(nib.load(directory / (name + ".nii.gz")).dataobj)

    def npy(directory, name):
        return np.load(directory / (name + ".npy"))

    report = {"scope": "saved first-level states and one accepted update; no complete estimator equivalence",
              "host": socket.gethostname(), "native_vs_fnit": {}, "stock_vs_oracle": {},
              "reference_vs_optimized": {}, "reduction_isolation": {}}
    names = ("normalized_moving", "normalized_fixed", "smoothed_moving", "initial_fixed",
             "initial_warped", "initial_mask", "initial_derivative")
    for name in names:
        report["native_vs_fnit"][name] = compare(image(native, name), npy(optimized, name))
    mask = npy(optimized, "initial_mask")
    report["native_vs_fnit"]["initial_derivative_valid_mask"] = compare(
        image(native, "initial_derivative")[mask], npy(optimized, "initial_derivative")[mask])
    for name, fnit_name in (("initial_gradient", "initial_gradient_direct"),
                            ("initial_gradient", "initial_gradient_lm"),
                            ("initial_diagonal", "initial_diagonal"),
                            ("initial_hessian_probe", "initial_hessian_probe"),
                            ("accepted_parameters", "accepted_parameters")):
        report["native_vs_fnit"][fnit_name] = compare(np.loadtxt(native / (name + ".txt")),
                                                     npy(optimized, fnit_name))
    for path in reference.glob("*.npy"):
        report["reference_vs_optimized"][path.stem] = compare(np.load(path), np.load(optimized / path.name))
    for name, prefix in (("initial_fixed", "FnirtDebugScaledRef"),
                         ("initial_warped", "FnirtDebugRobj"),
                         ("initial_mask", "FnirtDebugMask")):
        report["stock_vs_oracle"][name] = compare(image(stock, prefix + "_level01_iter00_attempt01"),
                                                  image(native, name))
    report["stock_vs_oracle"]["accepted_coefficients_fp32"] = compare(
        image(stock, "FnirtDebugDefCoefs_level01_iter01_attempt01"), image(native, "accepted_coefficients"))
    report["stock_vs_oracle"]["accepted_warped"] = compare(
        image(stock, "FnirtDebugRobj_level01_iter01_attempt01"), image(native, "accepted_warped"))
    report["stock_vs_oracle"]["gradient_ascii_precision10"] = compare(
        np.loadtxt(stock / "FnirtDebugGradient_level01_iter01_attempt00.txt"),
        np.loadtxt(native / "initial_gradient.txt"))
    report["oracle_scalars"] = (native / "scalars.txt").read_text()
    for name in ("reference", "optimized"):
        report["fnit_" + name] = json.loads((args.run / ("fnit-" + name) / "report.public.json").read_text())

    # Isolate Jte ordering on identical official and FNIT image arrays. FSL
    # forms float derivative*residual before the double adjoint and division.
    torch.set_num_threads(8)
    bases = spline_bases((24,28,24), (5,5,5), (8.,8.,8.), device="cpu", dtype=torch.float64)
    native_gradient = np.loadtxt(native / "initial_gradient.txt")
    for label, warped, fixed, derivative, valid in (
        ("official_arrays", image(native, "initial_warped"), image(native, "initial_fixed"),
         image(native, "initial_derivative"), image(native, "initial_mask") > 0),
        ("fnit_arrays", npy(optimized, "initial_warped"), npy(optimized, "initial_fixed"),
         npy(optimized, "initial_derivative"), mask),
    ):
        difference = torch.from_numpy((warped - fixed).copy())
        partial = torch.from_numpy(derivative.copy()).movedim(-1, 0)
        weight = torch.from_numpy(valid.copy()).to(torch.float32)
        n = int(valid.sum())
        gradient = 2 * adjoint_field((partial * difference[None] * weight[None]).double(), bases) / n
        scale_gradient = -2 * (torch.from_numpy(fixed.copy()) * difference * weight).sum(dtype=torch.float64) / n
        result = registration._pack(gradient, scale_gradient).numpy()
        report["reduction_isolation"][label + "_jte_order"] = compare(native_gradient, result)
    report["hashes"] = {}
    for directory in (native, stock, reference, optimized):
        report["hashes"][directory.name] = {p.name: digest(p) for p in directory.iterdir()
                                             if p.is_file() and p.suffix in (".gz", ".npy", ".txt", ".log", ".json")}
    report["flip_smoothing"] = json.loads((args.run / "flip_smoothing.public.json").read_text())
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"complete": True, "output": str(args.output),
                      "stock_oracle_coefficients_exact": report["stock_vs_oracle"]["accepted_coefficients_fp32"]["exact"],
                      "reduction_isolation": report["reduction_isolation"]}))


if __name__ == "__main__":
    main()
