"""Check actual candidate recipe capture against the saved shared-state probe."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def metric(candidate, reference):
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    return {"shape": list(candidate.shape), "dtype": str(candidate.dtype),
            "reference_dtype": str(reference.dtype),
            "exact": bool(np.array_equal(candidate, reference)),
            "different_values": int(np.count_nonzero(difference)),
            "max_absolute_error": float(np.max(np.abs(difference), initial=0)),
            "relative_l2": float(np.linalg.norm(difference.ravel()) /
                                 max(np.linalg.norm(reference.astype(np.float64).ravel()), 1e-300))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate = json.loads((args.candidate / "report.public.json").read_text())
    baseline = json.loads((args.baseline / "fnit/report.public.json").read_text())
    probe = json.loads((args.baseline / "shared_probe_hybrid_v5/report.public.json").read_text())
    mode = "epsilon_cpu64_interp_fp32_owner"
    prototype = probe["variants"][mode]
    with np.load(args.candidate / "shared_input.npz") as actual, np.load(args.baseline / "fnit/shared_input.npz") as old:
        inputs = {key: metric(actual[key], old[key]) for key in actual.files}
        if set(actual.files) != set(old.files) or not all(item["exact"] for item in inputs.values()):
            raise RuntimeError("actual candidate recipe state differs from saved oracle input")
    actual_gradient = np.load(args.candidate / "fnit_gradient.npy")
    prototype_gradient = np.load(args.baseline / ("shared_probe_hybrid_v5/" + mode + "_gradient.npy"))
    native_gradient = np.load(args.native / "native_total_gradient.npy")
    gradients = {"versus_prototype": metric(actual_gradient, prototype_gradient),
                 "versus_official": metric(actual_gradient, native_gradient)}
    cost_difference = candidate["first_cost"] - prototype["total_cost"]
    native_cost = json.loads((args.native / "report.public.json").read_text())["cost"]["total"]
    report = {"schema": "fnit.gems.actual-first-state.v1", "structure": candidate["structure"],
              "scope": "Actual frozen CPU candidate first recipe closure only; no optimizer update",
              "candidate_source_files": candidate["source"], "fixed_input_files": candidate["inputs"],
              "shared_arrays": inputs, "gradients": gradients,
              "cost": {"actual": candidate["first_cost"], "prototype": prototype["total_cost"],
                       "official": native_cost, "actual_minus_prototype": cost_difference,
                       "actual_minus_official": candidate["first_cost"] - native_cost},
              "runtime": {key: candidate[key] for key in ("host", "affinity", "torch_threads", "numba_threads")},
              "capture_files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in args.candidate.iterdir() if p.is_file()}}
    report["prototype_gate_passed"] = bool(abs(cost_difference) <= 1e-8 and
                                            gradients["versus_prototype"]["max_absolute_error"] <= 1e-7)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"structure": report["structure"], "cost": report["cost"],
                      "gradients": report["gradients"], "gate": report["prototype_gate_passed"]}))
    if not report["prototype_gate_passed"]:
        raise RuntimeError("candidate does not reproduce shared-state prototype")


if __name__ == "__main__":
    main()
