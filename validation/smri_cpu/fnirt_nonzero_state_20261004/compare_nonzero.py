"""Compare native and FNIT on identical saved real nonzero parameters."""
import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def metrics(reference, candidate):
    if reference.shape != candidate.shape:
        return {"shape_match": False, "reference_shape": list(reference.shape),
                "candidate_shape": list(candidate.shape)}
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    norm = float(np.linalg.norm(reference.astype(np.float64).ravel()))
    return {"exact": bool(np.array_equal(reference,candidate)),
            "different_values": int(np.count_nonzero(difference)),
            "max_abs": float(np.abs(difference).max()),
            "rmse": float(np.sqrt(np.mean(difference**2))),
            "relative_l2": float(np.linalg.norm(difference.ravel())/norm) if norm else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    native = args.run / "oracle"
    scalars = {name:float(value) for name,value in
               (line.split() for line in (native/"scalars.txt").read_text().splitlines())}
    report = {"scope":"same official first accepted FP64 parameters; no solve or complete pipeline",
              "native_scalars":scalars,"native_vs_fnit":{},"regularizer_isolation":{},
              "state_order_isolation":"artificial cf(trial)->grad(shared), not actual LM retry calls",
              "native_self":metrics(np.loadtxt(native/"shared_gradient.txt"),
                                    np.loadtxt(native/"reset_gradient.txt"))}
    for arm in ("baseline","candidate"):
        directory = args.run / arm
        record = json.loads((directory/"report.public.json").read_text())
        comparisons = {"state":record["shared"],"report":record}
        for name in ("shared_fixed","shared_warped","shared_mask","shared_derivative"):
            reference = np.asanyarray(nib.load(native/(name+".nii.gz")).dataobj)
            candidate = np.load(directory/(name+".npy"))
            comparisons[name] = metrics(reference,candidate)
            if name=="shared_derivative":
                mask = np.load(directory/"shared_mask.npy")
                comparisons["shared_derivative_valid_mask"] = metrics(reference[mask],candidate[mask])
        for name in ("shared_gradient","shared_diagonal","shared_hessian_probe"):
            comparisons[name] = metrics(np.loadtxt(native/(name+".txt")),
                                         np.load(directory/(name+".npy")))
        comparisons["shared_gradient_lm"] = metrics(np.loadtxt(native/"shared_gradient.txt"),
                                                     np.load(directory/"shared_gradient_lm.npy"))
        comparisons["stale_gradient_override"] = metrics(np.loadtxt(native/"stale_gradient.txt"),
                                                          np.load(directory/"stale_gradient_override.npy"))
        report["native_vs_fnit"][arm] = comparisons
        # Divide out each implementation's own lambda delta. Image-data terms
        # cancel, leaving the regularizer action on exactly the shared coefficients.
        native_delta = (np.loadtxt(native/"stale_gradient.txt") - np.loadtxt(native/"shared_gradient.txt"))
        fnit_delta = (np.load(directory/"stale_gradient_override.npy") - np.load(directory/"shared_gradient.npy"))
        native_unit = native_delta / (scalars["trial_lambda"] - scalars["shared_lambda"])
        fnit_unit = fnit_delta / (record["trial"]["effective_lambda"] - record["shared"]["effective_lambda"])
        report["regularizer_isolation"][arm] = metrics(native_unit,fnit_unit)
    report["file_sha256"] = {str(path.relative_to(args.run)):hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in sorted(args.run.rglob("*")) if path.is_file()
                              and path != args.output and path.suffix in (".gz",".npy",".txt",".json")}
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"native":scalars,"regularizer_isolation":report["regularizer_isolation"],
                      "official_vs_fnit":{arm:{name:group[name] for name in
                          ("state","shared_warped","shared_mask","shared_gradient","shared_gradient_lm",
                           "shared_diagonal","shared_hessian_probe")} for arm,group in report["native_vs_fnit"].items()}}))


if __name__ == "__main__":
    main()
