"""Locate full-stage residuals on a common mask without changing the gate."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import nibabel as nib
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, help="label=EDDY output directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reference = np.asarray(nib.load(str(args.reference / "data.nii.gz")).dataobj, dtype=np.float32)
    mask = np.asarray(nib.load(str(args.mask)).dataobj) > 0
    expected = np.loadtxt(args.reference / "data.eddy_outlier_map", skiprows=1)
    runs = {label: Path(folder) for label, folder in (item.split("=", 1) for item in args.run)}
    differences = {label: np.loadtxt(path / "data.eddy_outlier_map", skiprows=1) != expected
                   for label, path in runs.items()}
    union = np.logical_or.reduce(list(differences.values()))
    common = ~union[:, np.nonzero(mask)[2]].T
    report = {"scope": "diagnostic decomposition only; full-brain gate remains unchanged",
              "script_SHA256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "union_outlier_disagreement_frame_slice": np.argwhere(union).tolist(), "runs": {}}
    for label, path in runs.items():
        data = np.asarray(nib.load(str(path / "data.nii.gz")).dataobj, dtype=np.float32)
        delta = data[mask].astype(np.float64) - reference[mask].astype(np.float64)
        report["runs"][label] = {"brain_RMSE": float(np.sqrt(np.mean(delta * delta))),
            "brain_RMSE_common_slices_without_outlier_disagreement": float(np.sqrt(np.mean(delta[common]**2))),
            "brain_RMSE_by_frame": np.sqrt(np.mean(delta * delta, axis=0)).tolist(),
            "outlier_disagreements_by_frame": differences[label].sum(axis=1).astype(int).tolist(),
            "common_brain_values": int(common.sum())}
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
