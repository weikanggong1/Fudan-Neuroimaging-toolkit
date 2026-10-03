"""Read-only saved-array and fitted-parameter diagnostics for real EDDY arms."""
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
    parser.add_argument("--run", action="append", required=True, help="label=EDDY output directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reference = nib.load(str(args.reference / "data.nii.gz"))
    expected_parameters = np.loadtxt(args.reference / "data.eddy_parameters")
    frames = reference.shape[-1]
    if expected_parameters.shape != (frames, 16) or not np.isfinite(expected_parameters).all():
        raise ValueError("invalid official parameter reference")
    report = {"scope": "saved real-data solver outputs; same official reference, read-only",
              "script_SHA256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "parameter_columns": ["tx_mm", "ty_mm", "tz_mm", "rx_rad", "ry_rad", "rz_rad",
                  "ec_x_Hz_per_mm", "ec_y_Hz_per_mm", "ec_z_Hz_per_mm", "ec_xx_Hz_per_mm2",
                  "ec_yy_Hz_per_mm2", "ec_zz_Hz_per_mm2", "ec_xy_Hz_per_mm2",
                  "ec_xz_Hz_per_mm2", "ec_yz_Hz_per_mm2", "ec_constant_Hz"], "runs": {}}
    for spec in args.run:
        label, folder = spec.split("=", 1)
        path = Path(folder)
        image = nib.load(str(path / "data.nii.gz"))
        data = np.asarray(image.dataobj)
        parameters = np.loadtxt(path / "data.eddy_parameters")
        gradients = np.loadtxt(path / "data.eddy_rotated_bvecs")
        outliers = np.loadtxt(path / "data.eddy_outlier_map", skiprows=1)
        geometry_ok = image.shape == reference.shape and np.allclose(image.affine, reference.affine, atol=1e-5, rtol=0)
        shapes_ok = (parameters.shape == (frames, 16) and gradients.shape == (3, frames)
                     and outliers.shape == (frames, reference.shape[2]))
        finite = all(np.isfinite(a).all() for a in (data, parameters, gradients, outliers))
        if not geometry_ok or not shapes_ok or not finite:
            raise ValueError(f"invalid saved solver arrays: {label}")
        delta = parameters - expected_parameters
        norms = np.linalg.norm(gradients, axis=0)
        nonzero = norms > 0
        report["runs"][label] = {"all_finite": True, "geometry_and_shapes_match": True,
            "parameter_RMSE_by_column": np.sqrt(np.mean(delta * delta, axis=0)).tolist(),
            "parameter_max_abs_error_by_column": np.max(np.abs(delta), axis=0).tolist(),
            "parameter_abs_max_by_column": np.max(np.abs(parameters), axis=0).tolist(),
            "bvec_max_nonzero_norm_deviation": float(np.max(np.abs(norms[nonzero]-1))),
            "zero_bvec_count": int((~nonzero).sum()), "outlier_count": int(outliers.sum())}
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
