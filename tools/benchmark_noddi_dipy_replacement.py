"""Compare DIPY-backed and standalone AMICO-NODDI mathematics on real DWI."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import nibabel as nib
import numpy as np

from fnit.amico_noddi.core import AMICONODDIConfig
from fnit.amico_noddi.kernels import (
    _rotation_auxiliary,
    _subject_basis,
    amico_scheme,
    build_noddi_kernels,
    direction_indices,
    load_raw_bvecs,
    principal_directions,
)


def _compare(actual, expected):
    same = np.array_equal(actual, expected)
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return {
        "exact_equal": bool(same),
        "different_elements": int(np.count_nonzero(actual != expected)),
        "total_elements": int(actual.size),
        "mae": float(np.mean(difference)),
        "max_abs": float(np.max(difference)),
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--dwi", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--bvals", type=Path, required=True)
    parser.add_argument("--bvecs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("reference", "candidate"), required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dwi = nib.load(str(args.dwi))
    values = np.asarray(dwi.dataobj, dtype=np.float32)
    mask = np.asarray(nib.load(str(args.mask)).dataobj, dtype=np.uint8) == 1
    if values.shape[:3] != mask.shape:
        raise ValueError("DWI and mask spatial dimensions differ")
    flat = np.flatnonzero(mask.reshape(-1))
    bvals = np.loadtxt(args.bvals, dtype=np.float64).reshape(-1)
    bvecs = load_raw_bvecs(args.bvecs, bvals.size)
    config = AMICONODDIConfig()
    raw, b0, shells = amico_scheme(
        bvals, bvecs, b0_threshold=config.b0_threshold, b_step=config.b_step
    )
    baseline = np.mean(values[..., b0], axis=3)
    baseline[baseline <= 0] = 1
    signal = values.reshape(-1, values.shape[3])[flat] / baseline.reshape(-1)[flat, None]
    signal = np.maximum(signal.astype(np.float64), 0)
    t0 = perf_counter()
    directions = principal_directions(signal, raw)
    direction_seconds = perf_counter() - t0
    indices = direction_indices(directions)
    t0 = perf_counter()
    fit, rotated, constants, m0 = _rotation_auxiliary()
    rotation_seconds = perf_counter() - t0
    t0 = perf_counter()
    ordered_indices, basis = _subject_basis(raw, shells, b0)
    basis_seconds = perf_counter() - t0
    t0 = perf_counter()
    kernels = build_noddi_kernels(
        bvals, bvecs, config.ic_ods, config.ic_vfs,
        d_par=config.d_par, d_iso=config.d_iso,
        b0_threshold=config.b0_threshold, b_step=config.b_step,
    )
    kernel_seconds = perf_counter() - t0
    arrays = {
        "directions": directions,
        "lut_indices": indices,
        "rotation_fit": fit,
        "rotation_values": rotated,
        "rotation_constants": constants,
        "rotation_m0": m0,
        "subject_ordered_indices": ordered_indices,
        "subject_basis": basis,
        "wm": kernels["wm"],
        "iso": kernels["iso"],
        "norms": kernels["norms"],
    }
    output = args.output_dir / f"{args.mode}_arrays.internal.npz"
    np.savez(output, **arrays)
    metrics = None
    if args.mode == "candidate":
        with np.load(args.output_dir / "reference_arrays.internal.npz") as reference:
            metrics = {name: _compare(value, reference[name])
                       for name, value in arrays.items()}
    report = {
        "source_type": args.mode,
        "input_shape": list(values.shape),
        "mask_voxels": len(flat),
        "b0_count": int(b0.sum()),
        "dwi_count": int((~b0).sum()),
        "shells": shells.tolist(),
        "timing_seconds": {
            "direction": direction_seconds,
            "rotation_auxiliary": rotation_seconds,
            "subject_basis": basis_seconds,
            "full_kernel": kernel_seconds,
        },
        "metrics": metrics,
    }
    (args.output_dir / f"{args.mode}_report.internal.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
