"""Compare TOPUP with fixed real AP/PA inputs and independent official outputs.

No FSL executable is invoked. Output hashes and summary errors can be published;
the input images and output NIfTI files remain in the caller's private directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(candidate, reference, mask):
    x = candidate[mask].astype(np.float64).reshape(-1)
    y = reference[mask].astype(np.float64).reshape(-1)
    if not x.size:
        return {"voxels": 0, "spatial_voxels": 0, "scalar_values": 0,
                "pearson_r": None, "mae": None, "rmse": None,
                "p95_absdiff": None, "max_absdiff": None}
    delta = x - y
    absolute = np.abs(delta)
    varying = x.size > 1 and np.std(x) > 0 and np.std(y) > 0
    return {"voxels": int(x.size), "spatial_voxels": int(np.count_nonzero(mask)),
            "scalar_values": int(x.size),
            "pearson_r": float(np.corrcoef(x, y)[0, 1]) if varying else None,
            "mae": float(absolute.mean()), "rmse": float(np.sqrt(np.mean(delta**2))),
            "p95_absdiff": float(np.percentile(absolute, 95)),
            "max_absdiff": float(absolute.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imain", type=Path, required=True)
    parser.add_argument("--datain", type=Path, required=True)
    parser.add_argument("--official-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--brain-mask", type=Path)
    parser.add_argument("--compare-only", action="store_true",
                        help="summarize existing outputs without repeating reconstruction")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory-limit-bytes", type=int, default=20_000_000_000)
    args = parser.parse_args()
    if args.report.exists() or (args.output_dir.exists() and not args.compare_only):
        raise FileExistsError("output directory and report must be new")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not args.compare_only:
        torch.cuda.set_per_process_memory_fraction(
            args.memory_limit_bytes / torch.cuda.get_device_properties(device).total_memory,
            device)
        torch.cuda.synchronize(device)
    from fnit.topup import TorchTOPUP
    import fnit
    measurement_path = args.output_dir / "measurement.json"
    if args.compare_only:
        measurement = (json.loads(measurement_path.read_text()) if measurement_path.exists()
                       else {"api_read_compute_save_seconds": None, "qc": None})
    else:
        args.output_dir.mkdir(parents=True, mode=0o700)
        started = time.perf_counter()
        result = TorchTOPUP(device=device).run(
            args.imain, args.datain, out=args.output_dir / "fieldmap_out",
            fout=args.output_dir / "fieldmap_fout", iout=args.output_dir / "fieldmap_iout",
            jacout=args.output_dir / "fieldmap_jacout")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        measurement = {"api_read_compute_save_seconds": time.perf_counter() - started,
                       "qc": result.qc}
        # Persist timing before independent reporting so a failed summary
        # does not require repeating a successfully completed reconstruction.
        measurement_path.write_text(json.dumps(measurement, indent=2) + "\n")
    raw_image = nib.load(str(args.imain))
    raw = np.asarray(raw_image.dataobj, dtype=np.float32)
    regions = {"full_fov": np.ones(raw.shape[:3], dtype=bool),
               "fixed_raw_signal": raw.mean(axis=3) > 100}
    if args.brain_mask is not None:
        mask_image = nib.load(str(args.brain_mask))
        if mask_image.shape != raw.shape[:3] or not np.allclose(mask_image.affine, raw_image.affine, atol=1e-5, rtol=0):
            raise ValueError("brain mask must use the input voxel grid")
        regions["fixed_official_brain"] = np.asarray(mask_image.dataobj) > 0.5
    comparisons = {}
    for name in ("fieldmap_fout.nii.gz", "fieldmap_iout.nii.gz",
                 "fieldmap_jacout_01.nii.gz", "fieldmap_jacout_02.nii.gz",
                 "fieldmap_out_fieldcoef.nii.gz"):
        left = nib.load(str(args.output_dir / name))
        right = nib.load(str(args.official_dir / name))
        x, y = np.asarray(left.dataobj, dtype=np.float32), np.asarray(right.dataobj, dtype=np.float32)
        same_shape = x.shape == y.shape
        spatial_output = name in ("fieldmap_fout.nii.gz", "fieldmap_iout.nii.gz")
        affine_equal = bool(np.allclose(left.affine, right.affine, atol=1e-5, rtol=0))
        row = {"shape_equal": same_shape, "affine_equal": affine_equal,
               "finite": bool(np.isfinite(x).all() and np.isfinite(y).all()),
               "candidate_shape": list(x.shape), "reference_shape": list(y.shape),
               "candidate_intent": int(left.header["intent_code"]),
               "reference_intent": int(right.header["intent_code"])}
        row["geometry_gate"] = bool(same_shape and (affine_equal or not spatial_output))
        if row["geometry_gate"] and row["finite"]:
            coefficient = name.endswith("fieldcoef.nii.gz")
            masks = {"all_coefficients": np.ones(x.shape, dtype=bool)} if coefficient else regions
            if "jacout" in name and np.linalg.det(raw_image.affine[:3, :3]) > 0:
                # FSL Jacobians retain canonical storage while fout/iout
                # return to input storage. Only the ROI changes direction.
                masks = {label: mask[::-1] for label, mask in masks.items()}
            row["errors"] = {label: metrics(x, y, mask) for label, mask in masks.items()}
            row["array_equal"] = bool(np.array_equal(x, y))
        comparisons[name] = row
    left_motion = np.loadtxt(args.output_dir / "fieldmap_out_movpar.txt", ndmin=2)
    right_motion = np.loadtxt(args.official_dir / "fieldmap_out_movpar.txt", ndmin=2)
    motion_delta = left_motion - right_motion
    code_root = Path(fnit.__file__).parent
    paths = list((code_root / "topup").glob("*.py")) + list((code_root / "fnirt").glob("*.py"))
    payload = {"schema_version": 1, "subjects": 1,
               "scope": "same real b0 pair, acquisition parameters and b02b0 configuration",
               "input_shape": list(raw.shape), "input_sha256": digest(args.imain),
               "acquisition_sha256": digest(args.datain),
               "region_rule": "fixed raw AP/PA mean >100; optional independently supplied official brain mask >0.5",
               **measurement,
               "timing_available": measurement["api_read_compute_save_seconds"] is not None,
               "torch": torch.__version__, "numpy": np.__version__, "threads": args.threads,
               "memory_limit_bytes": args.memory_limit_bytes,
               "source_sha256": {str(p.relative_to(code_root)): digest(p) for p in sorted(paths)},
               "comparisons": comparisons,
               "motion_max_abs_translation_mm": float(np.max(np.abs(motion_delta[:, :3]))),
               "motion_max_abs_rotation_rad": float(np.max(np.abs(motion_delta[:, 3:]))) }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"event": "complete", "seconds": measurement["api_read_compute_save_seconds"],
                      "field_signal": comparisons["fieldmap_fout.nii.gz"]["errors"]["fixed_raw_signal"]}), flush=True)


if __name__ == "__main__":
    main()
