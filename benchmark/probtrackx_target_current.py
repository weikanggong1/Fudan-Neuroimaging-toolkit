"""Compare FSL and FNIT seed-voxel-to-target-ROI counts on real DWI."""

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def _wall(path):
    for line in path.read_text().splitlines():
        if line.startswith("wall_s="):
            return float(line.split("=", 1)[1].split()[0])
    raise ValueError(f"missing wall_s in {path}")


def _stem(path):
    return path.name.removesuffix(".nii.gz").removesuffix(".nii")


def compare(fsl_dir, fnit_dir, seed, target_list):
    reference = nib.load(str(seed))
    seed_mask = np.asarray(reference.dataobj) > 0
    targets = [Path(line) if Path(line).is_absolute() else target_list.parent / line
               for line in target_list.read_text().splitlines() if line.strip()]
    fsl_table = np.loadtxt(fsl_dir / "matrix_seeds_to_all_targets", ndmin=2)
    fnit_table = np.loadtxt(fnit_dir / "matrix_seeds_to_all_targets", ndmin=2)
    expected = (int(seed_mask.sum()), len(targets))
    if fsl_table.shape != expected or fnit_table.shape != expected:
        raise ValueError(f"target table shape differs from {expected}")
    result = {"matrix_shape": list(expected), "targets": {}}
    for col, target in enumerate(targets):
        filename = f"seeds_to_{_stem(target)}.nii.gz"
        arrays = []
        for directory in (fsl_dir, fnit_dir):
            image = nib.load(str(directory / filename))
            if image.shape != reference.shape or not np.allclose(image.affine, reference.affine):
                raise ValueError(f"target map geometry differs: {directory / filename}")
            values = np.asarray(image.dataobj, dtype=np.float64)
            if np.any(values[~seed_mask]):
                raise ValueError(f"target map has nonzero values outside seed: {directory / filename}")
            arrays.append(values[seed_mask])
        a, b = arrays
        if not np.isclose(a.sum(), fsl_table[:, col].sum()) or not np.isclose(
                b.sum(), fnit_table[:, col].sum()):
            raise ValueError(f"target map and text matrix totals disagree: {filename}")
        result["targets"][_stem(target)] = {
            "fsl_sum": float(a.sum()), "fnit_sum": float(b.sum()),
            "fsl_nonzero_seed_voxels": int(np.count_nonzero(a)),
            "fnit_nonzero_seed_voxels": int(np.count_nonzero(b)),
            "seed_voxel_mean_abs_error": float(np.abs(a - b).mean()),
            "seed_voxel_max_abs_error": float(np.abs(a - b).max()),
            "fsl_counts": a.tolist(), "fnit_counts": b.tolist(),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fsl-dir", required=True, type=Path)
    parser.add_argument("--fnit-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=Path)
    parser.add_argument("--target-list", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    report = compare(args.fsl_dir, args.fnit_dir, args.seed, args.target_list)
    report["wall_seconds_including_load_and_write"] = {
        "fsl": _wall(Path(str(args.fsl_dir) + ".time")),
        "fnit": _wall(Path(str(args.fnit_dir) + ".time")),
    }
    report["source_sha256"] = {
        name: hashlib.sha256((args.source_dir / name).read_bytes()).hexdigest()
        for name in ("pipeline.py", "_triton.py", "cli.py", "matrix_io.py", "_fast_counts.py")
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output_json)


if __name__ == "__main__":
    main()
