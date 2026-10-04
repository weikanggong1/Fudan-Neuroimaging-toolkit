"""Compare captured real official/FNIT arrays without publishing image data."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_outputs import scalar_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("preserve prior comparison reports")
    reference = json.loads((args.reference_dir / "control.private.json").read_text())
    candidate = json.loads((args.candidate_dir / "control.private.json").read_text())
    comparisons = {}
    for stage, original_name, candidate_name in (
        ("conform", "conform_0.private.npy", "conform.private.npy"),
        ("normalized_network_input", "network_input_0.private.npy", "network_input.private.npy"),
    ):
        first = np.load(args.reference_dir / original_name, mmap_mode="r")
        second = np.load(args.candidate_dir / candidate_name, mmap_mode="r")
        row = {"reference_shape": list(first.shape), "candidate_shape": list(second.shape),
               "reference_dtype": str(first.dtype), "candidate_dtype": str(second.dtype)}
        row["shape_matches"] = first.shape == second.shape
        if row["shape_matches"]:
            row["voxelwise"] = scalar_metrics(first, second)
            # Location summaries help identify nearest-neighbour planes without
            # copying any intensity values, filenames or subject identifiers.
            if first.ndim == 3:
                difference = first != second
                row["different_voxels_per_axis_plane"] = [difference.sum(axis=tuple(
                    other for other in range(3) if other != axis)).tolist() for axis in range(3)]
        comparisons[stage] = row
    original_geom = reference["conform_calls"][0]
    candidate_geom = candidate["stages"][0]
    comparisons["conform"]["affine_max_abs_mm"] = float(np.abs(
        np.asarray(original_geom["affine"]) - np.asarray(candidate_geom["affine"])).max())
    report = {"schema": "fnit.smri.cpu.strip.preprocess_comparison.v1",
              "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "source_input_hash_matches": reference["input_sha256"] == candidate["input_sha256"],
              "loaded_array_hash_matches": reference["loaded_input"]["array_sha256"] ==
                    candidate["loaded_input"]["array_sha256"],
              "loaded_dtype_matches": reference["loaded_input"]["dtype"] == candidate["loaded_input"]["dtype"],
              "loaded_affine_max_abs_mm": float(np.abs(np.asarray(reference["loaded_input"]["affine"]) -
                   np.asarray(candidate["loaded_input"]["affine"])).max()),
              "reference_voxel_size": reference["loaded_input"]["voxsize"],
              "candidate_voxel_size": candidate["loaded_input"]["zooms"],
              "comparisons": comparisons,
              "official_network_input_sha256": reference["network_calls"][0]["input_sha256"],
              "candidate_network_input_sha256": candidate["normalized_input"]["array_sha256"]}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "measured", "loaded_array_hash_matches": report["loaded_array_hash_matches"],
                      "comparisons": {name: {key: value for key, value in row.items()
                                               if key != "different_voxels_per_axis_plane"}
                                      for name, row in comparisons.items()}}))


if __name__ == "__main__":
    main()
