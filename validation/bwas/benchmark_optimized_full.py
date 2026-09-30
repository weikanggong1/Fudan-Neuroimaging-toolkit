"""Run all voxel pairs and compare private BWAS artifacts to a fixed baseline."""

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy
import torch

from fnit.bwas import core


def compare_outputs(reference_root, candidate_root):
    def files(root, suffix):
        matches = list((root / "group" / "func").glob(f"*{suffix}"))
        if len(matches) != 1:
            raise ValueError(f"expected one {suffix} in {root}")
        return matches[0]

    def edges(root):
        values = np.loadtxt(files(root, "BWASedges_relmat.tsv.gz"),
                            delimiter="\t", skiprows=1, ndmin=2)
        order = np.lexsort(tuple(values[:, column] for column in range(5, -1, -1)))
        return values[order]

    reference, candidate = edges(reference_root), edges(candidate_root)
    same_edges = reference.shape == candidate.shape and np.array_equal(
        reference[:, :6], candidate[:, :6])
    comparison = {"same_edge_list_and_CDT_decisions": same_edges,
                  "reference_edges": len(reference), "candidate_edges": len(candidate)}
    if same_edges:
        comparison["exact_z"] = bool(np.array_equal(reference[:, 6], candidate[:, 6]))
        error = np.zeros(len(reference))
        np.subtract(reference[:, 6], candidate[:, 6], out=error,
                    where=reference[:, 6] != candidate[:, 6])
        np.abs(error, out=error)
        comparison["z_max_absolute_error"] = float(error.max(initial=0))
        comparison["z_mean_absolute_error"] = float(error.mean()) if len(error) else 0.0
        probability_error = np.abs(scipy.stats.norm.cdf(reference[:, 6]) -
                                   scipy.stats.norm.cdf(candidate[:, 6]))
        comparison["cdf_max_absolute_error"] = float(probability_error.max(initial=0))
        pairs = np.unique(np.column_stack((reference[:, 7], candidate[:, 7])), axis=0)
        comparison["same_cluster_partition"] = bool(
            len(pairs) == len(np.unique(pairs[:, 0])) == len(np.unique(pairs[:, 1])))
        first = np.loadtxt(files(reference_root, "BWASclusters_stat.tsv"),
                           delimiter="\t", skiprows=1, ndmin=2)
        second = np.loadtxt(files(candidate_root, "BWASclusters_stat.tsv"),
                            delimiter="\t", skiprows=1, ndmin=2)
        if comparison["same_cluster_partition"]:
            second = second[pairs[:, 1].astype(int)-1]
            comparison["same_cluster_sizes"] = bool(np.array_equal(first[:, 1], second[:, 1]))
            comparison["same_cluster_p_values"] = bool(np.array_equal(first[:, 2:4], second[:, 2:4]))
            comparison["same_cluster_endpoint_counts"] = bool(np.array_equal(first[:, 5:], second[:, 5:]))
            peak_error = np.abs(first[:, 4]-second[:, 4])
            comparison["cluster_max_z_absolute_error"] = float(peak_error.max(initial=0))
        comparison["same_order_and_cluster_ids"] = bool(np.array_equal(reference[:, 7], candidate[:, 7]))
    first_ma = nib.load(files(reference_root, "BWASMA_statmap.nii.gz"))
    second_ma = nib.load(files(candidate_root, "BWASMA_statmap.nii.gz"))
    comparison["same_MA_map"] = bool(np.array_equal(np.asarray(first_ma.dataobj),
                                                      np.asarray(second_ma.dataobj)))
    comparison["same_MA_affine"] = bool(np.array_equal(first_ma.affine, second_ma.affine))
    return comparison


def runtime_versions():
    return {"numpy": np.__version__, "scipy": scipy.__version__,
            "torch": torch.__version__, "cuda": torch.version.cuda}


def parity_passed(comparison):
    required = ("same_edge_list_and_CDT_decisions", "same_cluster_partition",
                "same_cluster_sizes", "same_cluster_p_values", "same_cluster_endpoint_counts",
                "same_MA_map", "same_MA_affine")
    return (all(comparison.get(key, False) for key in required) and
            comparison.get("z_max_absolute_error", float("inf")) <= 1e-4)


def main():
    parser = argparse.ArgumentParser()
    for name in ("bids-root", "participants", "mask", "cache-dir",
                 "baseline-root", "output-root", "summary"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--packed-cache-dir", type=Path,
                        help="reuse packed cache; omit to tune subjects and build shards")
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--subject-block-size", type=int)
    parser.add_argument("--column-tiles", type=int, choices=(1, 2))
    parser.add_argument("--gpu-row-cache", action=argparse.BooleanOptionalAction,
                        default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    metadata_file, = (args.baseline_root / "group" / "func").glob("*BWASMA_statmap.json")
    reference = json.loads(metadata_file.read_text())
    result = core.run_bwas(args.bids_root, args.participants, args.mask, args.output_root,
        phenotype=reference["Phenotype"], covariates=tuple(reference["Covariates"]),
        cdt=reference["CDT"], fwhm=reference["FWHMInVoxels"], device=args.device,
        block_size=args.block_size, subject_block_size=args.subject_block_size,
        _prepared_cache_dir=args.cache_dir, _prepared_packed_cache_dir=args.packed_cache_dir,
        column_tiles=args.column_tiles, gpu_row_cache=args.gpu_row_cache)
    metadata = json.loads(result.metadata.read_text())
    summary = {"scope": "all unordered voxel pairs", "subjects": result.subjects,
        "runtime_versions": runtime_versions(),
        "optimized_core_sha256": hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        "voxels": result.voxels, "unordered_pairs": result.voxels*(result.voxels-1)//2,
        "elapsed_seconds": metadata["ElapsedSeconds"],
        "peak_allocated_bytes": metadata["PeakCUDAAllocatedBytes"],
        "peak_reserved_bytes": metadata["PeakCUDAReservedBytes"],
        "block_size": metadata["VoxelBlockSize"],
        "subject_block_size": metadata["SubjectBlockSize"],
        "glm_subject_block_size": metadata["GLMSubjectBlockSize"],
        "column_tiles": metadata["ColumnTilesPerRowBatch"],
        "gpu_row_cache": metadata.get("GPUResidentRowBlock", args.gpu_row_cache),
        "packed_cache_build_seconds": metadata["PackedCacheBuildSeconds"],
        "stage_seconds": metadata["StageSeconds"], "block_tuning": metadata["BlockTuning"],
        "comparison": compare_outputs(args.baseline_root, args.output_root)}
    args.summary.write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary), flush=True)
    if not parity_passed(summary["comparison"]):
        raise SystemExit("full BWAS parity failed; see aggregate comparison")


if __name__ == "__main__":
    main()
