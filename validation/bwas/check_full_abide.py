"""Verify complete ABIDE I+II BWAS outputs without publishing subject-level data."""

import argparse
import csv
import gzip
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mask-file", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root
    report = json.loads((output / "validation_summary.public.json").read_text())
    assert (report["matched_subjects"], report["subjects"], report["common_gray_voxels"]) == (
        1778, 1748, 112215)
    assert report["all_unordered_voxel_pairs"] == 6296047005
    assert report["cases"] == 792 and report["controls"] == 956
    assert report["sites"] == 36 and report["cdt"] == 5.0
    assert report["peak_cuda_allocated_bytes"] < 20 * 1024**3
    assert np.isfinite(report["fwhm_voxels"]) and report["fwhm_voxels"] >= 2
    assert (output / "dataset_description.json").is_file()

    group = output / "group" / "func"
    metadata_file, = group.glob("*_desc-BWASMA_statmap.json")
    ma_file, = group.glob("*_desc-BWASMA_statmap.nii.gz")
    edge_file, = group.glob("*_desc-BWASedges_relmat.tsv.gz")
    cluster_file, = group.glob("*_desc-BWASclusters_stat.tsv")
    metadata = json.loads(metadata_file.read_text())
    assert len(metadata["Participants"]) == len(set(metadata["Participants"])) == 1748
    assert metadata["VoxelCount"] == 112215
    assert metadata["SuprathresholdEdgeCount"] == report["suprathreshold_edges"]
    assert metadata["FWHMInVoxels"] == report["fwhm_voxels"]
    assert np.isfinite(metadata["ElapsedSeconds"])

    mask = nib.load(args.mask_file)
    image = nib.load(ma_file)
    values = image.get_fdata(dtype=np.float32)
    assert image.shape == mask.shape and np.allclose(image.affine, mask.affine)
    assert np.isfinite(values).all() and np.all(values >= 0)
    with cluster_file.open(newline="") as stream:
        clusters = list(csv.DictReader(stream, delimiter="\t"))
    with gzip.open(edge_file, "rt", newline="") as stream:
        edge_count = sum(1 for _ in csv.DictReader(stream, delimiter="\t"))
    assert edge_count == report["suprathreshold_edges"]
    assert sum(int(row["edges"]) for row in clusters) == edge_count
    checked = {"subjects": 1748, "gray_voxels": 112215,
               "all_unordered_pairs": 6296047005,
               "suprathreshold_edges": edge_count,
               "clusters": len(clusters), "ma_nonzero_voxels": int(np.count_nonzero(values)),
               "peak_cuda_allocated_bytes": report["peak_cuda_allocated_bytes"],
               "elapsed_seconds": report["fnit_elapsed_seconds"]}
    print(json.dumps(checked, indent=2))


if __name__ == "__main__":
    main()
