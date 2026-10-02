"""Compare FNIT and official atlas-to-native geometry on one real subject."""

from __future__ import annotations

import argparse
import json

import nibabel as nib
import numpy as np

from fnit.gems.atlas import GEMSAtlas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t1", required=True)
    parser.add_argument("--official-aligned-atlas", required=True)
    parser.add_argument("--fnit-report", required=True)
    parser.add_argument("--atlas-mesh", required=True)
    parser.add_argument("--structure", default="thalamus")
    args = parser.parse_args()
    t1 = nib.load(args.t1)
    aligned = nib.load(args.official_aligned_atlas)
    with open(args.fnit_report) as stream:
        report = json.load(stream)
    fnit = np.asarray(report["initialization"][args.structure]["atlas_to_native_voxel"])
    official = np.linalg.inv(t1.affine) @ aligned.affine
    atlas = GEMSAtlas.from_freesurfer(args.atlas_mesh)
    points = np.c_[atlas.reference_vertices, np.ones(len(atlas.vertices))]
    delta = np.linalg.norm((points @ official.T - points @ fnit.T)[:, :3], axis=1)
    print(json.dumps({"fnit": fnit.tolist(), "official": official.tolist(),
                      "reference_node_displacement_voxels": {
                          "mean": float(delta.mean()), "p95": float(np.percentile(delta, 95)),
                          "max": float(delta.max())}}))


if __name__ == "__main__":
    main()
