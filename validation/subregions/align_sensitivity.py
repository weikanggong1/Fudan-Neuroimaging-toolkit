"""Compare mask preprocessing options for atlas registration on a real subject."""

from __future__ import annotations

import argparse
import json

import nibabel as nib
import numpy as np
from scipy import ndimage
import torch

from fnit.gems.atlas import GEMSAtlas
from fnit.gems.initialize import estimate_mask_affine
from fnit.gems.recipes.base import spherical_neighborhood


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t1", required=True)
    parser.add_argument("--aseg", required=True)
    parser.add_argument("--atlas-dump", required=True)
    parser.add_argument("--atlas-mesh", required=True)
    parser.add_argument("--official-aligned-atlas", required=True)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    image = nib.load(args.t1)
    source = nib.load(args.atlas_dump)
    atlas = GEMSAtlas.from_freesurfer(args.atlas_mesh)
    official = np.linalg.inv(image.affine) @ nib.load(args.official_aligned_atlas).affine
    points = np.c_[atlas.reference_vertices, np.ones(len(atlas.vertices))]
    labels = np.asarray(nib.load(args.aseg).dataobj, dtype=np.int32)
    target = np.isin(labels, [int(value) for value in args.ids.split(",")])
    struct = spherical_neighborhood(1)
    masks = {
        "original": target,
        "closing": ndimage.binary_erosion(ndimage.binary_dilation(target, struct), struct,
                                          border_value=1),
        "opening": ndimage.binary_dilation(ndimage.binary_erosion(target, struct,
                                         border_value=1), struct),
    }
    results = {}
    for name, mask in masks.items():
        matrix, score = estimate_mask_affine(source, image, mask.astype(np.uint8), (1,),
                                             device=args.device)
        delta = np.linalg.norm((points @ official.T - points @ matrix.T)[:, :3], axis=1)
        results[name] = {"mask_dice": score, "mean_node_error_voxels": float(delta.mean()),
                         "p95_node_error_voxels": float(np.percentile(delta, 95))}
    print(json.dumps(results))


if __name__ == "__main__":
    main()
