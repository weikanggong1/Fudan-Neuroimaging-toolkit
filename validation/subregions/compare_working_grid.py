"""Compare one FNIT working T1 grid to the saved official resampling stage."""

from __future__ import annotations

import argparse
import json

import nibabel as nib
import numpy as np

from fnit.gems.context import SubregionContext
from fnit.gems.recipes.hippo_amygdala import HippoAmygdalaRecipe
from fnit.gems.recipes.thalamus import ThalamusRecipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t1", required=True)
    parser.add_argument("--aseg", required=True)
    parser.add_argument("--official-image", required=True)
    parser.add_argument("--structure", choices=("thalamus", "hippo-amygdala-left",
                                                "hippo-amygdala-right"), required=True)
    args = parser.parse_args()
    context = SubregionContext.prepare(args.t1, need_coarse=True, need_parc=False,
                                       coarse_segmentation=args.aseg)
    recipe = (ThalamusRecipe("thalamus", ".") if args.structure == "thalamus" else
              HippoAmygdalaRecipe(args.structure.rsplit("-", 1)[-1], "."))
    ours, _, _ = recipe.prepare_working_image(context)
    official = nib.load(args.official_image)
    report = {"fnit_shape": list(map(int, ours.shape)),
              "official_shape": list(map(int, official.shape)),
              "max_affine_difference": float(np.max(np.abs(ours.affine - official.affine)))}
    if ours.shape == official.shape and np.allclose(ours.affine, official.affine, atol=1e-4):
        first = np.asarray(ours.dataobj)
        second = np.asarray(official.dataobj)
        common = (first > 0) & (second > 0)
        report["joint_nonzero_voxels"] = int(common.sum())
        report["mean_absolute_intensity_difference"] = float(np.abs(first[common] - second[common]).mean())
        report["intensity_correlation"] = float(np.corrcoef(first[common], second[common])[0, 1])
    print(json.dumps(report))


if __name__ == "__main__":
    main()
