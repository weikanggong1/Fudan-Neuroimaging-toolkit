"""Measure how much official subregion anatomy a coarse merge mask can retain."""

from __future__ import annotations

import argparse
import json

import nibabel as nib
import numpy as np
from scipy import ndimage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aseg", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--coarse-ids", required=True)
    parser.add_argument("--reference-min", required=True, type=int)
    parser.add_argument("--reference-max", required=True, type=int)
    args = parser.parse_args()
    coarse = np.asarray(nib.load(args.aseg).dataobj, dtype=np.int32)
    reference = np.asarray(nib.load(args.reference).dataobj, dtype=np.int32)
    roi = (reference >= args.reference_min) & (reference <= args.reference_max)
    base = np.isin(coarse, [int(value) for value in args.coarse_ids.split(",")])
    retained = {}
    for steps in (0, 2, 3, 5, 7):
        support = ndimage.binary_dilation(base, iterations=steps) if steps else base
        retained[str(steps)] = float(np.count_nonzero(roi & support) / max(roi.sum(), 1))
    print(json.dumps({"reference_voxels": int(roi.sum()), "fraction_inside_support": retained}))


if __name__ == "__main__":
    main()
