"""Select compact JHU white-matter seed masks on a subject's native DWI grid."""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt


REGIONS = {
    "genu_cc": 3,
    "cst_right": 7,
    "cst_left": 8,
    "slf_right": 41,
    "slf_left": 42,
}
OFFSETS = np.array([(0, 0, 0), (1, 0, 0), (-1, 0, 0),
                    (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--fa", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    labels_img = nib.load(args.labels)
    mask_img = nib.load(args.mask)
    fa_img = nib.load(args.fa)
    for image in (labels_img, fa_img):
        if image.shape != mask_img.shape or not np.allclose(image.affine, mask_img.affine, atol=1e-4):
            raise ValueError("labels, FA and BEDPOSTX mask must share the native DWI grid")
    labels = np.asarray(labels_img.dataobj).astype(np.int16)
    mask = np.asarray(mask_img.dataobj) > 0
    fa = np.asarray(fa_img.dataobj)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, label in REGIONS.items():
        region = (labels == label) & mask & (fa >= 0.3)
        distance = distance_transform_edt(region)
        candidates = np.argwhere(distance >= np.sqrt(2))
        if not len(candidates):
            raise ValueError(f"no 7-voxel interior seed for {name}: label {label}")
        scores = distance[tuple(candidates.T)] + 0.01 * fa[tuple(candidates.T)]
        center = candidates[np.argmax(scores)]
        points = center + OFFSETS
        seed = np.zeros(mask.shape, dtype=np.uint8)
        seed[tuple(points.T)] = 1
        if not np.all(region[tuple(points.T)]):
            raise AssertionError(f"selected seed leaves {name}")
        path = out / f"seed_{name}.nii.gz"
        nib.save(nib.Nifti1Image(seed, mask_img.affine, mask_img.header), path)
        report[name] = {"jhu_label": label, "center_voxel": center.tolist(),
                        "seed_voxels": int(seed.sum()), "median_fa": float(np.median(fa[seed > 0])),
                        "atlas_region_voxels_in_mask_and_fa_threshold": int(region.sum())}
    (out / "seed_provenance.private.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
