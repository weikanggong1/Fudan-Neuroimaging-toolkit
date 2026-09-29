"""比较同一体素网格上的官方与 FNIT 核团硬标签。"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def compare_nuclei(reference: str | Path, candidate: str | Path) -> dict:
    """返回每个非零标签的 Dice 和相对硬体积差。"""
    ref_image, got_image = nib.load(str(reference)), nib.load(str(candidate))
    if ref_image.shape != got_image.shape or not np.allclose(ref_image.affine, got_image.affine, atol=1e-5):
        raise ValueError("两幅标签图须位于相同体素网格")
    ref = np.asarray(ref_image.dataobj, dtype=np.int32)
    got = np.asarray(got_image.dataobj, dtype=np.int32)
    rows = []
    for label in np.union1d(np.unique(ref), np.unique(got)):
        if label == 0:
            continue
        a, b = ref == label, got == label
        na, nb = int(a.sum()), int(b.sum())
        dice = 2 * int(np.count_nonzero(a & b)) / (na + nb) if na + nb else 1.0
        difference = abs(nb - na) / na if na else float("inf")
        rows.append({"label": int(label), "reference_voxels": na, "fnit_voxels": nb,
                     "dice": dice, "volume_difference": difference,
                     "accepted": dice >= 0.95 and difference <= 0.05})
    a, b = ref != 0, got != 0
    return {"reference": str(reference), "candidate": str(candidate), "labels": len(rows),
            "accepted": sum(row["accepted"] for row in rows),
            "foreground_dice": 2 * int(np.count_nonzero(a & b)) / (int(a.sum()) + int(b.sum())),
            "different_voxels": int(np.count_nonzero(ref != got)), "regions": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = compare_nuclei(reference=args.reference, candidate=args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"{report['accepted']}/{report['labels']} 标签达标；前景 Dice={report['foreground_dice']:.6f}")


if __name__ == "__main__":
    main()
