"""在相同 conform 网格上按离散标签报告 Dice，包含最差局部脑区。"""

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


FILES = ("aseg.mgz", "aparc+aseg.mgz", "aparc.a2009s+aseg.mgz",
         "aparc.DKTatlas+aseg.mgz", "wmparc.mgz", "ribbon.mgz", "filled.mgz")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--label-table", type=Path, required=True)
    args = parser.parse_args()
    names = {}
    for line in args.label_table.read_text().splitlines():
        fields = line.split()
        if fields and fields[0].isdigit():
            names[int(fields[0])] = fields[1]
    result = {}
    for name in FILES:
        paths = [root / "mri" / name for root in (args.reference, args.candidate)]
        images = [nib.load(str(path)) for path in paths]
        a, b = [np.asarray(image.dataobj) for image in images]
        if a.shape != b.shape or not np.allclose(images[0].affine, images[1].affine,
                                                atol=1e-6, rtol=0):
            raise ValueError(f"conform grids differ: {name}")
        if not all(np.isfinite(array).all() and array.min() >= 0 and
                   np.array_equal(array, np.floor(array))
                   for array in (a, b)):
            raise ValueError(f"nonnegative integer labels required: {name}")
        dtypes = [str(a.dtype), str(b.dtype)]
        a, b = a.astype(np.int64), b.astype(np.int64)
        ref = np.bincount(a.ravel())
        got = np.bincount(b.ravel())
        common = np.bincount(a[a == b])
        rows = {}
        for label in np.union1d(np.flatnonzero(ref), np.flatnonzero(got)):
            if label == 0:
                continue
            first = int(ref[label]) if label < len(ref) else 0
            second = int(got[label]) if label < len(got) else 0
            overlap = int(common[label]) if label < len(common) else 0
            rows[str(label)] = {"name": names.get(int(label)),
                                "reference_voxels": first, "candidate_voxels": second,
                                "intersection_voxels": overlap,
                                "dice": 2 * overlap / (first + second)}
        values = np.array([row["dice"] for row in rows.values()])
        result[name] = {"different_voxels": int(np.count_nonzero(a != b)),
                        "stored_dtypes": dtypes,
                        "minimum_dice": float(values.min()),
                        "p05_dice": float(np.quantile(values, .05)),
                        "median_dice": float(np.median(values)),
                        "worst_labels": sorted(rows, key=lambda key: rows[key]["dice"])[:10],
                        "per_label": rows,
                        "input_sha256": [hashlib.sha256(path.read_bytes()).hexdigest()
                                         for path in paths]}
    report = {"code_commit": args.code_commit, "reference": str(args.reference),
              "candidate": str(args.candidate), "files": result,
              "space": "same conform grid; label 0 excluded from summaries",
              "label_table_sha256": hashlib.sha256(args.label_table.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "equivalence_status": "not_assessed; no prospective ROI gates supplied"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: {field: value for field, value in row.items()
                           if field in ("minimum_dice", "p05_dice", "median_dice")}
                      for key, row in result.items()}))


if __name__ == "__main__":
    main()
