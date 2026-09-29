"""比较真实 GMWMI 种子在 2/4/8/16 mm RAS 网格上的空间分布。"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--official", type=Path, required=True, help="第一份 MRtrix Seedtest -output_seeds CSV")
parser.add_argument("--official-repeat", type=Path, required=True,
                    help="第二份独立 MRtrix Seedtest -output_seeds CSV")
parser.add_argument("--fnit", type=Path, required=True, help="FNIT float32 [N,3] RAS-mm 种子 NPY")
parser.add_argument("--output", type=Path, required=True, help="输入哈希、样本量和多尺度配对指标 JSON")
args = parser.parse_args()
paths = {"official_0": args.official, "official_1": args.official_repeat, "fnit": args.fnit}
points = {
    name: (np.load(path).astype(np.float64) if name == "fnit" else
           np.loadtxt(path, delimiter=",", comments="#", usecols=(2, 3, 4)))
    for name, path in paths.items()
}
minimum = np.min(np.concatenate(list(points.values())), axis=0)
maximum = np.max(np.concatenate(list(points.values())), axis=0)
report = {
    "input_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                     for name, path in paths.items()},
    "counts": {name: len(value) for name, value in points.items()},
    "binning": "RAS-mm histogram; edges aligned to integer multiples of bin width; Pearson on union nonzero cells",
    "widths_mm": {},
}
for width in (2, 4, 8, 16):
    bins = [np.arange(np.floor(minimum[axis] / width) * width,
                      np.ceil(maximum[axis] / width) * width + width * 1.01, width)
            for axis in range(3)]
    hist = {name: np.histogramdd(value, bins=bins)[0].ravel()
            for name, value in points.items()}
    scores = {}
    for left, right in (("official_0", "official_1"),
                        ("official_0", "fnit"), ("official_1", "fnit")):
        a, b = hist[left], hist[right]
        union = (a + b) > 0
        scores[left + "_vs_" + right] = {
            "pearson": float(np.corrcoef(a[union], b[union])[0, 1]),
            "total_variation": float(0.5 * np.abs(a/a.sum() - b/b.sum()).sum()),
            "shared_nonzero_bins": int(((a > 0) & (b > 0)).sum()),
            "union_nonzero_bins": int(union.sum()),
        }
    report["widths_mm"][str(width)] = scores
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report["widths_mm"], indent=2))
