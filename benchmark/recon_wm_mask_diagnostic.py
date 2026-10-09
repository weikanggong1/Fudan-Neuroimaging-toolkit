"""对已完成的同输入WM输出做掩膜诊断，不修改候选或重新计时。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def main():
    """读取三份同网格uint8 WM，写逐标签之外的>=5掩膜Dice及完整差异。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True,
                        help="已完成完整配对目录，含sub-*/full_wm_planar_a100_v4")
    parser.add_argument("--case", choices=("sub-07", "sub-06"), required=True,
                        help="公开冻结测试例标识；不参与候选计算")
    args = parser.parse_args()
    directory = args.output_root / args.case / "full_wm_planar_a100_v4"
    files = {name: directory / filename for name, filename in
             (("native", "0-native.mgz"), ("python", "1-python.mgz"),
              ("cached", "2-cached.mgz"))}
    images = {name: nib.load(str(path)) for name, path in files.items()}
    arrays = {name: np.asarray(image.dataobj) for name, image in images.items()}
    reference = images["native"]
    for name, image in images.items():
        if (arrays[name].ndim != 3 or arrays[name].dtype != np.uint8 or
                image.shape != reference.shape or
                not np.array_equal(image.affine, reference.affine)):
            raise ValueError("all WM inputs must be matching 3D uint8 voxel grids")

    def compare(a, b):
        # 与生产WM二值转换相同：uint8强度/特殊标签>=5为白质。
        first, second = arrays[a] >= 5, arrays[b] >= 5
        overlap = int(np.count_nonzero(first & second))
        first_count, second_count = int(first.sum()), int(second.sum())
        error = np.abs(arrays[a].astype(np.int16) - arrays[b].astype(np.int16))
        return {"wm_definition": "uint8 WM >=5; source binary conversion",
                "wm_different_voxels": int(np.count_nonzero(first != second)),
                "wm_dice": (2 * overlap / (first_count + second_count)
                            if first_count + second_count else 1.0),
                "a_wm_voxels": first_count, "b_wm_voxels": second_count,
                "full_uint8_different_voxels": int(np.count_nonzero(error)),
                "full_uint8_max_abs": int(error.max()),
                "full_uint8_p99_abs": float(np.percentile(error, 99))}

    report = {
        "scope": "post-timing same-output WM mask diagnostic; not whole recon",
        "case": args.case,
        "files": {name: {"name": path.name,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                  for name, path in files.items()},
        "cached_vs_python": compare("cached", "python"),
        "cached_vs_native": compare("cached", "native"),
        "python_vs_native": compare("python", "native"),
        "whole_metric_equivalence": "not_assessed",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output = directory / "wm_mask_post_timing.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["cached_vs_native"]), flush=True)


if __name__ == "__main__":
    main()
