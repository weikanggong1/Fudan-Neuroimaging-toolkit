"""同一真实 T1 上验证 FNIT 已给定 FNIRT coefficient 的 Tian S1 路径。"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from fnit.connectome.atlas_tian import fnirt_tian_to_t1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("t1-brain", "tian-mni", "forward-coefficients", "reference-atlas", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda"):
        torch.empty(1, device=args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
    start = time.perf_counter()
    image = fnirt_tian_to_t1(
        t1_brain=args.t1_brain,
        tian_mni=args.tian_mni,
        forward_coefficients=args.forward_coefficients,
        device=args.device,
    )
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(args.device)
    seconds = time.perf_counter() - start
    reference = nib.load(str(args.reference_atlas))
    if image.shape != reference.shape or not np.allclose(image.affine, reference.affine, atol=1e-5):
        raise ValueError("FNIT and reference atlas must share one T1 grid")
    a, b = np.asarray(image.dataobj), np.asarray(reference.dataobj)
    labels = sorted(set(np.unique(a).tolist()) | set(np.unique(b).tolist()))
    dice = {}
    for label in labels[1:]:
        x, y = a == label, b == label
        dice[str(label)] = float(2 * (x & y).sum() / (x.sum() + y.sum()))
    report = {
        "input_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in (("t1_brain", args.t1_brain), ("tian_mni", args.tian_mni),
                               ("forward_coefficients", args.forward_coefficients),
                               ("reference_atlas", args.reference_atlas))
        },
        "shape": list(image.shape),
        "labels": len(dice),
        "voxel_xor": int(np.count_nonzero(a != b)),
        "per_label_dice_min": min(dice.values()),
        "fnit_seconds_with_io": seconds,
        "peak_torch_allocated_gib": (torch.cuda.max_memory_allocated(args.device) / 2**30
                                     if args.device.startswith("cuda") else None),
    }
    nib.save(image, str(args.output_dir / "tian_s1_t1.nii.gz"))
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
