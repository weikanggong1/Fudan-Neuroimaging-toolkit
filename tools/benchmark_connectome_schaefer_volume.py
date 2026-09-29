"""原 UKB Schaefer 模板到真实 T1 ribbon 标签体积的同输入验证。"""

import argparse
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from connectome_benchmark_common import _sha256
from fnit.connectome.atlas_builder import schaefer_to_t1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("subject-dir", "fsaverage-dir", "left-annot", "right-annot", "official-volume", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--parcels", type=int, choices=(200, 500, 1000), default=200)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda"):
        torch.empty(1, device=args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
    started = time.perf_counter()
    candidate, nodes = schaefer_to_t1(
        subject_dir=args.subject_dir, fsaverage_dir=args.fsaverage_dir,
        left_annot=args.left_annot, right_annot=args.right_annot, device=args.device,
    )
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(args.device)
    seconds = time.perf_counter() - started
    reference = nib.load(str(args.official_volume))
    if candidate.shape != reference.shape or not np.allclose(candidate.affine, reference.affine, atol=1e-5):
        raise ValueError("FNIT and original UKB T1 atlas grids differ")
    actual = np.asarray(candidate.dataobj)
    expected = np.asarray(reference.dataobj)
    report = {
        "input_sha256": {name: _sha256(path) for name, path in (
            ("left_annot", args.left_annot), ("right_annot", args.right_annot),
            ("ribbon", args.subject_dir / "mri/ribbon.mgz"),
            ("official_volume", args.official_volume))},
        "device": args.device, "shape": candidate.shape, "nodes": len(nodes),
        "fnit_seconds": seconds,
        "torch_peak_allocated_gib": (torch.cuda.max_memory_allocated(args.device) / 2**30
                                     if args.device.startswith("cuda") else None),
        "voxel_xor": int(np.count_nonzero(actual != expected)),
        "voxel_count": int(actual.size),
        "nonzero_xor": int(np.count_nonzero((actual > 0) != (expected > 0))),
        "label_min_max": [int(actual.min()), int(actual.max())],
    }
    nib.save(candidate, str(args.output_dir / f"fnit_schaefer{args.parcels}_t1.nii.gz"))
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
