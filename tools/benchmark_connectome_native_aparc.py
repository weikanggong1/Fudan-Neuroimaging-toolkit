"""真实 recon-all aparc/a2009s 到 T1 atlas 的原 UKB 同输入对照。"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from fnit.connectome.atlas_builder import native_annotation_to_t1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-dir", type=Path, required=True)
    parser.add_argument("--annotation", choices=("aparc", "aparc.a2009s"), required=True)
    parser.add_argument("--official-volume", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.device.startswith("cuda"):
        torch.empty(1, device=args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
    start = time.perf_counter()
    candidate, nodes = native_annotation_to_t1(
        subject_dir=args.subject_dir, annotation=args.annotation, device=args.device,
    )
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(args.device)
    seconds = time.perf_counter() - start
    reference = nib.load(str(args.official_volume))
    if candidate.shape != reference.shape or not np.allclose(candidate.affine, reference.affine, atol=1e-5):
        raise ValueError("FNIT and original UKB T1 atlas grids differ")
    actual, expected = np.asarray(candidate.dataobj), np.asarray(reference.dataobj)
    files = {
        "left_annot": args.subject_dir / "label" / f"lh.{args.annotation}.annot",
        "right_annot": args.subject_dir / "label" / f"rh.{args.annotation}.annot",
        "ribbon": args.subject_dir / "mri/ribbon.mgz",
        "official_volume": args.official_volume,
    }
    report = {
        "annotation": args.annotation,
        "input_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for name, path in files.items()},
        "shape": list(candidate.shape),
        "nodes": len(nodes),
        "voxel_xor": int(np.count_nonzero(actual != expected)),
        "nonzero_xor": int(np.count_nonzero((actual > 0) != (expected > 0))),
        "label_min_max": [int(actual.min()), int(actual.max())],
        "fnit_seconds": seconds,
        "torch_peak_allocated_gib": (torch.cuda.max_memory_allocated(args.device) / 2**30
                                     if args.device.startswith("cuda") else None),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    nib.save(candidate, str(args.output_dir / "atlas_t1.nii.gz"))
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
