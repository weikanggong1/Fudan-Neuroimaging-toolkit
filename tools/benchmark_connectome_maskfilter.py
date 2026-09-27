"""Compare PyTorch maskfilter with MRtrix on a real same-grid brain mask.

Reference: ``maskfilter brain_mask.nii.gz erode eroded.nii.gz -npass 2`` and
``maskfilter brain_mask.nii.gz dilate dilated.nii.gz -npass 2``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from fnit.connectome.masks import maskfilter_six_connected
from fnit.connectome.pipeline import _scalar_on_grid


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brain-mask", type=Path, required=True)
    parser.add_argument("--reference-eroded", type=Path, required=True)
    parser.add_argument("--reference-dilated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset-label", default="OpenNeuro ds004666 sub-01 ses-2mm corrected DWI mask")
    args = parser.parse_args()
    paths = {"brain_mask": args.brain_mask, "erode": args.reference_eroded,
             "dilate": args.reference_dilated}
    image = nib.load(str(args.brain_mask))
    device = torch.device(args.device)
    source = _scalar_on_grid(args.brain_mask, image, device, binary=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    results = {}
    for operation in ("erode", "dilate"):
        start = time.perf_counter()
        candidate = maskfilter_six_connected(source, operation, passes=2)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - start
        reference = _scalar_on_grid(paths[operation], image, device, binary=True)
        mismatch = candidate ^ reference
        results[operation] = {
            "seconds_core": seconds,
            "reference_voxels": int(reference.sum()),
            "candidate_voxels": int(candidate.sum()),
            "xor_voxels": int(mismatch.sum()),
            "dice": float(2 * (candidate & reference).sum() /
                          (candidate.sum() + reference.sum())),
        }
    report = {
        "dataset": args.dataset_label,
        "shape": list(source.shape),
        "source_voxels": int(source.sum()),
        "device": str(device),
        "input_sha256": {name: _sha256(path) for name, path in paths.items()},
        "results": results,
        "peak_torch_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / 1024 ** 3
            if device.type == "cuda" else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
