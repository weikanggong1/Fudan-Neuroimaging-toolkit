"""Compare the GPU 5TT/GMWMI operators with MRtrix on one fixed FreeSurfer input."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch

from fnit.connectome.anatomy import freesurfer_five_tissue, gmwmi_from_five_tissue


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--aparc-aseg", type=Path, required=True)
    parser.add_argument("--reference-5tt", type=Path, required=True)
    parser.add_argument("--reference-gmwmi", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    paths = (args.aparc_aseg, args.reference_5tt, args.reference_gmwmi)
    images = [nib.load(str(path)) for path in paths]
    arrays = [np.asarray(image.dataobj) for image in images]
    assert arrays[0].shape + (5,) == arrays[1].shape
    assert arrays[0].shape == arrays[2].shape
    assert all(np.array_equal(images[0].affine, image.affine) for image in images[1:])
    labels = torch.as_tensor(arrays[0].astype(np.int32), device=args.device)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    five_tissue = freesurfer_five_tissue(labels)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    t_five = time.perf_counter() - start
    start = time.perf_counter()
    gmwmi = gmwmi_from_five_tissue(five_tissue)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    t_seed = time.perf_counter() - start
    output_5tt = five_tissue.cpu().numpy()
    output_seed = gmwmi.cpu().numpy()
    reference_5tt = arrays[1]
    reference_seed = arrays[2]
    difference_5tt = np.abs(output_5tt - reference_5tt)
    difference_seed = np.abs(output_seed - reference_seed)
    support_a = output_seed > 0
    support_b = reference_seed > 0
    dice = 2 * np.count_nonzero(support_a & support_b) / (np.count_nonzero(support_a) + np.count_nonzero(support_b))
    report = {
        "input_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "shape": list(arrays[0].shape),
        "device": args.device,
        "peak_torch_cuda_allocated_gib": (torch.cuda.max_memory_allocated() / 2**30
                                          if args.device.startswith("cuda") else None),
        "peak_torch_cuda_reserved_gib": (torch.cuda.max_memory_reserved() / 2**30
                                         if args.device.startswith("cuda") else None),
        "tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "5tt_mismatched_values": int(np.count_nonzero(difference_5tt)),
        "5tt_mismatched_voxels": int(np.count_nonzero(np.any(difference_5tt != 0, axis=-1))),
        "5tt_max_abs_difference": float(difference_5tt.max()),
        "gmwmi_mismatched_values": int(np.count_nonzero(difference_seed)),
        "gmwmi_max_abs_difference": float(difference_seed.max()),
        "gmwmi_mean_abs_difference": float(difference_seed.mean()),
        "gmwmi_support_dice": float(dice),
        "gmwmi_positive_voxels": [int(support_a.sum()), int(support_b.sum())],
        "torch_seconds": {"5tt": t_five, "gmwmi": t_seed},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
