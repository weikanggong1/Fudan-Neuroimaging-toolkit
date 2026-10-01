"""Measure raw-T1 shared SynthSeg/SynthSeg+ and wmparc-proxy preparation."""

import argparse
import json
from pathlib import Path
from time import monotonic

import nibabel as nib
import numpy as np
import torch

from fnit.gems.context import SubregionContext
from fnit.weights import MODEL_FILES, download_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t1", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.23)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    for filename in MODEL_FILES["synthseg-plus"]:
        download_file(filename, args.weights, verify_only=True)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
    start = monotonic()
    context = SubregionContext.prepare(args.t1, need_coarse=True, need_parc=True,
                                       synthseg_weights=args.weights,
                                       synthseg_parc_weights=args.weights,
                                       device=args.device)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(args.device)
    elapsed = monotonic() - start
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in (("aseg_proxy", context.coarse_segmentation),
                         ("cortical_parcellation", context.cortical_parcellation),
                         ("wmparc_proxy", context.wmparc_proxy)):
        nib.save(nib.Nifti1Image(values.astype(np.int32), context.image.affine),
                 args.output_dir / f"{name}.nii.gz")
    support = {str(label): int(np.count_nonzero(context.wmparc_proxy == label))
               for label in (3006, 3007, 3016, 4006, 4007, 4016)}
    report = {"t1": str(args.t1), "device": args.device,
              "seconds": elapsed, "shape": list(context.image.shape),
              "peak_gpu_gib": torch.cuda.max_memory_allocated(args.device) / 2**30
                              if args.device.startswith("cuda") else None,
              "wmparc_support_voxels": support,
              "wm_only": bool(np.all(
                  np.isin(context.wmparc_proxy,
                          (3006, 3007, 3016, 4006, 4007, 4016)) <=
                  np.isin(context.coarse_segmentation, (2, 41))))}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
