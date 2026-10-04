"""Whole TorchFAST GPU calls on a real brain or 64-cube spatial subset.

This guards changed FAST math dispatch. Subsets are recorded explicitly.
The unchanged default tensor path and explicit ordered path are both checked.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, choices=(0, 64), default=64,
                        help="0 retains the complete input image and header")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    torch.cuda.init()
    properties = torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(20e9 / properties.total_memory, 0)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    import nibabel as nib
    import numpy as np
    from fnit.fast import TorchFAST

    image = nib.load(args.image)
    starts = [(size - args.size) // 2 for size in image.shape] if args.size else [0, 0, 0]
    slices = tuple(slice(start, start + args.size) for start in starts) if args.size else (...,)
    values = np.asarray(image.dataobj, dtype=np.float32)[slices].copy()
    affine = image.affine.copy()
    affine[:3, 3] += affine[:3, :3] @ np.asarray(starts)
    crop = nib.Nifti1Image(values, affine) if args.size else image
    report = {"scope": ("real_spatial64cube_whole_gpu_api_regression_only" if args.size else
                        "real_complete_brain_whole_gpu_api_regression"),
              "parent_image_sha256": hashlib.sha256(Path(args.image).read_bytes()).hexdigest(),
              "crop_start": starts, "crop_shape": list(values.shape),
              "crop_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
              "gpu_name": properties.name, "modes": {}}
    fields = ("pve_csf", "pve_gm", "pve_wm", "hard_segmentation", "pve_segmentation",
              "mixel_type", "bias_field", "restored")
    for execution in ("tensor", "fsl"):
        model = TorchFAST(device="cuda", threads=8, execution=execution)
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.synchronize(0)
        started = time.perf_counter()
        result = model(crop)
        torch.cuda.synchronize(0)
        entry = {"api_seconds": time.perf_counter() - started,
                 "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                 "peak_reserved_bytes": torch.cuda.max_memory_reserved(0), "outputs": {}}
        for name in fields:
            output = getattr(result, name)
            array = np.asarray(output.dataobj)
            entry["outputs"][name] = {"array_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                                       "finite": bool(np.isfinite(array).all())}
            nib.save(output, args.output / (execution + "_" + name + ".nii.gz"))
        report["modes"][execution] = entry
    (args.output / "record.public.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
