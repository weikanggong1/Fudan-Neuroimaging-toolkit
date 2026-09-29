"""Check MNI-mask tracking against both saved FNIT dMRI registration branches."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import nibabel as nib
import numpy as np
import torch

from fnit import TorchApplyWarp, TorchConvertWarp, TorchInvWarp, TorchProbtrackX


def _data(path):
    return np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-dir", required=True)
    parser.add_argument("--samples-dir", required=True)
    parser.add_argument("--mni-mask", required=True)
    parser.add_argument("--backend", choices=("tbss", "mmorf"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-root", required=True)
    args = parser.parse_args()
    root = Path(args.pipeline_dir)
    registration = root / "registration"
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reference = registration / "standard/FA.nii.gz"
    source = root / "native/dti_FA.nii.gz"
    model = TorchConvertWarp(args.device)
    selected_device = torch.device(args.device)
    gpu = selected_device.type == "cuda"
    gpu_index = torch.cuda.current_device() if gpu else None
    started = perf_counter()
    if args.backend == "tbss":
        field = model.run(
            reference=reference, warp1=registration / "dti_FA_to_MNI_warp.nii.gz",
            output=output / "diff2mni_warp.nii.gz")
        source = registration / "dti_FA_preprocessed.nii.gz"
    else:
        field = model.run_mmorf(
            reference=reference, source=source,
            mmorf_warp=registration / "mmorf_warp.nii.gz",
            affine=registration / "dti_FA_to_MNI_affine.mat",
            output=output / "diff2mni_warp.nii.gz")
    forward_seconds = perf_counter() - started
    forward_peak = torch.cuda.max_memory_allocated(gpu_index) / 2**30 if gpu else None
    warper = TorchApplyWarp(args.device)
    sampled = warper(input=source, reference=reference, warp=field.image,
                     warp_convention="relative", interpolation="trilinear").image
    reference_data = _data(reference)
    candidate_data = np.asarray(sampled.dataobj, dtype=np.float32)
    support = (reference_data != 0) | (candidate_data != 0)
    started = perf_counter()
    inverse = TorchInvWarp(args.device).run(
        reference=Path(args.samples_dir) / "nodif_brain_mask.nii.gz",
        warp=field.image, warp_convention="relative",
        output=output / "mni2diff_warp.nii.gz")
    inverse_seconds = perf_counter() - started
    inverse_peak = torch.cuda.max_memory_allocated(gpu_index) / 2**30 if gpu else None
    mask = warper.run(
        input=args.mni_mask, reference=Path(args.samples_dir) / "nodif_brain_mask.nii.gz",
        warp=inverse.image, warp_convention="relative", interpolation="nearest",
        output_dtype="char", output=output / "mni_mask_in_diff.nii.gz")
    tracker = TorchProbtrackX(device=args.device, nsamples=20, nsteps=200,
                              batch_size=8192, seed=711)
    started = perf_counter()
    automatic = tracker.run(
        samples_dir=args.samples_dir, output_dir=output / "tracking_auto",
        seed=args.mni_mask, dmri_pipeline_dir=root)
    auto_seconds = perf_counter() - started
    auto_peak = torch.cuda.max_memory_allocated(gpu_index) / 2**30 if gpu else None
    direct = tracker.run(
        samples_dir=args.samples_dir, output_dir=output / "tracking_direct",
        seed=automatic.mni_to_diffusion_dir / "masks/000" / Path(args.mni_mask).name)
    auto_density = _data(automatic.paths)
    direct_density = _data(direct.paths)
    hashes = {}
    for name in ("convertwarp/core.py", "invwarp/core.py", "probtrackx/mni_masks.py"):
        path = Path(args.source_root) / "fnit" / name
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {
        "date": "2026-09-29", "backend": args.backend,
        "data": "one real UKB-format DWI; FNIT pipeline registration; JHU atlas label 9",
        "source_sha256": hashes,
        "forward_shape": list(field.image.shape),
        "inverse_shape": list(inverse.image.shape),
        "forward_intent": int(field.image.header["intent_code"]),
        "inverse_intent": int(inverse.image.header["intent_code"]),
        "standard_FA_union_r": float(np.corrcoef(
            reference_data[support], candidate_data[support])[0, 1]),
        "standard_FA_union_MAE": float(np.abs(
            reference_data[support] - candidate_data[support]).mean()),
        "mapped_mask_voxels": int((np.asanyarray(mask.image.dataobj) > 0).sum()),
        "auto_seed_voxels": automatic.seed_points,
        "auto_direct_density_equal": bool(np.array_equal(auto_density, direct_density)),
        "auto_waytotal": automatic.waytotal.read_text().strip(),
        "direct_waytotal": direct.waytotal.read_text().strip(),
        "time_seconds": {
            "convertwarp_python_call": forward_seconds,
            "invwarp_python_call": inverse_seconds,
            "automatic_mask_conversion_and_tracking_python_call": auto_seconds,
        },
        "cumulative_peak_pytorch_allocated_gib": {
            "convertwarp": forward_peak, "invwarp": inverse_peak,
            "automatic_mask_conversion_and_tracking": auto_peak,
        },
        "timing_boundary": "single Python call after imports; shared GPU",
    }
    (output / "report.public.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
