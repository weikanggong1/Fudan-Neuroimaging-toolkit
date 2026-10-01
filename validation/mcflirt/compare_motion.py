"""真实 BOLD 的局部 MCFLIRT 估计控制；不写出任何影像。"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--bold", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--brain-mask", type=Path, required=True)
    parser.add_argument("--original-matrices", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.frames < 2 or args.threads < 1:
        parser.error("frames must be at least 2 and threads must be positive")
    if args.output_dir.exists():
        parser.error("choose a fresh output directory")
    from fnit.mcflirt import TorchMCFLIRT
    import fnit.mcflirt.core as source
    from fnit.applywarp.core import _fsl_voxel_matrix
    torch.set_num_threads(args.threads)
    raw = nib.load(args.bold)
    reference = nib.load(args.reference)
    mask = nib.load(args.brain_mask)
    if raw.ndim != 4 or args.frames > raw.shape[3]:
        parser.error("the real BOLD does not contain the requested frame count")
    if mask.shape != reference.shape or not np.allclose(mask.affine, reference.affine):
        parser.error("brain-mask must use the reference grid")
    region = np.asarray(mask.dataobj) > 0
    if not np.any(region):
        parser.error("brain-mask is empty")
    bold = nib.Nifti1Image(np.asarray(raw.dataobj[..., :args.frames], dtype=np.float32),
                           raw.affine, raw.header.copy())
    original = np.stack([np.loadtxt(args.original_matrices / f"MAT_{frame:04d}")
                         for frame in range(args.frames)])
    if original.shape != (args.frames, 4, 4) or not np.isfinite(original).all():
        parser.error("original matrices must contain finite 4x4 transforms")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = TorchMCFLIRT(device=device).run(bold, reference)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    sampling = _fsl_voxel_matrix(reference)
    points = sampling[:3, :3] @ np.asarray(np.where(region)) + sampling[:3, 3:4]
    rows = []
    for frame in range(args.frames - 1):
        difference = (np.linalg.inv(result.matrices[frame]) - np.linalg.inv(original[frame]))[:3]
        distance = np.linalg.norm(difference[:, :3] @ points + difference[:, 3:4], axis=0)
        relative = result.matrices[frame] @ np.linalg.inv(original[frame])
        rows.append({"frame": frame,
                     "matrix_max_abs": float(np.max(np.abs(result.matrices[frame] - original[frame]))),
                     "pull_rms_mm": float(np.sqrt(np.mean(distance**2))),
                     "pull_p95_mm": float(np.percentile(distance, 95)),
                     "relative_translation_mm": float(np.linalg.norm(relative[:3, 3]))})
    report = {"subjects": 1, "real_frames": args.frames, "compared_frames": args.frames - 1,
              "stage_iterations": [1, 1, 1], "device": str(device), "wall_seconds": seconds,
              "cost_evaluations": result.cost_evaluations, "rows": rows,
              "input_sha256": {"bold": sha256(args.bold), "sbref": sha256(args.reference),
                               "brain_mask": sha256(args.brain_mask)},
              "sources": {"fnit.mcflirt.core": sha256(source.__file__)},
              "note": "Only the first N-1 frames are compared: source MCFLIRT leaves the last coarse-stage initial matrix unchanged. CUDA time may include first compilation."}
    if device.type == "cuda":
        report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        report["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    args.output_dir.mkdir(parents=True)
    np.save(args.output_dir / "candidate_matrices.private.npy", result.matrices)
    (args.output_dir / "summary.public.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
