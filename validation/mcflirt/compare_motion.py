"""真实 BOLD 的 MCFLIRT 估计控制；不写出任何影像。"""
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
    frame_selection = parser.add_mutually_exclusive_group()
    frame_selection.add_argument("--frames", type=int, default=8)
    frame_selection.add_argument("--all-frames", action="store_true")
    parser.add_argument("--original-parameters", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.frames < 2 or args.threads < 1:
        parser.error("frames must be at least 2 and threads must be positive")
    if args.output_dir.exists():
        parser.error("choose a fresh output directory")
    from fnit.mcflirt import TorchMCFLIRT
    import fnit.mcflirt.core as source
    import fnit.flirt.core as flirt_source
    from fnit.applywarp.core import _fsl_voxel_matrix
    torch.set_num_threads(args.threads)
    raw = nib.load(args.bold)
    reference = nib.load(args.reference)
    mask = nib.load(args.brain_mask)
    if args.all_frames and raw.ndim == 4:
        args.frames = raw.shape[3]
    if raw.ndim != 4 or args.frames > raw.shape[3]:
        parser.error("the real BOLD does not contain the requested frame count")
    if mask.shape != reference.shape or not np.allclose(mask.affine, reference.affine):
        parser.error("brain-mask must use the reference grid")
    region = np.asarray(mask.dataobj) > 0
    if not np.any(region):
        parser.error("brain-mask is empty")
    if args.all_frames:
        matrix_count = len(list(args.original_matrices.glob("MAT_[0-9][0-9][0-9][0-9]")))
        if matrix_count != args.frames:
            parser.error("--all-frames requires a complete matching original matrix series")
        bold = args.bold
    else:
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
    compared_frames = args.frames if args.all_frames else args.frames - 1
    for frame in range(compared_frames):
        difference = (np.linalg.inv(result.matrices[frame]) - np.linalg.inv(original[frame]))[:3]
        distance = np.linalg.norm(difference[:, :3] @ points + difference[:, 3:4], axis=0)
        relative = result.matrices[frame] @ np.linalg.inv(original[frame])
        rows.append({"frame": frame,
                     "matrix_max_abs": float(np.max(np.abs(result.matrices[frame] - original[frame]))),
                     "pull_rms_mm": float(np.sqrt(np.mean(distance**2))),
                     "pull_p95_mm": float(np.percentile(distance, 95)),
                     "relative_translation_mm": float(np.linalg.norm(relative[:3, 3]))})
    report = {"subjects": 1, "real_frames": args.frames, "compared_frames": compared_frames,
              "stage_iterations": [1, 1, 1], "device": str(device), "wall_seconds": seconds,
              "cost_evaluations": result.cost_evaluations,
              "timing_scope": ("Path input loading, preparation and estimation; no resampling or writes."
                               if args.all_frames else
                               "Preloaded frame subset: preparation and estimation; no loading, resampling or writes."),
              "input_sha256": {"bold": sha256(args.bold), "sbref": sha256(args.reference),
                               "brain_mask": sha256(args.brain_mask)},
              "sources": {"fnit.mcflirt.core": sha256(source.__file__),
                          "fnit.flirt.core": sha256(flirt_source.__file__)},
              "note": ("Complete matching series: every frame is compared. CUDA time may include first compilation."
                       if args.all_frames else
                       "Only the first N-1 frames are compared: source MCFLIRT leaves the last coarse-stage initial matrix unchanged. CUDA time may include first compilation.")}
    if args.all_frames:
        values = np.asarray([row["pull_rms_mm"] for row in rows])
        report["pull_rms_mm"] = {"mean": float(values.mean()), "median": float(np.median(values)),
                                 "p95": float(np.percentile(values, 95)), "max": float(values.max())}
        report["worst_frame"] = int(np.argmax(values))
        report["worst_frame_pull_p95_mm"] = rows[report["worst_frame"]]["pull_p95_mm"]
    else:
        report["rows"] = rows
    if args.original_parameters:
        original_parameters = np.loadtxt(args.original_parameters)
        if original_parameters.ndim != 2 or original_parameters.shape[1] != 6 or original_parameters.shape[0] < compared_frames:
            parser.error("original parameters must contain one six-column row per compared frame")
        parameter_rows = []
        for column, name in enumerate(("rotation_x_rad", "rotation_y_rad", "rotation_z_rad",
                                       "translation_x_mm", "translation_y_mm", "translation_z_mm")):
            candidate = result.parameters[:compared_frames, column]
            target = original_parameters[:compared_frames, column]
            correlation = np.corrcoef(candidate, target)[0, 1]
            error = candidate - target
            parameter_rows.append({"parameter": name,
                                   "pearson_r": float(correlation) if np.isfinite(correlation) else None,
                                   "rmse": float(np.sqrt(np.mean(error ** 2))),
                                   "mae": float(np.mean(np.abs(error))),
                                   "max_abs": float(np.max(np.abs(error)))})
        report["parameters"] = parameter_rows
        report["original_parameters_sha256"] = sha256(args.original_parameters)
    if device.type == "cuda":
        report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        report["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    args.output_dir.mkdir(parents=True)
    np.save(args.output_dir / "candidate_matrices.private.npy", result.matrices)
    np.save(args.output_dir / "candidate_parameters.private.npy", result.parameters)
    if args.all_frames:
        (args.output_dir / "framewise.private.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output_dir / "summary.public.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
