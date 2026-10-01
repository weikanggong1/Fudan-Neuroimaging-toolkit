"""真实 MCFLIRT 对照：在 EPI 计算指标，仅将三维指标图变换到 MNI 展示。"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

from fnit.fmri.normalization import resample_world
import fnit.fmri.normalization as resampling_source
import fnit.mcflirt.core as motion_source
from fnit.applywarp.core import _fsl_voxel_matrix


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p05": float(np.percentile(values, 5)), "p95": float(np.percentile(values, 95)),
            "min": float(values.min()), "max": float(values.max())}


def save_map(values, reference, path):
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(values.astype(np.float32), reference.affine, header), path)


def epi_metrics(original, candidate, region):
    """按体素分块，以 float64 中心化求时间 SD 和 r；不做空间平滑。"""
    frames = original.shape[-1]
    left = np.asarray(original.dataobj, dtype=np.float32).reshape(-1, frames, order="F")
    right = np.asarray(candidate.dataobj, dtype=np.float32).reshape(-1, frames, order="F")
    active = region.reshape(-1, order="F")
    arrays = {name: np.zeros(left.shape[0], dtype=np.float64 if name == "temporal_r" else np.float32) for name in
              ("original_mean", "candidate_mean", "original_sd", "candidate_sd", "temporal_r", "valid")}
    squared_error, maximum_error, equal_count, noninteger_count = 0.0, 0.0, 0, 0
    for start in range(0, left.shape[0], 4096):
        stop = min(start + 4096, left.shape[0])
        a, b = left[start:stop].astype(np.float64), right[start:stop].astype(np.float64)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("The real motion-corrected input contains nonfinite values")
        roi = active[start:stop]
        if roi.any():
            difference = a[roi] - b[roi]
            squared_error += float(np.square(difference).sum())
            maximum_error = max(maximum_error, float(np.abs(difference).max()))
            equal_count += int(np.count_nonzero(difference == 0))
            noninteger_count += int(np.count_nonzero(b[roi] != np.trunc(b[roi])))
        mean_a, mean_b = a.mean(axis=1), b.mean(axis=1)
        a -= mean_a[:, None]
        b -= mean_b[:, None]
        sum_a, sum_b = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
        valid = (sum_a > frames * 1e-12) & (sum_b > frames * 1e-12)
        correlation = np.zeros(stop - start, dtype=np.float64)
        correlation[valid] = np.clip((a[valid] * b[valid]).sum(axis=1) /
                                     np.sqrt(sum_a[valid] * sum_b[valid]), -1, 1)
        for name, block in (("original_mean", mean_a), ("candidate_mean", mean_b),
                            ("original_sd", np.sqrt(sum_a / frames)),
                            ("candidate_sd", np.sqrt(sum_b / frames)),
                            ("temporal_r", correlation), ("valid", valid)):
            arrays[name][start:stop] = block
    maps = {name: array.reshape(original.shape[:3], order="F") for name, array in arrays.items()}
    valid_region = region & (maps["valid"] > 0)
    if not valid_region.any():
        raise ValueError("No comparison voxel has two nonconstant time series")
    error_sd = maps["candidate_sd"][region].astype(np.float64) - maps["original_sd"][region]
    report = {"brain_voxels": int(region.sum()), "valid_time_correlation_voxels": int(valid_region.sum()),
              "temporal_pearson_r": describe(maps["temporal_r"][valid_region]),
              "brain_4d_rmse": float(np.sqrt(squared_error / (region.sum() * frames))),
              "brain_4d_max_abs": maximum_error,
              "brain_exact_fraction": float(equal_count / (region.sum() * frames)),
              "candidate_brain_noninteger_fraction": float(noninteger_count / (region.sum() * frames)),
              "original_temporal_sd": describe(maps["original_sd"][region]),
              "candidate_temporal_sd": describe(maps["candidate_sd"][region]),
              "temporal_sd_map_pearson_r": float(np.corrcoef(maps["original_sd"][region],
                                                             maps["candidate_sd"][region])[0, 1]),
              "temporal_sd_map_rmse": float(np.sqrt(np.mean(error_sd ** 2))),
              "temporal_sd_definition": "Population SD (ddof=0) across all frames in EPI space, before display transformation."}
    return maps, report


def matrix_metrics(args, reference, region, private_output):
    frames = reference.shape[-1]
    original = np.stack([np.loadtxt(args.original_matrices / f"MAT_{t:04d}") for t in range(frames)])
    candidate = np.stack([np.loadtxt(args.candidate_matrices / f"MAT_{t:04d}") for t in range(frames)])
    original_par, candidate_par = np.loadtxt(args.original_parameters), np.loadtxt(args.candidate_parameters)
    if original_par.shape != (frames, 6) or candidate_par.shape != (frames, 6):
        raise ValueError("Motion parameter arrays must contain the full six-column frame series")
    coordinate_matrix = _fsl_voxel_matrix(reference)
    points = coordinate_matrix[:3, :3] @ np.asarray(np.where(region)) + coordinate_matrix[:3, 3:4]
    rms, p95 = [], []
    for first, second in zip(candidate, original):
        delta = (np.linalg.inv(first) - np.linalg.inv(second))[:3]
        distance = np.linalg.norm(delta[:, :3] @ points + delta[:, 3:4], axis=0)
        rms.append(np.sqrt(np.mean(distance ** 2)))
        p95.append(np.percentile(distance, 95))
    np.save(private_output / "pull_rms_framewise.private.npy", rms)
    np.save(private_output / "pull_p95_framewise.private.npy", p95)
    columns = []
    for column, name in enumerate(("rotation_x_rad", "rotation_y_rad", "rotation_z_rad",
                                   "translation_x_mm", "translation_y_mm", "translation_z_mm")):
        error = candidate_par[:, column] - original_par[:, column]
        correlation = np.corrcoef(candidate_par[:, column], original_par[:, column])[0, 1]
        columns.append({"parameter": name, "pearson_r": float(correlation) if np.isfinite(correlation) else None,
                        "rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(np.abs(error))),
                        "max_abs": float(np.max(np.abs(error)))})
    hashes = {}
    for name, directory in (("original", args.original_matrices), ("candidate", args.candidate_matrices)):
        digest = hashlib.sha256()
        for frame in range(frames):
            digest.update((directory / f"MAT_{frame:04d}").read_bytes())
        hashes[name + "_matrices_concat_frame_order"] = digest.hexdigest()
    hashes["original_parameters"] = sha256(args.original_parameters)
    hashes["candidate_parameters"] = sha256(args.candidate_parameters)
    return {"compared_frames": frames, "pull_rms_mm": describe(rms), "pull_p95_mm": describe(p95),
            "worst_frame": int(np.argmax(rms)), "worst_frame_pull_p95_mm": float(p95[np.argmax(rms)]),
            "parameters": columns, "sha256": hashes}


def render(mapped, mask, reference, report, output):
    # Template coordinates select anatomy; no native-space image is published.
    centre_world = np.asarray([-4., -18., 22., 1.])
    centre = np.rint(np.linalg.inv(reference.affine) @ centre_world).astype(int)[:3]
    centre = np.clip(centre, 0, np.asarray(reference.shape) - 1)
    mean_scale = float(np.percentile(np.concatenate([mapped[n][mask] for n in
                                                    ("original_mean", "candidate_mean")]), 99))
    sd_scale = float(np.percentile(np.concatenate([mapped[n][mask] for n in
                                                  ("original_sd", "candidate_sd")]), 99))
    rows = (("original_mean", "Original FSL MCFLIRT: mean", "gray", 0, mean_scale),
            ("candidate_mean", "FNIT TorchMCFLIRT: mean", "gray", 0, mean_scale),
            ("original_sd", "Original FSL MCFLIRT: temporal SD", "gray", 0, sd_scale),
            ("candidate_sd", "FNIT TorchMCFLIRT: temporal SD", "gray", 0, sd_scale),
            ("temporal_r", "EPI voxel temporal Pearson r", "viridis", .995, 1.))
    fig, axes = plt.subplots(5, 3, figsize=(11, 14), facecolor="white")
    for row, (name, label, cmap, low, high) in enumerate(rows):
        display = np.where(mask, mapped[name], np.nan)
        for axis, plane_name in enumerate(("Sagittal", "Coronal", "Axial")):
            plane = np.rot90(np.take(display, centre[axis], axis=axis))
            image = axes[row, axis].imshow(plane, cmap=cmap, vmin=low, vmax=high, interpolation="nearest")
            axes[row, axis].axis("off")
            axes[row, axis].set_title((label + "\n" if axis == 1 else "") + plane_name, fontsize=10)
        fig.colorbar(image, ax=axes[row, -1], shrink=.75)
    mean_r = report["epi"]["temporal_pearson_r"]["mean"]
    fig.suptitle(f"Real {report['frames']}-frame BOLD: same raw input / SBRef; EPI mean time r = {mean_r:.8f}\n"
                 "Mean, SD and r calculated in EPI, then mapped to MNI for display with the same original BBR + FNIRT", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(output, dpi=145, bbox_inches="tight")
    plt.close(fig)
    return {"slice_reference_world_mm": centre_world[:3].tolist(), "shared_mean_vmax": mean_scale,
            "shared_temporal_sd_vmax": sd_scale, "temporal_r_colour_range": [.995, 1.],
            "display_only": True, "spatial_smoothing": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("original", "candidate", "epi-mask", "template", "mni-mask", "reference-to-source-world",
                 "mni-to-t1-pull", "private-output", "figure-out", "report-out"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("original-matrices", "candidate-matrices", "original-parameters", "candidate-parameters"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    transform_inputs = (args.original_matrices, args.candidate_matrices, args.original_parameters, args.candidate_parameters)
    if any(transform_inputs) and not all(transform_inputs):
        parser.error("motion matrix/parameter comparison requires all four input locations")
    if args.threads < 1 or args.private_output.exists() or args.figure_out.exists() or args.report_out.exists():
        parser.error("use positive threads and fresh private, figure and report destinations")
    original, candidate, epi_mask = nib.load(args.original), nib.load(args.candidate), nib.load(args.epi_mask)
    if original.ndim != 4 or original.shape != candidate.shape or epi_mask.shape != original.shape[:3]:
        parser.error("motion-corrected BOLD must share the complete 4D shape and EPI mask")
    for image in (candidate, epi_mask):
        if not np.allclose(original.affine, image.affine, atol=1e-4, rtol=0):
            parser.error("motion-corrected BOLD and EPI mask grids differ")
    region = np.asarray(epi_mask.dataobj) > 0
    if not region.any():
        parser.error("EPI comparison mask is empty")
    args.private_output.mkdir(parents=True)
    args.figure_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    maps, epi_report = epi_metrics(original, candidate, region)
    metric_seconds = time.perf_counter() - started
    report = {"subjects": 1, "frames": original.shape[3], "source_revision": args.source_revision,
              "device": "cpu", "threads": torch.get_num_threads(), "epi": epi_report,
              "dtype": {"original": str(original.get_data_dtype()), "candidate": str(candidate.get_data_dtype())},
              "metric_wall_seconds_including_image_reads": metric_seconds}
    matrix = np.loadtxt(args.reference_to_source_world)
    mapped = {}
    started = time.perf_counter()
    for name, values in maps.items():
        source_path = args.private_output / (name + "_epi.private.nii.gz")
        output_path = args.private_output / (name + "_mni.private.nii.gz")
        save_map(values, original, source_path)
        # Nearest neighbour preserves the r/valid summaries; means and SD use
        # trilinear sampling solely for template-space display.
        interpolation = "nearest" if name in ("temporal_r", "valid") else "linear"
        resample_world(source_path, args.template, matrix, output_path,
                       pre_affine_pull_ras=args.mni_to_t1_pull, output_mask=args.mni_mask,
                       interpolation=interpolation, device="cpu", batch_size=1)
        mapped[name] = np.asarray(nib.load(output_path).dataobj, dtype=np.float32)
    report["display_mapping_wall_seconds_including_3d_io"] = time.perf_counter() - started
    template = nib.load(args.template)
    mni_mask = nib.load(args.mni_mask)
    if template.shape != mni_mask.shape or not np.allclose(template.affine, mni_mask.affine, atol=1e-4, rtol=0):
        parser.error("MNI template and mask grids differ")
    display_mask = (np.asarray(mni_mask.dataobj) > 0) & (mapped["valid"] > .5)
    if not display_mask.any():
        parser.error("The MNI display mask is empty")
    if all(transform_inputs):
        report["motion"] = matrix_metrics(args, original, region, args.private_output)
    report["display"] = render(mapped, display_mask, template, report, args.figure_out)
    input_names = ("original", "candidate", "epi_mask", "template", "mni_mask",
                   "reference_to_source_world", "mni_to_t1_pull")
    report["input_sha256"] = {name: sha256(getattr(args, name)) for name in input_names}
    report["sources"] = {"fnit.mcflirt.core": sha256(motion_source.__file__),
                         "fnit.fmri.normalization": sha256(resampling_source.__file__),
                         "renderer": sha256(__file__)}
    report["figure_sha256"] = sha256(args.figure_out)
    report["note"] = "Statistics are measured in EPI space. Three-dimensional summaries alone are mapped with the same original BBR and FNIRT for MNI display. Native and MNI numerical maps remain private; only the authorized deidentified MNI PNG and anonymous scalar/hash report are published."
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
