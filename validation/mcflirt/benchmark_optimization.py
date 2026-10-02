"""真实 BOLD 的 MCFLIRT 优化验证和固定掩膜 FEAT 后续处理。

输入：四维 BOLD、同网格三维参考与脑掩膜、冻结版本逐帧 MAT_####
目录（或 N×4×4 .npy）、N×6 .par、完整 FEAT filtered_func_data。
输出：summary.public.json 只有匿名标量、计时与 SHA-256；candidate_*.private
影像、矩阵、参数及 framewise.private.json 只应保存在私有验证目录。
默认使用全部帧。--frames 是前 N 帧初步控制，其最后一帧粗阶段初值会变，
因此不比较该帧，不与完整时序的 FEAT 高通结果比较。

本脚本不调用原软件。原 MCFLIRT 的对应独立验证命令为：
mcflirt -in "$RAW_BOLD" -reffile "$SBREF" -out "$NATIVE_MOTION" -mats -plots -spline_final
该命令须另行运行；--prior-native-seconds 仅记录既有观察，不自动计算加速比。
源码：https://git.fmrib.ox.ac.uk/fsl/mcflirt/-/blob/2111.0/mcflirt.cc
参考：Jenkinson et al. NeuroImage 17:825-841 (2002), doi:10.1006/nimg.2002.1132。

完整示例（所有输入和输出路径均由操作者指定，不能上传私有结果）：
python validation/mcflirt/benchmark_optimization.py \\
  --bold "$RAW_BOLD" --reference "$SBREF" --brain-mask "$EPI_BRAIN_MASK" \\
  --baseline-matrices "$FROZEN_MOTION_MATRICES" \\
  --baseline-parameters "$FROZEN_MOTION_PARAMETERS" \\
  --baseline-filtered "$FROZEN_FILTERED_BOLD" \\
  --baseline-corrected "$FROZEN_CORRECTED_BOLD" \\
  --baseline-kind frozen_fnit --baseline-source-revision "$FROZEN_GIT_REVISION" \\
  --output-dir "$FRESH_PRIVATE_OUTPUT" --device cuda:0 --threads 8 \\
  --source-revision "$CANDIDATE_GIT_REVISION"
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys
import time
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch


def sha256(path):
    """流式读取文件，返回 SHA-256，不在报告中记录路径。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes():
    """核对实际导入的运行时源文件和本驱动；键只使用模块名。"""
    root = Path(importlib.import_module("fnit.mcflirt").__file__).resolve().parent.parent
    files = list((root / "mcflirt").glob("*.py")) + list((root / "flirt").glob("*.py"))
    files += [root / name for name in ("fmri/spatial.py", "fmri/pipeline.py",
                                      "feat/temporal.py", "applywarp/core.py")]
    result = {"fnit." + ".".join(path.relative_to(root).with_suffix("").parts): sha256(path)
              for path in sorted(files)}
    result["benchmark_driver"] = sha256(__file__)
    return result


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_call(name, function, timings, device):
    """CUDA 阶段开始和结束均同步；计时包含该阶段的 CPU 工作。"""
    synchronize(device)
    started = time.perf_counter()
    result = function()
    synchronize(device)
    timings[name] = time.perf_counter() - started
    return result


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0}
    return {"count": int(values.size), "mean": float(values.mean()),
            "median": float(np.median(values)), "p05": float(np.percentile(values, 5)),
            "p95": float(np.percentile(values, 95)), "min": float(values.min()),
            "max": float(values.max())}


def load_matrices(path, total_frames):
    """读取完整冻结结果；目录按帧号拼接原始文件字节计算哈希。"""
    if path.is_dir():
        files = sorted(path.glob("MAT_[0-9][0-9][0-9][0-9]"))
        if len(files) != total_frames:
            raise ValueError("baseline matrix directory must contain the complete input series")
        files = [path / f"MAT_{frame:04d}" for frame in range(total_frames)]
        matrices = np.stack([np.loadtxt(file, dtype=np.float64) for file in files])
        digest = hashlib.sha256()
        for file in files:
            digest.update(file.read_bytes())
        artifact_hash = digest.hexdigest()
    else:
        matrices = np.load(path, allow_pickle=False)
        artifact_hash = sha256(path)
    if matrices.shape != (total_frames, 4, 4) or not np.isfinite(matrices).all():
        raise ValueError("baseline matrices must contain one finite 4x4 transform per input frame")
    if not np.allclose(matrices[:, 3], (0, 0, 0, 1), atol=1e-8, rtol=0):
        raise ValueError("baseline transforms must be homogeneous affine matrices")
    if np.any(np.abs(np.linalg.det(matrices[:, :3, :3])) < 1e-10):
        raise ValueError("baseline matrices must be invertible")
    return matrices, artifact_hash


def save_image(data, reference, output):
    """保持参考几何，按输入数组类型保存私有 NIfTI。"""
    array = np.asarray(data)
    header = reference.header.copy()
    header.set_data_dtype(array.dtype)
    header.set_slope_inter(1.0, 0.0)
    image = nib.Nifti1Image(array, reference.affine, header)
    image.set_qform(reference.get_qform(), int(reference.header["qform_code"]))
    image.set_sform(reference.get_sform(), int(reference.header["sform_code"]))
    nib.save(image, str(output))
    return image


def compare_motion(matrices, parameters, baseline_matrices, baseline_parameters,
                   raw, reference, region, compared_frames):
    """比较参考脑区采样点在原 BOLD RAS-mm 中的 pull 距离。"""
    from fnit.applywarp.core import _fsl_voxel_matrix

    reference_fsl = _fsl_voxel_matrix(reference)
    source_fsl_to_ras = raw.affine @ np.linalg.inv(_fsl_voxel_matrix(raw))
    points = reference_fsl[:3, :3] @ np.asarray(np.where(region)) + reference_fsl[:3, 3:4]
    rows = []
    for frame in range(compared_frames):
        difference = (np.linalg.inv(matrices[frame]) - np.linalg.inv(baseline_matrices[frame]))[:3]
        displacement = source_fsl_to_ras[:3, :3] @ (
            difference[:, :3] @ points + difference[:, 3:4])
        distances = np.linalg.norm(displacement, axis=0)
        rows.append({"frame": frame,
                     "matrix_max_abs": float(np.max(np.abs(matrices[frame] - baseline_matrices[frame]))),
                     "pull_rms_ras_mm": float(np.sqrt(np.mean(distances ** 2))),
                     "pull_p95_ras_mm": float(np.percentile(distances, 95))})
    parameter_rows = []
    for column, name in enumerate(("rotation_x_rad", "rotation_y_rad", "rotation_z_rad",
                                   "translation_x_mm", "translation_y_mm", "translation_z_mm")):
        candidate = parameters[:compared_frames, column]
        baseline = baseline_parameters[:compared_frames, column]
        error = candidate - baseline
        candidate_centred = candidate - candidate.mean()
        baseline_centred = baseline - baseline.mean()
        denominator = np.linalg.norm(candidate_centred) * np.linalg.norm(baseline_centred)
        parameter_rows.append({"parameter": name, "rmse": float(np.sqrt(np.mean(error ** 2))),
                               "mae": float(np.mean(np.abs(error))), "max_abs": float(np.max(np.abs(error))),
                               "pearson_r": float(candidate_centred @ baseline_centred / denominator)
                               if denominator > 0 else None})
    rms = np.asarray([row["pull_rms_ras_mm"] for row in rows])
    matrix_error = matrices[:compared_frames] - baseline_matrices[:compared_frames]
    report = {"compared_frames": compared_frames, "brain_voxels": int(region.sum()),
              "matrix_max_abs": float(np.max(np.abs(matrix_error))),
              "matrix_rmse": float(np.sqrt(np.mean(matrix_error ** 2))),
              "matrix_exact_equal": bool(np.array_equal(matrices[:compared_frames], baseline_matrices[:compared_frames])),
              "pull_rms_ras_mm": summary(rms),
              "all_brain_points_pull_rms_ras_mm": float(np.sqrt(np.mean(rms ** 2))),
              "worst_frame": int(np.argmax(rms)), "parameters": parameter_rows,
              "coordinate_definition": "Reference brain voxels pulled into raw BOLD RAS-mm; FSL scaled-mm matrices converted with actual NIfTI geometry."}
    return report, rows


def compare_image(candidate, baseline_path, region, frames, *, voxel_chunk=4096):
    """逐空间块计算完整时序误差和脑内时间 r，不生成完整 float64 影像。"""
    baseline = nib.load(str(baseline_path))
    if baseline.ndim != 4 or baseline.shape != (*region.shape, frames):
        raise ValueError("baseline image must contain the complete input series on the reference grid")
    if not np.allclose(candidate.affine, baseline.affine, atol=1e-4, rtol=0):
        raise ValueError("baseline image must use the same reference affine")
    left = np.asanyarray(candidate.dataobj)
    right = np.asanyarray(baseline.dataobj)
    totals = {"whole_image": [0.0, 0.0, 0, 0], "brain": [0.0, 0.0, 0, 0]}
    correlations = []
    for x in range(region.shape[0]):
        left_plane = left[x].reshape(-1, frames)
        right_plane = right[x].reshape(-1, frames)
        mask_plane = region[x].reshape(-1)
        for start in range(0, left_plane.shape[0], voxel_chunk):
            stop = min(start + voxel_chunk, left_plane.shape[0])
            a = left_plane[start:stop].astype(np.float64)
            b = right_plane[start:stop].astype(np.float64)
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError("candidate and baseline images must contain finite values")
            error = a - b
            selected = mask_plane[start:stop]
            for name, differences in (("whole_image", error), ("brain", error[selected])):
                if differences.size:
                    total = totals[name]
                    total[0] += float(np.sum(differences * differences, dtype=np.float64))
                    total[1] = max(total[1], float(np.max(np.abs(differences))))
                    total[2] += int(np.count_nonzero(differences == 0))
                    total[3] += int(differences.size)
            if np.any(selected):
                a, b = a[selected], b[selected]
                a -= a.mean(axis=1, keepdims=True)
                b -= b.mean(axis=1, keepdims=True)
                denominator = np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1))
                valid = denominator > 0
                correlations.append(np.clip(np.sum(a[valid] * b[valid], axis=1) / denominator[valid], -1, 1))
    report = {}
    for name, (squared_sum, maximum, exact, count) in totals.items():
        report[name] = {"samples": count, "rmse": float(np.sqrt(squared_sum / count)),
                        "max_abs": maximum, "exact_fraction": exact / count,
                        "exact_equal": exact == count}
    report["brain_temporal_pearson_r"] = summary(np.concatenate(correlations) if correlations else [])
    report["dtype"] = {"candidate": str(candidate.get_data_dtype()), "baseline": str(baseline.get_data_dtype())}
    report["metric_scope"] = "All frames; image errors cover all voxels and separately the supplied brain mask. Temporal Pearson r excludes constant time series."
    return report


def parser_for_cli():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag, help_text in (("bold", "真实四维 BOLD NIfTI"), ("reference", "与 BOLD 同网格的三维 SBRef"),
                            ("brain-mask", "参考网格三维脑掩膜，值大于零为脑内"),
                            ("baseline-matrices", "完整冻结矩阵目录 MAT_#### 或 N×4×4 .npy"),
                            ("baseline-parameters", "完整冻结 N×6 .par，前三列 rad、后三列 mm"),
                            ("baseline-filtered", "完整冻结 FEAT 高通结果，默认全帧核对"),
                            ("output-dir", "尚不存在的私有输出目录")):
        parser.add_argument("--" + flag, type=Path, required=True, help=help_text)
    parser.add_argument("--baseline-corrected", type=Path, help="可选：完整冻结 MCFLIRT 重采样影像")
    parser.add_argument("--baseline-memory-matrices", type=Path, help="可选：冻结未舍入 float64 N×4×4 .npy；与 memory-parameters 同时提供")
    parser.add_argument("--baseline-memory-parameters", type=Path, help="可选：冻结未舍入 float64 N×6 .npy")
    parser.add_argument("--baseline-kind", choices=("frozen_fnit", "native_mcflirt"), default="frozen_fnit",
                        help="基线来源；默认是冻结 FNIT，不能据此声称与原 FSL 比较")
    parser.add_argument("--baseline-source-revision", default=None, help="可选：冻结 FNIT 的 Git revision")
    parser.add_argument("--source-revision", required=True, help="候选 Git revision；实际文件另以 SHA-256 核对")
    parser.add_argument("--device", default="cuda:0", help="PyTorch device，默认 cuda:0")
    parser.add_argument("--threads", type=int, default=8, help="PyTorch CPU 线程数，默认 8")
    parser.add_argument("--frames", type=int, default=None, help="可选：仅前 N 帧初步控制；默认全部帧")
    parser.add_argument("--tr-seconds", type=float, default=None, help="可选：TR 秒数；默认读取 BOLD 头并转换时间单位")
    parser.add_argument("--highpass-cutoff-seconds", type=float, default=100.0, help="FEAT 高通周期秒数，默认 100")
    parser.add_argument("--prior-native-seconds", type=float, default=None,
                        help="可选：另行完成的原 MCFLIRT 命令历史耗时，仅记观察，不计算加速比")
    return parser


def main(argv=None):
    parser = parser_for_cli()
    args = parser.parse_args(argv)
    if bool(args.baseline_memory_matrices) != bool(args.baseline_memory_parameters):
        parser.error("both baseline memory arrays are required together")
    if args.threads < 1 or (args.frames is not None and args.frames < 2):
        parser.error("threads must be positive and frames must be at least 2")
    for revision in (args.source_revision, args.baseline_source_revision):
        if revision is not None and not re.fullmatch(r"[0-9a-fA-F]{7,40}", revision):
            parser.error("source revisions must be 7-40 hexadecimal Git commit characters")
    if args.output_dir.exists():
        parser.error("choose a fresh private output directory")
    if not np.isfinite(args.highpass_cutoff_seconds) or args.highpass_cutoff_seconds <= 0:
        parser.error("highpass cutoff must be positive and finite")
    if args.prior_native_seconds is not None and (not np.isfinite(args.prior_native_seconds) or args.prior_native_seconds <= 0):
        parser.error("prior native seconds must be positive and finite")

    from fnit.mcflirt import TorchMCFLIRT
    from fnit.feat.temporal import gaussian_highpass, grand_mean_scale
    import fnit.fmri.spatial as spatial_source

    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA was requested but is unavailable")
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats(device)
    timings = {}
    hashes_before = source_hashes()
    header_started = time.perf_counter()
    raw = nib.load(str(args.bold))
    reference = nib.load(str(args.reference))
    mask_image = nib.load(str(args.brain_mask))
    if raw.ndim != 4 or raw.shape[3] < 2 or reference.ndim != 3:
        parser.error("BOLD must be 4D with at least two frames and reference must be 3D")
    total_frames = raw.shape[3]
    frames = total_frames if args.frames is None else args.frames
    if frames > total_frames:
        parser.error("requested frame count exceeds the real BOLD length")
    full_series = frames == total_frames
    if raw.shape[:3] != reference.shape or not np.allclose(raw.affine, reference.affine, atol=1e-4, rtol=0):
        parser.error("BOLD and reference must use the same voxel grid")
    if mask_image.shape != reference.shape or not np.allclose(mask_image.affine, reference.affine, atol=1e-4, rtol=0):
        parser.error("brain-mask must use the reference grid")
    region = np.asarray(mask_image.dataobj) > 0
    if not region.any():
        parser.error("brain-mask is empty")
    seconds_per_unit = {"msec": 0.001, "usec": 0.000001}.get(raw.header.get_xyzt_units()[1], 1.0)
    tr = float(raw.header.get_zooms()[3] * seconds_per_unit if args.tr_seconds is None else args.tr_seconds)
    if not np.isfinite(tr) or tr <= 0:
        parser.error("TR must be positive and finite")
    selected_raw = raw if full_series else nib.Nifti1Image(
        np.asanyarray(raw.dataobj[..., :frames]), raw.affine, raw.header.copy())
    timings["input_headers_mask_and_optional_frame_subset_seconds"] = time.perf_counter() - header_started
    baseline_matrices, baseline_matrix_hash = load_matrices(args.baseline_matrices, total_frames)
    baseline_parameters = np.loadtxt(args.baseline_parameters, dtype=np.float64)
    if baseline_parameters.shape != (total_frames, 6) or not np.isfinite(baseline_parameters).all():
        parser.error("baseline parameters must contain one finite six-column row per original frame")
    # 校验冻结影像几何，但解压与比较均在候选计时之后进行。
    for baseline_path in (args.baseline_filtered, args.baseline_corrected):
        if baseline_path is not None:
            baseline_header = nib.load(str(baseline_path))
            if baseline_header.shape != (*reference.shape, total_frames) or not np.allclose(
                    baseline_header.affine, reference.affine, atol=1e-4, rtol=0):
                parser.error("baseline images must contain the full BOLD series on the reference grid")
    args.output_dir.mkdir(parents=True)
    matrix_dir = args.output_dir / "candidate_mc.private.mat"
    matrix_dir.mkdir()
    sampler = spatial_source.apply_motion_warp
    sampling_calls = []

    def timed_sampler(*sampling_args, **sampling_kwargs):
        synchronize(device)
        started = time.perf_counter()
        value = sampler(*sampling_args, **sampling_kwargs)
        synchronize(device)
        sampling_calls.append(time.perf_counter() - started)
        return value

    synchronize(device)
    compute_started = time.perf_counter()
    with patch.object(spatial_source, "apply_motion_warp", timed_sampler):
        fit = timed_call("motion_run_seconds", lambda: TorchMCFLIRT(device=device).run(
            selected_raw, reference, resample=True, interpolation="spline"), timings, device)
    if len(sampling_calls) != 1 or fit.corrected is None:
        raise RuntimeError("expected one motion-only resampling call and a corrected image")
    timings["motion_resampling_seconds"] = sampling_calls[0]
    timings["motion_run_minus_resampling_seconds"] = timings["motion_run_seconds"] - sampling_calls[0]
    corrected = fit.corrected
    corrected.header.set_zooms((*corrected.header.get_zooms()[:3], tr))
    corrected.header.set_xyzt_units(xyz=corrected.header.get_xyzt_units()[0], t="sec")

    def write_motion():
        np.save(args.output_dir / "candidate_matrices.private.npy", fit.matrices)
        np.save(args.output_dir / "candidate_parameters.private.npy", fit.parameters)
        for frame, matrix in enumerate(fit.matrices):
            np.savetxt(matrix_dir / f"MAT_{frame:04d}", matrix, fmt="%.12g")
        np.savetxt(args.output_dir / "candidate_mc.private.par", fit.parameters, fmt="%.9g")
        nib.save(corrected, str(args.output_dir / "candidate_motion.private.nii.gz"))

    timed_call("motion_artifact_writes_seconds", write_motion, timings, device)
    timings["motion_run_and_artifact_writes_seconds"] = time.perf_counter() - compute_started
    continuation_started = time.perf_counter()

    def prepare_masked():
        data = np.asarray(corrected.dataobj, dtype=np.float32).copy()
        save_image(data.mean(axis=3), reference, args.output_dir / "candidate_unwarp_mean.private.nii.gz")
        save_image(region.astype(np.uint8), reference, args.output_dir / "candidate_mask.private.nii.gz")
        data *= region[..., None]
        return data

    masked = timed_call("feat_mean_mask_prepare_and_writes_seconds", prepare_masked, timings, device)
    scaled, scale_factor = timed_call("feat_grand_mean_scale_seconds",
                                      lambda: grand_mean_scale(masked, region), timings, device)
    del masked
    filtered_data = timed_call("feat_highpass_seconds", lambda: gaussian_highpass(
        scaled, sigma_volumes=args.highpass_cutoff_seconds / (2 * tr),
        device=device, preserve_mean=True), timings, device)
    del scaled

    def write_filtered():
        image = save_image(filtered_data, corrected, args.output_dir / "candidate_filtered.private.nii.gz")
        save_image(filtered_data.mean(axis=3), reference, args.output_dir / "candidate_mean.private.nii.gz")
        return image

    filtered = timed_call("feat_filtered_and_mean_writes_seconds", write_filtered, timings, device)
    timings["feat_fixed_mask_continuation_seconds"] = time.perf_counter() - continuation_started
    timings["candidate_motion_and_fixed_mask_feat_with_writes_seconds"] = time.perf_counter() - compute_started
    peak = {"peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device))} if device.type == "cuda" else {}
    compared_frames = frames if full_series else frames - 1
    comparison_started = time.perf_counter()
    motion_report, framewise = compare_motion(fit.matrices, fit.parameters, baseline_matrices,
                                               baseline_parameters, raw, reference, region, compared_frames)
    written_matrices = np.stack([np.loadtxt(matrix_dir / f"MAT_{frame:04d}")
                                 for frame in range(compared_frames)])
    written_parameters = np.loadtxt(args.output_dir / "candidate_mc.private.par")[:compared_frames]
    motion_report["written_matrix_text_max_abs"] = float(np.max(np.abs(
        written_matrices - baseline_matrices[:compared_frames])))
    motion_report["written_matrix_text_exact_equal"] = bool(np.array_equal(
        written_matrices, baseline_matrices[:compared_frames]))
    motion_report["written_parameter_text_max_abs"] = float(np.max(np.abs(
        written_parameters - baseline_parameters[:compared_frames])))
    motion_report["written_parameter_text_exact_equal"] = bool(np.array_equal(
        written_parameters, baseline_parameters[:compared_frames]))
    if args.baseline_memory_matrices is not None:
        memory_matrices = np.load(args.baseline_memory_matrices, allow_pickle=False)
        memory_parameters = np.load(args.baseline_memory_parameters, allow_pickle=False)
        for values, shape in ((memory_matrices, (total_frames, 4, 4)),
                              (memory_parameters, (total_frames, 6))):
            if values.dtype != np.dtype(np.float64) or values.shape != shape or not np.isfinite(values).all():
                parser.error("baseline memory arrays must be finite float64 with full-series shapes")
        motion_report["unrounded_memory_comparison"] = {
            "matrices_bitwise_equal": bool(np.array_equal(
                fit.matrices[:compared_frames].view(np.uint64),
                memory_matrices[:compared_frames].view(np.uint64))),
            "parameters_bitwise_equal": bool(np.array_equal(
                fit.parameters[:compared_frames].view(np.uint64),
                memory_parameters[:compared_frames].view(np.uint64))),
            "matrix_max_abs": float(np.max(np.abs(fit.matrices[:compared_frames] - memory_matrices[:compared_frames]))),
            "parameters_max_abs": float(np.max(np.abs(fit.parameters[:compared_frames] - memory_parameters[:compared_frames]))),
            "baseline_memory_matrices_sha256": sha256(args.baseline_memory_matrices),
            "baseline_memory_parameters_sha256": sha256(args.baseline_memory_parameters),
        }
    report = {"subjects": 1, "real_input_frames": total_frames, "candidate_frames": frames,
              "full_series": full_series, "source_revision": args.source_revision,
              "baseline_kind": args.baseline_kind, "baseline_source_revision": args.baseline_source_revision,
              "device": str(device), "threads": torch.get_num_threads(), "stage_iterations": [1, 1, 1],
              "interpolation": "spline", "tf32": bool(torch.backends.cuda.matmul.allow_tf32),
              "half_precision": False, "tr_seconds": tr,
              "highpass_cutoff_seconds": args.highpass_cutoff_seconds, "intensity_factor": float(scale_factor),
              "cost_evaluations": int(fit.cost_evaluations), "sampling_calls": len(sampling_calls),
              "timings": timings, "motion_comparison": motion_report, **peak,
              "timing_scope": {
                  "motion_run": "Actual TorchMCFLIRT.run with pre-opened NIfTI headers; full-series mode includes input array read/decompression, preparation, estimation, spline sampling and output dtype cast; excludes writes. First-N mode instead preloads its subset before this timer.",
                  "motion_run_minus_resampling": "Motion run minus the synchronized apply_motion_warp wrapper; includes input array loading, preparation, estimation and output dtype cast, not a pure optimizer timer.",
                  "motion_run_and_artifact_writes": "Motion run plus private NIfTI, full-precision array, MAT text and par text writes.",
                  "feat_continuation": "Supplied fixed brain mask: corrected mean/mask writes, masking, grand-mean scaling, highpass and filtered/mean writes. Brain extraction, BIDS lookup, warp estimation and complete run_feat_core entry point are outside this boundary.",
                  "excluded": "Baseline reads, hashes and numerical comparisons are outside candidate compute/write timings. The one sampler wrapper adds only boundary synchronization and a wall-clock read. First execution may include compilation."},
              "software": {"python": sys.version.split()[0], "numpy": np.__version__,
                           "nibabel": nib.__version__, "torch": torch.__version__},
              "privacy": "Only anonymous aggregates and hashes are public. Images, per-frame matrices, parameters and framewise metrics remain in the private output directory.",
              "comparison_note": "Baseline matrix/par text can be rounded relative to its in-memory fit. Frozen FNIT comparison does not establish native-FSL equivalence. A first-N control changes the final coarse-stage initial matrix and cannot validate full-series FEAT filtering."}
    if device.type == "cuda":
        report["cuda_device_name"] = torch.cuda.get_device_name(device)
        report["within_20_gb_peak_allocated"] = peak["peak_cuda_allocated_bytes"] < 20_000_000_000
        report["within_20_gb_peak_reserved"] = peak["peak_cuda_reserved_bytes"] < 20_000_000_000
    if full_series:
        report["filtered_comparison"] = compare_image(filtered, args.baseline_filtered, region, frames)
        if args.baseline_corrected is not None:
            report["corrected_comparison"] = compare_image(corrected, args.baseline_corrected, region, frames)
    else:
        report["filtered_comparison"] = {"skipped": "First-N temporal filtering differs from the full-series baseline."}
        report["corrected_comparison"] = {"skipped": "Preliminary frame control changes its final-frame initialization; use full series for image comparison."}
    report["comparison_seconds_including_baseline_image_reads"] = time.perf_counter() - comparison_started
    if args.prior_native_seconds is not None:
        report["prior_native_command_observation"] = {
            "wall_seconds": args.prior_native_seconds,
            "scope": "Historical independent native MCFLIRT command includes estimation, final resampling and writes; server load, cache state and exact output set were not matched by this driver.",
            "matched_speedup_claim": False}
    input_paths = {"bold": args.bold, "reference": args.reference, "brain_mask": args.brain_mask,
                   "baseline_parameters": args.baseline_parameters, "baseline_filtered": args.baseline_filtered}
    if args.baseline_corrected is not None:
        input_paths["baseline_corrected"] = args.baseline_corrected
    report["input_sha256"] = {name: sha256(path) for name, path in input_paths.items()}
    report["input_sha256"]["baseline_matrices_frame_order"] = baseline_matrix_hash
    report["candidate_artifact_sha256"] = {name: sha256(args.output_dir / filename) for name, filename in (
        ("matrices_array", "candidate_matrices.private.npy"), ("parameters_array", "candidate_parameters.private.npy"),
        ("parameters_text", "candidate_mc.private.par"), ("corrected", "candidate_motion.private.nii.gz"),
        ("filtered", "candidate_filtered.private.nii.gz"))}
    candidate_matrix_digest = hashlib.sha256()
    for frame in range(frames):
        candidate_matrix_digest.update((matrix_dir / f"MAT_{frame:04d}").read_bytes())
    report["candidate_artifact_sha256"]["matrices_text_frame_order"] = candidate_matrix_digest.hexdigest()
    report["sources_sha256"] = hashes_before
    report["runtime_sources_unchanged"] = hashes_before == source_hashes()
    report["valid_run"] = report["runtime_sources_unchanged"]
    (args.output_dir / "framewise.private.json").write_text(json.dumps(framewise, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "summary.public.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, allow_nan=False), flush=True)
    return 0 if report["valid_run"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
