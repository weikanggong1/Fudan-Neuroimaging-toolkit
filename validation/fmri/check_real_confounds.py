"""用捕获的真实 BOLD 验证混杂回归，公开报告只包含哈希和数值。

--run-root 是 benchmark_bids.py 保存中间结果的目录；--native-clean 是
最终原生空间 BOLD；--source-revision 标记实测提交；--report-out 保存匿名
JSON。--chunk-voxels 控制独立 NumPy float64 参考计算的内存，不改变投影。
参考用 SVD 正交基计算残差，不调用 FNIT 的 clean_confounds。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import nibabel as nib
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bandpass(values, keep):
    if keep is None:
        return values
    spectrum = np.fft.rfft(values, axis=0)
    spectrum[~keep] = 0
    return np.fft.irfft(spectrum, n=values.shape[0], axis=0)


def orthogonal_basis(design, keep=None, *, normalize=True):
    values = design.astype(np.float64, copy=True)
    if normalize:
        active = np.r_[True, np.ptp(values[:, 1:], axis=0) > 0]
        values = values[:, active]
        values[:, 1:] -= values[:, 1:].mean(axis=0)
        norms = np.linalg.norm(values, axis=0)
        values = values[:, norms > 0] / norms[norms > 0]
    values = bandpass(values, keep)
    if normalize:
        norms = np.linalg.norm(values, axis=0)
        active = norms > np.finfo(np.float64).eps * max(values.shape)
        values = values[:, active] / norms[active]
    left, singular_values, _ = np.linalg.svd(values, full_matrices=False)
    cutoff = singular_values[0] * 1e-8 if len(singular_values) else 0.0
    rank = int(np.count_nonzero(singular_values > cutoff))
    return left[:, :rank], {
        "rank": rank, "columns": int(values.shape[1]),
        "singular_value_cutoff": float(cutoff),
        "smallest_six_singular_values": singular_values[-6:].tolist(),
    }


class Comparison:
    def __init__(self):
        self.count = 0
        self.error_abs = self.error_sq = self.max_error = 0.0
        self.candidate_sum = self.reference_sum = 0.0
        self.candidate_sq = self.reference_sq = self.product = 0.0
        self.voxel_correlations = []

    def add(self, candidate, reference):
        difference = candidate - reference
        self.count += difference.size
        self.error_abs += float(np.abs(difference).sum())
        self.error_sq += float(np.square(difference).sum())
        self.max_error = max(self.max_error, float(np.abs(difference).max()))
        self.candidate_sum += float(candidate.sum())
        self.reference_sum += float(reference.sum())
        self.candidate_sq += float(np.square(candidate).sum())
        self.reference_sq += float(np.square(reference).sum())
        self.product += float((candidate * reference).sum())
        candidate_centered = candidate - candidate.mean(axis=0)
        reference_centered = reference - reference.mean(axis=0)
        candidate_norm = np.linalg.norm(candidate_centered, axis=0)
        reference_norm = np.linalg.norm(reference_centered, axis=0)
        threshold = 1e-6 * math.sqrt(candidate.shape[0])
        valid = (candidate_norm > threshold) & (reference_norm > threshold)
        correlations = ((candidate_centered[:, valid] * reference_centered[:, valid]).sum(axis=0)
                        / (candidate_norm[valid] * reference_norm[valid]))
        self.voxel_correlations.append(correlations)

    def report(self):
        numerator = self.product - self.candidate_sum * self.reference_sum / self.count
        candidate_variance = self.candidate_sq - self.candidate_sum**2 / self.count
        reference_variance = self.reference_sq - self.reference_sum**2 / self.count
        denominator = math.sqrt(max(0.0, candidate_variance * reference_variance))
        correlations = np.concatenate(self.voxel_correlations)
        return {
            "evaluated_values": self.count,
            "pooled_correlation": float(np.clip(numerator / denominator, -1, 1)) if denominator else None,
            "mae": self.error_abs / self.count,
            "rmse": math.sqrt(self.error_sq / self.count),
            "max_absolute_difference": self.max_error,
            "candidate_rms": math.sqrt(self.candidate_sq / self.count),
            "reference_rms": math.sqrt(self.reference_sq / self.count),
            "valid_temporal_correlation_voxels": int(correlations.size),
            "median_voxel_temporal_correlation": float(np.median(correlations)) if correlations.size else None,
            "per_voxel_rms_threshold": 1e-6,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--native-clean", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--chunk-voxels", type=int, default=4096)
    args = parser.parse_args()
    if args.chunk_voxels < 1:
        parser.error("--chunk-voxels must be positive")
    started = time.perf_counter()
    root = args.run_root
    paths = {
        "aroma_input": root / "intermediates/aroma/filtered_func_data_aroma.nii.gz",
        "brain_mask": root / "intermediates/feat/mask.nii.gz",
        "motion_parameters": root / "intermediates/feat/mc/prefiltered_func_data_mcf.par",
        "final_native": args.native_clean,
    }
    metadata = json.loads(args.native_clean.with_name(args.native_clean.name.removesuffix(".nii.gz") + ".json").read_text())
    configuration = metadata["FNIT"]["Configuration"]
    source = metadata["FNIT"]["Source"]
    captured_report = root / "fmri_volume.public.json"
    if captured_report.is_file() and json.loads(captured_report.read_text())["source_revision"] != args.source_revision:
        raise ValueError("source revision differs from the captured benchmark")
    manifest_digest = hashlib.sha256(json.dumps(
        source["SourceSHA256"], sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    if manifest_digest != source["SourceManifestSHA256"]:
        raise ValueError("runtime source manifest digest is inconsistent")
    package = root / "source/src/fnit"
    mismatches = [name for name, recorded in source["SourceSHA256"].items()
                  if sha256(package / name) != recorded]
    if mismatches:
        raise ValueError("runtime source manifest does not match the captured source")
    image = nib.load(str(paths["aroma_input"]))
    final_image = nib.load(str(paths["final_native"]))
    if image.shape != final_image.shape or not np.allclose(image.affine, final_image.affine, rtol=0, atol=1e-4):
        raise ValueError("AROMA input and final native BOLD must share their grid")
    data = np.asarray(image.dataobj, dtype=np.float32)
    final_data = np.asarray(final_image.dataobj, dtype=np.float32)
    if not np.isfinite(data).all() or not np.isfinite(final_data).all():
        raise ValueError("real BOLD contains nonfinite values")
    frame_count = data.shape[3]

    def load_mask(path):
        mask_image = nib.load(str(path))
        if mask_image.shape != data.shape[:3] or not np.allclose(mask_image.affine, image.affine, rtol=0, atol=1e-4):
            raise ValueError("captured mask must match the AROMA input grid")
        mask = np.asarray(mask_image.dataobj) > 0
        if not mask.any():
            raise ValueError("captured mask is empty")
        return mask

    brain_mask = load_mask(paths["brain_mask"])
    time_axis = np.linspace(-1.0, 1.0, frame_count)
    columns = [np.ones(frame_count), time_axis, (3 * time_axis**2 - 1) / 2]
    tissue_columns = {}
    masks = {}
    for name, filename in (("wm", "wm_mask.nii.gz"), ("csf", "regression_csf_mask.nii.gz")):
        if configuration[f"regress_{name}"]:
            paths[f"{name}_mask"] = root / "intermediates/masks" / filename
            mask = load_mask(paths[f"{name}_mask"])
            if np.any(mask & ~brain_mask):
                raise ValueError("tissue mask contains voxels outside the brain mask")
            tissue_columns[name] = len(columns)
            columns.append(data[mask].mean(axis=0, dtype=np.float64))
            masks[name] = {"shape": list(mask.shape), "voxels": int(mask.sum())}
    if configuration["global_signal"]:
        columns.append(data[brain_mask].mean(axis=0, dtype=np.float64))
    motion = np.loadtxt(paths["motion_parameters"], ndmin=2)
    if motion.shape != (frame_count, 6):
        raise ValueError("captured motion must have one six-column row per frame")

    def make_design(motion_parameters):
        selected_columns = list(columns)
        if configuration["regress_motion"]:
            model = configuration["motion_model"]
            if model == 6:
                motion_design = motion_parameters
            elif model == 12:
                differences = np.vstack((np.zeros((1, 6)), np.diff(motion_parameters, axis=0)))
                motion_design = np.column_stack((motion_parameters, differences))
            elif model == 24:
                previous = np.vstack((np.zeros((1, 6)), motion_parameters[:-1]))
                motion_design = np.column_stack((motion_parameters, previous, motion_parameters**2, previous**2))
            else:
                raise ValueError("unsupported motion model")
            selected_columns.extend(motion_design.T)
        return np.column_stack(selected_columns)

    design = make_design(motion)
    keep = None
    if configuration["bandpass"] is not None:
        low, high = configuration["bandpass"]
        frequencies = np.fft.rfftfreq(frame_count, metadata["RepetitionTime"])
        keep = (frequencies >= low) & (frequencies <= high)
    reference_basis, reference_rank = orthogonal_basis(design, keep)
    legacy_basis, legacy_rank = orthogonal_basis(design, keep, normalize=False)
    degrees_motion = motion.copy()
    degrees_motion[:, :3] *= 180 / np.pi
    degrees_design = make_design(degrees_motion)
    degrees_basis, degrees_rank = orthogonal_basis(degrees_design, keep)
    _, legacy_degrees_rank = orthogonal_basis(degrees_design, keep, normalize=False)
    changed_design = design.copy()
    for name, index in tissue_columns.items():
        scale, offset = (8, 10000000) if name == "wm" else (4, 5000000)
        changed_design[:, index] = design[:, index] * scale + offset
    changed_basis, changed_rank = orthogonal_basis(changed_design, keep)
    comparisons = {name: Comparison() for name in (
        "final_vs_independent_reference", "legacy_vs_fixed_same_input",
        "fixed_radians_vs_degrees", "fixed_original_vs_scaled_tissues",
    )}
    flat = data.reshape((-1, frame_count))
    final_flat = final_data.reshape((-1, frame_count))
    voxel_indices = np.flatnonzero(brain_mask.ravel())
    float32_oracle_max_difference = 0.0
    for start in range(0, len(voxel_indices), args.chunk_voxels):
        indices = voxel_indices[start:start + args.chunk_voxels]
        series = bandpass(flat[indices].T.astype(np.float64), keep)
        reference = series - reference_basis @ (reference_basis.T @ series)
        actual = final_flat[indices].T.astype(np.float64)
        comparisons["final_vs_independent_reference"].add(actual, reference)
        float32_oracle_max_difference = max(float32_oracle_max_difference,
            float(np.abs(actual - reference.astype(np.float32).astype(np.float64)).max()))
        for name, basis in (
            ("legacy_vs_fixed_same_input", legacy_basis),
            ("fixed_radians_vs_degrees", degrees_basis),
            ("fixed_original_vs_scaled_tissues", changed_basis),
        ):
            residual = series - basis @ (basis.T @ series)
            comparisons[name].add(residual, reference)
    outside_max = 0.0
    outside = ~brain_mask.ravel()
    for start in range(0, len(outside), args.chunk_voxels):
        values = final_flat[start:start + args.chunk_voxels][outside[start:start + args.chunk_voxels]]
        if values.size:
            outside_max = max(outside_max, float(np.abs(values).max()))
    checks = {name: comparison.report() for name, comparison in comparisons.items()}
    checks["final_vs_independent_reference"]["max_difference_from_float32_oracle"] = float32_oracle_max_difference
    if float32_oracle_max_difference > 1e-3 or outside_max != 0:
        raise ValueError("final output does not match the independent reference or brain mask")
    report = {
        "schema_version": 1, "source_revision": args.source_revision,
        "data": {"kind": "one real UK Biobank run", "anonymous_id": "real_run_01",
                 "bold_shape": list(data.shape), "brain_mask_voxels": len(voxel_indices),
                 "tr_seconds": metadata["RepetitionTime"], "tissue_masks": masks},
        "configuration": {key: configuration[key] for key in (
            "regress_wm", "regress_csf", "regress_motion", "motion_model", "global_signal", "bandpass")},
        "input_sha256": {name: sha256(path) for name, path in paths.items()},
        "runtime_source_manifest_verified": True,
        "runtime_source_manifest_sha256": source["SourceManifestSHA256"],
        "confounds_source_sha256": source["SourceSHA256"]["fmri/confounds.py"],
        "independent_reference": {"method": "NumPy float64 SVD orthogonal projection",
                                  "numpy": np.__version__, "nibabel": nib.__version__,
                                  "script_sha256": sha256(__file__),
                                  "all_brain_mask_voxels_evaluated": True},
        "design": {"shape": list(design.shape), "legacy_radians": legacy_rank,
                   "legacy_degrees": legacy_degrees_rank, "fixed_radians": reference_rank,
                   "fixed_degrees": degrees_rank, "fixed_scaled_tissues": changed_rank},
        "unit_scale_controls": {
            "motion_rotation_change": "radians multiplied by 180/pi; translations unchanged",
            "tissue_signal_changes": {"wm": {"scale": 8, "offset": 10000000},
                                      "csf": {"scale": 4, "offset": 5000000}},
        },
        "checks": checks, "outside_brain_mask_max_absolute_value": outside_max,
        "validation_wall_seconds": time.perf_counter() - started,
        "limits": ["Legacy/fixed and unit/scale comparisons hold the same captured AROMA BOLD, tissue signals and motion fixed; no new pipeline or official UKB comparison is implied."],
        "privacy": "Only anonymous scalar metrics, shapes and hashes are included; no private paths or image arrays.",
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
