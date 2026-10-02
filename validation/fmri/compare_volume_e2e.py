"""比较从原始数据独立运行的 FNIT 与 fMRIPrep 完整体积结果。

私有清单含 fnit_report、reference_report、raw_inputs 和 mni_preproc
（candidate、reference、mask）。只在双方已有的共同 MNI 网格上计算，
不追加配准、重采样、平滑或强度拟合。逐体素统计图只写到私有目录。
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import nibabel as nib
import numpy as np

import compare_matched_pipeline as comparison_helper
from compare_matched_pipeline import correlations, load_mask, same_grid


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_at(value, parent):
    path = Path(value)
    return path if path.is_absolute() else parent / path


def require_raw_identity(fnit, reference, raw):
    """核对真实执行记录及当前原始文件，不从 preproc 推断 raw 来源。"""
    inputs = fnit.get("input_sha256", fnit.get("volume", {}).get("input_sha256", {}))
    hashes = {key: sha256(path) for key, path in raw.items()}
    for key, reference_key in (("bold", "raw_bold_sha256"), ("t1w", "raw_t1_sha256")):
        if hashes.get(key) != inputs.get(key) or hashes.get(key) != reference.get(reference_key):
            raise ValueError("independent pipelines do not attest the same raw " + key)
    if "sbref" in hashes and hashes["sbref"] != inputs.get("sbref"):
        raise ValueError("candidate SBRef hash differs from the raw file")
    if hashes.get("sbref") != reference.get("raw_sbref_sha256"):
        raise ValueError("original SBRef hash differs from the raw file")
    if reference.get("exit_code") != 0 or reference.get("validation_complete") is not True:
        raise ValueError("original complete workflow has not passed its exit/output checks")
    if reference.get("STC") is not False or reference.get("SDC") is not False:
        raise ValueError("this benchmark requires STC and SDC disabled")
    if fnit.get("configuration", {}).get("slice_timing") is not False:
        raise ValueError("candidate workflow must attest STC disabled")
    if not fnit.get("source_unchanged_during_run", False):
        raise ValueError("candidate workflow source identity changed or was not checked")
    return hashes


def temporal_maps(candidate, reference, mask):
    # 每个位置的全部原始帧参与计算；常数时序不报告相关性。
    correlation = np.full(mask.shape, np.nan, np.float32)
    error = np.full(mask.shape, np.nan, np.float32)
    coordinates = np.flatnonzero(mask.ravel())
    left, right = candidate.reshape(-1, candidate.shape[-1]), reference.reshape(-1, reference.shape[-1])
    for start in range(0, len(coordinates), 4096):
        rows = coordinates[start:start + 4096]
        x, y = left[rows].astype(np.float64), right[rows].astype(np.float64)
        error.ravel()[rows] = np.sqrt(np.square(x - y).mean(axis=1))
        x -= x.mean(axis=1, keepdims=True)
        y -= y.mean(axis=1, keepdims=True)
        xx, yy = np.square(x).sum(axis=1), np.square(y).sum(axis=1)
        valid = (xx > 1e-12 * x.shape[1]) & (yy > 1e-12 * y.shape[1])
        correlation.ravel()[rows[valid]] = np.clip(
            (x[valid] * y[valid]).sum(axis=1) / np.sqrt(xx[valid] * yy[valid]), -1, 1)
    return correlation, error


def common_mni_views(candidate, reference):
    """只排列/翻转索引；空间格点不同仍拒绝，绝不执行插值。"""
    transform = nib.orientations.ornt_transform(
        nib.orientations.io_orientation(reference.affine),
        nib.orientations.io_orientation(candidate.affine))
    view = reference.as_reoriented(transform)
    same_grid(candidate, view, complete_shape=True)
    return [candidate, view], {
        "candidate_saved_axis_codes": list(nib.aff2axcodes(candidate.affine)),
        "reference_saved_axis_codes": list(nib.aff2axcodes(reference.affine)),
        "reference_index_transform": transform.tolist(),
        "reference_reindexed": not np.array_equal(transform, [[0, 1], [1, 1], [2, 1]]),
        "same_physical_lattice_verified": True,
        "interpolation_performed": False,
        "original_files_and_hashes_preserved": True,
    }


def plot_maps(maps, mask, output, frames):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "figure.facecolor": "white"})
    cuts = [int(np.median(np.nonzero(mask)[axis])) for axis in range(3)]
    columns = [("Original mean", maps["reference_mean"], "gray", 0,
                float(np.percentile(maps["reference_mean"][mask], 99))),
               ("FNIT mean", maps["candidate_mean"], "gray", 0,
                float(np.percentile(maps["reference_mean"][mask], 99))),
               ("Original temporal SD", maps["reference_sd"], "magma", 0,
                float(np.percentile(maps["reference_sd"][mask], 99))),
               ("FNIT temporal SD", maps["candidate_sd"], "magma", 0,
                float(np.percentile(maps["reference_sd"][mask], 99))),
               ("Temporal Pearson r", maps["temporal_r"], "viridis", -1, 1)]
    figure, axes = plt.subplots(3, len(columns), figsize=(15, 8), constrained_layout=True)
    for row, axis in enumerate(range(3)):
        for col, (name, values, cmap, low, high) in enumerate(columns):
            section = np.rot90(np.take(values, cuts[axis], axis=axis))
            visible = np.rot90(np.take(mask, cuts[axis], axis=axis))
            view = axes[row, col].imshow(np.ma.masked_where(~visible, section), cmap=cmap,
                                         vmin=low, vmax=high, interpolation="nearest")
            axes[row, col].set_axis_off()
            if row == 0:
                axes[row, col].set_title(name)
            if row == 2:
                figure.colorbar(view, ax=axes[:, col], shrink=.55, pad=.01)
    figure.suptitle(f"Independent raw-to-MNI preproc: all {frames} frames; no extra smoothing")
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return {"voxel_cuts_xyz": cuts, "interpolation": "nearest for display only",
            "columns": [{"name": name, "limits": [low, high]}
                        for name, _, _, low, high in columns]}


def compare(manifest, directory, private_output, figure_out=None):
    fnit = json.loads(path_at(manifest["fnit_report"], directory).read_text())
    reference = json.loads(path_at(manifest["reference_report"], directory).read_text())
    raw = {name: path_at(value, directory) for name, value in manifest["raw_inputs"].items()}
    if set(raw) != {"bold", "t1w", "sbref"}:
        raise ValueError("enumerate raw BOLD, T1w and SBRef without identifying labels")
    identity = require_raw_identity(fnit, reference, raw)
    pair = {key: str(path_at(manifest["mni_preproc"][key], directory))
            for key in ("candidate", "reference", "mask")}
    pair["kind"] = "bold"
    candidate_sha = sha256(pair["candidate"])
    expected_candidate = fnit["checks"]["volume"]["preproc_mni"]["sha256"]
    reference_sha = sha256(pair["reference"])
    reference_outputs = [row for row in reference["outputs"]
                         if row["kind"] == "MNI152NLin6Asym"]
    if candidate_sha != expected_candidate or reference_sha not in {
            row["sha256"] for row in reference_outputs}:
        raise ValueError("comparison output is not bound to its completed workflow report")
    if sha256(pair["mask"]) != fnit["input_sha256"]["mni_brain_mask"]:
        raise ValueError("comparison mask must be the pinned input MNI brain mask")
    original_images = [nib.load(pair[name]) for name in ("candidate", "reference")]
    images, orientation = common_mni_views(*original_images)
    arrays = [np.asarray(image.dataobj, np.float32) for image in images]
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("complete MNI outputs must contain only finite values")
    if not all(image.header.get_xyzt_units() == ("mm", "sec") for image in images):
        raise ValueError("published MNI outputs must use millimetres and seconds")
    tr = [float(image.header.get_zooms()[3]) for image in images]
    if not np.isclose(tr[0], tr[1], rtol=1e-5, atol=1e-6):
        raise ValueError("published MNI repetition times differ")
    mask = load_mask(pair["mask"], images[0])
    metric = {
        "shape": list(images[0].shape), "same_grid_after_lossless_reindex": True,
        "all_values_finite": True,
        "sha256": {"candidate": candidate_sha, "reference": reference_sha},
        "saved_dtype": {key: str(image.get_data_dtype()) for key, image in
                        zip(("candidate", "reference"), original_images)},
        "mask_voxels": int(mask.sum()),
        "comparison_mask_sha256": {"mask": sha256(pair["mask"])},
        "tr_seconds": tr[0],
        "metrics": correlations(arrays[0][mask], arrays[1][mask], temporal=True),
        "ranges": {key: {"min": float(array.min()), "max": float(array.max())}
                   for key, array in zip(("candidate", "reference"), arrays)},
        "metrics_exclude_zero_series": False,
    }
    if images[0].shape[3] != reference["input_frames"]:
        raise ValueError("candidate/reference do not contain every input frame")
    raw_bold = nib.load(str(raw["bold"]))
    if raw_bold.ndim != 4 or raw_bold.shape[3] != images[0].shape[3]:
        raise ValueError("candidate/reference frame count differs from raw BOLD")
    correlation, error = temporal_maps(*arrays, mask)
    varying = [array.std(axis=3, dtype=np.float64) > 1e-6 for array in arrays]
    maps = {"candidate_mean": arrays[0].mean(axis=3, dtype=np.float64).astype(np.float32),
            "reference_mean": arrays[1].mean(axis=3, dtype=np.float64).astype(np.float32),
            "candidate_sd": arrays[0].std(axis=3, dtype=np.float64).astype(np.float32),
            "reference_sd": arrays[1].std(axis=3, dtype=np.float64).astype(np.float32),
            "temporal_r": correlation, "temporal_rmse": error}
    map_metrics = {name: correlations(maps["candidate_" + name][mask],
                                      maps["reference_" + name][mask]) for name in ("mean", "sd")}
    private_output.mkdir(parents=True, exist_ok=True)
    for name, values in maps.items():
        header = images[0].header.copy()
        header.set_data_dtype(np.float32)
        nib.save(nib.Nifti1Image(values, images[0].affine, header),
                 private_output / (name + ".private.nii.gz"))
    revision = fnit.get("source_revision", "")
    if not re.fullmatch("[0-9a-f]{7,40}", revision):
        raise ValueError("candidate source revision must be attested")
    report = {"schema_version": 1, "candidate_revision": revision,
              "reference": "fMRIPrep 25.2.4 plus official newMSM",
              "input_sha256": identity, "same_raw_bold_t1w_verified": True,
              "execution_report_sha256": {
                  "fnit": sha256(path_at(manifest["fnit_report"], directory)),
                  "reference": sha256(path_at(manifest["reference_report"], directory))},
              "final_outputs_bound_to_execution_reports": True,
              "lossless_orientation_views": orientation,
              "comparison_mask": "Pinned input MNI brain mask, independent of each estimated coverage mask; zeros within this domain remain part of numerical errors.",
              "mni_preproc": metric, "mean_and_temporal_sd": map_metrics,
              "temporal_coverage_in_fixed_mask": {
                  "mask_voxels": int(mask.sum()),
                  "candidate_varying_voxels": int((mask & varying[0]).sum()),
                  "reference_varying_voxels": int((mask & varying[1]).sum()),
                  "both_varying_voxels": int((mask & varying[0] & varying[1]).sum()),
                  "candidate_only_varying_voxels": int((mask & varying[0] & ~varying[1]).sum()),
                  "reference_only_varying_voxels": int((mask & varying[1] & ~varying[0]).sum()),
                  "both_constant_voxels": int((mask & ~varying[0] & ~varying[1]).sum()),
                  "demeaned_rms_threshold": 1e-6},
              "comparison_script_sha256": sha256(__file__),
              "comparison_helper_sha256": sha256(comparison_helper.__file__),
              "scope": "Independent raw-to-volume estimates, all frames in a fixed MNI brain mask; no additional interpolation, smoothing or fitted intensity scale.",
              "privacy": "Only anonymous aggregates, hashes and permitted template-space PNG; per-voxel maps remain private."}
    if figure_out:
        figure_out.parent.mkdir(parents=True, exist_ok=True)
        report["figure"] = {**plot_maps(maps, mask, figure_out, images[0].shape[3]),
                            "sha256": sha256(figure_out)}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--figure-out", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    report = compare(json.loads(args.manifest.read_text()), args.manifest.parent,
                     args.private_output, args.figure_out)
    report["validation_wall_seconds_excluded_from_pipeline"] = time.perf_counter() - started
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"candidate_revision": report["candidate_revision"],
                      "metrics": report["mni_preproc"]["metrics"]}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
