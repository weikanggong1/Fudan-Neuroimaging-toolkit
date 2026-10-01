"""比较一例真实 volume 新旧结果及官方发布数据，影像仅写入私有目录。"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch

from fnit.fmri.normalization import resample_world


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def one(root, pattern):
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected one {pattern}; found {len(matches)}")
    return matches[0]


def load_region(path, reference, mask):
    image = nib.load(path)
    if image.shape[:3] != reference.shape[:3] or not np.allclose(
        image.affine, reference.affine, rtol=0, atol=1e-4
    ):
        raise ValueError("Comparison images have different grids")
    values = np.asarray(image.dataobj, dtype=np.float32)[mask]
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite comparison input")
    return values


def metrics(left, right):
    if left.shape != right.shape:
        raise ValueError("Comparison frame counts differ")
    per_voxel = []
    cross = left_energy = right_energy = squared_error = absolute_error = 0.0
    count = 0
    for start in range(0, len(left), 4096):
        x = left[start:start + 4096].astype(np.float64)
        y = right[start:start + 4096].astype(np.float64)
        error = x - y
        squared_error += np.square(error).sum()
        absolute_error += np.abs(error).sum()
        count += error.size
        x -= x.mean(axis=1, keepdims=True)
        y -= y.mean(axis=1, keepdims=True)
        xx = np.square(x).sum(axis=1)
        yy = np.square(y).sum(axis=1)
        xy = (x * y).sum(axis=1)
        valid = (xx > 1e-12) & (yy > 1e-12)
        per_voxel.append(xy[valid] / np.sqrt(xx[valid] * yy[valid]))
        cross += xy.sum()
        left_energy += xx.sum()
        right_energy += yy.sum()
    r = np.concatenate(per_voxel)
    return {
        "valid_voxels": int(len(r)),
        "mean_voxel_temporal_r": float(r.mean()),
        "median_voxel_temporal_r": float(np.median(r)),
        "pooled_demeaned_r": float(cross / np.sqrt(left_energy * right_energy)),
        "mae": float(absolute_error / count),
        "rmse": float(np.sqrt(squared_error / count)),
    }


def common_pairs(paths, reference, region, pairs):
    values = {name: load_region(path, reference, region) for name, path in paths.items()}
    valid = np.ones(int(region.sum()), dtype=bool)
    for array in values.values():
        valid &= array.std(axis=1, dtype=np.float64) > 1e-8
    selected = {name: array[valid] for name, array in values.items()}
    result = {
        "mask_voxels": int(region.sum()), "common_varying_voxels": int(valid.sum()),
        "frames": int(next(iter(values.values())).shape[1]),
        "input_sha256": {name: digest(path) for name, path in paths.items()},
        "pairs": {a + "_vs_" + b: metrics(selected[a], selected[b]) for a, b in pairs},
    }
    return result


def geometry_change(current, previous, mask, template):
    xyz = template.affine[:3, :3] @ np.indices(template.shape, dtype=np.float64).reshape(3, -1)
    xyz += template.affine[:3, 3:4]
    t1, epi, hashes = [], [], {}
    for label, root in [("current", current), ("previous", previous)]:
        field_path = root / "resampling_inputs/mni_to_t1_pull_ras.nii.gz"
        matrix_path = root / "resampling_inputs/reference_to_source_world.txt"
        field = nib.load(field_path)
        if field.shape != (*template.shape, 3) or not np.allclose(
            field.affine, template.affine, atol=1e-4, rtol=0
        ):
            raise ValueError("Pull field grid differs")
        mapped = xyz + np.asarray(field.dataobj, dtype=np.float64).reshape(-1, 3).T
        matrix = np.loadtxt(matrix_path)
        t1.append(mapped[:, mask.ravel()])
        epi.append((matrix[:3, :3] @ mapped + matrix[:3, 3:4])[:, mask.ravel()])
        hashes[label] = {"pull": digest(field_path), "matrix": digest(matrix_path)}
    def distances(values):
        distance = np.linalg.norm(values[0] - values[1], axis=0)
        return {"mean_mm": float(distance.mean()), "median_mm": float(np.median(distance)),
                "p95_mm": float(np.percentile(distance, 95)), "max_mm": float(distance.max())}
    return {"mni_to_t1_world_change": distances(t1),
            "mni_to_epi_world_change": distances(epi), "sha256": hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("current-root", "previous-root", "official-native", "official-mask",
                 "official-mni", "official-pre-ica", "official-pre-mask", "template",
                 "private-output", "report-out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--current-revision", required=True)
    parser.add_argument("--previous-revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(8)
    args.private_output.mkdir(parents=True, exist_ok=True)
    current_func = args.current_root / "derivatives/sub-benchmark/func"
    previous_func = args.previous_root / "derivatives/sub-benchmark/func"
    new_native = one(current_func, "*_space-boldref_desc-clean_bold.nii.gz")
    old_native = one(previous_func, "*_space-boldref_desc-clean_bold.nii.gz")
    new_mni = one(current_func, "*_space-MNI152NLin6Asym_res-2_desc-clean_bold.nii.gz")
    old_mni = one(previous_func, "*_space-MNI152NLin6Asym_res-2_desc-clean_bold.nii.gz")
    new_mask = one(current_func, "*_space-MNI152NLin6Asym_res-2_desc-brain_mask.nii.gz")
    old_mask = one(previous_func, "*_space-MNI152NLin6Asym_res-2_desc-brain_mask.nii.gz")
    new_feat = args.current_root / "intermediates/feat/filtered_func_data.nii.gz"
    old_feat = args.previous_root / "feat_saved/filtered_func_data.nii.gz"
    native_reference = nib.load(new_native)
    native_mask = np.ones(native_reference.shape[:3], dtype=bool)
    for path in (args.current_root / "intermediates/feat/mask.nii.gz",
                 args.previous_root / "feat_saved/mask.nii.gz", args.official_mask):
        native_mask &= load_region(path, native_reference, np.ones(native_mask.shape, dtype=bool)).reshape(native_mask.shape) > 0
    native = common_pairs(
        {"current_clean": new_native, "previous_clean": old_native,
         "official_fix": args.official_native, "current_pre_ica": new_feat,
         "previous_pre_ica": old_feat}, native_reference, native_mask,
        [("current_clean", "previous_clean"), ("current_clean", "official_fix"),
         ("previous_clean", "official_fix"), ("current_pre_ica", "previous_pre_ica"),
         ("current_pre_ica", "current_clean")],
    )
    print("native comparisons complete", flush=True)
    pre_mask = np.asarray(nib.load(args.current_root / "intermediates/feat/mask.nii.gz").dataobj) > 0
    pre_mask &= load_region(args.official_pre_mask, native_reference, np.ones(pre_mask.shape, dtype=bool)).reshape(pre_mask.shape) > 0
    preprocessing = common_pairs(
        {"current_pre_ica": new_feat, "official_no_gdc_no_b0_pre_ica": args.official_pre_ica},
        native_reference, pre_mask, [("current_pre_ica", "official_no_gdc_no_b0_pre_ica")],
    )
    control = args.private_output / "official_native_with_current_warp.nii.gz"
    started = time.perf_counter()
    resample_world(
        args.official_native, args.template,
        np.loadtxt(args.current_root / "resampling_inputs/reference_to_source_world.txt"),
        control, pre_affine_pull_ras=args.current_root / "resampling_inputs/mni_to_t1_pull_ras.nii.gz",
        output_mask=new_mask, interpolation="spline", batch_size=8, device=args.device,
    )
    if str(args.device).startswith("cuda"):
        torch.cuda.synchronize(args.device)
    control_seconds = time.perf_counter() - started
    print("official native resampled with current warp", flush=True)
    template = nib.load(args.template)
    region = (np.asarray(nib.load(new_mask).dataobj) > 0) & (np.asarray(nib.load(old_mask).dataobj) > 0)
    mni = common_pairs(
        {"current_clean": new_mni, "previous_clean": old_mni,
         "official_fix_official_warp": args.official_mni, "official_fix_current_warp": control},
        template, region,
        [("current_clean", "previous_clean"), ("current_clean", "official_fix_official_warp"),
         ("previous_clean", "official_fix_official_warp"),
         ("current_clean", "official_fix_current_warp"),
         ("official_fix_current_warp", "official_fix_official_warp")],
    )
    report = {
        "schema_version": 1, "current_revision": args.current_revision,
        "previous_revision": args.previous_revision, "subjects": 1,
        "native": native, "same_input_preprocessing": preprocessing, "mni": mni,
        "registration_change": geometry_change(args.current_root, args.previous_root, region, template),
        "control_resampling_seconds_including_io": control_seconds,
        "scope": [
            "One real 490-frame UKB run. Pairwise voxel temporal Pearson r uses the same intersection mask and common varying voxels within each table.",
            "ICA-AROMA is the selected denoising method. Official UKB FIX compares different processing, including GDC/B0 correction and registration.",
            "Same official cleaned BOLD with two maps isolates map choice after cleanup; it does not separately identify BBR, T1 affine, nonlinear registration or distortion contributions.",
            "Previous and current revisions include different FLIRT and confound implementations. Their whole-run difference is not attributable to one component alone.",
        ],
        "privacy": "Anonymous scalar summaries and hashes; all voxel data remain private.",
    }
    args.report_out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
