"""比较单例同输入、同处理步骤的 FNIT 与原程序结果。

--manifest 指定私有 JSON 文件清单，路径可相对该清单；--report-out 输出
匿名标量、尺寸和 SHA-256。清单可包含 input_files/reference_input_files、
images、transforms、motion、pull、controls；已指定文件必须存在且有效。
controls 的四种交叉采样用相同 FNIT 样条插值，影像只保存在 --private-output。
该脚本不启动 FSL、FreeSurfer 或其他原程序，不调度其他被试。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import nibabel as nib
import numpy as np


IMAGE_STAGES = {
    "epi_synthstrip_mask", "t1_synthstrip_mask", "fast_csf", "fast_gm", "fast_wm",
    "motion_corrected", "feat_filtered", "aroma_native", "clean_native", "clean_mni",
    "t1_registered", "epi_registered",
}
INPUT_NAMES = {"bold", "sbref", "t1w", "mni_template", "mni_mask",
               "synthstrip_weights", "synthmorph_weights"}
AFFINE_STAGES = {"bbr_initial", "bbr_final", "t1_affine"}
ORACLE_PROGRAMS = {"fast", "flirt", "mcflirt", "fnirt", "melodic", "applywarp",
                   "mri_synthstrip", "mri_synthmorph", "ica_aroma", "fslmaths"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_path(value, directory):
    path = Path(value).expanduser()
    return path if path.is_absolute() else directory / path


def same_grid(image, reference, *, complete_shape=False):
    shape = image.shape if complete_shape else image.shape[:3]
    target = reference.shape if complete_shape else reference.shape[:3]
    if shape != target or not np.allclose(image.affine, reference.affine, rtol=0, atol=1e-4):
        raise ValueError("Comparison grids differ; no implicit resampling is performed")


def values(image):
    result = np.asarray(image.dataobj, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Comparison input contains nonfinite values")
    return result


def load_mask(path, reference):
    image = nib.load(path)
    same_grid(image, reference)
    array = values(image)
    if array.ndim != 3 or not np.isin(array, [0, 1]).all():
        raise ValueError("Comparison mask must be a finite binary 3D image")
    return array > 0


def correlations(candidate, reference, *, temporal=False, chunk=4096):
    """Float64 scalar errors and correlations; constant series return null."""
    if candidate.shape != reference.shape or candidate.size == 0:
        raise ValueError("Comparison values differ in shape or the region is empty")
    left = candidate.reshape(-1, candidate.shape[-1]) if temporal else candidate.reshape(-1, 1)
    right = reference.reshape(left.shape)
    count = left.size
    sums = np.zeros(5, dtype=np.float64)
    error_absolute = error_squared = max_error = 0.0
    temporal_cross = temporal_left = temporal_right = 0.0
    per_voxel = []
    for start in range(0, len(left), chunk):
        x = left[start:start + chunk].astype(np.float64)
        y = right[start:start + chunk].astype(np.float64)
        error = x - y
        error_absolute += float(np.abs(error).sum())
        error_squared += float(np.square(error).sum())
        max_error = max(max_error, float(np.abs(error).max()))
        sums += (x.sum(), y.sum(), np.square(x).sum(), np.square(y).sum(), (x * y).sum())
        if temporal:
            x -= x.mean(axis=1, keepdims=True)
            y -= y.mean(axis=1, keepdims=True)
            xx, yy, xy = np.square(x).sum(axis=1), np.square(y).sum(axis=1), (x * y).sum(axis=1)
            valid = (xx > 1e-12 * x.shape[1]) & (yy > 1e-12 * x.shape[1])
            per_voxel.append(np.clip(xy[valid] / np.sqrt(xx[valid] * yy[valid]), -1, 1))
            temporal_cross += float(xy[valid].sum())
            temporal_left += float(xx[valid].sum())
            temporal_right += float(yy[valid].sum())
    sx, sy, xx, yy, xy = sums
    denominator = np.sqrt(max(0.0, xx - sx * sx / count) * max(0.0, yy - sy * sy / count))
    result = {
        "evaluated_values": int(count),
        "pearson_r": float(np.clip((xy - sx * sy / count) / denominator, -1, 1)) if denominator else None,
        "mae": error_absolute / count, "rmse": float(np.sqrt(error_squared / count)),
        "max_absolute_difference": max_error,
    }
    if temporal:
        r = np.concatenate(per_voxel)
        pooled = np.sqrt(temporal_left * temporal_right)
        result.update({
            "valid_voxel_temporal_r": int(r.size),
            "mean_voxel_temporal_r": float(r.mean()) if r.size else None,
            "median_voxel_temporal_r": float(np.median(r)) if r.size else None,
            "p05_voxel_temporal_r": float(np.percentile(r, 5)) if r.size else None,
            "pooled_time_demeaned_r": float(np.clip(temporal_cross / pooled, -1, 1)) if pooled else None,
            "temporal_rms_threshold": 1e-6,
        })
    return result


def image_pair(specification, directory):
    paths = [local_path(specification[key], directory) for key in ("candidate", "reference")]
    images = [nib.load(path) for path in paths]
    same_grid(images[0], images[1], complete_shape=True)
    arrays = [values(image) for image in images]
    kind = specification["kind"]
    result = {"shape": list(images[0].shape), "same_grid": True, "all_values_finite": True,
              "sha256": {key: sha256(path) for key, path in zip(("candidate", "reference"), paths)}}
    if kind == "mask":
        if arrays[0].ndim != 3 or not all(np.isin(a, [0, 1]).all() for a in arrays):
            raise ValueError("Brain masks must be binary 3D images")
        a, b = [array > 0 for array in arrays]
        if not a.any() or not b.any():
            raise ValueError("Brain masks must be nonempty")
        intersection, union = int((a & b).sum()), int((a | b).sum())
        result.update({"candidate_voxels": int(a.sum()), "reference_voxels": int(b.sum()),
                       "intersection_voxels": intersection, "dice": 2 * intersection / (a.sum() + b.sum()),
                       "jaccard": intersection / union,
                       "candidate_only_voxels": int((a & ~b).sum()),
                       "reference_only_voxels": int((b & ~a).sum())})
        return result
    if kind not in ("scalar", "bold") or arrays[0].ndim != (4 if kind == "bold" else 3):
        raise ValueError("kind scalar requires 3D; kind bold requires 4D")
    region = np.ones(images[0].shape[:3], dtype=bool)
    for key in ("mask", "candidate_mask", "reference_mask"):
        if key in specification:
            region &= load_mask(local_path(specification[key], directory), images[0])
    if not region.any():
        raise ValueError("Comparison mask intersection is empty")
    result["mask_voxels"] = int(region.sum())
    if kind == "bold":
        tr = [float(image.header.get_zooms()[3]) for image in images]
        if not np.isclose(tr[0], tr[1], rtol=1e-5, atol=1e-6):
            raise ValueError("BOLD repetition times differ")
        if not all(image.header.get_xyzt_units()[1] == "sec" for image in images):
            raise ValueError("BOLD time units must be seconds")
        result["tr_seconds"] = tr[0]
    result["metrics"] = correlations(arrays[0][region], arrays[1][region], temporal=kind == "bold")
    result["ranges"] = {key: {"min": float(array.min()), "max": float(array.max())}
                        for key, array in zip(("candidate", "reference"), arrays)}
    return result


def matrix(path):
    array = np.loadtxt(path)
    if array.shape != (4, 4) or not np.isfinite(array).all():
        raise ValueError("Transform must be finite 4x4")
    if not np.allclose(array[3], (0, 0, 0, 1), rtol=0, atol=1e-8):
        raise ValueError("Transform must be homogeneous")
    if abs(np.linalg.det(array[:3, :3])) < 1e-8:
        raise ValueError("Transform is singular")
    return array


def world_points(reference, region):
    indices = np.argwhere(region).T
    return reference.affine[:3, :3] @ indices + reference.affine[:3, 3:4]


def distances(left, right):
    distance = np.linalg.norm(left - right, axis=0)
    return {"points": int(distance.size), "rms_mm": float(np.sqrt(np.square(distance).mean())),
            "mean_mm": float(distance.mean()), "median_mm": float(np.median(distance)),
            "p95_mm": float(np.percentile(distance, 95)), "max_mm": float(distance.max())}


def world_forward(value, moving, reference, convention):
    if convention == "ras":
        return value
    if convention != "fsl":
        raise ValueError("Affine convention must be fsl or ras")
    from fnit.flirt.coordinates import flirt_to_world_affine
    return flirt_to_world_affine(value, moving.affine, reference.affine,
                                 moving.shape[:3], reference.shape[:3],
                                 moving.header.get_zooms()[:3], reference.header.get_zooms()[:3])


def affine_pair(specification, directory):
    moving = nib.load(local_path(specification["moving"], directory))
    reference = nib.load(local_path(specification["reference"], directory))
    region = (load_mask(local_path(specification["mask"], directory), reference)
              if "mask" in specification else np.ones(reference.shape[:3], dtype=bool))
    if not region.any():
        raise ValueError("Affine evaluation region is empty")
    points = world_points(reference, region)
    paths = [local_path(specification[key], directory)
             for key in ("candidate_matrix", "reference_matrix")]
    raw = [matrix(path) for path in paths]
    pull = [np.linalg.inv(world_forward(value, moving, reference, specification.get("convention", "fsl")))
            for value in raw]
    mapped = [value[:3, :3] @ points + value[:3, 3:4] for value in pull]
    return {"inverse_world_displacement": distances(*mapped),
            "matrix_max_absolute_difference": float(np.abs(raw[0] - raw[1]).max()),
            "convention": specification.get("convention", "fsl"),
            "sha256": {key: sha256(path) for key, path in zip(("candidate", "reference"), paths)}}


def matrix_files(value, directory):
    if isinstance(value, list):
        paths = [local_path(path, directory) for path in value]
    else:
        root = local_path(value, directory)
        paths = sorted(root.glob("MAT_*"))
    if not paths:
        raise ValueError("Motion matrix sequence is empty")
    return paths


def motion_pair(specification, directory):
    moving = nib.load(local_path(specification["moving"], directory))
    reference = nib.load(local_path(specification["reference"], directory))
    sequences = [matrix_files(specification[key], directory)
                 for key in ("candidate_matrices", "reference_matrices")]
    if len(sequences[0]) != len(sequences[1]) or len(sequences[0]) != moving.shape[3]:
        raise ValueError("Motion matrix counts differ from the full BOLD frame count")
    region = load_mask(local_path(specification["mask"], directory), reference)
    if not region.any():
        raise ValueError("Motion evaluation mask is empty")
    points = world_points(reference, region)
    per_frame = []
    for first, second in zip(*sequences):
        mapped = []
        for path in (first, second):
            transform = np.linalg.inv(world_forward(matrix(path), moving, reference, "fsl"))
            mapped.append(transform[:3, :3] @ points + transform[:3, 3:4])
        per_frame.append(distances(*mapped)["rms_mm"])
    return {"frames": len(per_frame), "evaluation_points": int(region.sum()),
            "mean_frame_rms_mm": float(np.mean(per_frame)),
            "median_frame_rms_mm": float(np.median(per_frame)),
            "p95_frame_rms_mm": float(np.percentile(per_frame, 95)),
            "max_frame_rms_mm": float(np.max(per_frame)),
            "matrix_sequence_sha256": {
                key: hashlib.sha256("\n".join(sha256(path) for path in paths).encode()).hexdigest()
                for key, paths in zip(("candidate", "reference"), sequences)}}


def pull_pair(specification, directory):
    if specification.get("convention") != "ras_mm_pull_displacement":
        raise ValueError("Pull fields require explicit ras_mm_pull_displacement convention")
    reference = nib.load(local_path(specification["mni_template"], directory))
    region = load_mask(local_path(specification["mask"], directory), reference)
    if not region.any():
        raise ValueError("Pull evaluation mask is empty")
    points = world_points(reference, region)
    mapped, hashes = [], {}
    for key in ("candidate", "reference"):
        path = local_path(specification[key], directory)
        image = nib.load(path)
        same_grid(image, reference)
        if image.shape != (*reference.shape[:3], 3):
            raise ValueError("Pull fields must have shape X,Y,Z,3 and RAS-mm displacement semantics")
        mapped.append(points + values(image)[region].astype(np.float64).T)
        hashes[key] = sha256(path)
    result = {"mni_to_t1_world_displacement": distances(*mapped), "sha256": hashes,
              "convention": "MNI-grid pull displacement in world RAS millimetres"}
    affine_keys = ("candidate_to_epi_world", "reference_to_epi_world")
    if any(key in specification for key in affine_keys):
        if not all(key in specification for key in affine_keys):
            raise ValueError("Both T1-world-to-EPI-world transforms are required")
        epi = []
        for key, point in zip(affine_keys, mapped):
            path = local_path(specification[key], directory)
            transform = matrix(path)
            epi.append(transform[:3, :3] @ point + transform[:3, 3:4])
            hashes[key] = sha256(path)
        result["mni_to_epi_world_displacement"] = distances(*epi)
    return result


def cross_controls(specification, directory, output, device):
    import torch
    from fnit.fmri.normalization import resample_world
    if specification.get("convention") != "ras_mm_pull_displacement":
        raise ValueError("Cross controls require explicit ras_mm_pull_displacement convention")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    template = local_path(specification["template"], directory)
    mask = local_path(specification["mask"], directory)
    saved = {}
    started = time.perf_counter()
    for clean in ("candidate", "reference"):
        for warp in ("candidate", "reference"):
            destination = output / (clean + "_native_" + warp + "_warp.nii.gz")
            resample_world(
                local_path(specification[clean + "_native"], directory), template,
                matrix(local_path(specification[warp + "_to_epi_world"], directory)),
                destination, pre_affine_pull_ras=local_path(specification[warp + "_pull"], directory),
                output_mask=mask, interpolation="spline", batch_size=8, device=device,
            )
            saved[clean, warp] = destination
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    pairings = {
        "same_candidate_warp_cleaning_difference": (saved["candidate", "candidate"], saved["reference", "candidate"]),
        "same_reference_warp_cleaning_difference": (saved["candidate", "reference"], saved["reference", "reference"]),
        "same_candidate_native_map_difference": (saved["candidate", "candidate"], saved["candidate", "reference"]),
        "same_reference_native_map_difference": (saved["reference", "candidate"], saved["reference", "reference"]),
    }
    for key in ("candidate", "reference"):
        if key + "_mni" in specification:
            pairings[key + "_saved_vs_shared_sampler"] = (
                local_path(specification[key + "_mni"], directory), saved[key, key])
    return {"resampling_seconds_including_io": seconds,
            "sampler": "FNIT periodic cubic B-spline, float32/TF32, eight frames per chunk",
            "pairs": {key: image_pair({"kind": "bold", "candidate": str(first),
                                        "reference": str(second), "mask": str(mask)}, directory)
                      for key, (first, second) in pairings.items()}}


def oracle_provenance(programs, directory):
    if set(programs) - ORACLE_PROGRAMS:
        raise ValueError("Unknown oracle program label")
    result = {}
    for name, details in programs.items():
        version = details["version"]
        if not re.fullmatch(r"[A-Za-z0-9 ._+():-]{1,120}", version):
            raise ValueError("Use a concise component version, without paths")
        actual = sha256(local_path(details["executable"], directory))
        if details.get("expected_sha256", actual) != actual:
            raise ValueError("Oracle executable checksum mismatch")
        item = {"version": version, "executable_sha256": actual,
                "version_source": "Oracle launcher component-version record; not inferred from FSL installation directory"}
        if "version_record" in details:
            item["version_record_sha256"] = sha256(local_path(details["version_record"], directory))
        result[name] = item
    return result


def compare_manifest(manifest, directory, private_output=None, device="cuda:0"):
    for key in ("candidate_revision", "reference_revision"):
        if not re.fullmatch(r"[0-9a-f]{7,40}", manifest[key]):
            raise ValueError("Revisions must be Git commit identifiers")
    inputs = manifest.get("input_files", {})
    if not {"bold", "t1w", "mni_template"}.issubset(inputs):
        raise ValueError("Raw BOLD, T1w and MNI template are required for matched comparison")
    if set(inputs) - INPUT_NAMES:
        raise ValueError("Unknown input name; identifiers must not enter public reports")
    input_hashes = {key: sha256(local_path(value, directory)) for key, value in inputs.items()}
    expected = manifest.get("expected_input_sha256", {})
    if any(input_hashes.get(key) != value for key, value in expected.items()):
        raise ValueError("Input checksum does not match the pinned manifest")
    paired_inputs = manifest.get("reference_input_files", {})
    if set(paired_inputs) != set(inputs):
        raise ValueError("Candidate and reference must enumerate identical raw inputs/resources")
    input_matches = {key: sha256(local_path(paired_inputs[key], directory)) == value
                     for key, value in input_hashes.items()}
    if not all(input_matches.values()):
        raise ValueError("Candidate/reference raw inputs or resources differ")
    stages = manifest.get("images", {})
    transforms = manifest.get("transforms", {})
    if set(stages) - IMAGE_STAGES or set(transforms) - AFFINE_STAGES:
        raise ValueError("Unknown stage label; use the documented anonymous stage names")
    report = {
        "schema_version": 1, "candidate_revision": manifest["candidate_revision"],
        "reference_revision": manifest["reference_revision"], "subjects": 1,
        "input_sha256": input_hashes, "same_raw_inputs_and_resources": input_matches,
        "images": {key: image_pair(value, directory) for key, value in stages.items()},
        "transforms": {key: affine_pair(value, directory) for key, value in transforms.items()},
        "omitted_image_stages": sorted(IMAGE_STAGES - stages.keys()),
        "scope": [
            "Single real-run matched-step comparison. Stage pairs are separately evaluated on their stated common mask.",
            "Mask intersections restrict the numerical domain; Dice and mask-only counts report coverage differences separately.",
            "Temporal Pearson r uses common varying series with demeaned RMS greater than 1e-6; constants have no correlation.",
            "The published UKB FIX/GDC/B0 comparison near mean temporal r=0.272 uses a different pipeline and is not this matched-step oracle.",
            "Cross controls hold a shared interpolation implementation and mask fixed; they isolate cleanup/map choice after estimation, not individual upstream causes.",
        ],
        "privacy": "Anonymous scalar metrics, shapes and hashes only. Manifest paths and voxel data remain private.",
        "comparison_script_sha256": sha256(__file__),
        "dependencies": {"numpy": np.__version__, "nibabel": nib.__version__},
    }
    if "oracle_programs" in manifest:
        report["oracle_programs"] = oracle_provenance(manifest["oracle_programs"], directory)
    if "motion" in manifest:
        report["motion"] = motion_pair(manifest["motion"], directory)
    if "pull" in manifest:
        report["pull"] = pull_pair(manifest["pull"], directory)
    if "controls" in manifest:
        if private_output is None:
            raise ValueError("Cross controls require --private-output")
        report["controls"] = cross_controls(manifest["controls"], directory, private_output, device)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    started = time.perf_counter()
    report = compare_manifest(json.loads(args.manifest.read_text()), args.manifest.parent,
                              args.private_output, args.device)
    report["comparison_wall_seconds_including_validation"] = time.perf_counter() - started
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
