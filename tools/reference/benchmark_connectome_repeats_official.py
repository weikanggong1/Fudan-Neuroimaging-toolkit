"""独立 MRtrix 随机重复 benchmark：真实检查点 → 多次追踪 → 全部 atlas 矩阵。

此工具不在 FNIT 运行时调用。只读本轮自产 tracking_inputs.pt 和 CLI 输出，
写入新目录；默认五个种子、100000 次尝试。dry-run 仅输出计划。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from connectome_repeat_common import NAMES, load_metadata, load_profiles, sha256, validate_matrices


PROGRAMS = ("mrinfo", "tckgen", "tckinfo", "tcksift2", "tckstats",
            "tcksample", "tck2connectome")


def effective_parameters(payload: dict, n_seeds: int) -> dict:
    """从实际绑定的 FNIT kwargs 和 Float64 FOD 仿射求有效追踪参数。"""
    kwargs = payload["tracking_kwargs"]
    if int(kwargs["n_seeds"]) != n_seeds:
        raise ValueError("--n-seeds differs from actual tracking checkpoint")
    affine = payload["fod_affine"].to(dtype=torch.float64, device="cpu")
    voxel_size = float(torch.linalg.vector_norm(affine[:3, :3], dim=0).prod().pow(1 / 3))
    step = voxel_size / 2 if kwargs["step_mm"] is None else float(kwargs["step_mm"])
    minimum = 2 * voxel_size if kwargs["min_length_mm"] is None else float(kwargs["min_length_mm"])
    if kwargs["lmax"] != 8 or payload["wm_sh"].shape[-1] != 45:
        raise ValueError("this reference profile requires actual lmax=8, 45-coefficient WM FOD")
    params = {"n_seed_attempts": n_seeds, "step_mm": step, "min_length_mm": minimum,
              "max_length_mm": float(kwargs["max_length_mm"]),
              "max_angle_degrees": float(kwargs["max_angle_degrees"]),
              "cutoff": float(kwargs["cutoff"]), "power": float(kwargs["power"]),
              "samples": 3, "tracking_threads": 0,
              "fnit_batch_size": int(kwargs["batch_size"]),
              "fnit_compile_arc": bool(kwargs["compile_arc"]),
              "fnit_seed": int(kwargs["seed"]),
              "fnit_tracking_kwargs": kwargs,
              "rng_policy": "MRTRIX_RNG_SEED controls official RNG; equal integer seeds do not imply equal PyTorch random streams",
              "unchanged_mrtrix_defaults": {"trials": 1000, "max_attempts_per_seed": 1000,
                                             "downsample": 2},
              "act_options": "no backtrack, crop_at_gmwmi, mask or additional stop constraints"}
    if not all(np.isfinite(params[key]) and params[key] > 0 for key in (
            "step_mm", "max_length_mm", "max_angle_degrees", "cutoff", "power")) or not np.isfinite(minimum) or minimum < 0:
        raise ValueError("invalid actual tracking parameters")
    if params["max_length_mm"] <= step or minimum > params["max_length_mm"]:
        raise ValueError("actual step/minimum exceed maximum tracking length")
    return params


def command_plan(mrtrix_bin: Path, output: Path, profiles: dict, seeds: list[int],
                 params: dict, threads: int) -> list[dict]:
    """只构造 argv；同一重复的 SIFT2、FA 和长度各计算一次供所有 atlas 使用。"""
    images = output / "inputs"
    records = []
    # Read NIfTI-2 directly: MRtrix's MIF writer rounds the vox field before
    # writing full-precision transform vectors. That avoidable conversion
    # changes the effective affine even when the data array is identical.
    readers = [("wm_fod", images / "wm_fod.nii.gz"),
               ("five_tissue_act", images / "five_tissue_act.nii.gz"),
               ("five_tissue_sift2", images / "five_tissue_sift2.nii.gz"),
               ("gmwmi", images / "gmwmi.nii.gz"), ("fa", images / "fa.nii.gz")]
    readers += [(f"atlas:{name}", images / "atlases" / name / "atlas_dwi.nii.gz")
                for name in profiles]
    reader_options = ["-config", "NIfTIUseSform", "1"]
    for name, path in readers:
        records.append({"stage": "input_readback", "input": name,
                        "json": str(path.with_name(path.name + ".mrinfo.json")),
                        "argv": [str(mrtrix_bin / "mrinfo"), str(path), "-json_all",
                                 str(path.with_name(path.name + ".mrinfo.json")),
                                 *reader_options, "-nthreads", str(threads)],
                        "log": str(path.with_name(path.name + ".mrinfo.log"))})
    for seed in seeds:
        root = output / f"seed-{seed}"
        tracks = root / "tracks.tck"
        common = ["-nthreads", str(threads)]
        records.append({"stage": "tracking", "seed": seed, "argv": [str(mrtrix_bin / "tckgen"),
            "-algorithm", "iFOD2", "-seed_gmwmi", str(images / "gmwmi.nii.gz"),
            "-act", str(images / "five_tissue_act.nii.gz"), "-seeds", str(params["n_seed_attempts"]),
            "-select", "0", "-maxlength", str(params["max_length_mm"]),
            "-minlength", str(params["min_length_mm"]), "-step", str(params["step_mm"]),
            "-angle", str(params["max_angle_degrees"]), "-cutoff", str(params["cutoff"]),
            "-samples", str(params["samples"]), "-power", str(params["power"]),
            "-trials", "1000", "-max_attempts_per_seed", "1000", "-downsample", "2",
            "-nthreads", "0", *reader_options,
            str(images / "wm_fod.nii.gz"), str(tracks)], "log": str(root / "tckgen.log")})
        records.append({"stage": "track_count", "seed": seed,
                        "argv": [str(mrtrix_bin / "tckinfo"), "-count", str(tracks)],
                        "log": str(root / "tckinfo.txt")})
        for stage, argv in (
            ("sift2", ["tcksift2", str(tracks), str(images / "wm_fod.nii.gz"), str(root / "sift2_weights.txt"),
                       "-act", str(images / "five_tissue_sift2.nii.gz"), "-csv", str(root / "sift2_stats.csv"),
                       *reader_options]),
            ("length", ["tckstats", "-dump", str(root / "lengths.txt"), str(tracks)]),
            ("fa", ["tcksample", "-precise", "-stat_tck", "mean", str(tracks),
                    str(images / "fa.nii.gz"), str(root / "mean_fa.txt"), *reader_options]),
        ):
            records.append({"stage": stage, "seed": seed,
                            "argv": [str(mrtrix_bin / argv[0]), *argv[1:], *common],
                            "log": str(root / f"{stage}.log")})
        for profile in profiles:
            target = root / "atlases" / profile
            atlas = images / "atlases" / profile / "atlas_dwi.nii.gz"
            for name in ("count", "sift2_fbc", "mean_length", "mean_fa"):
                options = ["-symmetric", "-assignment_radial_search", "4"]
                if name != "count":
                    options += ["-tck_weights_in", str(root / "sift2_weights.txt")]
                if name in ("mean_length", "mean_fa"):
                    scalar = "lengths.txt" if name == "mean_length" else "mean_fa.txt"
                    options += ["-scale_file", str(root / scalar), "-stat_edge", "mean"]
                records.append({"stage": f"matrix_{name}", "seed": seed, "atlas": profile,
                    "argv": [str(mrtrix_bin / "tck2connectome"), *options, str(tracks), str(atlas),
                             str(target / f"{name}.csv"), *common, *reader_options],
                    "log": str(target / f"{name}.log")})
    return records


def _nifti_affine_contract(affine: np.ndarray) -> dict:
    """NIfTI sform 存 3x4；只接受 dtype 舍入级别的齐次末行残差。"""
    if affine.shape != (4, 4) or affine.dtype.kind != "f" or affine.dtype.itemsize not in (4, 8) or not np.isfinite(affine).all():
        raise ValueError("NIfTI export requires a finite float32/float64 4x4 affine")
    canonical = np.array([0., 0., 0., 1.], dtype=affine.dtype)
    epsilon = float(np.finfo(affine.dtype).eps)
    # A four-term matrix product has the standard gamma_4 machine bound.
    # This fixed unit-scale bottom-row serialization contract is independent
    # of image size, observed errors and scientific parity acceptance.
    bound = 4 * epsilon / (1 - 4 * epsilon)
    difference = float(np.abs(affine[3] - canonical).max())
    if difference > bound:
        raise ValueError("NIfTI cannot represent a projective/non-affine bottom row beyond dtype machine roundoff")
    return {"source_dtype": str(affine.dtype), "source_bottom_row": affine[3].tolist(),
            "implicit_file_bottom_row": canonical.tolist(),
            "bottom_row_max_abs_difference": difference,
            "machine_epsilon": epsilon, "machine_bound_gamma4": bound,
            "policy": "first 3x4 stored exactly in Float64 sform; bottom row implicit [0,0,0,1]; only dtype gamma_4 roundoff accepted"}


def _save_image(path: Path, values, affine, spacing=None) -> dict:
    values = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
    affine = affine.detach().cpu().numpy() if isinstance(affine, torch.Tensor) else np.asarray(affine)
    affine_contract = _nifti_affine_contract(affine)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti2Image(values, affine, dtype=values.dtype)
    if spacing is not None:
        image.header.set_zooms((*spacing, *image.header.get_zooms()[3:]))
    nib.save(image, path)
    check = nib.load(path)
    # Preserve NaN payloads too: array_equal without equal_nan rejects an
    # unchanged NaN, while equal_nan alone does not check its stored bits.
    native_dtype = values.dtype.newbyteorder("=")
    original_bits = np.ascontiguousarray(values, dtype=native_dtype).view(np.uint8)
    decoded_bits = np.ascontiguousarray(np.asarray(check.dataobj), dtype=native_dtype).view(np.uint8)
    decoded_dtype = np.dtype(check.get_data_dtype())
    source_sform_bits = np.ascontiguousarray(affine[:3], dtype=np.float64).view(np.uint8)
    decoded_sform_bits = np.ascontiguousarray(check.affine[:3], dtype=np.float64).view(np.uint8)
    if (decoded_dtype.kind != values.dtype.kind or decoded_dtype.itemsize != values.dtype.itemsize
            or not np.array_equal(decoded_sform_bits, source_sform_bits)
            or not np.array_equal(check.affine[3], [0., 0., 0., 1.])
            or not np.array_equal(decoded_bits, original_bits)):
        raise ValueError(f"{path}: NIfTI-2 export changed voxel bits or representable 3x4 sform")
    nonfinite = ~np.isfinite(values)
    nonfinite_count = int(nonfinite.sum())
    # The limit bounds metadata size only; the full array is exported intact.
    nonfinite_coordinates = []
    if nonfinite_count:
        flat_indices = np.flatnonzero(nonfinite.ravel())[:100]
        nonfinite_coordinates = np.array(np.unravel_index(flat_indices, values.shape)).T.tolist()
    return {"path": str(path), "sha256": sha256(path), "dtype": str(values.dtype),
            "shape": list(values.shape), "affine": affine.tolist(),
            "file_affine": check.affine.tolist(), "affine_serialization": affine_contract,
            "sform_3x4_bits_verified_equal": True,
            "header_spacing": list(check.header.get_zooms()[:3]),
            "voxel_bits_verified_equal": True,
            "nonfinite_count": nonfinite_count,
            "nonfinite_coordinates": nonfinite_coordinates,
            "nonfinite_coordinates_complete": nonfinite_count <= 100}


def source_readiness(checkpoint: Path, payload: dict) -> dict:
    """要求同一实际 core 的后处理输入齐备；不代替整例显存/科学验收。"""
    required = ("geometry.npz", "fa.nii.gz", "tracks.tck", "track_metrics.npz")
    for name in required:
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"actual core outputs not ready: {path}")
    with np.load(checkpoint / "geometry.npz", allow_pickle=False) as geometry:
        for name, key in (("dwi_affine", "fod_affine"), ("five_tissue_affine", "five_tissue_affine")):
            if not np.array_equal(geometry[name], payload[key].numpy()):
                raise ValueError(f"{name}: completed core geometry differs from actual tracking snapshot")
    fa = nib.load(checkpoint / "fa.nii.gz")
    if fa.shape != tuple(payload["wm_sh"].shape[:3]):
        raise ValueError("completed core FA shape differs from actual FOD snapshot")
    # NIfTI-1 checkpoint geometry is rounded; export_inputs uses the PT affine.
    if float(np.abs(fa.affine - payload["fod_affine"].numpy()).max()) > 1e-3:
        raise ValueError("completed core FA header is not on the tracking FOD grid")
    tracks = nib.streamlines.load(checkpoint / "tracks.tck", lazy_load=True)
    # TCK stores the literal `count`; the standardized NB_STREAMLINES key is
    # used by TRK and is not present in nibabel's lazy TCK header.
    count = int(tracks.header["count"])
    if count < 1:
        raise ValueError("actual core has no completed accepted streamlines")
    with np.load(checkpoint / "track_metrics.npz", allow_pickle=False) as metrics:
        for name in ("weights", "lengths", "mean_fa"):
            if metrics[name].shape != (count,):
                raise ValueError(f"{name}: completed metrics differ from actual TCK track count")
        if metrics["endpoints"].shape != (count, 2, 3):
            raise ValueError("completed endpoints differ from actual TCK track count")
    return {"status": "computed_outputs_ready", "accepted_tracks": count,
            "file_sha256": {name: sha256(checkpoint / name) for name in required},
            "fa_dtype": str(fa.get_data_dtype()), "fa_shape": list(fa.shape),
            "scope": "actual core artifacts and source geometry only; not an end-to-end memory or scientific parity pass"}


def input_readback(record: dict, exported: dict) -> dict:
    """核对实际 MRtrix header 契约，记录 reader 几何；不新增误差容差。"""
    actual = json.loads(Path(record["json"]).read_text())
    if Path(actual["name"]).resolve() != Path(exported["path"]).resolve() or not actual["format"].startswith("NIfTI-2"):
        raise ValueError("official reader did not inspect the actual exported NIfTI-2")
    dtype = np.dtype(exported["dtype"])
    datatype = {"f": "Float", "i": "Int", "u": "UInt"}.get(dtype.kind, "") + str(dtype.itemsize * 8)
    if not datatype or not actual["datatype"].startswith(datatype):
        raise ValueError("official reader datatype differs from exported voxel dtype")
    shape = np.asarray(exported["shape"], dtype=np.int64)
    size = np.asarray(actual["size"], dtype=np.int64)
    strides = np.asarray(actual["strides"], dtype=np.int64)
    spacing = np.asarray(actual["spacing"], dtype=np.float64)
    transform = np.asarray(actual["transform"], dtype=np.float64)
    if size.shape != shape.shape or spacing.shape != shape.shape or strides.shape != shape.shape:
        raise ValueError("official reader changed image dimensionality")
    # NIfTI spatial dimensions have strides 1,2,3 before MRtrix realignment.
    axes = np.abs(strides[:3]) - 1
    if sorted(axes.tolist()) != [0, 1, 2] or not np.array_equal(size[:3], shape[axes]) or not np.array_equal(size[3:], shape[3:]):
        raise ValueError("official reader shape/axis mapping differs from exported NIfTI-2")
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.isfinite(spacing).all() or (spacing <= 0).any()
            or not np.array_equal(transform[3], [0., 0., 0., 1.])
            or np.linalg.det(transform[:3, :3]) == 0
            or float(actual["intensity_offset"]) != 0 or float(actual["intensity_scale"]) != 1):
        raise ValueError("official reader returned invalid geometry or altered intensity scaling")
    mapping = np.eye(4)
    mapping[:3, :3] = 0
    for axis, original_axis in enumerate(axes):
        sign = 1 if strides[axis] > 0 else -1
        mapping[original_axis, axis] = sign
        if sign < 0:
            mapping[original_axis, 3] = shape[original_axis] - 1
    source_affine = np.asarray(exported["affine"], dtype=np.float64)
    file_affine = np.asarray(exported.get("file_affine", exported["affine"]), dtype=np.float64)
    source_spacing = np.asarray(exported["header_spacing"], dtype=np.float64)
    lengths = np.linalg.norm(file_affine[:3, :3], axis=0)
    # This 1e-5 is the pre-existing official NIfTI reader rule, not a parity
    # acceptance tolerance. A larger pixdim/sform discrepancy makes MRtrix
    # use sform column norms; otherwise it uses the exported pixdim values.
    rescaled = bool((np.abs(source_spacing / lengths - 1) > 1e-5).any())
    selected_spacing = lengths if rescaled else source_spacing
    reader_affine = file_affine.copy()
    reader_affine[:3, :3] *= selected_spacing / lengths
    expected = reader_affine @ mapping
    decoded = transform.copy()
    decoded[:3, :3] *= spacing[:3]
    corners = np.array(np.meshgrid(*[(0, int(value) - 1) for value in size[:3]], indexing="ij")).reshape(3, -1).T
    displacement = np.linalg.norm(nib.affines.apply_affine(decoded, corners) -
                                  nib.affines.apply_affine(expected, corners), axis=1)
    source_displacement = np.linalg.norm(nib.affines.apply_affine(decoded, corners) -
                                         nib.affines.apply_affine(source_affine @ mapping, corners), axis=1)
    file_displacement = np.linalg.norm(nib.affines.apply_affine(decoded, corners) -
                                       nib.affines.apply_affine(file_affine @ mapping, corners), axis=1)
    fnit_affine = np.asarray(exported.get("fnit_effective_affine", exported["affine"]), dtype=np.float64)
    fnit_displacement = np.linalg.norm(nib.affines.apply_affine(decoded, corners) -
                                      nib.affines.apply_affine(fnit_affine @ mapping, corners), axis=1)
    return {"status": "header_source_shape_dtype_scaling_verified_geometry_recorded", "mrinfo_json": actual,
            "mrinfo_json_sha256": sha256(Path(record["json"])),
            "axis_mapping_to_source": mapping.tolist(),
            "source_voxel_to_world_affine": source_affine.tolist(),
            "file_voxel_to_world_affine": file_affine.tolist(),
            "fnit_effective_voxel_to_world_affine": fnit_affine.tolist(),
            "affine_serialization": exported.get("affine_serialization"),
            "decoded_voxel_to_world_affine": decoded.tolist(),
            "official_nifti_sform_pixdim_rule_rescaled": rescaled,
            "expected_reader_affine": expected.tolist(),
            "maximum_corner_difference_from_reader_rule_mm": float(displacement.max()),
            "maximum_corner_difference_from_source_affine_mm": float(source_displacement.max()),
            "maximum_corner_difference_from_file_affine_mm": float(file_displacement.max()),
            "maximum_corner_difference_from_fnit_operator_affine_mm": float(fnit_displacement.max()),
            "geometry_assessment": "descriptive actual reader result; JSON numbers are serialized, not a bit-exact affine claim or a new scientific gate",
            "voxel_validation_scope": "source NIfTI-2 bits verified on export; mrinfo checks header/shape/scaling, not full decoded image values"}


def export_inputs(output: Path, payload: dict, checkpoint: Path, profiles: dict) -> dict:
    images = output / "inputs"
    affine = payload["fod_affine"]
    five_affine = payload["five_tissue_affine"]
    exports = {
        "wm_fod": _save_image(images / "wm_fod.nii.gz", payload["wm_sh"], affine),
        "five_tissue_act": _save_image(images / "five_tissue_act.nii.gz", payload["five_tissue"],
                                       five_affine, payload["five_tissue_spacing_mm"]),
        "five_tissue_sift2": _save_image(images / "five_tissue_sift2.nii.gz", payload["five_tissue"], five_affine),
        "gmwmi": _save_image(images / "gmwmi.nii.gz", payload["gmwmi"], five_affine),
    }
    spacing = payload["five_tissue_spacing_mm"]
    act_affine = five_affine.to(dtype=torch.float64).clone()
    if spacing is not None:
        act_affine[:3, :3] *= torch.as_tensor(spacing, dtype=torch.float64) / torch.linalg.vector_norm(act_affine[:3, :3], dim=0)
    exports["five_tissue_act"]["fnit_effective_affine"] = act_affine.tolist()
    fa = payload.get("fa")
    if fa is None:
        fa_image = nib.load(checkpoint / "fa.nii.gz")
        if float(np.abs(fa_image.affine - affine.numpy()).max()) > 1e-3:
            raise ValueError("FA checkpoint header is not on the actual FOD grid")
        fa = np.asarray(fa_image.dataobj)
    if tuple(fa.shape) != tuple(payload["wm_sh"].shape[:3]):
        raise ValueError("FA voxel grid differs from actual FOD checkpoint")
    exports["fa"] = _save_image(images / "fa.nii.gz", fa, affine)
    exports["atlases"] = {}
    for profile, (matrices, metadata) in profiles.items():
        source = Path(metadata["directory"])
        atlas = nib.load(source / "atlas_dwi.nii.gz")
        labels = np.asarray(atlas.dataobj)
        if labels.shape != tuple(payload["wm_sh"].shape[:3]):
            raise ValueError(f"{profile}: atlas voxel grid differs from actual FOD checkpoint")
        if not np.isfinite(labels).all() or not np.array_equal(labels, np.rint(labels)):
            raise ValueError(f"{profile}: expected finite integer atlas labels")
        if int(labels.max()) > matrices["count"].shape[0] or int(labels.min()) < 0:
            raise ValueError(f"{profile}: atlas labels and matrix node count differ")
        if float(np.abs(atlas.affine - affine.numpy()).max()) > 1e-3:
            raise ValueError(f"{profile}: CLI atlas header is not on the checkpoint DWI grid")
        atlas_export = _save_image(images / "atlases" / profile / "atlas_dwi.nii.gz", labels, affine)
        atlas_export["original_sha256"] = sha256(source / "atlas_dwi.nii.gz")
        atlas_export["maximum_present_label"] = int(labels.max())
        atlas_export["original_affine_max_abs_difference"] = float(np.abs(atlas.affine - affine.numpy()).max())
        exports["atlases"][profile] = atlas_export
    return exports


def align_official_matrices(directory: Path, canonical_nodes: int, maximum_present_label: int) -> dict | None:
    """保留原 CSV，只对经 atlas 证实缺失的末尾节点生成补零的 canonical 副本。"""
    if not 0 <= maximum_present_label <= canonical_nodes:
        raise ValueError("maximum present atlas label is outside canonical nodes")
    raw_paths = {name: directory / f"{name}.csv" for name in NAMES}
    arrays = {name: np.loadtxt(path, delimiter=",", ndmin=2) for name, path in raw_paths.items()}
    validate_matrices(arrays, str(directory))
    original_nodes = arrays["count"].shape[0]
    if original_nodes > canonical_nodes or original_nodes < maximum_present_label:
        raise ValueError(f"{directory}: official matrix truncates present labels or exceeds canonical nodes")
    if original_nodes == canonical_nodes:
        return None
    if load_metadata(directory, canonical_nodes)["node_rows"] is None:
        raise ValueError("canonical matrix alignment requires an explicit ordered nodes.tsv")
    canonical_hashes = {}
    for name, values in arrays.items():
        canonical = np.zeros((canonical_nodes, canonical_nodes), dtype=values.dtype)
        canonical[:original_nodes, :original_nodes] = values
        path = directory / f"connectome_{name}.csv"
        np.savetxt(path, canonical, delimiter=",", fmt="%.17g")
        if not np.array_equal(np.loadtxt(path, delimiter=",", ndmin=2), canonical):
            raise ValueError(f"{path}: canonical CSV changed original matrix values")
        canonical_hashes[name] = sha256(path)
    alignment = {"policy": "only trailing nodes absent from atlas receive zero rows/columns; raw official CSV retained",
                 "original_nodes": original_nodes, "canonical_nodes": canonical_nodes,
                 "maximum_present_label": maximum_present_label,
                 "absent_trailing_nodes": list(range(original_nodes + 1, canonical_nodes + 1)),
                 "raw_matrix_sha256": {name: sha256(path) for name, path in raw_paths.items()},
                 "canonical_matrix_sha256": canonical_hashes}
    (directory / "canonical_matrix_alignment.json").write_text(json.dumps(alignment, indent=2) + "\n")
    return alignment


def _run(record: dict, environment: dict) -> dict:
    log = Path(record["log"])
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    timer = shutil.which("time", path="/usr/bin:/bin")
    argv = ([timer, "-f", "wall_seconds=%e peak_rss_kib=%M", "-o", str(log.with_suffix(".time")),
             *record["argv"]] if timer else record["argv"])
    with log.open("w") as stream:
        process = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, env=environment)
    result = {**record, "seconds_inclusive": time.perf_counter() - started,
              "returncode": process.returncode}
    if timer:
        result["time_file"] = str(log.with_suffix(".time"))
        result["time_text"] = log.with_suffix(".time").read_text().strip()
    return result


def main(argv=None) -> None:
    entry_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mrtrix-bin", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--fnit-dir", type=Path, required=True, help="existing CLI output containing atlases/*")
    parser.add_argument("--output-dir", type=Path, required=True, help="fresh output directory")
    parser.add_argument("--tracking-inputs", type=Path, help="defaults to checkpoint-dir/tracking_inputs.pt")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-seeds", type=int, default=100000)
    parser.add_argument("--downstream-threads", type=int, default=8)
    parser.add_argument("--atlas", nargs="+", help="explicit subset; default uses all CLI output atlases")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if len(args.seeds) < 3 or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("at least three distinct nonnegative official seeds are required")
    if args.n_seeds < 1 or args.downstream_threads < 1 or args.output_dir.exists():
        parser.error("positive seed/thread counts and a fresh output directory are required")
    tracking_path = args.tracking_inputs or args.checkpoint_dir / "tracking_inputs.pt"
    payload = torch.load(tracking_path, map_location="cpu", weights_only=True)
    params = effective_parameters(payload, args.n_seeds)
    readiness = source_readiness(args.checkpoint_dir, payload)
    profiles = load_profiles(args.fnit_dir)
    if args.atlas:
        if len(set(args.atlas)) != len(args.atlas) or not set(args.atlas).issubset(profiles):
            parser.error("--atlas must select unique existing CLI atlas outputs")
        profiles = {name: profiles[name] for name in args.atlas}
    for name, (_, metadata) in profiles.items():
        source = Path(metadata["directory"])
        if not (source / "atlas_dwi.nii.gz").is_file() or metadata["node_rows"] is None:
            raise ValueError(f"{name}: current CLI atlas image and nodes.tsv required")
    programs = {}
    for name in PROGRAMS:
        executable = args.mrtrix_bin / name
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise FileNotFoundError(executable)
        programs[name] = {"path": str(executable.resolve()), "invoked_path": str(executable),
                          "sha256": sha256(executable)}
    plan = command_plan(args.mrtrix_bin, args.output_dir, profiles, args.seeds, params, args.downstream_threads)
    manifest = {"dataset": args.dataset, "parameters": params,
                "source_readiness": readiness,
                "tracking_input_sha256": sha256(tracking_path), "programs": programs,
                "source_profile_metadata": {name: metadata for name, (_, metadata) in profiles.items()},
                "source_fa_checkpoint_sha256": (sha256(args.checkpoint_dir / "fa.nii.gz")
                                               if payload.get("fa") is None else None),
                "seeds": args.seeds, "downstream_threads": args.downstream_threads,
                "commands": plan, "completed_commands": [], "execution_completed": False,
                "input_readbacks": {},
                "input_format_policy": "original voxel bits and representable 3x4 Float64 sform exported as NIfTI-2; original complete source 4x4 and implicit file bottom row separately recorded; official programs read these directly; reader geometry recorded before tracking",
                "matrix_definitions": {"count": "number of assigned tracks",
                                       "sift2_fbc": "sum(w)",
                                       "mean_length": "sum(w * length) / sum(w)",
                                       "mean_fa": "sum(w * precise_streamline_mean_fa) / sum(w)",
                                       "unassigned": "dropped", "self_connections": "retained"},
                "scientific_parity": "not_assessed", "script_sha256": sha256(Path(__file__)),
                "helper_sha256": sha256(Path(__file__).resolve().parents[1] / "connectome_repeat_common.py"),
                "environment": {"host": platform.node(), "python_executable": sys.executable,
                                "python": platform.python_version(), "torch": torch.__version__,
                                "numpy": np.__version__, "nibabel": nib.__version__, "cuda_visible_devices": ""},
                "timing_scope": "main entry through output/input validation; Python import/startup excluded"}
    if args.dry_run:
        print(json.dumps(manifest, indent=2, allow_nan=False))
        return
    started = time.perf_counter()
    manifest["preflight_seconds"] = started - entry_started
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report_path = args.output_dir / "reference_manifest.json"
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": str(args.downstream_threads),
                   "OPENBLAS_NUM_THREADS": str(args.downstream_threads), "MKL_NUM_THREADS": str(args.downstream_threads)}
    try:
        version = subprocess.run([str(args.mrtrix_bin / "tckgen"), "-version"],
            check=True, capture_output=True, text=True, env=environment)
        manifest["mrtrix_version"] = (version.stdout + version.stderr).strip()
        manifest["input_exports"] = export_inputs(args.output_dir, payload, args.checkpoint_dir, profiles)
        for seed in args.seeds:
            for profile, (_, metadata) in profiles.items():
                directory = args.output_dir / f"seed-{seed}" / "atlases" / profile
                directory.mkdir(parents=True)
                shutil.copyfile(Path(metadata["directory"]) / "nodes.tsv", directory / "nodes.tsv")
                (directory / "nodes.txt").write_text(str(metadata["nodes"]) + "\n")
                # Compare original CLI byte identity while retaining exact NIfTI-2 export separately.
                (directory / "atlas.sha256").write_text(metadata["atlas_sha256"] + "\n")
                (directory / "reference_atlas_export.json").write_text(json.dumps(
                    manifest["input_exports"]["atlases"][profile], indent=2) + "\n")
        for record in plan:
            if record["stage"] == "tracking":
                if len(manifest["input_readbacks"]) != 5 + len(profiles):
                    raise RuntimeError("official image readback contract must complete before tracking")
                (args.output_dir / f"seed-{record['seed']}").mkdir(exist_ok=True)
            run_environment = {**environment, "MRTRIX_RNG_SEED": str(record.get("seed", 0))}
            manifest["completed_commands"].append(_run(record, run_environment))
            report_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
            if manifest["completed_commands"][-1]["returncode"]:
                raise RuntimeError(f"official command failed: {record['log']}")
            if record["stage"] == "input_readback":
                name = record["input"]
                exported = (manifest["input_exports"]["atlases"][name.removeprefix("atlas:")]
                            if name.startswith("atlas:") else manifest["input_exports"][name])
                manifest["input_readbacks"][name] = input_readback(record, exported)
                report_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        manifest["outputs"] = {}
        for seed in args.seeds:
            for name, (_, metadata) in profiles.items():
                align_official_matrices(args.output_dir / f"seed-{seed}" / "atlases" / name,
                                        metadata["nodes"],
                                        manifest["input_exports"]["atlases"][name]["maximum_present_label"])
            actual = load_profiles(args.output_dir / f"seed-{seed}")
            if set(actual) != set(profiles) or any(actual[name][0]["count"].shape != profiles[name][0]["count"].shape
                                                   for name in profiles):
                raise ValueError("official outputs do not contain the exact requested atlas matrix dimensions")
            root = args.output_dir / f"seed-{seed}"
            manifest["outputs"][str(seed)] = {
                "profiles": {name: metadata for name, (_, metadata) in actual.items()},
                "file_sha256": {name: sha256(root / name) for name in (
                    "tracks.tck", "sift2_weights.txt", "lengths.txt", "mean_fa.txt", "tckinfo.txt")},
            }
        if sha256(tracking_path) != manifest["tracking_input_sha256"]:
            raise ValueError("tracking checkpoint changed during official reference run")
        if any(sha256(args.checkpoint_dir / name) != digest
               for name, digest in readiness["file_sha256"].items()):
            raise ValueError("completed core checkpoint changed during official reference run")
        if manifest["source_fa_checkpoint_sha256"] is not None and sha256(args.checkpoint_dir / "fa.nii.gz") != manifest["source_fa_checkpoint_sha256"]:
            raise ValueError("FA checkpoint changed during official reference run")
        current_profiles = load_profiles(args.fnit_dir)
        for name, (_, metadata) in profiles.items():
            current = current_profiles[name][1]
            if current != metadata:
                raise ValueError(f"{name}: source atlas or matrices changed during official reference run")
        if any(sha256(Path(info["invoked_path"])) != info["sha256"] for info in programs.values()):
            raise ValueError("official executable changed during reference run")
        manifest["execution_completed"] = True
    except Exception as error:
        manifest["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        manifest["execution_wall_seconds"] = time.perf_counter() - started
        manifest["total_wall_seconds"] = time.perf_counter() - entry_started
        report_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"execution_completed": True, "report": str(report_path)}, allow_nan=False))


if __name__ == "__main__":
    main()
