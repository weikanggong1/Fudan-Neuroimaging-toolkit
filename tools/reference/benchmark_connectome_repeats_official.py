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


PROGRAMS = ("mrconvert", "mrinfo", "tckgen", "tckinfo", "tcksift2", "tckstats",
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
            "step_mm", "max_length_mm", "max_angle_degrees", "cutoff", "power")) or minimum < 0:
        raise ValueError("invalid actual tracking parameters")
    return params


def command_plan(mrtrix_bin: Path, output: Path, profiles: dict, seeds: list[int],
                 params: dict, threads: int) -> list[dict]:
    """只构造 argv；同一重复的 SIFT2、FA 和长度各计算一次供所有 atlas 使用。"""
    images = output / "inputs"
    records = []
    for name in ("wm_fod", "five_tissue_act", "five_tissue_sift2", "gmwmi", "fa"):
        records.append({"stage": "prepare_mif", "argv": [str(mrtrix_bin / "mrconvert"),
            str(images / f"{name}.nii.gz"), str(images / f"{name}.mif"), "-nthreads", str(threads)],
            "log": str(images / f"{name}.convert.log")})
    for seed in seeds:
        root = output / f"seed-{seed}"
        tracks = root / "tracks.tck"
        common = ["-nthreads", str(threads)]
        records.append({"stage": "tracking", "seed": seed, "argv": [str(mrtrix_bin / "tckgen"),
            "-algorithm", "iFOD2", "-seed_gmwmi", str(images / "gmwmi.mif"),
            "-act", str(images / "five_tissue_act.mif"), "-seeds", str(params["n_seed_attempts"]),
            "-select", "0", "-maxlength", str(params["max_length_mm"]),
            "-minlength", str(params["min_length_mm"]), "-step", str(params["step_mm"]),
            "-angle", str(params["max_angle_degrees"]), "-cutoff", str(params["cutoff"]),
            "-samples", str(params["samples"]), "-power", str(params["power"]), "-nthreads", "0",
            str(images / "wm_fod.mif"), str(tracks)], "log": str(root / "tckgen.log")})
        records.append({"stage": "track_count", "seed": seed,
                        "argv": [str(mrtrix_bin / "tckinfo"), "-count", str(tracks)],
                        "log": str(root / "tckinfo.txt")})
        for stage, argv in (
            ("sift2", ["tcksift2", str(tracks), str(images / "wm_fod.mif"), str(root / "sift2_weights.txt"),
                       "-act", str(images / "five_tissue_sift2.mif"), "-csv", str(root / "sift2_stats.csv")]),
            ("length", ["tckstats", "-dump", str(root / "lengths.txt"), str(tracks)]),
            ("fa", ["tcksample", "-precise", "-stat_tck", "mean", str(tracks),
                    str(images / "fa.mif"), str(root / "mean_fa.txt")]),
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
                             str(target / f"{name}.csv"), *common],
                    "log": str(target / f"{name}.log")})
    return records


def _save_image(path: Path, values, affine, spacing=None) -> dict:
    values = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
    affine = affine.detach().cpu().numpy() if isinstance(affine, torch.Tensor) else np.asarray(affine)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti2Image(values, affine)
    if spacing is not None:
        image.header.set_zooms((*spacing, *image.header.get_zooms()[3:]))
    nib.save(image, path)
    check = nib.load(path)
    if not np.array_equal(check.affine, affine) or not np.array_equal(np.asarray(check.dataobj), values):
        raise ValueError(f"{path}: NIfTI-2 export changed voxel values or affine")
    return {"path": str(path), "sha256": sha256(path), "dtype": str(values.dtype),
            "shape": list(values.shape), "affine": affine.tolist(),
            "header_spacing": list(check.header.get_zooms()[:3])}


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
                "tracking_input_sha256": sha256(tracking_path), "programs": programs,
                "source_profile_metadata": {name: metadata for name, (_, metadata) in profiles.items()},
                "source_fa_checkpoint_sha256": (sha256(args.checkpoint_dir / "fa.nii.gz")
                                               if payload.get("fa") is None else None),
                "seeds": args.seeds, "downstream_threads": args.downstream_threads,
                "commands": plan, "completed_commands": [], "execution_completed": False,
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
                (args.output_dir / f"seed-{record['seed']}").mkdir(exist_ok=True)
            run_environment = {**environment, "MRTRIX_RNG_SEED": str(record.get("seed", 0))}
            manifest["completed_commands"].append(_run(record, run_environment))
            report_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
            if manifest["completed_commands"][-1]["returncode"]:
                raise RuntimeError(f"official command failed: {record['log']}")
        for name in ("wm_fod", "five_tissue_act", "five_tissue_sift2", "gmwmi", "fa"):
            image_path = args.output_dir / "inputs" / f"{name}.mif"
            transform = subprocess.run([str(args.mrtrix_bin / "mrinfo"), str(image_path), "-transform"],
                                      check=True, capture_output=True, text=True, env=environment).stdout
            spacing = subprocess.run([str(args.mrtrix_bin / "mrinfo"), str(image_path), "-spacing"],
                                    check=True, capture_output=True, text=True, env=environment).stdout
            manifest["input_exports"][name]["mif_transform_text"] = transform.strip()
            manifest["input_exports"][name]["mif_spacing_text"] = spacing.strip()
            manifest["input_exports"][name]["mif_sha256"] = sha256(image_path)
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
