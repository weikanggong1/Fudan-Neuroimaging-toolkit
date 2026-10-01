"""独立验证用：原 MELODIC、原 ICA-AROMA 和相同混杂回归策略。

此脚本只放在 validation，不由 FNIT 的生产接口调用。输入是已完成运动
校正、掩膜、强度归一化和 100 s 高通的 4D BOLD；这里不再高通。可先用
--phase ica 估计一次 MELODIC，再用 --phase clean 分别检验配准后端。
原 ICA_AROMA_functions.py 和三张分类掩膜由调用者从作者网站下载，脚本
只导入外部文件，不把第三方源码复制进仓库。末端是独立 NumPy float64
SVD 回归：截距、二次趋势、WM/CSF 均值和 Friston-24；输出不加回均值。
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import shlex
import shutil
import subprocess
import time

import nibabel as nib
import numpy as np


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def image_check(path, *, reference=None, fourth_dimension=None):
    path = Path(path)
    if path.name.endswith(".gz"):
        with gzip.open(path, "rb") as stream:
            while stream.read(8 * 1024 * 1024):
                pass
    image = nib.load(str(path))
    if reference is not None and (image.shape[:3] != reference.shape[:3] or
            not np.allclose(image.affine, reference.affine, rtol=0, atol=1e-4)):
        raise ValueError(f"Unexpected image grid: {path.name}")
    if fourth_dimension is not None and image.shape != (*image.shape[:3], fourth_dimension):
        raise ValueError(f"Unexpected image frame count: {path.name}")
    for frame in range(image.shape[3] if image.ndim == 4 else 1):
        values = np.asarray(image.dataobj[..., frame] if image.ndim == 4 else image.dataobj)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite image values: {path.name}")
    return image


def load_mask(path, reference):
    image = nib.load(str(path))
    if image.shape != reference.shape[:3] or not np.allclose(image.affine, reference.affine, rtol=0, atol=1e-4):
        raise ValueError(f"Mask grid mismatch: {Path(path).name}")
    mask = np.asarray(image.dataobj) > 0
    if not mask.any():
        raise ValueError(f"Empty mask: {Path(path).name}")
    return mask


def save_image(path, values, reference):
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    header.set_slope_inter(1.0, 0.0)
    nib.save(nib.Nifti1Image(np.asarray(values, dtype=np.float32), reference.affine, header), str(path))


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class OfficialCommands:
    def __init__(self, args):
        self.directory = args.fsl_bin or args.fsl_dir / "bin"
        self.output = args.output_dir
        self.allow_255 = args.allow_complete_exit255
        self.records = []
        os.environ["FSLDIR"] = str(args.fsl_dir)
        os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"
        os.environ["PATH"] = str(self.directory) + os.pathsep + os.environ.get("PATH", "")
        os.environ["LD_LIBRARY_PATH"] = str(args.fsl_dir / "lib") + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")

    def run(self, name, arguments, validator, phase):
        command = [str(self.directory / name), *map(str, arguments)]
        log = self.output / (phase + ".log")
        started = time.perf_counter()
        with log.open("wb") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
        wall = time.perf_counter() - started
        record = {"phase": phase, "argv": command, "exit_status": result.returncode,
                  "command_wall_seconds": wall, "output_validation_passed": False}
        self.records.append(record)
        write_json(self.output / "commands.private.json", self.records)
        if result.returncode != 0 and not (result.returncode == 255 and self.allow_255):
            raise RuntimeError(f"{name} exited {result.returncode}; inspect {log}")
        validator()
        record["output_validation_passed"] = True
        record["abnormal_exit_accepted_after_complete_output_checks"] = result.returncode != 0
        write_json(self.output / "commands.private.json", self.records)
        return wall

    def provenance(self, fsl_dir):
        version = subprocess.run([str(self.directory / "melodic"), "--version"],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        text = version.stdout.decode("utf-8", errors="replace")
        (self.output / "melodic_version.log").write_text(text)
        match = re.search(r"MELODIC \(Version ([^)]+)\)", text)
        metadata = []
        for path in sorted((fsl_dir / "conda-meta").glob("fsl-melodic-*.json")):
            details = json.loads(path.read_text())
            metadata.append({key: details.get(key) for key in ("name", "version", "build")})
        return {"fslversion_file": (fsl_dir / "etc/fslversion").read_text().strip(),
                "melodic_version_string": match.group(1) if match else None,
                "melodic_conda_packages": metadata,
                "binary_sha256": {name: sha256(self.directory / name) for name in
                                  ("melodic", "fsl_regfilt", "applywarp", "fslroi", "fslmaths", "fslstats")}}


@contextmanager
def checked_aroma_commands(commands):
    """保留原 AROMA 的 FSL 命令，在外层核对每张临时 IC 的新输出。"""
    original_system = os.system
    original_getoutput = subprocess.getoutput
    count = 0

    def checked(command):
        nonlocal count
        values = shlex.split(command)
        name = Path(values[0]).name
        if name not in ("fslroi", "fslmaths"):
            raise ValueError(f"Unexpected original AROMA command: {name}")
        target = values[2] if name == "fslroi" else values[-1]
        count += 1
        commands.run(name, values[1:], lambda: image_check(target), f"aroma_spatial_{count:04d}")
        return 0

    def scalar_query(command):
        nonlocal count
        count += 1
        started = time.perf_counter()
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, check=False)
        wall = time.perf_counter() - started
        text = result.stdout.decode("utf-8", errors="replace").strip()
        if result.returncode != 0 and not (result.returncode == 255 and commands.allow_255):
            raise RuntimeError("Original AROMA scalar query failed")
        if not re.fullmatch(r"[0-9.eE+\-\s]+", text) or not np.isfinite(np.fromstring(text, sep=" ")).all():
            raise ValueError("Original AROMA scalar query did not return finite numbers")
        commands.records.append({"phase": f"aroma_scalar_query_{count:04d}", "shell_pipeline": command,
                                 "exit_status": result.returncode, "command_wall_seconds": wall,
                                 "output_validation_passed": True,
                                 "abnormal_exit_accepted_after_complete_output_checks": result.returncode != 0})
        write_json(commands.output / "commands.private.json", commands.records)
        return text

    os.system = checked
    subprocess.getoutput = scalar_query
    try:
        yield
    finally:
        os.system = original_system
        subprocess.getoutput = original_getoutput


def melodic_files(directory, source):
    mixing = np.loadtxt(directory / "melodic_mix", ndmin=2)
    spectrum = np.loadtxt(directory / "melodic_FTmix", ndmin=2)
    if mixing.shape[0] != source.shape[3] or not np.isfinite(mixing).all():
        raise ValueError("MELODIC mixing does not match the BOLD time axis")
    count = mixing.shape[1]
    if spectrum.shape[1] != count or not np.isfinite(spectrum).all():
        raise ValueError("MELODIC FTmix component count or values are invalid")
    image_check(directory / "melodic_IC.nii.gz", reference=source, fourth_dimension=count)
    for component in range(1, count + 1):
        image_check(directory / "stats" / f"thresh_zstat{component}.nii.gz", reference=source)
    return count


def run_melodic(args, commands, source, brain_mask):
    directory = args.melodic_dir
    if directory.exists():
        raise FileExistsError("Use a new MELODIC directory for --phase ica/all; --phase clean reuses it")
    directory.parent.mkdir(parents=True, exist_ok=True)
    arguments = [f"--in={args.input_bold}", f"--outdir={directory}", f"--mask={args.brain_mask}",
                 f"--tr={args.tr}", f"--dim={args.dim}", "--dimest=lap", "--nl=pow3", "--eps=0.001",
                 f"--maxit={args.ica_max_iter}", f"--seed={args.random_state}", "--Ostats", "--nobet",
                 "--mmthresh=0.5", "--report"]
    wall = commands.run("melodic", arguments, lambda: melodic_files(directory, source), "melodic")
    count = melodic_files(directory, source)
    maps = []
    for component in range(1, count + 1):
        image = nib.load(str(directory / "stats" / f"thresh_zstat{component}.nii.gz"))
        # 原 ICA-AROMA runICA 在 mixture 拟合失败时取最后一张 fallback 图。
        values = np.asarray(image.dataobj[..., -1] if image.ndim == 4 else image.dataobj, dtype=np.float32)
        maps.append(values * brain_mask)
    merged = directory / "melodic_IC_thr.nii.gz"
    save_image(merged, np.stack(maps, axis=3), source)
    manifest = {"input_sha256": sha256(args.input_bold), "brain_mask_sha256": sha256(args.brain_mask),
                "components": count, "dim_requested": args.dim, "random_state": args.random_state,
                "source_revision": args.source_revision, "highpass_cutoff_seconds_already_applied": args.highpass_cutoff_seconds,
                "additional_temporal_filtering": False, "melodic_command_wall_seconds": wall,
                "merged_thresholded_maps_sha256": sha256(merged)}
    write_json(directory / "oracle_ica.public.json", manifest)
    return manifest


def independent_confounds(args, aroma_path, source, brain_mask):
    image = nib.load(str(aroma_path))
    data = np.asarray(image.dataobj, dtype=np.float32)
    input_mean = np.asarray(source.dataobj, dtype=np.float32).mean(axis=3, dtype=np.float64)
    aroma_mean = data.mean(axis=3, dtype=np.float64)
    retained = brain_mask & np.any(data != 0, axis=3)
    mean_difference = (aroma_mean - input_mean)[retained]
    mean_check = {"evaluated_retained_brain_voxels": int(retained.sum()),
                  "zeroed_brain_voxels_with_nonzero_input_mean": int(np.count_nonzero(brain_mask & ~retained & (input_mean != 0))),
                  "aroma_minus_input_mean_mae": float(np.abs(mean_difference).mean()) if mean_difference.size else None,
                  "aroma_minus_input_mean_rmse": float(np.sqrt(np.square(mean_difference).mean())) if mean_difference.size else None,
                  "aroma_minus_input_mean_max_absolute": float(np.abs(mean_difference).max()) if mean_difference.size else None,
                  "scope": "Measured on retained native AROMA voxels; no assumed fsl_regfilt mean or implicit-mask rule."}
    nt = data.shape[3]
    time_axis = np.linspace(-1, 1, nt)
    columns = [np.ones(nt), time_axis, (3 * time_axis**2 - 1) / 2]
    labels = ["intercept", "linear_trend", "quadratic_trend"]
    tissue_counts = {}
    for name, path in (("wm", args.wm_mask), ("csf", args.csf_mask)):
        mask = load_mask(path, source)
        if np.any(mask & ~brain_mask):
            raise ValueError("Native WM/CSF regression masks must be intersected with the EPI brain mask")
        columns.append(data[mask].mean(axis=0, dtype=np.float64))
        labels.append(name)
        tissue_counts[name] = int(mask.sum())
    six = np.loadtxt(args.motion, ndmin=2)
    if six.shape != (nt, 6) or not np.isfinite(six).all():
        raise ValueError("Motion must contain T finite rows and six FSL parameters")
    previous = np.vstack((np.zeros((1, 6)), six[:-1]))
    model = np.column_stack((six, previous, six**2, previous**2))
    columns.extend(model.T)
    labels += [f"motion_{group}_{parameter}" for group in ("current", "previous", "current_squared", "previous_squared")
               for parameter in ("rx_rad", "ry_rad", "rz_rad", "tx_mm", "ty_mm", "tz_mm")]
    raw = np.column_stack(columns)
    np.savetxt(args.output_dir / "confounds_raw.tsv", raw, fmt="%.17g", delimiter="\t", header="\t".join(labels), comments="")
    active = np.r_[True, np.ptp(raw[:, 1:], axis=0) > 0]
    effective = raw[:, active].copy()
    effective[:, 1:] -= effective[:, 1:].mean(axis=0)
    effective /= np.linalg.norm(effective, axis=0)
    np.savetxt(args.output_dir / "confounds_normalized.tsv", effective, fmt="%.17g", delimiter="\t",
               header="\t".join(np.asarray(labels)[active]), comments="")
    left, singular_values, _ = np.linalg.svd(effective, full_matrices=False)
    rank = int(np.count_nonzero(singular_values > singular_values[0] * 1e-8))
    basis = left[:, :rank]
    flat = data.reshape((-1, nt))
    residual = np.empty_like(flat)
    for start in range(0, len(flat), args.chunk_voxels):
        series = flat[start:start + args.chunk_voxels].T.astype(np.float64)
        residual[start:start + args.chunk_voxels] = (series - basis @ (basis.T @ series)).T
    output = args.clean_native or args.output_dir / "clean_native.nii.gz"
    output.parent.mkdir(parents=True, exist_ok=True)
    save_image(output, residual.reshape(data.shape), image)
    image_check(output, reference=source, fourth_dimension=nt)
    return output, {"method": "independent NumPy float64 SVD joint projection, rcond=1e-8",
                    "raw_design_shape": list(raw.shape), "effective_rank": rank,
                    "effective_design_columns": int(effective.shape[1]), "tissue_mask_voxels": tissue_counts,
                    "bandpass": None, "global_signal": False, "quadratic_drift": True,
                    "temporal_mean_restored": False, "output_dtype": "float32",
                    "raw_design_sha256": sha256(args.output_dir / "confounds_raw.tsv"),
                    "normalized_design_sha256": sha256(args.output_dir / "confounds_normalized.tsv"),
                    "outside_brain_mask_max_absolute_value": float(np.abs(residual[~brain_mask.ravel()]).max()),
                    "aroma_temporal_mean_check": mean_check}


def main():
    driver_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("all", "ica", "clean"), default="all")
    for name in ("input-bold", "brain-mask", "output-dir", "fsl-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("fsl-bin", "melodic-dir", "motion", "wm-mask", "csf-mask", "aroma-functions",
                 "classification-masks-dir", "mni-template", "epi-to-t1", "t1-to-mni-warp", "clean-native"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--warp-convention", choices=("auto", "relative", "absolute"), default="auto")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--tr", type=float, required=True)
    parser.add_argument("--dim", type=int, default=0, help="0 自动定阶；正整数仅用于另行注明的固定阶数控制")
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--ica-max-iter", type=int, default=500)
    parser.add_argument("--highpass-cutoff-seconds", type=float, default=100, help="记录上游已做的高通周期，本脚本不再次滤波")
    parser.add_argument("--chunk-voxels", type=int, default=4096)
    parser.add_argument("--allow-complete-exit255", action="store_true", help="仅在新输出全部通过 CRC/网格/有限值检查后接受255；原退出码仍记录")
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    if args.tr <= 0 or args.dim < 0 or args.chunk_voxels < 1 or args.ica_max_iter < 1:
        parser.error("TR/chunk/max-iter must be positive; dim must be nonnegative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.melodic_dir = args.melodic_dir or args.output_dir / "melodic.ica"
    source = nib.load(str(args.input_bold))
    if source.ndim != 4 or source.shape[3] < 4:
        parser.error("input-bold must be a 4D BOLD with at least four frames")
    brain_mask = load_mask(args.brain_mask, source)
    commands = OfficialCommands(args)
    report = {"schema_version": 1, "source_revision": args.source_revision,
              "scope": "official MELODIC/ICA-AROMA plus independent NumPy regression; benchmark only",
              "data": {"anonymous_id": "real_run_01", "shape": list(source.shape), "tr_seconds": args.tr,
                       "brain_mask_voxels": int(brain_mask.sum())},
              "input_sha256": {"bold": sha256(args.input_bold), "brain_mask": sha256(args.brain_mask)},
              "software": commands.provenance(args.fsl_dir), "driver_sha256": sha256(__file__)}
    if args.phase in ("all", "ica"):
        report["ica"] = run_melodic(args, commands, source, brain_mask)
    else:
        report["ica"] = json.loads((args.melodic_dir / "oracle_ica.public.json").read_text())
        for name, path in (("input_sha256", args.input_bold), ("brain_mask_sha256", args.brain_mask)):
            if report["ica"][name] != sha256(path):
                raise ValueError("Reused ICA must belong to this exact BOLD and mask")
        melodic_files(args.melodic_dir, source)
    if args.phase in ("all", "clean"):
        for key in ("motion", "wm_mask", "csf_mask", "aroma_functions", "classification_masks_dir",
                    "mni_template", "epi_to_t1", "t1_to_mni_warp"):
            if getattr(args, key) is None:
                parser.error("--phase all/clean requires --" + key.replace("_", "-"))
        mapped = args.output_dir / "melodic_IC_thr_MNI152_2mm.nii.gz"
        template = nib.load(str(args.mni_template))
        arguments = [f"--in={args.melodic_dir / 'melodic_IC_thr.nii.gz'}", f"--ref={args.mni_template}",
                     f"--warp={args.t1_to_mni_warp}", f"--premat={args.epi_to_t1}", f"--out={mapped}", "--interp=trilinear"]
        if args.warp_convention != "auto":
            arguments.append("--rel" if args.warp_convention == "relative" else "--abs")
        timings = {"classification_resampling": commands.run("applywarp", arguments,
                   lambda: image_check(mapped, reference=template, fourth_dimension=report["ica"]["components"]), "classification_resampling")}
        classification = args.output_dir / "classification"
        classification.mkdir(exist_ok=True)
        for name in ("mask_csf.nii.gz", "mask_edge.nii.gz", "mask_out.nii.gz"):
            original = args.classification_masks_dir / name
            load_mask(original, template)
            target = classification / name
            if not target.exists():
                target.symlink_to(original)
        specification = importlib.util.spec_from_file_location("official_ica_aroma", args.aroma_functions)
        official = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(official)
        random.seed(args.random_state)
        np.random.seed(args.random_state)
        started = time.perf_counter()
        # 原 feature_spatial 使用未加引号的路径拼接；提前拒绝歧义路径。
        for path in (commands.directory, classification, args.classification_masks_dir, mapped):
            if any(character.isspace() for character in str(path)):
                raise ValueError("Original ICA-AROMA spatial function requires paths without whitespace")
        with working_directory(classification), checked_aroma_commands(commands):
            motion_feature = official.feature_time_series(str(args.melodic_dir / "melodic_mix"), str(args.motion))
            frequency_feature = official.feature_frequency(str(args.melodic_dir / "melodic_FTmix"), args.tr)
            edge, csf = official.feature_spatial(str(commands.directory) + "/", str(classification), str(args.classification_masks_dir), str(mapped))
            noise = np.atleast_1d(official.classification(str(classification), motion_feature, edge, frequency_feature, csf)).astype(int)
        timings["official_aroma_features"] = time.perf_counter() - started
        features = np.column_stack((motion_feature, edge, frequency_feature, csf))
        if not np.isfinite(features).all():
            raise ValueError("Official AROMA produced nonfinite features")
        aroma_path = args.output_dir / "filtered_func_data_aroma.nii.gz"
        if len(noise):
            arguments = [f"--in={args.input_bold}", f"--design={args.melodic_dir / 'melodic_mix'}",
                         "--filter=" + ",".join(map(str, noise + 1)), f"--out={aroma_path}"]
            timings["fsl_regfilt_nonaggr"] = commands.run("fsl_regfilt", arguments,
                lambda: image_check(aroma_path, reference=source, fourth_dimension=source.shape[3]), "fsl_regfilt_nonaggr")
        else:
            shutil.copyfile(args.input_bold, aroma_path)
            timings["fsl_regfilt_nonaggr"] = 0.0
        started = time.perf_counter()
        clean, confounds = independent_confounds(args, aroma_path, source, brain_mask)
        timings["independent_confounds"] = time.perf_counter() - started
        report["aroma"] = {"mode": "nonaggr", "components": report["ica"]["components"], "noise_components": len(noise),
                           "motion_feature_splits": 1000, "python_random_seed": args.random_state,
                           "official_functions_sha256": sha256(args.aroma_functions), "uses_original_fsl_regfilt": bool(len(noise)),
                           "classification_mask_sha256": {name: sha256(args.classification_masks_dir / name)
                                                          for name in ("mask_csf.nii.gz", "mask_edge.nii.gz", "mask_out.nii.gz")}}
        upstream = args.aroma_functions.parent / "upstream_manifest.json"
        if upstream.is_file():
            report["aroma"]["external_upstream_manifest"] = json.loads(upstream.read_text())
        report["confounds"] = confounds
        report["timing_seconds"] = timings
        for name, path in (("aroma", aroma_path), ("clean_native", clean), ("motion", args.motion),
                           ("wm_mask", args.wm_mask), ("csf_mask", args.csf_mask), ("epi_to_t1", args.epi_to_t1),
                           ("t1_to_mni_warp", args.t1_to_mni_warp), ("mni_template", args.mni_template)):
            group = "output_sha256" if name in ("aroma", "clean_native") else "input_sha256"
            report.setdefault(group, {})[name] = sha256(path)
    report["commands"] = [{key: record[key] for key in ("phase", "exit_status", "command_wall_seconds", "output_validation_passed", "abnormal_exit_accepted_after_complete_output_checks")}
                          for record in commands.records]
    report["native_subprocess_wall_sum_seconds"] = sum(record["command_wall_seconds"] for record in commands.records)
    report["native_subprocess_timing_scope"] = "FSL command calls and original AROMA scalar-query shell pipelines; excludes image validation, hashing, feature Python arithmetic and NumPy confounds."
    report["driver_wall_seconds"] = time.perf_counter() - driver_started
    report["privacy"] = "Only anonymous scalars and hashes; images, designs, command paths and logs stay local."
    write_json(args.output_dir / "official_denoising.public.json", report)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
