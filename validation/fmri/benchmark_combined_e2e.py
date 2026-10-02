"""连续测量真实单例 raw BIDS → volume → fsLR32k / 91k。

--plan 是服务器私有 JSON；必填路径字段为 bids_root、derivatives_root、
subject、mni_template、mni_brain_mask、recon_all、hcp_assets_dir、wb_command、
synthstrip_weights、capture_root。source_root 可写在清单或 --source-root。
--source-revision 绑定实际冻结提交，--report-out 写匿名统计。

驱动固定 FNIRT、STC 关闭、100 秒高通、nonaggr AROMA、WM/CSF/24 项运动
回归及默认 preproc surface。两个公开 API 在同一进程中依次实际计算；
输出目录必须不存在，解剖缓存不复用，球面必须本次估计。不运行原软件。
每个包内函数只加 perf_counter 观察，不增加 GPU 同步或改变其参数。
私有捕获使用硬链接，只有内存中运动校正图和 FAST 附加图需保存；完整
观察时间与扣除捕获后的时间同时保留。哈希、QC 与数值比较均在计时外。
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import functools
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

import nibabel as nib
import numpy as np
import torch

from fnit.fmri import fMRIVolume_pipeline, fMRISurface_pipeline
from fnit.fmri.bids import locate_bids_inputs
from fnit.fmri.derivatives import sidecar
import fnit.fmri.end_to_end as volume_module
import fnit.fmri.aroma_pipeline as aroma_module
import fnit.fmri.bbr as bbr_module
import fnit.fmri.pipeline as feat_module
import fnit.fmri.spatial as spatial_module
import fnit.fmri.normalization as normalization_module
import fnit.fmri.surface_pipeline as surface_module
from fnit.fast import TorchFAST
from fnit.flirt import TorchFLIRT
from fnit.fnirt import TorchFNIRT
from fnit.mcflirt import TorchMCFLIRT
from fnit.synthstrip import SynthStrip


_helper_spec = importlib.util.spec_from_file_location(
    "combined_benchmark_helpers", Path(__file__).with_name("benchmark_bids.py"))
_helpers = importlib.util.module_from_spec(_helper_spec)
_helper_spec.loader.exec_module(_helpers)
sha256 = _helpers.sha256
source_hashes = _helpers.source_hashes


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def gpu_state():
    result = subprocess.run([
        "nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    if result.returncode:
        return {"available": False, "exit_code": result.returncode}
    fields = ("physical_index", "name", "total_memory_mib", "used_memory_mib", "utilization_percent")
    return {"available": True, "devices": [dict(zip(fields, values))
            for line in result.stdout.splitlines()
            if len(values := [item.strip() for item in line.split(",")]) == len(fields)]}


class Observer:
    """只观察函数边界；捕获内容不参与候选计算或原版参照。"""

    def __init__(self, root):
        self.root = root
        self.capture_seconds = 0.0
        self.capture_by_stage = {}
        self.capture_methods = {"hardlink": 0, "copy": 0, "image_save": 0, "array_save": 0}
        self.calls = {}
        self.patches = []
        self.volume_stage = "volume_setup"
        self.surface_stage = "surface_setup"
        self.mode = "volume"
        self.mcf = None
        self.msm_inputs = {}
        self.initial_spheres = []
        self.projection_inputs = None
        self.strip_calls = 0

    @contextmanager
    def capture(self, stage):
        started = time.perf_counter()
        try:
            yield
        finally:
            seconds = time.perf_counter() - started
            self.capture_seconds += seconds
            self.capture_by_stage[stage] = self.capture_by_stage.get(stage, 0.0) + seconds

    def file(self, source, destination):
        source, destination = Path(source), Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination)
            self.capture_methods["hardlink"] += 1
        except OSError:
            shutil.copyfile(source, destination)
            self.capture_methods["copy"] += 1

    def patch(self, owner, name, replacement):
        self.patches.append((owner, name, getattr(owner, name)))
        setattr(owner, name, replacement)

    def restore(self):
        for owner, name, original in reversed(self.patches):
            setattr(owner, name, original)
        self.patches.clear()

    def timed(self, original, name, *, context=None, after=None):
        @functools.wraps(original)
        def observed(*args, **kwargs):
            label = name(*args, **kwargs) if callable(name) else name
            previous = self.volume_stage
            if context is not None:
                self.volume_stage = context
            captured_before = self.capture_seconds
            started = time.perf_counter()
            try:
                result = original(*args, **kwargs)
                if after is not None:
                    after(result, args, kwargs)
            finally:
                elapsed = time.perf_counter() - started
                capture = self.capture_seconds - captured_before
                record = self.calls.setdefault(label, {"calls": 0, "observed_seconds": 0.0,
                                                        "capture_seconds": 0.0, "capture_adjusted_seconds": 0.0})
                record["calls"] += 1
                record["observed_seconds"] += elapsed
                record["capture_seconds"] += capture
                record["capture_adjusted_seconds"] += elapsed - capture
                if context is not None:
                    self.volume_stage = previous
            return result
        return observed

    def observe_final_volume_resampling(self, module):
        def resample_name(*args, **kwargs):
            if kwargs.get("motion_pull_world") is not None:
                return "preproc_mni_resampling" if kwargs.get("pre_affine_pull_ras") is not None else "preproc_t1w_resampling"
            return "clean_mni_resampling" if kwargs.get("output_mask") is not None else "clean_mni_mask_resampling"
        entry_point = _helpers.final_volume_resampler_name(module)
        self.patch(module, entry_point, self.timed(getattr(module, entry_point), resample_name))

    def install(self):
        # Keep computational input paths and objects unchanged.
        def captured_strip(result, args, kwargs):
            self.strip_calls += 1
            if self.strip_calls == 1:
                with self.capture("epi_synthstrip"):
                    result.mask.save(self.root / "volume/masks/epi_mask.nii.gz")
                    result.image.save(self.root / "volume/masks/epi_brain.nii.gz")
                    self.capture_methods["image_save"] += 2
        self.patch(SynthStrip, "__call__", self.timed(
            SynthStrip.__call__, lambda *a, **k: "synthstrip_in_" + self.volume_stage,
            after=captured_strip))

        def captured_fast(result, args, kwargs):
            with self.capture("fast"):
                for attribute, filename in (("pve_gm", "T1_pve_gm.nii.gz"),
                                             ("bias_field", "T1_bias.nii.gz"),
                                             ("restored", "T1_restored.nii.gz")):
                    getattr(result, attribute).save(self.root / "volume/masks" / filename)
                    self.capture_methods["image_save"] += 1
        self.patch(TorchFAST, "__call__", self.timed(TorchFAST.__call__, "fast_computation", after=captured_fast))

        def captured_motion(result, args, kwargs):
            self.mcf = {"cost_evaluations": result.cost_evaluations,
                        "frames": len(result.matrices), "interpolation": kwargs.get("interpolation")}
            with self.capture("feat_core"):
                destination = self.root / "volume/feat"
                destination.mkdir(parents=True, exist_ok=True)
                if result.corrected is not None:
                    nib.save(result.corrected, str(destination / "prefiltered_func_data_mcf.nii.gz"))
                    self.capture_methods["image_save"] += 1
                np.save(destination / "matrices.private.npy", result.matrices, allow_pickle=False)
                np.save(destination / "parameters.private.npy", result.parameters, allow_pickle=False)
                self.capture_methods["array_save"] += 2
        self.patch(TorchMCFLIRT, "run", self.timed(TorchMCFLIRT.run, "mcflirt_complete", after=captured_motion))
        self.patch(spatial_module, "apply_motion_warp", self.timed(
            spatial_module.apply_motion_warp, "mcflirt_final_sampling"))

        def captured_feat(result, args, kwargs):
            with self.capture("feat_core"):
                for name in ("filtered_func_data.nii.gz", "mask.nii.gz", "mean_func.nii.gz",
                             "example_func.nii.gz", "mc/prefiltered_func_data_mcf.par"):
                    self.file(result.output_dir / name, self.root / "volume/feat" / name)
                for matrix in sorted(result.motion_matrices.glob("MAT_*")):
                    self.file(matrix, self.root / "volume/feat/mc/prefiltered_func_data_mcf.mat" / matrix.name)
        self.patch(volume_module, "run_feat_core", self.timed(
            volume_module.run_feat_core, "feat_complete", context="feat_core", after=captured_feat))
        for name in ("grand_mean_scale", "gaussian_highpass"):
            self.patch(feat_module, name, self.timed(getattr(feat_module, name), name))

        def captured_anatomical(result, args, kwargs):
            if result.reused:
                raise RuntimeError("fresh end-to-end benchmark unexpectedly reused anatomy")
            with self.capture("anatomical_capture"):
                for name in ("T1_brain.nii.gz", "T1_mask.nii.gz", "MNI_brain.nii.gz", "MNI_mask.nii.gz",
                             "T1_pve_wm.nii.gz", "T1_pve_csf.nii.gz", "T1_wmseg.nii.gz"):
                    self.file(result.path(name), self.root / "volume/masks" / name)
                registration = result.registration
                self.file(registration.affine, self.root / "volume/reg" / registration.affine.name)
                self.file(registration.pull_ras, self.root / "volume/reg" / registration.pull_ras.name)
        self.patch(volume_module, "prepare_anatomical", self.timed(
            volume_module.prepare_anatomical, "anatomical_complete", context="anatomy", after=captured_anatomical))
        def captured_bbr(result, args, kwargs):
            with self.capture("bbr_capture"):
                directory = self.root / "volume/reg"
                directory.mkdir(parents=True, exist_ok=True)
                nib.save(result.moved, str(directory / "example_func2highres.nii.gz"))
                np.savetxt(directory / "example_func2highres.mat", result.matrix, fmt="%.12g")
                self.capture_methods["image_save"] += 1
        self.patch(bbr_module, "register_bbr", self.timed(
            bbr_module.register_bbr, "bbr_complete", context="bbr", after=captured_bbr))

        def captured_affine(result, args, kwargs):
            role = "bbr_initial" if self.volume_stage == "bbr" else "t1_affine"
            stage = "bbr_initial_flirt" if role == "bbr_initial" else "t1_to_mni_affine"
            with self.capture(stage):
                directory = self.root / "volume/reg"
                directory.mkdir(parents=True, exist_ok=True)
                nib.save(result.moved, str(directory / (role + "_moved.nii.gz")))
                np.savetxt(directory / (role + ".mat"), result.matrix, fmt="%.12g")
                self.capture_methods["image_save"] += 1
        self.patch(TorchFLIRT, "__call__", self.timed(
            TorchFLIRT.__call__, lambda *a, **k: "bbr_initial_flirt_computation" if self.volume_stage == "bbr" else "t1_affine_computation",
            after=captured_affine))

        def captured_fnirt(result, args, kwargs):
            with self.capture("t1_to_mni_nonlinear"):
                directory = self.root / "volume/reg"
                directory.mkdir(parents=True, exist_ok=True)
                nib.save(result.moved, str(directory / "t1_nonlinear_moved.nii.gz"))
                self.capture_methods["image_save"] += 1
        self.patch(TorchFNIRT, "__call__", self.timed(TorchFNIRT.__call__, "fnirt_computation", after=captured_fnirt))

        for name, label in (("decompose_spatial_ica", "pica"), ("classify_aroma", "aroma_classification"),
                            ("denoise_aroma", "aroma_denoising"), ("clean_confounds", "nuisance_regression")):
            self.patch(aroma_module, name, self.timed(getattr(aroma_module, name), label))

        def captured_aroma(result, args, kwargs):
            with self.capture("pica_aroma_confounds"):
                for source, destination in (
                    (result.denoised_bold, "aroma/filtered_func_data_aroma.nii.gz"),
                    (result.features, "aroma/aroma_features.tsv"),
                    (result.noise_components, "aroma/aroma_noise_components.txt")):
                    self.file(source, self.root / "volume" / destination)
                for name in ("mixing", "frequency_power", "thresholded_maps"):
                    path = getattr(result.ica, name)
                    self.file(path, self.root / "volume/aroma/ica" / path.name)
                for name in ("wm_mask", "regression_csf_mask"):
                    if kwargs.get(name) is not None:
                        self.file(kwargs[name], self.root / "volume/masks" / (name + ".nii.gz"))
        self.patch(volume_module, "run_aroma_pipeline", self.timed(
            volume_module.run_aroma_pipeline, "pica_aroma_confounds_complete", after=captured_aroma))

        self.observe_final_volume_resampling(volume_module)
        # AROMA imports the normalization module at its own call boundary.
        self.patch(normalization_module, "resample_world", self.timed(
            normalization_module.resample_world, "aroma_map_mni_resampling"))

        def captured_prepare(result, args, kwargs):
            with self.capture("native_surface_preparation"):
                for index, source in enumerate(result.initial_spheres):
                    destination = self.root / "surface/initial_spheres" / f"{'LR'[index]}.sphere.surf.gii"
                    self.file(source, destination)
                    self.initial_spheres.append(str(destination.resolve()))
        self.patch(surface_module, "prepare_fmriprep_surface_inputs", self.timed(
            surface_module.prepare_fmriprep_surface_inputs, "surface_native_preparation", after=captured_prepare))

        def captured_msm_inputs(result, args, kwargs):
            with self.capture("msmsulc_preparation_and_registration"):
                for hemisphere, entry in result.items():
                    self.msm_inputs[hemisphere] = {}
                    for field, source in asdict(entry).items():
                        if source is None:
                            self.msm_inputs[hemisphere][field] = None
                            continue
                        destination = self.root / "surface/msm_inputs" / hemisphere / Path(source).name
                        self.file(source, destination)
                        self.msm_inputs[hemisphere][field] = str(destination.resolve())
                write_json(self.root / "surface/msm_inputs.private.json", self.msm_inputs)
        self.patch(surface_module, "prepare_msmsulc_inputs", self.timed(
            surface_module.prepare_msmsulc_inputs, "surface_msm_input_preparation", after=captured_msm_inputs))
        self.patch(surface_module, "run_msmsulc", self.timed(surface_module.run_msmsulc, "surface_msmsulc_estimation"))

        original_projection = surface_module.run_fmriprep_surface_projection

        @functools.wraps(original_projection)
        def observed_projection(*args, **kwargs):
            # The projection's own internal timer starts inside the original
            # function, after this capture. Only the enclosing surface/outer
            # walls include it; do not subtract it from L/R projection times.
            with self.capture("surface_projection_input_capture"):
                bound = inspect.signature(original_projection).bind(*args, **kwargs)
                bound.apply_defaults()
                values = bound.arguments
                self.projection_inputs = {
                    "bold_file": str(Path(values["clean_t1w"]).resolve()),
                    "bold_std": str(Path(values["clean_mni"]).resolve()),
                    "volume_roi": str(Path(values["goodvoxels"]).resolve()) if values["goodvoxels"] is not None else None,
                    "repetition_time": float(values["tr_seconds"]), "expected_frames": self.frames,
                    "signal": "preproc", "geometry_space": "T1w world RAS", "sphere_kind": "estimated_msmsulc",
                }
                attributes = {"white": "white", "pial": "pial", "midthickness": "midthickness",
                              "midthickness_fsLR": "atlas_midthickness", "sphere_reg_fsLR": "registered_sphere",
                              "cortex_mask": "native_roi"}
                for field, attribute in attributes.items():
                    self.projection_inputs[field] = []
                    for hemisphere, geometry in zip(("L", "R"), (values["left"], values["right"])):
                        destination = self.root / "surface/projection_inputs" / f"{hemisphere}.{field}.gii"
                        self.file(getattr(geometry, attribute), destination)
                        self.projection_inputs[field].append(str(destination.resolve()))
                self.projection_inputs["native_rois"] = self.projection_inputs["cortex_mask"]
                self.projection_inputs["initial_spheres"] = self.initial_spheres
                self.projection_inputs["area_surfaces"] = {
                    "native": self.projection_inputs["midthickness"], "fsLR": self.projection_inputs["midthickness_fsLR"]}
                write_json(self.root / "surface/inputs.private.json", self.projection_inputs)
            return original_projection(*args, **kwargs)
        self.patch(surface_module, "run_fmriprep_surface_projection", observed_projection)

    def adjusted_stages(self, result, mode):
        timings = dict(result.timing_seconds)
        allowed = ({"epi_synthstrip", "fast", "bbr_initial_flirt", "t1_to_mni_affine", "t1_to_mni_nonlinear",
                    "feat_core", "pica_aroma_confounds"} if mode == "volume" else
                   {"native_surface_preparation", "msmsulc_preparation_and_registration"})
        for name in allowed:
            if name in timings:
                timings[name] -= self.capture_by_stage.get(name, 0.0)
        return timings


def read_plan(path, source_root):
    plan = json.loads(path.read_text())
    if "capture_root" not in plan and "private_output" in plan:
        plan["capture_root"] = str(Path(plan["private_output"]) / "intermediates")
    required = ("bids_root", "derivatives_root", "subject", "mni_template", "mni_brain_mask", "recon_all",
                "hcp_assets_dir", "wb_command", "synthstrip_weights", "capture_root")
    if any(name not in plan for name in required):
        raise ValueError("private plan lacks a required field")
    for name in required:
        if name == "subject":
            if not isinstance(plan[name], str) or not plan[name] or plan[name].startswith("sub-"):
                raise ValueError("subject must be a nonempty BIDS label without sub-")
        elif not isinstance(plan[name], str) or not Path(plan[name]).is_absolute():
            raise ValueError(f"{name} must be an absolute path")
    if source_root is not None:
        plan["source_root"] = str(source_root.resolve())
    if not Path(plan.get("source_root", "")).is_absolute():
        raise ValueError("source_root must be in the plan or CLI")
    for name in ("derivatives_root", "capture_root"):
        if Path(plan[name]).exists() or Path(plan[name]).is_symlink():
            raise FileExistsError(f"{name} must not exist for a fresh continuous run")
    for name in ("bids_root", "mni_template", "mni_brain_mask", "recon_all", "hcp_assets_dir", "synthstrip_weights", "source_root"):
        if not Path(plan[name]).exists():
            raise FileNotFoundError(f"missing {name}")
    if not Path(plan["wb_command"]).is_file():
        raise FileNotFoundError("missing wb_command")
    return plan


def volume_private_files(result, captures):
    """明确映射实际产物；私有对照驱动不需要推断 BIDS 文件名。"""
    files = {name: str(getattr(result, name).resolve()) for name in
             ("clean_native", "clean_mni", "mask_mni", "t1_brain", "bbr_matrix", "metadata",
              "preproc_t1w", "preproc_mni", "motion_pull", "mni_pull")}
    roles = {
        "epi_mask": "masks/epi_mask.nii.gz", "epi_brain": "masks/epi_brain.nii.gz",
        "t1_mask": "masks/T1_mask.nii.gz", "mni_template_mask": "masks/MNI_mask.nii.gz",
        "fast_gm": "masks/T1_pve_gm.nii.gz", "fast_wm": "masks/T1_pve_wm.nii.gz",
        "fast_csf": "masks/T1_pve_csf.nii.gz", "fast_bias": "masks/T1_bias.nii.gz",
        "fast_restored": "masks/T1_restored.nii.gz", "t1_wmseg": "masks/T1_wmseg.nii.gz",
        "native_wm_mask": "masks/wm_mask.nii.gz", "native_csf_mask": "masks/regression_csf_mask.nii.gz",
        "bbr_initial": "reg/bbr_initial.mat", "bbr_initial_moved": "reg/bbr_initial_moved.nii.gz",
        "bbr_final_moved": "reg/example_func2highres.nii.gz", "t1_affine_moved": "reg/t1_affine_moved.nii.gz",
        "t1_nonlinear_moved": "reg/t1_nonlinear_moved.nii.gz", "t1_affine": "reg/t1_affine.mat",
        "motion_corrected": "feat/prefiltered_func_data_mcf.nii.gz", "feat_filtered": "feat/filtered_func_data.nii.gz",
        "feat_mask": "feat/mask.nii.gz", "feat_reference": "feat/example_func.nii.gz",
        "motion_parameters": "feat/mc/prefiltered_func_data_mcf.par",
        "motion_matrices": "feat/mc/prefiltered_func_data_mcf.mat",
        "unrounded_matrices": "feat/matrices.private.npy", "unrounded_parameters": "feat/parameters.private.npy",
        "aroma_native": "aroma/filtered_func_data_aroma.nii.gz", "aroma_features": "aroma/aroma_features.tsv",
        "aroma_noise_components": "aroma/aroma_noise_components.txt",
    }
    for role, relative in roles.items():
        path = captures / "volume" / relative
        if not path.exists():
            raise FileNotFoundError(f"missing actual private capture for {role}")
        files[role] = str(path.resolve())
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--plan", type=Path, required=True, help="私有路径清单")
    parser.add_argument("--source-root", type=Path, help="覆盖清单中的冻结源码根目录")
    parser.add_argument("--source-revision", required=True, help="实际冻结源码提交")
    parser.add_argument("--report-out", type=Path, required=True, help="匿名汇总 JSON")
    args = parser.parse_args()
    plan = read_plan(args.plan, args.source_root)
    from fnit.msm import _fastpd_native
    root, captures = Path(plan["source_root"]), Path(plan["capture_root"])
    device = torch.device("cuda:0")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.init()
    torch.cuda.set_per_process_memory_fraction(min(1.0, 20e9 / torch.cuda.get_device_properties(device).total_memory), device)
    torch.cuda.synchronize(device)
    inputs = locate_bids_inputs(plan["bids_root"], subject=plan["subject"])
    if len(inputs.t1w_images) != 1:
        raise ValueError("this benchmark requires one matching T1w")
    raw = nib.load(str(inputs.bold))
    if raw.ndim != 4 or raw.shape[3] < 2:
        raise ValueError("input must be a complete real 4D BOLD run")
    sources_before = source_hashes(root)
    loaded_volume = Path(volume_module.__file__).resolve()
    if loaded_volume != (root / "src/fnit/fmri/end_to_end.py").resolve():
        raise ValueError("PYTHONPATH does not point to the supplied frozen source")
    hashes = {"bold": sha256(inputs.bold), "sbref": sha256(inputs.sbref) if inputs.sbref else None,
              "t1w": sha256(inputs.t1w_images[0]), "mni_template": sha256(plan["mni_template"]),
              "mni_brain_mask": sha256(plan["mni_brain_mask"]), "synthstrip_weights": sha256(plan["synthstrip_weights"])}
    captures.mkdir(parents=True, exist_ok=False, mode=0o700)
    (captures / "volume/masks").mkdir(parents=True)
    observer = Observer(captures)
    observer.frames = raw.shape[3]
    initial_gpu = gpu_state()
    torch.cuda.reset_peak_memory_stats(device)
    observer.install()
    start = time.perf_counter()
    try:
        volume_started = time.perf_counter()
        volume = fMRIVolume_pipeline(
            plan["bids_root"], plan["derivatives_root"], subject=plan["subject"],
            mni_template=plan["mni_template"], mni_brain_mask=plan["mni_brain_mask"],
            synthstrip_weights=plan["synthstrip_weights"], registration_backend="fnirt",
            reuse_anatomical=False, bbr_execution="batched", fnirt_execution="optimized",
            regress_wm=True, regress_csf=True, regress_motion=True, motion_model=24,
            highpass_cutoff_seconds=100.0, aroma_mode="nonaggr", slice_timing=False,
            global_signal=False, bandpass=None, random_state=0, n_splits=1000,
            ica_max_iter=500, ica_n_components=None, motion_iterations=(1, 1, 1),
            device=str(device), batch_size=8, overwrite=False,
        )
        volume_elapsed = time.perf_counter() - volume_started
        volume_capture = observer.capture_seconds
        volume_memory = {"peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
                         "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device)}
        with observer.capture("controller_manifest_between_apis"):
            private_volume = volume_private_files(volume, captures)
            write_json(captures / "volume/outputs.private.json", private_volume)
            write_json(captures / "volume/phase.complete.private.json", {
                "complete": True, "source_revision": args.source_revision, "files": private_volume,
                "public_api_observed_seconds": volume_elapsed,
                "public_api_capture_adjusted_seconds": volume_elapsed - volume_capture,
            })
        surface_capture_before = observer.capture_seconds
        observer.mode = "surface"
        surface_started = time.perf_counter()
        surface = fMRISurface_pipeline(
            plan["bids_root"], plan["derivatives_root"], subject=plan["subject"],
            recon_all=plan["recon_all"], hcp_assets_dir=plan["hcp_assets_dir"],
            wb_command=plan["wb_command"], device=str(device), signal="preproc",
            registered_spheres=None, msm_config=None, msm_execution="optimized",
            parallel=True, cpu_threads=8, overwrite=False,
        )
        surface_elapsed = time.perf_counter() - surface_started
        surface_capture = observer.capture_seconds - surface_capture_before
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
    finally:
        observer.restore()
    final_gpu = gpu_state()
    surface_memory = {"peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
                      "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device)}
    memory = {name: max(volume_memory[name], surface_memory[name]) for name in volume_memory}
    memory.update(after_volume=volume_memory, after_surface=surface_memory)
    sources_after = source_hashes(root)
    if sources_before != sources_after:
        raise ValueError("frozen runtime sources changed during the run")
    volume_meta, surface_meta = (json.loads(result.metadata.read_text()) for result in (volume, surface))
    if volume_meta["FNIT"]["Report"]["anatomical_cache"]["reused"]:
        raise ValueError("anatomy was reused")
    if not volume_meta["FNIT"]["Report"]["ica_converged"]:
        raise ValueError("ICA did not converge")
    template = nib.load(plan["mni_template"])
    mask = np.asarray(nib.load(str(volume.mask_mni)).dataobj) > 0
    checks = {"volume": {
        "clean_native": _helpers.check_native_volume(volume.clean_native, inputs),
        "clean_mni": _helpers.check_volume(volume.clean_mni, template, mask),
        "preproc_t1w": _helpers.check_volume(volume.preproc_t1w),
        "preproc_mni": _helpers.check_volume(volume.preproc_mni, template),
        "mask_voxels": int(mask.sum()), "mask_sha256": sha256(volume.mask_mni),
        "motion_pull_sha256": sha256(volume.motion_pull), "mni_pull_sha256": sha256(volume.mni_pull)}}
    cifti = nib.load(str(surface.dtseries))
    values = np.asarray(cifti.dataobj, dtype=np.float32)
    if values.shape != (raw.shape[3], 91282) or not np.isfinite(values).all():
        raise ValueError("invalid complete CIFTI shape or values")
    if not np.isclose(cifti.header.get_axis(0).step, inputs.tr, rtol=0, atol=1e-6):
        raise ValueError("CIFTI changed TR")
    registered = surface_meta["FNIT"]["RegisteredSpheres"]
    if set(registered) != {"L", "R"} or not all(entry["EstimatedHere"] for entry in registered.values()):
        raise ValueError("both spheres must be estimated in this run")
    checks["surface"] = {"cifti_shape": list(values.shape), "all_finite": True,
                         "cifti_sha256": sha256(surface.dtseries), "tr_seconds": float(cifti.header.get_axis(0).step),
                         "brain_models": {name: int(model.size) for name, _, model in cifti.header.get_axis(1).iter_structures()},
                         "hemispheres": {}}
    for hemi, path in (("L", surface.left), ("R", surface.right)):
        metric = nib.load(str(path))
        if len(metric.darrays) != raw.shape[3] or any(a.data.shape != (32492,) or a.data.dtype != np.float32
                                                   or not np.isfinite(a.data).all() for a in metric.darrays):
            raise ValueError("invalid complete GIFTI")
        checks["surface"]["hemispheres"][hemi] = {"frames": len(metric.darrays), "vertices": 32492,
                                                "all_finite": True, "sha256": sha256(path)}
    # Capture temporary BOLD paths now point to deleted staging files. Replace
    # them with persistent outputs only after all timed calls have completed.
    observer.projection_inputs["bold_file"] = str(volume.preproc_t1w.resolve())
    observer.projection_inputs["bold_std"] = str(volume.preproc_mni.resolve())
    write_json(captures / "surface/inputs.private.json", observer.projection_inputs)
    startpoints = {"t1w_preproc": sha256(volume.preproc_t1w), "mni_preproc": sha256(volume.preproc_mni)}
    write_json(captures / "surface/outputs.private.json", {
        "left": str(surface.left.resolve()), "right": str(surface.right.resolve()),
        "dtseries": str(surface.dtseries.resolve()),
        "registered_spheres": [str(path.resolve()) for path in surface.registered_spheres],
        "projection_inputs_json": str((captures / "surface/inputs.private.json").resolve()),
        "msm_inputs_json": str((captures / "surface/msm_inputs.private.json").resolve()),
        "startpoint_sha256": startpoints, "msm_config": surface_meta["FNIT"]["RegistrationDetails"]["Configuration"],
        "source_revision": args.source_revision,
    })
    combined_files = {"volume": private_volume, "surface": {
        "left": str(surface.left.resolve()), "right": str(surface.right.resolve()),
        "dtseries": str(surface.dtseries.resolve()), "metadata": str(surface.metadata.resolve()),
        "qc_report": str(surface.qc_report.resolve()),
        "registered_spheres": [str(path.resolve()) for path in surface.registered_spheres],
        "outputs_manifest": str((captures / "surface/outputs.private.json").resolve()),
    }, "private_capture_root": str(captures.resolve())}
    write_json(captures / "files.private.json", combined_files)
    write_json(captures / "outputs.private.json", combined_files)
    volume_stages = observer.adjusted_stages(volume, "volume")
    surface_stages = observer.adjusted_stages(surface, "surface")
    volume_stages["total"] -= volume_capture
    surface_stages["total"] -= surface_capture
    motion = observer.calls.get("mcflirt_complete", {}).get("capture_adjusted_seconds")
    sampling = observer.calls.get("mcflirt_final_sampling", {}).get("capture_adjusted_seconds")
    report = {
        "schema_version": 1, "source_revision": args.source_revision,
        "source_sha256_before": sources_before, "source_sha256_after": sources_after,
        "source_unchanged_during_run": True, "driver_sha256": sha256(__file__),
        "native_extension_sha256": sha256(_fastpd_native.__file__), "input_sha256": hashes,
        "data": {"kind": "one real UK Biobank run", "subjects": 1, "bold_shape": list(raw.shape),
                 "t1w_shape": list(nib.load(str(inputs.t1w_images[0])).shape)},
        "configuration": {**_helpers.public_configuration(volume_meta["FNIT"]["Configuration"]),
                          "surface_signal": "preproc", "surface_msm_execution": "optimized",
                          "surface_parallel": True, "surface_cpu_thread_budget": 8},
        "algorithm": {"volume": {name: volume_meta["FNIT"]["Report"][name] for name in
                                   ("ica_components", "ica_iterations", "ica_converged", "aroma_noise_components", "anatomical_cache")},
                      "mcflirt": observer.mcf, "surface_registration": surface_meta["FNIT"]["Registration"],
                      "surface_registration_details": surface_meta["FNIT"]["RegistrationDetails"]},
        "checks": checks,
        "timing_seconds": {
            "raw_to_cifti_observed_including_capture_and_output_save": elapsed,
            "raw_to_cifti_capture_adjusted_including_output_save": elapsed - observer.capture_seconds,
            "volume_public_api_observed": volume_elapsed,
            "volume_public_api_capture_adjusted": volume_elapsed - volume_capture,
            "surface_public_api_observed": surface_elapsed,
            "surface_public_api_capture_adjusted": surface_elapsed - surface_capture,
            "private_capture_overhead": observer.capture_seconds,
            "private_capture_seconds_by_stage": observer.capture_by_stage,
            "volume_pipeline_stages_observed": dict(volume.timing_seconds),
            "volume_pipeline_stages_capture_adjusted": volume_stages,
            "surface_pipeline_stages_observed": dict(surface.timing_seconds),
            "surface_pipeline_stages_capture_adjusted": surface_stages,
            "observed_function_calls": observer.calls,
            "mcflirt_loading_reference_estimation_and_output_conversion_excluding_sampling": motion - sampling,
        },
        "timing_boundaries": {
            "outer": "One continuous same-process wall from volume API invocation through surface publication and final CUDA synchronization; both APIs actually compute from fresh derivatives. Recon-all reconstruction is an existing input.",
            "initialization": "Package import, CUDA initialization, input/source hashing and shared-GPU observations precede the outer timer.",
            "capture_adjustment": "Subtracts measured private capture only; observed wall remains primary evidence. It is an estimate of running without capture, not an independently measured no-capture run.",
            "function_calls": "perf_counter only; no added synchronization. Nested calls overlap and must not be summed. MCFLIRT subtraction includes input loading, reference creation, fitting and output conversion.",
            "stages": "Unmodified runtime timers have their own boundaries. Internal total excludes final BIDS publication; concurrent L/R projection timings overlap.",
            "postprocessing": "All QC, hashes and comparisons occur after the outer timer. One small private completion/path manifest between the APIs is separately measured and subtracted as controller_manifest_between_apis; final manifests are outside.",
        },
        "memory": memory, "observer": {"calculation_modified": False, "extra_gpu_synchronization_inside_functions": False,
                                          "capture_methods": observer.capture_methods},
        "environment": {"host": platform.node(), "python": platform.python_version(), "torch": torch.__version__,
                        "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(device),
                        "logical_device": str(device), "visible_device_mapping": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "cpu_threads": 8, "gpu_memory_limit_gb": 20, "tf32": True, "low_precision_enabled": False,
                        "gpu_state_before": initial_gpu, "gpu_state_after": final_gpu},
        "limits": ["One complete run on a shared host; measured durations are observations.",
                   "Recon-all reconstruction, resource deployment and compilation are excluded; existing same-source geometry is verified by the API.",
                   "No STC, SDC/GDC, FIX, spatial smoothing, global-signal regression or additional bandpass.",
                   "ICA-AROMA clean and minimally preprocessed surface are separate output branches; original reference comparisons must use their corresponding branches."],
        "privacy": "Only anonymous scalars, source/input/output hashes and resource identities are public. Raw images, subject labels, geometry, paths and full arrays remain private.",
    }
    write_json(args.report_out, report)
    print(json.dumps({"completed": "raw_to_cifti", "observed_seconds": elapsed,
                      "capture_adjusted_seconds": elapsed - observer.capture_seconds,
                      "volume_observed_seconds": volume_elapsed, "surface_observed_seconds": surface_elapsed,
                      "frames": raw.shape[3], "memory": memory}, indent=2), flush=True)


if __name__ == "__main__":
    main()
