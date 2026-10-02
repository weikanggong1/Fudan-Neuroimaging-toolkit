"""Generate recon-all's MNI152 nonlinear warp and registration check image."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from fnit._transforms import AffineTransform
from fnit.synthmorph import SynthMorph

from .mni_aux_chain import TEMPLATE_DIR, write_mni_voxel_lta


def run_mni_nonlinear_chain(subject_dir: str | Path, weights_dir: str | Path,
                            assets_dir: str | Path, *, warp_convert: str | Path,
                            ca_register: str | Path, mri_convert: str | Path,
                            device: str = "cpu", threads: int = 4,
                            postprocess_backend: str = "conda", chunk_slices: int = 16) -> dict:
    """将个体裁剪 T1 配准到 MNI152，并写出前向、逆向 warp 和检查图。

    subject_dir 提供 conform orig、裁剪 T1 与初始 affine LTA；weights_dir
    和 assets_dir 提供固定权重与 1 mm MNI 模板。三个具名原生程序路径分别
    完成 warp 转换、求逆和最近邻检查图。device 默认 CPU，threads 默认 4。
    CUDA 的本阶段固定使用 FP32：同输入验证表明 TF32 会放大位移误差；
    其他阶段的 TF32 设置在返回时恢复。返回三个绝对输出路径、模型、设备、
    精度设置及各子步骤秒数；NIfTI 位移单位为 mm。
    postprocess_backend默认conda，gpu为完整GPU后处理候选；仅gpu要求显式CUDA。
    chunk_slices默认16，为GPU转换/检查图的X方向分块大小，不减少求逆范围。
    三个原生程序路径只在conda后端消费；gpu不调用它们。
    缺少输入、后处理失败或输出网格不符时抛出异常，不静默回退。
    """
    if postprocess_backend not in ("conda", "gpu"):
        raise ValueError("postprocess_backend must be conda or gpu")
    if postprocess_backend == "gpu" and torch.device(device).type != "cuda":
        raise ValueError("GPU postprocessing requires an explicit CUDA device")
    if threads < 1 or chunk_slices < 1:
        raise ValueError("threads and chunk_slices must be positive")
    subject = Path(subject_dir)
    mri = subject / "mri"
    transforms = mri / "transforms/synthmorph.1.0mm.1.0mm"
    temporary = transforms / "tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    target_dir = Path(assets_dir) / TEMPLATE_DIR
    cropped_target = target_dir / "mni152.1.0mm.cropped.nii.gz"
    full_target = target_dir / "mni152.1.0mm.nii.gz"
    crop = transforms / "invol.crop.nii.gz"
    original = mri / "orig.mgz"
    affine = transforms / "aff.lta"
    weight = Path(weights_dir) / "synthmorph.deform.3.h5"
    for path in (cropped_target, full_target, crop, original, affine, weight):
        if not path.is_file():
            raise FileNotFoundError(path)
    torch.set_num_threads(threads)
    timings = {}
    tick = time.perf_counter()
    previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    forwards = []
    cuda = torch.device(device).type == "cuda"
    try:
        registration = SynthMorph(weights=weights_dir, device=device, model="deform", extent=256, configure_precision=False)
        if cuda:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        result = registration(crop, cropped_target, init=affine, transform_only=True, compute_inverse=False, precision_report=forwards)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
    ras_warp = temporary / "deform.mgz"
    result.transform.save(ras_warp)
    del registration, result
    timings["deform_model_and_save"] = time.perf_counter() - tick

    crop_image, original_image = nib.load(str(crop)), nib.load(str(original))
    write_mni_voxel_lta(temporary / "reg.crop-to-invol.lta",
                        np.linalg.inv(original_image.affine) @ crop_image.affine,
                        crop, original)
    AffineTransform(np.eye(4), source=nib.load(str(cropped_target)),
                    target=nib.load(str(full_target)), space="world").save(
                        temporary / "reg.crop-to-full.lta")
    forward = transforms / "warp.to.mni152.1.0mm.1.0mm.nii.gz"
    inverse = transforms / "warp.to.mni152.1.0mm.1.0mm.inv.nii.gz"
    check = transforms / "test.nii.gz"
    environment = dict(os.environ, FREESURFER_HOME=str(Path(assets_dir)))
    postprocess_report = {}
    if postprocess_backend == "gpu":
        from .mni_warp_sampling import convert_mni_warp, resample_mni_check
        from .mni_warp_inverse import invert_mni_warp
        postprocess_report["conversion"] = convert_mni_warp(
            ras_warp, crop, original, full_target, temporary / "reg.crop-to-invol.lta",
            temporary / "reg.crop-to-full.lta", forward, device=device,
            chunk_slices=chunk_slices)
        timings["warp_convert"] = postprocess_report["conversion"]["total_seconds"]
        postprocess_report["inverse"] = invert_mni_warp(forward, inverse, device=device)
        timings["warp_inverse"] = postprocess_report["inverse"]["total_seconds"]
        postprocess_report["check"] = resample_mni_check(
            original, forward, check, device=device, chunk_slices=chunk_slices)
        timings["resample_check"] = postprocess_report["check"]["total_seconds"]
    else:
        tick = time.perf_counter()
        subprocess.run([str(warp_convert), "--inras", str(ras_warp),
                        "--insrcgeom", str(crop), "--outfswarp", str(forward),
                        "--vg-thresh", "1e-4", "--lta1-inv",
                        str(temporary / "reg.crop-to-invol.lta"), "--lta2",
                        str(temporary / "reg.crop-to-full.lta")],
                       env=environment, check=True)
        timings["warp_convert"] = time.perf_counter() - tick
        tick = time.perf_counter()
        subprocess.run([str(ca_register), "-invert-and-save", str(forward),
                        str(inverse)], env=environment, check=True)
        timings["warp_inverse"] = time.perf_counter() - tick
        tick = time.perf_counter()
        subprocess.run([str(mri_convert), "-rt", "nearest", str(original),
                        "-at", str(forward), str(check)], env=environment, check=True)
        timings["resample_check"] = time.perf_counter() - tick
    if nib.load(str(forward)).shape != (*nib.load(str(full_target)).shape, 1, 3) \
            or nib.load(str(inverse)).shape != (*original_image.shape, 1, 3) \
            or nib.load(str(check)).shape != nib.load(str(full_target)).shape:
        raise ValueError("MNI152 nonlinear outputs have unexpected grids")
    return {"forward": str(forward), "inverse": str(inverse), "check": str(check),
            "model": "pytorch-synthmorph-deform", "device": device,
            "precision": {"cuda_fp32_exception": cuda, "fp16_or_bf16": False, "actual_forwards": forwards},
            "timings_seconds": timings, "postprocess_backend": postprocess_backend,
            "postprocess": postprocess_report}
