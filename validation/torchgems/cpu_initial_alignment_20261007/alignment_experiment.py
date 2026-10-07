"""GEMS 初始对齐实验选择器；默认软 Dice，CPU 保序候选必须显式选择。

本模块的右侧真实初始对齐案例通过原20配准门及源定义空间/网格门。CPU
候选仍依赖完整 FNIT checkout 的 validation 适配器；不是 wheel 生产 API，
不更改 GEMSRecipe.run 或 GPU 路径。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import monotonic


@dataclass
class InitialAlignmentResult:
    """atlas voxel -> 未裁剪 context.image voxel 矩阵和阶段报告。"""
    atlas_to_context_voxel: object
    report: dict
    registration_result: dict | None = None


def context_label_image(context, *, coarse_reference=None):
    """给 ndarray 标签恢复 context.image 几何，或核验原标签影像。

    context.image 是当前拟合处理网格；context.native_image 可以是预处理前的
    另一网格，此函数不读取后者，也不使用准备目标或高分辨率 ROI 的 crop affine。
    """
    import nibabel as nib
    import numpy as np
    from fnit._nib import load_image
    from fnit.robust_register.preparation import _geometry, _mgh_image

    image = context.image
    values = np.asarray(context.coarse_segmentation)
    if (values.ndim != 3 or values.shape != image.shape
            or not np.issubdtype(values.dtype, np.integer)):
        raise ValueError("coarse_segmentation must be integer labels on context.image grid")
    geometry = _geometry(image)
    if coarse_reference is not None:
        labels_image = load_image(coarse_reference, "coarse_reference")
        if (labels_image.ndim != 3 or labels_image.shape != image.shape
                or not np.array_equal(_geometry(labels_image).affine, geometry.affine)
                or not np.array_equal(np.asarray(labels_image.dataobj), values)):
            raise ValueError("coarse_reference must match context label values and geometry")
    elif isinstance(image.header, nib.freesurfer.mghformat.MGHHeader):
        # Reuse the mature source-defined Double geometry / MGH field writer.
        labels_image = _mgh_image(values.astype(np.int32, copy=False), geometry, image)
    elif isinstance(image, (nib.Nifti1Image, nib.Nifti2Image)):
        header = image.header.copy()
        header.set_data_dtype(np.int32)
        labels_image = type(image)(values.astype(np.int32, copy=False),
                                   geometry.affine.copy(), header)
    else:
        raise TypeError("context.image must have an MGH or NIfTI header")
    if not np.array_equal(_geometry(labels_image).affine, geometry.affine):
        raise ValueError("label wrapper changed source-defined context geometry")
    return labels_image


def initialize_recipe_alignment(recipe, context, *, method="soft-dice",
                                device="cpu", cpu_candidate=None,
                                output_directory=None, coarse_reference=None,
                                registration_parameters=None):
    """只生成初始变换，不执行 GEMS mesh / intensity / postprocess。

    soft-dice 直接委托现有函数，可接收原 CPU/GPU 设备。cpu-robust-experiment
    限海马/杏仁核左右侧和 CPU，调用方持有一个 live、可重复使用的适配器。
    """
    if method == "soft-dice":
        from fnit.gems.initialize import estimate_mask_affine
        matrix, score = estimate_mask_affine(
            recipe.alignment_image(), context.image,
            context.coarse_segmentation, recipe.alignment_ids, device=device)
        return InitialAlignmentResult(matrix, {
            "method": "soft-dice", "alignment_dice": score,
            "output_frame": "uncropped_context_image_voxels"})
    if method != "cpu-robust-experiment":
        raise ValueError("method must be soft-dice or cpu-robust-experiment")
    if str(device) != "cpu":
        raise ValueError("cpu-robust-experiment requires CPU; no GPU fallback")
    if (getattr(recipe, "name", None) not in
            ("hippo-amygdala-left", "hippo-amygdala-right")
            or getattr(recipe, "side", None) not in ("left", "right")):
        raise ValueError("this CPU bridge is limited to hippocampus/amygdala recipes")
    if cpu_candidate is None:
        raise ValueError("pass one live reusable CPU candidate explicitly")
    cpu_candidate._ensure_open()
    if output_directory is None:
        raise ValueError("output_directory is required for the CPU experiment")
    directory = Path(output_directory)
    if directory.exists():
        raise FileExistsError("output_directory must not exist")
    parameters = dict(device="cpu", saturation=50.0, iterations_per_level=5,
                      stop_distance=0.01, initialize_translation=True,
                      pyramid_min_size=16, pyramid_max_size=-1,
                      highres_iterations=-1, spatial_chunk_size=131072,
                      memory_budget_gb=20.0, tf32=True)
    parameters.update(registration_parameters or {})
    if str(parameters.get("device")) != "cpu" or "mode" in parameters:
        raise ValueError("CPU pair fixes device=cpu and rigid then affine modes")
    # No global thread or precision setters: preserve the caller's policy.
    import numpy as np
    import nibabel as nib
    from fnit._nib import load_image
    from fnit.robust_register import (
        prepare_subregion_alignment_target, reflect_atlas_header)
    from fnit.robust_register.preparation import _geometry

    started = monotonic()
    labels_image = context_label_image(context, coarse_reference=coarse_reference)
    atlas = load_image(Path(recipe.directory) / "AtlasDump.mgz", "atlas_dump")
    if not isinstance(atlas, nib.freesurfer.mghformat.MGHImage) or atlas.ndim != 3:
        raise TypeError("atlas dump must be one MGH/MGZ 3D image")
    prepared = prepare_subregion_alignment_target(
        labels_image, recipe.alignment_ids, target_voxel_mm=1.0,
        bbox_margin_voxels=6, smoothing="backward", device="cpu",
        memory_budget_gb=parameters["memory_budget_gb"])
    moving = reflect_atlas_header(atlas) if recipe.side == "right" else atlas
    directory.mkdir(parents=True, exist_ok=False)
    source_path = directory / "atlas.source.mgz"
    target_path = directory / "target.mask.mgz"
    nib.save(moving, source_path)
    nib.save(prepared.image, target_path)
    preparation_seconds = monotonic() - started
    registration_started = monotonic()
    pair = cpu_candidate.cpu_robust_rigid_affine(
        source_path, target_path, stage_directory=directory / "registration",
        **parameters)
    registration_seconds = monotonic() - registration_started
    conversion_started = monotonic()
    # Original SAMSEG reloads the second mapmovhdr MGH and maps its geometry.
    # Use that storage boundary, not unrounded combined_RAS @ original source.
    # Return into context.image, NOT target-mask crop or context.native_image.
    mapped_atlas = nib.load(directory / "registration" / "affine.header.mgz")
    # Surfa Affine.inv / __matmul__ use NumPy Double inv / matmul here.
    # This final GEMS frame conversion is distinct from robust Eigen arithmetic.
    matrix = np.matmul(np.linalg.inv(_geometry(context.image).affine),
                       _geometry(mapped_atlas).affine)
    conversion_seconds = monotonic() - conversion_started
    return InitialAlignmentResult(matrix, {
        "method": "cpu-robust-experiment", "side": recipe.side,
        "alignment_dice": None,  # Do not mislabel robust error as soft Dice.
        "target_preparation": prepared.report,
        "output_frame": "uncropped_context_image_voxels",
        "source_frame": "original_AtlasDump_voxel_indices",
        "crop_affine_not_used_as_output_frame": True,
        "aligned_atlas_geometry_source": "reloaded_final_mapmovhdr_MGH",
        "unrounded_combined_RAS_not_used_for_GEMS_frame": True,
        "coarse_reference_preserved": coarse_reference is not None,
        "parameters": parameters,
        "seconds": {"preparation_and_save": preparation_seconds,
                    "rigid_affine_API": registration_seconds,
                    "frame_conversion": conversion_seconds,
                    "total": monotonic() - started},
    }, pair)
