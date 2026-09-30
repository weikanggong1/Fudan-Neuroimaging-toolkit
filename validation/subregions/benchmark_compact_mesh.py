"""真实保存阶段的丘脑或海马–杏仁核网格组件对照；不代表全脑或整条流程。

输入：官方保存阶段目录中的 processedImageMasked.mgz、alignedAtlasImage.mgz、
warpedOriginalMesh.txt.gz，以及原图谱 compressionLookupTable.txt。输出 JSON 记录
输入/源码哈希、首次与交替重复耗时、独立显存峰值、先验/目标函数/梯度差异。
两条路径共用一次 dense 初始化得到的 Gaussian 参数，全部计算保持 float32。
资源由用户在原许可范围内提供；本脚本不下载或复制图谱、影像。

示例（变量使用绝对路径）：
  prepared_stage=/absolute/path/official_saved_thalamus_stage  # 官方保存的真实阶段
  compression_lut=/absolute/path/thalamus/compressionLookupTable.txt  # 图谱标签顺序
  benchmark_output=/absolute/path/compact_thalamus_fp32.json  # 组件结果
  python validation/subregions/benchmark_compact_mesh.py \\
      --prepared-stage "$prepared_stage" --lut "$compression_lut" \\
      --structure thalamus --device cuda:0 --no-tf32 --output "$benchmark_output"

官方 FreeSurfer 8.2 对应流程：segment_subregions thalamus --cross subject_id
--sd subjects_directory --threads 4，或把 thalamus 替换为 hippo-amygdala。
这里只重放已保存的网格目标函数。
原实现：https://github.com/freesurfer/freesurfer/tree/dev/attic/gems
参考：Ashburner et al. (2000), Image registration using a symmetric prior—in three dimensions.
https://pmc.ncbi.nlm.nih.gov/articles/PMC6871943/
"""

import argparse
import gc
import hashlib
import inspect
import json
import os
import platform
import statistics
import subprocess
from pathlib import Path
from time import monotonic

import nibabel as nib
import numpy as np
import torch

from fnit.gems.atlas import GEMSAtlas
from fnit.gems.deformation import ashburner_prior, prepare_deformation_reference
from fnit.gems.gaussian import gaussian_log_likelihood, initialise_gaussians
from fnit.gems.rasterize import BlockIndex, build_block_index, rasterize_priors, rasterize_priors_compact
from fnit.gems.recipes import HippoAmygdalaRecipe, ThalamusRecipe


def file_identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def differences(dense, compact):
    gradient = dense["gradient"].double()
    gradient_difference = gradient - compact["gradient"].double()
    report = {
        "prior_max_abs_difference": float((dense["priors"] - compact["priors"]).abs().max()),
        "gradient_relative_l2_difference": float(torch.linalg.vector_norm(gradient_difference) /
                                                  torch.linalg.vector_norm(gradient).clamp_min(1e-30)),
        "gradient_max_abs_difference": float(gradient_difference.abs().max()),
        "coverage_disagreement_voxels": int((dense["coverage"] != compact["coverage"]).sum()),
    }
    for name in ("data_cost", "deformation_cost", "objective"):
        absolute = abs(dense[name] - compact[name])
        report[f"{name}_abs_difference"] = absolute
        report[f"{name}_relative_difference"] = absolute / max(abs(dense[name]), 1e-30)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepared-stage", required=True, type=Path, help="含三个官方保存阶段文件的目录")
    parser.add_argument("--lut", required=True, type=Path, help="对应图谱的 compressionLookupTable.txt")
    parser.add_argument("--structure", required=True,
                        choices=("thalamus", "hippo-amygdala-left", "hippo-amygdala-right"),
                        help="选择图谱的 Gaussian 标签分组")
    parser.add_argument("--device", default="cuda:0", choices=("cpu", "cuda:0"), help="PyTorch 计算设备")
    parser.add_argument("--output", required=True, type=Path, help="JSON 结果文件")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="允许 CUDA TF32，默认开启；--no-tf32 执行严格 FP32 数值门槛")
    parser.add_argument("--memory-fraction", type=float, default=.23, help="CUDA 进程显存比例，默认 0.23")
    parser.add_argument("--repeats", type=int, default=3, help="预热后交替重复次数，至少 3")
    parser.add_argument("--source-commit", help="无 .git 的冻结源码副本所对应提交；源码文件另行 SHA-256 核对")
    args = parser.parse_args()
    if args.repeats < 3 or not 0 < args.memory_fraction <= 1:
        parser.error("--repeats 必须至少为 3，--memory-fraction 必须在 (0,1] 内")

    device = torch.device(args.device)
    tf32_active = args.tf32 and device.type == "cuda"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def timed(operation):
        synchronize()
        started = monotonic()
        result = operation()
        synchronize()
        return result, monotonic() - started

    paths = [args.prepared_stage / name for name in
             ("processedImageMasked.mgz", "alignedAtlasImage.mgz", "warpedOriginalMesh.txt.gz")]
    image, aligned = nib.load(paths[0]), nib.load(paths[1])
    transform = np.linalg.inv(image.affine) @ aligned.affine
    atlas = GEMSAtlas.from_freesurfer(paths[2], args.lut).transformed(transform, transform_reference=True)
    recipe = (ThalamusRecipe("thalamus", args.lut.parent) if args.structure == "thalamus" else
              HippoAmygdalaRecipe(args.structure.rsplit("-", 1)[1], args.lut.parent))
    classes = recipe.intensity_groups(atlas, 0)
    background = int(classes[atlas.label_names.index("Unknown")])
    grouped = np.zeros((len(atlas.vertices), int(classes.max()) + 1), np.float32)
    for channel, group in enumerate(classes):
        grouped[:, group] += atlas.alphas[:, channel]
    alphas = torch.as_tensor(grouped, device=device)
    tetrahedra = torch.as_tensor(atlas.tetrahedra, device=device)
    reference = torch.as_tensor(atlas.reference_vertices, device=device, dtype=torch.float32)
    data = torch.as_tensor(np.asarray(image.dataobj, np.float32).squeeze(), device=device)
    valid = data > 0
    if data.ndim != 3 or not bool(valid.any()):
        raise ValueError("保存阶段必须是含正值体素的三维真实影像")
    shape = tuple(data.shape)
    index_started = monotonic()
    initial_index = build_block_index(atlas.vertices, atlas.tetrahedra, shape, margin=3)
    index_seconds = monotonic() - index_started
    class_ids = torch.arange(grouped.shape[1], device=device)

    def initialize():
        with torch.no_grad():
            priors, _ = rasterize_priors(torch.as_tensor(atlas.vertices, device=device, dtype=torch.float32),
                                         tetrahedra, alphas, shape, block_index=initial_index,
                                         background_channel=background, tolerance=2e-5)
            parameters = initialise_gaussians(data[valid].reshape(-1, 1, 1),
                                               priors[:, valid].reshape(grouped.shape[1], -1, 1, 1), class_ids)
            return gaussian_log_likelihood(data, parameters)[:, valid].detach()

    likelihood, initialization_seconds = timed(initialize)
    # Fresh per-path indices ensure the Gaussian initialization does not warm one path.
    indices = {name: BlockIndex(initial_index.shape, initial_index.block_size, initial_index.candidates)
               for name in ("dense", "compact")}
    del initial_index
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    def evaluate(name, reference_geometry):
        vertices = torch.tensor(atlas.vertices, device=device, dtype=torch.float32, requires_grad=True)
        synchronize()
        total_started = monotonic()

        def raster():
            if name == "dense":
                priors, coverage = rasterize_priors(vertices, tetrahedra, alphas, shape,
                                                     block_index=indices[name], background_channel=background,
                                                     tolerance=2e-5)
                return priors[:, valid], coverage[valid]
            return rasterize_priors_compact(vertices, tetrahedra, alphas, shape, valid_mask=valid,
                                            block_index=indices[name], background_channel=background,
                                            tolerance=2e-5)

        (selected, coverage), raster_seconds = timed(raster)
        data_cost, data_seconds = timed(lambda: -(selected.clamp_min(torch.finfo(selected.dtype).tiny).log() +
                                                 likelihood).logsumexp(0).sum())
        (deformation_cost, jacobians), deformation_seconds = timed(
            lambda: ashburner_prior(vertices, reference, tetrahedra, atlas.stiffness,
                                     reference_geometry=reference_geometry))
        objective = data_cost + deformation_cost
        _, backward_seconds = timed(objective.backward)
        total_seconds = monotonic() - total_started
        timings = {"raster_and_valid_selection_seconds": raster_seconds, "data_cost_seconds": data_seconds,
                   "deformation_forward_seconds": deformation_seconds, "backward_seconds": backward_seconds,
                   "forward_backward_seconds": total_seconds}
        snapshot = {"priors": selected.detach().cpu(), "coverage": coverage.detach().cpu(),
                    "gradient": vertices.grad.detach().cpu(), "data_cost": float(data_cost.detach()),
                    "deformation_cost": float(deformation_cost.detach()), "objective": float(objective.detach())}
        scalar = {key: snapshot[key] for key in ("data_cost", "deformation_cost", "objective")}
        scalar.update(covered_valid_voxels=int(coverage.sum()), minimum_jacobian=float(jacobians.min().detach()))
        return snapshot, {"timings": timings, "values": scalar}

    reference_geometry, reference_setup_seconds = timed(lambda: prepare_deformation_reference(reference, tetrahedra))
    runs = {name: {"first": None, "warm": []} for name in indices}
    comparisons = []
    for repetition in range(args.repeats + 1):
        order = ("dense", "compact") if repetition % 2 == 0 else ("compact", "dense")
        snapshots = {}
        for name in order:
            snapshots[name], measurement = evaluate(name, reference_geometry if name == "compact" else None)
            if repetition == 0:
                runs[name]["first"] = measurement
            else:
                runs[name]["warm"].append(measurement)
        comparisons.append({"repeat": repetition, "order": list(order),
                            **differences(snapshots["dense"], snapshots["compact"])})
        del snapshots
    for name in runs:
        runs[name]["warm_median_seconds"] = {
            key: statistics.median(run["timings"][key] for run in runs[name]["warm"])
            for key in runs[name]["first"]["timings"]}

    dense_batches = indices["dense"].device_batches(device, torch.float32)
    compact_batches, reorder = indices["compact"].device_compact_batches(valid, device, torch.float32)
    counts = {"grid_blocks": len(indices["dense"].candidates),
              "dense_active_blocks": sum(len(batch[0]) for batch in dense_batches),
              "dense_batches": len(dense_batches),
              "dense_rasterized_points": sum(batch[0].shape[0] * batch[0].shape[1] for batch in dense_batches),
              "compact_active_blocks": sum(len(batch[0]) for batch in compact_batches),
              "compact_batches": len(compact_batches),
              "compact_rasterized_points": sum(batch[0].shape[0] * batch[0].shape[1] for batch in compact_batches),
              "valid_points_without_candidates": int((reorder < 0).sum())}
    del dense_batches, compact_batches, reorder, reference_geometry

    # Measure each path with only its own dispatch/reference cache resident.
    memory = {}
    for name in indices:
        for index in indices.values():
            index._device_cache.clear()
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            synchronize()
            resident = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            cache = prepare_deformation_reference(reference, tetrahedra) if name == "compact" else None
            memory_snapshot, _ = evaluate(name, cache)
            memory[name] = {"resident_before_bytes": resident,
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                            "incremental_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) - resident,
                            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
            del cache, memory_snapshot
        else:
            memory[name] = None

    thresholds = {"prior_max_abs_difference": 1e-5, "gradient_relative_l2_difference": 1e-4,
                  "objective_relative_difference": 1e-6}
    worst = {key: max(row[key] for row in comparisons) for key in thresholds}
    finite = all(np.isfinite(row[key]) for row in comparisons for key in thresholds)
    parity_passed = finite and all(worst[key] < limit for key, limit in thresholds.items())
    parity_passed = parity_passed and all(row["coverage_disagreement_voxels"] == 0 for row in comparisons)
    source_root = Path(__file__).resolve().parents[2]
    source_commit, source_status = args.source_commit, None
    if (source_root / ".git").exists():
        source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_root, text=True).strip()
        source_status = subprocess.check_output(["git", "status", "--short"], cwd=source_root, text=True).splitlines()
    source_paths = [Path(__file__).resolve(), *[Path(inspect.getsourcefile(item)) for item in
                    (rasterize_priors, ashburner_prior, gaussian_log_likelihood, GEMSAtlas, type(recipe))]]
    report = {
        "mode": f"real_{args.structure}_mesh_cost_component", "structure": args.structure,
        "scope": "saved real-data mesh objective component; excludes preparation, EM iterations and whole-subject runtime",
        "first_call_context": "after shared dense Gaussian initialization; fresh per-path dispatch caches",
        "device": str(device), "dtype": "float32", "tf32": tf32_active, "tf32_requested": args.tf32,
        "threads": torch.get_num_threads(),
        "memory_fraction": args.memory_fraction if device.type == "cuda" else None,
        "shape": list(shape), "dense_voxels": data.numel(), "valid_voxels": int(valid.sum()),
        "vertices": len(atlas.vertices), "tetrahedra": len(atlas.tetrahedra), "classes": grouped.shape[1],
        "background_channel": background, "stiffness": atlas.stiffness, "block_size": indices["dense"].block_size,
        "block_index_seconds": index_seconds, "gaussian_initialization_seconds": initialization_seconds,
        "reference_cache_setup_seconds": reference_setup_seconds, "counts": counts,
        "timings_and_costs": runs, "pytorch_memory": memory, "comparisons": comparisons,
        "numerical_gate": {"enforced": not tf32_active, "thresholds": thresholds, "worst": worst,
                           "thresholds_met": parity_passed,
                           "interpretation": "TF32 records observed differences; strict FP32 parity is not claimed"
                           if tf32_active else "strict FP32 thresholds apply to all first/warm comparisons"},
        "inputs": [file_identity(path) for path in [*paths, args.lut]],
        "source_git_commit": source_commit, "source_git_status": source_status,
        "source_files": [file_identity(path) for path in source_paths],
        "mesh_to_image_transform": transform.tolist(),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "torch": torch.__version__, "cuda": torch.version.cuda,
                        "cudnn": torch.backends.cudnn.version(), "nibabel": nib.__version__,
                        "numpy": np.__version__, "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "warm_seconds": {
        name: runs[name]["warm_median_seconds"]["forward_backward_seconds"] for name in runs},
        "numerical_gate": report["numerical_gate"]}, allow_nan=False))
    if not tf32_active and not parity_passed:
        raise AssertionError("FP32 compact 网格组件未通过先验、梯度、目标函数或 coverage 数值门槛")


if __name__ == "__main__":
    main()
