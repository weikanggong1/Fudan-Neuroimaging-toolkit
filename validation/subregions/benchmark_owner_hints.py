"""真实保存 T1 网格在平滑顶点移动时，对照完整 owner 搜索与 interior hint。

全部影像、图谱、Gaussian likelihood 来自已有真实保存阶段；顶点沿固定
平滑场移动，单独测重复闭包的时间、cost/gradient/owner 差异及正 Jacobian。
这是网格组件，不能作为一个人完整 T1 分割 benchmark。脚本不下载资源，
不调用原软件运行时，计算保持 FP32/TF32；可单独开启全局 data-cost FP64 累加。
"""
import argparse
import gc
import inspect
import json
import os
from pathlib import Path
import statistics
from time import monotonic

import nibabel as nib
import numpy as np
import torch

from benchmark_fused_mesh import identity
from fnit.gems.atlas import GEMSAtlas
from fnit.gems import rasterize as raster_module
from fnit.gems.deformation import ashburner_prior, prepare_current_geometry, prepare_deformation_reference
from fnit.gems.gaussian import gaussian_log_likelihood, initialise_gaussians
from fnit.gems.rasterize import BlockIndex, build_block_index, compact_data_cost, rasterize_priors_compact
from fnit.gems.recipes import HippoAmygdalaRecipe, ThalamusRecipe


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared-stage", type=Path, required=True)
    p.add_argument("--lut", type=Path, required=True)
    p.add_argument("--structure", choices=["thalamus", "hippo-amygdala-left", "hippo-amygdala-right"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--memory-fraction", type=float, default=.23)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--sampling-stride", type=int, default=1)
    p.add_argument("--hint-tolerance", type=float, default=2e-4)
    p.add_argument("--refresh-interval", type=int, default=8)
    p.add_argument("--double-data-cost", action="store_true")
    a = p.parse_args()
    if a.repeats < 3 or a.sampling_stride < 1 or a.refresh_interval < 1:
        p.error("repeats>=3 and positive strides are required")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = a.tf32
    torch.backends.cudnn.allow_tf32 = a.tf32
    device = torch.device(a.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(a.memory_fraction, device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    paths = [a.prepared_stage / name for name in
             ("processedImageMasked.mgz", "alignedAtlasImage.mgz", "warpedOriginalMesh.txt.gz")]
    image, aligned = nib.load(paths[0]), nib.load(paths[1])
    transform = np.linalg.inv(image.affine) @ aligned.affine
    atlas = GEMSAtlas.from_freesurfer(paths[2], a.lut).transformed(transform, transform_reference=True)
    recipe = ThalamusRecipe("thalamus", a.lut.parent) if a.structure == "thalamus" else HippoAmygdalaRecipe(a.structure.rsplit("-", 1)[1], a.lut.parent)
    classes = recipe.intensity_groups(atlas, 0)
    background = int(classes[atlas.label_names.index("Unknown")])
    grouped = np.zeros((len(atlas.vertices), int(classes.max()) + 1), np.float32)
    for channel, group in enumerate(classes):
        grouped[:, group] += atlas.alphas[:, channel]
    alphas = torch.as_tensor(grouped, device=device)
    tetra = torch.as_tensor(atlas.tetrahedra, device=device)
    reference = torch.as_tensor(atlas.reference_vertices, device=device, dtype=torch.float32)
    initial = torch.as_tensor(atlas.vertices, device=device, dtype=torch.float32)
    data = torch.as_tensor(np.asarray(image.dataobj, np.float32).squeeze(), device=device)
    valid = data > 0
    shape = tuple(data.shape)
    initial_index = build_block_index(atlas.vertices, atlas.tetrahedra, shape, margin=3)
    with torch.no_grad():
        priors, _ = rasterize_priors_compact(initial, tetra, alphas, shape, valid_mask=valid,
            block_index=initial_index, background_channel=background)
        params = initialise_gaussians(data[valid].reshape(-1, 1, 1),
            priors.reshape(grouped.shape[1], -1, 1, 1), torch.arange(grouped.shape[1], device=device))
        likelihood = gaussian_log_likelihood(data[valid].reshape(-1, 1, 1), params).reshape(grouped.shape[1], -1).detach()
    del priors
    mesh_valid, scale = valid, 1.0
    if a.sampling_stride > 1:
        x = torch.arange(shape[0], device=device)[:, None, None]
        y = torch.arange(shape[1], device=device)[None, :, None]
        z = torch.arange(shape[2], device=device)[None, None, :]
        quadrature = (x + 3 * y + 5 * z) % a.sampling_stride == 0
        selection = quadrature[valid]
        mesh_valid = valid & quadrature
        scale = selection.numel() / int(selection.sum())
        likelihood = likelihood[:, selection].contiguous()
    ref_geometry = prepare_deformation_reference(reference, tetra)
    field_xyz = atlas.vertices - atlas.vertices.mean(0)
    field = np.stack((np.sin(field_xyz[:, 1] / 24) * np.cos(field_xyz[:, 2] / 30),
                      np.sin(field_xyz[:, 2] / 20) * np.cos(field_xyz[:, 0] / 30),
                      np.sin(field_xyz[:, 0] / 22) * np.cos(field_xyz[:, 1] / 18)), axis=1)
    amplitudes = [0, .01, .03, .06, .12, .25, .50, .75, 1.5, .75, .50, .25, .12, .06, .03, .01]
    indices = {mode: BlockIndex(initial_index.shape, initial_index.block_size, initial_index.candidates)
               for mode in ("full", "hint")}

    def evaluate(mode, positions, force_full):
        vertices = torch.tensor(positions, dtype=torch.float32, device=device, requires_grad=True)
        stats = {"include_assignments": True}
        sync()
        started = monotonic()
        geometry = prepare_current_geometry(vertices, tetra)
        data_cost = compact_data_cost(vertices, tetra, alphas, shape, valid_mask=mesh_valid,
            block_index=indices[mode], background_channel=background,
            current_geometry=geometry, likelihood=likelihood,
            cache_owner_hints=mode == "hint", owner_hints=mode == "hint" and not force_full,
            hint_tolerance=a.hint_tolerance, hint_stats=stats,
            double_accumulation=a.double_data_cost)
        if data_cost is None:
            raise RuntimeError("FP32 CUDA/Triton data path is unavailable")
        data_cost = data_cost * scale
        prior_cost, jac = ashburner_prior(vertices, reference, tetra, atlas.stiffness,
            reference_geometry=ref_geometry, current_geometry=geometry, analytic_gradient=True)
        objective = data_cost + prior_cost
        objective.backward()
        sync()
        elapsed = monotonic() - started
        snapshot = dict(gradient=vertices.grad.detach().cpu(), owners=stats["selected_ids"].cpu(),
                        coverage=stats["covered"].cpu(), data_cost=float(data_cost),
                        prior_cost=float(prior_cost), objective=float(objective),
                        min_jacobian=float(jac.min()), reused_points=int(stats["reused_points"]),
                        evaluated_points=stats["evaluated_points"])
        return snapshot, elapsed

    comparisons, measurements, index_times = [], [], []
    for rep in range(a.repeats + 1):
        for step, amplitude in enumerate(amplitudes):
            positions = atlas.vertices + amplitude * field
            reset = step in (0, 8)
            if reset:
                started = monotonic()
                common = build_block_index(positions, atlas.tetrahedra, shape, margin=3)
                indices = {mode: BlockIndex(common.shape, common.block_size, common.candidates) for mode in indices}
                index_times.append(dict(repetition=rep, step=step, seconds=monotonic() - started))
            force_full = step % a.refresh_interval == 0
            order = ["full", "hint"] if (rep + step) % 2 == 0 else ["hint", "full"]
            snapshots = {}
            times = {}
            for mode in order:
                snapshots[mode], times[mode] = evaluate(mode, positions, force_full)
            r, f = snapshots["full"], snapshots["hint"]
            difference = r["gradient"].double() - f["gradient"].double()
            relative = float(torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(r["gradient"].double()).clamp_min(1e-30))
            comparison = dict(repetition=rep, step=step, amplitude_voxels=amplitude,
                index_reset=reset, force_full=force_full, order=order,
                gradient_relative_l2=relative, gradient_max_absolute=float(difference.abs().max()),
                gradients_finite=bool(torch.isfinite(f["gradient"]).all()),
                owner_disagreement_points=int((r["owners"] != f["owners"]).sum()),
                coverage_disagreement_points=int((r["coverage"] != f["coverage"]).sum()),
                reused_points=f["reused_points"], evaluated_points=f["evaluated_points"],
                costs={key: {"full": r[key], "hint": f[key], "absolute_difference": abs(r[key] - f[key]),
                             "relative_difference": abs(r[key] - f[key]) / max(abs(r[key]), 1e-30)}
                       for key in ("data_cost", "prior_cost", "objective", "min_jacobian")})
            comparisons.append(comparison)
            measurements.append(dict(repetition=rep, step=step, **times))
            if step in (0, 8, 15):
                print(json.dumps(dict(repetition=rep, step=step, amplitude=amplitude, timings=times,
                    hint_fraction=f["reused_points"] / max(f["evaluated_points"], 1),
                    gradient_relative_l2=relative, owner_disagreement=comparison["owner_disagreement_points"])), flush=True)
    warm = [row for row in measurements if row["repetition"] > 0 and row["step"] not in (0, 8)]
    summary = {mode: statistics.median(row[mode] for row in warm) for mode in indices}
    passed = all(row["gradients_finite"] and row["gradient_relative_l2"] < 2e-4
                 and row["costs"]["objective"]["relative_difference"] < 2e-6
                 and row["owner_disagreement_points"] == 0 and row["coverage_disagreement_points"] == 0
                 and row["costs"]["min_jacobian"]["hint"] > 0 for row in comparisons)
    sources = [Path(__file__), Path(inspect.getsourcefile(identity)), Path(inspect.getsourcefile(raster_module)),
               Path(inspect.getsourcefile(prepare_current_geometry)), Path(inspect.getsourcefile(raster_module.lookup_candidates))]
    report = dict(scope="saved real T1 mesh component with deterministic smooth vertex moves; excludes EM/preparation/end-to-end",
        approximation="strict interior owner reuse assumes local mesh cells do not globally overlap; boundary points use full maxscore search; periodic full search resets hints",
        structure=a.structure, dtype="float32", tf32=a.tf32, device=str(device),
        memory_fraction=a.memory_fraction, sampling_stride=a.sampling_stride, sampling_scale=scale,
        double_data_cost_accumulation=a.double_data_cost, hint_tolerance=a.hint_tolerance,
        refresh_interval=a.refresh_interval, shape=shape, valid_voxels=int(valid.sum()),
        selected_voxels=likelihood.shape[1], vertices=len(atlas.vertices), tetrahedra=len(atlas.tetrahedra),
        classes=grouped.shape[1], warm_median_seconds=summary, measurements=measurements,
        comparisons=comparisons, index_rebuild_timings=index_times,
        numerical_gate=dict(enforced=not a.tf32, passed=passed),
        source_files=[identity(path) for path in sources], inputs=[identity(path) for path in [*paths, a.lut]],
        memory=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
        torch=torch.__version__, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(output=str(a.output), warm_median_seconds=summary, numerical_gate=report["numerical_gate"])), flush=True)
    if not a.tf32 and not passed:
        raise AssertionError("moving real mesh owner hint failed cost/gradient/assignment/Jacobian numerical gate")


if __name__ == "__main__":
    main()
