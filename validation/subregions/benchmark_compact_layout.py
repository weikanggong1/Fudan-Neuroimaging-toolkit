"""真实保存 T1 网格的 fresh-index compact layout 冷启动对照。

对照旧逐 batch GPU 传输与新 CPU packed 传输。输入是已有真实阶段影像及
tetrahedral atlas，不调用原软件。逐 tensor 位图、先验、cost/gradient 均验证；
这里是重新建 index 后的组件耗时，不作为完整个体分割时间。
"""
import argparse
import gc
import hashlib
import inspect
import json
import math
from pathlib import Path
import statistics
from time import monotonic

import nibabel as nib
import numpy as np
import torch

from fnit.gems.atlas import GEMSAtlas
from fnit.gems import rasterize as raster_module
from fnit.gems.deformation import prepare_current_geometry
from fnit.gems.gaussian import gaussian_log_likelihood, initialise_gaussians
from fnit.gems.rasterize import BlockIndex, build_block_index, compact_data_cost, rasterize_priors_compact
from fnit.gems.recipes import HippoAmygdalaRecipe, ThalamusRecipe


def legacy_compact_batches(index, valid_mask, device, dtype):
    """Frozen pre-packing FNIT layout, used only as a regression oracle."""
    coordinates = np.argwhere(valid_mask.detach().cpu().numpy())
    nblocks = tuple(int(math.ceil(s / index.block_size)) for s in index.shape)
    block_xyz = coordinates // index.block_size
    block_ids = (block_xyz[:, 0] * nblocks[1] + block_xyz[:, 1]) * nblocks[2] + block_xyz[:, 2]
    order = np.argsort(block_ids, kind="stable")
    sorted_ids = block_ids[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_ids)) + 1, len(order)]
    groups = {}
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if start == stop:
            continue
        ids = index.candidates[int(sorted_ids[start])]
        if not len(ids):
            continue
        rows = order[start:stop]
        width = 1 << (len(ids) - 1).bit_length()
        n_points = 1 << (len(rows) - 1).bit_length()
        groups.setdefault((n_points, width), []).append((coordinates[rows], ids, rows))
    batches = []
    reorder = np.full(len(coordinates), -1, dtype=np.int64)
    offset = 0
    for (n_points, width), blocks in groups.items():
        batch_size = max(1, min(16, 8_388_608 // (n_points * width)))
        for start in range(0, len(blocks), batch_size):
            subset = blocks[start:start + batch_size]
            points = torch.as_tensor(np.stack([np.pad(block[0], ((0, n_points - len(block[0])), (0, 0)), mode="edge")
                                               for block in subset]), device=device, dtype=dtype)
            ids = torch.as_tensor(np.stack([np.pad(block[1], (0, width - len(block[1]))) for block in subset]),
                                  device=device, dtype=torch.long)
            counts = torch.as_tensor([len(block[1]) for block in subset], device=device)
            candidate_mask = torch.arange(width, device=device)[None] < counts[:, None]
            batch_ids = torch.arange(len(subset), device=device)[:, None]
            rows = np.concatenate([block[2] for block in subset])
            point_rows = torch.as_tensor(np.concatenate([np.arange(len(block[0])) + i * n_points
                                                        for i, block in enumerate(subset)]), device=device)
            reorder[rows] = np.arange(offset, offset + len(rows))
            offset += len(rows)
            batches.append((points, ids, candidate_mask, batch_ids, point_rows))
    return tuple(batches), torch.as_tensor(reorder, device=device)


def install_legacy_cache(index, mask, device, dtype):
    result = legacy_compact_batches(index, mask, device, dtype)
    version = None if torch.is_inference(mask) else mask._version
    index._device_cache[("compact", str(device), dtype, id(mask))] = (mask, version, result)
    return result


def identity(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=h.hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared-stage", type=Path, required=True)
    p.add_argument("--lut", type=Path, required=True)
    p.add_argument("--structure", choices=["thalamus", "hippo-amygdala-left", "hippo-amygdala-right"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--memory-fraction", type=float, default=.14)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--sampling-stride", type=int, default=1)
    a = p.parse_args()
    if a.repeats < 3 or a.sampling_stride < 1:
        p.error("repeats>=3 and positive stride are required")
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
    initial = torch.as_tensor(atlas.vertices, device=device, dtype=torch.float32)
    data = torch.as_tensor(np.asarray(image.dataobj, np.float32).squeeze(), device=device)
    valid = data > 0
    shape = tuple(data.shape)
    index_started = monotonic()
    common = build_block_index(atlas.vertices, atlas.tetrahedra, shape, margin=3)
    index_seconds = monotonic() - index_started
    with torch.no_grad():
        priors, _ = rasterize_priors_compact(initial, tetra, alphas, shape, valid_mask=valid,
            block_index=common, background_channel=background)
        params = initialise_gaussians(data[valid].reshape(-1, 1, 1),
            priors.reshape(grouped.shape[1], -1, 1, 1), torch.arange(grouped.shape[1], device=device))
        likelihood = gaussian_log_likelihood(data[valid].reshape(-1, 1, 1), params).reshape(grouped.shape[1], -1).detach()
    del priors
    common._device_cache.clear()
    mesh_valid = valid
    if a.sampling_stride > 1:
        x = torch.arange(shape[0], device=device)[:, None, None]
        y = torch.arange(shape[1], device=device)[None, :, None]
        z = torch.arange(shape[2], device=device)[None, None, :]
        quadrature = (x + 3 * y + 5 * z) % a.sampling_stride == 0
        selection = quadrature[valid]
        mesh_valid = valid & quadrature
        likelihood = likelihood[:, selection].contiguous()

    def evaluate(mode):
        index = BlockIndex(common.shape, common.block_size, common.candidates)
        sync()
        started = monotonic()
        layout = (install_legacy_cache(index, mesh_valid, device, torch.float32) if mode == "legacy" else
                  index.device_compact_batches(mesh_valid, device, torch.float32))
        sync()
        layout_seconds = monotonic() - started
        vertices = initial.detach().clone().requires_grad_(True)
        geometry = prepare_current_geometry(vertices, tetra)
        sync()
        started = monotonic()
        cost = compact_data_cost(vertices, tetra, alphas, shape, valid_mask=mesh_valid,
            block_index=index, background_channel=background, current_geometry=geometry, likelihood=likelihood)
        if cost is None:
            raise RuntimeError("FP32 CUDA fused data cost is unavailable")
        cost.backward()
        sync()
        data_seconds = monotonic() - started
        snapshot = dict(layout=(tuple(tuple(t.detach().cpu() for t in batch) for batch in layout[0]), layout[1].cpu()),
                        cost=float(cost), gradient=vertices.grad.cpu())
        measurement = dict(layout_seconds=layout_seconds, warm_cost_gradient_seconds=data_seconds,
                           fresh_layout_plus_cost_seconds=layout_seconds + data_seconds)
        return snapshot, measurement

    measurements, comparisons = [], []
    for rep in range(a.repeats + 1):
        order = ["legacy", "packed"] if rep % 2 == 0 else ["packed", "legacy"]
        snapshots, times = {}, {}
        for mode in order:
            snapshots[mode], times[mode] = evaluate(mode)
        r, f = snapshots["legacy"], snapshots["packed"]
        layout_equal = len(r["layout"][0]) == len(f["layout"][0]) and torch.equal(r["layout"][1], f["layout"][1])
        layout_equal = layout_equal and all(torch.equal(x, y) for old, new in zip(r["layout"][0], f["layout"][0]) for x, y in zip(old, new))
        difference = r["gradient"].double() - f["gradient"].double()
        comparison = dict(repetition=rep, bitwise_layout_equal=layout_equal,
            cost_abs_difference=abs(r["cost"] - f["cost"]),
            gradient_relative_l2=float(torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(r["gradient"].double()).clamp_min(1e-30)),
            gradient_max_absolute=float(difference.abs().max()),
            gradients_finite=bool(torch.isfinite(f["gradient"]).all()))
        measurements.append(dict(repetition=rep, order=order, **times))
        comparisons.append(comparison)
        print(json.dumps(dict(repetition=rep, timings=times, comparison=comparison)), flush=True)
    warm_medians = {mode: {key: statistics.median(row[mode][key] for row in measurements[1:])
                           for key in measurements[0][mode]} for mode in ("legacy", "packed")}
    passed = all(row["bitwise_layout_equal"] and row["cost_abs_difference"] == 0
                 and row["gradient_relative_l2"] < 2e-4 and row["gradients_finite"] for row in comparisons)
    report = dict(scope="real saved T1 fresh index compact layout and unchanged full data cost, excludes preparation/EM/full pipeline",
        structure=a.structure, shape=shape, valid_voxels=int(valid.sum()), sampled_voxels=likelihood.shape[1],
        sampling_stride=a.sampling_stride, dtype="float32", tf32=a.tf32, device=str(device),
        cpu_bounding_box_index_seconds=index_seconds, measurements=measurements,
        comparisons=comparisons, warm_median_seconds=warm_medians,
        numerical_gate=dict(passed=passed), source_files=[identity(Path(__file__)), identity(inspect.getsourcefile(raster_module))],
        inputs=[identity(path) for path in [*paths, a.lut]],
        memory=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(device), peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
               if device.type == "cuda" else None, torch=torch.__version__)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(output=str(a.output), warm_median_seconds=warm_medians, numerical_gate=report["numerical_gate"])), flush=True)
    if not passed:
        raise AssertionError("packed layout changed candidate/voxel bitmaps, data cost or gradient")


if __name__ == "__main__":
    main()
