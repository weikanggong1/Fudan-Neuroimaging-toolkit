"""真实 T1 保存网格 dense raster 的逐 block CUDA 初始化与 CPU packed 对照。

使用已保存真实影像确定网格、已有许可图谱及其 Gaussian 标签分组；核对
batch 布局位图、dense priors 和 coverage。这里仅计组件，不作为整例分割时间。
"""
import argparse
import inspect
import json
from pathlib import Path
import statistics
from time import monotonic

import nibabel as nib
import numpy as np
import torch

from fnit.gems.atlas import GEMSAtlas
from fnit.gems import rasterize as raster_module
from fnit.gems.rasterize import BlockIndex, build_block_index, rasterize_priors
from fnit.gems.recipes import HippoAmygdalaRecipe, ThalamusRecipe


def legacy_dense_batches(index, device, dtype):
    """Pre-packing FNIT dense setup, retained only as a regression oracle."""
    groups = {}
    for points, ids, _, _ in index.device_blocks(device, dtype):
        width = 1 << (len(ids) - 1).bit_length()
        groups.setdefault((len(points), width), []).append((points, ids))
    batches = []
    for (n_points, width), blocks in groups.items():
        batch_size = max(1, min(16, 8_388_608 // (n_points * width)))
        for start in range(0, len(blocks), batch_size):
            subset = blocks[start:start + batch_size]
            points = torch.stack([block[0] for block in subset])
            ids = torch.stack([torch.nn.functional.pad(block[1], (0, width - len(block[1]))) for block in subset])
            counts = torch.as_tensor([len(block[1]) for block in subset], device=device)
            candidate_mask = torch.arange(width, device=device)[None] < counts[:, None]
            batch_ids = torch.arange(len(subset), device=device)[:, None]
            coordinates = points.reshape(-1, 3).long().unbind(-1)
            batches.append((points, ids, candidate_mask, batch_ids, coordinates))
    result = tuple(batches)
    index._device_cache[("batches", str(device), dtype)] = result
    return result


def main():
    from benchmark_compact_layout import identity
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared-stage", type=Path, required=True)
    p.add_argument("--lut", type=Path, required=True)
    p.add_argument("--structure", choices=["thalamus", "hippo-amygdala-left", "hippo-amygdala-right"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--memory-fraction", type=float, default=.14)
    p.add_argument("--repeats", type=int, default=3)
    a = p.parse_args()
    if a.repeats < 3:
        p.error("repeats>=3 required")
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
    vertices = torch.as_tensor(atlas.vertices, device=device, dtype=torch.float32)
    shape = tuple(int(value) for value in image.shape[:3])
    started = monotonic()
    common = build_block_index(atlas.vertices, atlas.tetrahedra, shape, margin=3)
    index_seconds = monotonic() - started

    def evaluate(mode):
        index = BlockIndex(common.shape, common.block_size, common.candidates)
        sync()
        started = monotonic()
        layout = (legacy_dense_batches(index, device, torch.float32) if mode == "legacy" else
                  index.device_batches(device, torch.float32))
        sync()
        layout_seconds = monotonic() - started
        started = monotonic()
        with torch.no_grad():
            priors, coverage = rasterize_priors(vertices, tetra, alphas, shape, block_index=index,
                                               background_channel=background)
        sync()
        raster_seconds = monotonic() - started
        layout_cpu = tuple(tuple(tuple(x.cpu() for x in tensor) if isinstance(tensor, tuple) else tensor.cpu()
                                 for tensor in batch) for batch in layout)
        return dict(layout=layout_cpu, priors=priors.cpu(), coverage=coverage.cpu()), dict(
            layout_seconds=layout_seconds, warm_dense_raster_seconds=raster_seconds,
            fresh_layout_plus_raster_seconds=layout_seconds + raster_seconds)

    measurements, comparisons = [], []
    for rep in range(a.repeats + 1):
        order = ["legacy", "packed"] if rep % 2 == 0 else ["packed", "legacy"]
        snapshots, times = {}, {}
        for mode in order:
            snapshots[mode], times[mode] = evaluate(mode)
        r, f = snapshots["legacy"], snapshots["packed"]
        equal = len(r["layout"]) == len(f["layout"])
        equal = equal and all(all(torch.equal(a, b) for a, b in zip(x, y)) if isinstance(x, tuple)
                              else torch.equal(x, y) for old, new in zip(r["layout"], f["layout"])
                              for x, y in zip(old, new))
        row = dict(repetition=rep, bitwise_layout_equal=equal,
            prior_max_abs_difference=float((r["priors"] - f["priors"]).abs().max()),
            coverage_disagreement_voxels=int((r["coverage"] != f["coverage"]).sum()))
        comparisons.append(row)
        measurements.append(dict(repetition=rep, order=order, **times))
        print(json.dumps(dict(repetition=rep, timings=times, comparison=row)), flush=True)
    medians = {mode: {key: statistics.median(row[mode][key] for row in measurements[1:])
                      for key in measurements[0][mode]} for mode in ("legacy", "packed")}
    passed = all(row["bitwise_layout_equal"] and row["prior_max_abs_difference"] == 0
                 and row["coverage_disagreement_voxels"] == 0 for row in comparisons)
    report = dict(scope="real saved T1 fresh dense index layout+raster component; excludes EM/preparation/full pipeline",
        structure=a.structure, shape=shape, classes=grouped.shape[1], dtype="float32", tf32=a.tf32,
        device=str(device), cpu_bounding_box_index_seconds=index_seconds,
        measurements=measurements, comparisons=comparisons, warm_median_seconds=medians,
        numerical_gate=dict(passed=passed), source_files=[identity(Path(__file__)), identity(inspect.getsourcefile(raster_module))],
        inputs=[identity(path) for path in [*paths, a.lut]],
        memory=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(device), peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
               if device.type == "cuda" else None)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(output=str(a.output), warm_median_seconds=medians, numerical_gate=report["numerical_gate"])), flush=True)
    if not passed:
        raise AssertionError("dense packed layout changed candidate/point bitmaps or priors/coverage")


if __name__ == "__main__":
    main()
