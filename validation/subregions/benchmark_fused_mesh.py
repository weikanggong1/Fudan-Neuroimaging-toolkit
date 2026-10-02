"""Replay a saved real T1 tetrahedral mesh; compare full-voxel mixture gradients.

Uses only supplied licensed assets, nibabel and FNIT/PyTorch. No atlas/image
download and no external neuroimaging runtime. Timings are a mesh component,
not the full T1 pipeline. Run with --no-tf32 for the numerical gate.
"""
import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics
from time import monotonic

import nibabel as nib
import numpy as np
import torch

from fnit.gems.atlas import GEMSAtlas
from fnit.gems import rasterize as raster_module
from fnit.gems.deformation import ashburner_prior, prepare_current_geometry, prepare_deformation_reference
from fnit.gems.gaussian import gaussian_log_likelihood, initialise_gaussians
from fnit.gems.rasterize import BlockIndex, build_block_index, compact_data_cost, rasterize_priors_compact
from fnit.gems.recipes import HippoAmygdalaRecipe, ThalamusRecipe


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
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--memory-fraction", type=float, default=.23)
    p.add_argument("--sampling-stride", type=int, default=1)
    a = p.parse_args()
    if a.repeats < 3 or a.sampling_stride < 1:
        p.error("at least 3 repeats and positive stride are required")
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
    mesh_valid = valid
    selection = None
    scale = 1.0
    if a.sampling_stride > 1:
        x = torch.arange(shape[0], device=device)[:, None, None]
        y = torch.arange(shape[1], device=device)[None, :, None]
        z = torch.arange(shape[2], device=device)[None, None, :]
        quadrature = (x + 3 * y + 5 * z) % a.sampling_stride == 0
        selection = quadrature[valid]
        mesh_valid = valid & quadrature
        scale = selection.numel() / int(selection.sum())
        likelihood = likelihood[:, selection].contiguous()
    indices = {mode: BlockIndex(initial_index.shape, initial_index.block_size, initial_index.candidates)
               for mode in ("autograd", "fused")}
    ref_geometry = prepare_deformation_reference(reference, tetra)

    def evaluate(mode):
        vertices = initial.detach().clone().requires_grad_(True)
        sync()
        started = monotonic()
        geometry = prepare_current_geometry(vertices, tetra)
        if mode == "fused":
            data_cost = compact_data_cost(vertices, tetra, alphas, shape, valid_mask=mesh_valid,
                block_index=indices[mode], background_channel=background,
                current_geometry=geometry, likelihood=likelihood)
            if data_cost is None:
                raise RuntimeError("FP32 CUDA/Triton data path is unavailable")
        else:
            priors, _ = rasterize_priors_compact(vertices, tetra, alphas, shape,
                valid_mask=mesh_valid, block_index=indices[mode], background_channel=background,
                current_geometry=geometry)
            data_cost = -(priors.clamp_min(torch.finfo(priors.dtype).tiny).log() + likelihood).logsumexp(0).sum()
        data_cost = data_cost * scale
        prior_cost, jac = ashburner_prior(vertices, reference, tetra, atlas.stiffness,
            reference_geometry=ref_geometry, current_geometry=geometry, analytic_gradient=True)
        objective = data_cost + prior_cost
        objective.backward()
        sync()
        elapsed = monotonic() - started
        snapshot = dict(gradient=vertices.grad.detach().cpu(), data_cost=float(data_cost),
                        prior_cost=float(prior_cost), objective=float(objective),
                        min_jacobian=float(jac.min()))
        return snapshot, elapsed

    runs = {mode: {"first_seconds": None, "warm_seconds": []} for mode in indices}
    comparisons = []
    for rep in range(a.repeats + 1):
        order = ["autograd", "fused"] if rep % 2 == 0 else ["fused", "autograd"]
        snapshots = {}
        for mode in order:
            snapshots[mode], elapsed = evaluate(mode)
            if rep == 0:
                runs[mode]["first_seconds"] = elapsed
            else:
                runs[mode]["warm_seconds"].append(elapsed)
        r, f = snapshots["autograd"], snapshots["fused"]
        difference = r["gradient"].double() - f["gradient"].double()
        comparison = dict(repetition=rep, order=order,
            gradient_relative_l2=float(torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(r["gradient"].double()).clamp_min(1e-30)),
            gradient_max_absolute=float(difference.abs().max()),
            gradients_finite=bool(torch.isfinite(f["gradient"]).all()),
            costs={key: {"autograd": r[key], "fused": f[key], "absolute_difference": abs(r[key] - f[key]),
                         "relative_difference": abs(r[key] - f[key]) / max(abs(r[key]), 1e-30)}
                   for key in ("data_cost", "prior_cost", "objective", "min_jacobian")})
        comparisons.append(comparison)
    for mode in runs:
        runs[mode]["warm_median_seconds"] = statistics.median(runs[mode]["warm_seconds"])
    memory = {}
    for mode in indices:
        for index in indices.values():
            index._device_cache.clear()
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            sync()
            resident = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            snapshot, elapsed = evaluate(mode)
            memory[mode] = dict(resident_bytes=resident, peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                                peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    passed = all(row["gradients_finite"] and row["gradient_relative_l2"] < 2e-4
                 and row["costs"]["objective"]["relative_difference"] < 2e-6 for row in comparisons)
    sources = [Path(__file__), Path(inspect.getsourcefile(raster_module)),
               Path(inspect.getsourcefile(prepare_current_geometry)),
               Path(inspect.getsourcefile(raster_module.lookup_candidates))]
    report = dict(scope="saved real T1 full-valid-voxel mesh component, excludes EM/preparation/end-to-end",
        structure=a.structure, dtype="float32", tf32=a.tf32, device=str(device),
        memory_fraction=a.memory_fraction, sampling_stride=a.sampling_stride, sampling_scale=scale,
        shape=shape, valid_voxels=int(valid.sum()), selected_voxels=likelihood.shape[1],
        vertices=len(atlas.vertices), tetrahedra=len(atlas.tetrahedra), classes=grouped.shape[1],
        runs=runs, memory=memory, comparisons=comparisons, numerical_gate=dict(enforced=not a.tf32, passed=passed),
        source_files=[identity(path) for path in sources], inputs=[identity(path) for path in [*paths, a.lut]],
        torch=torch.__version__, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(output=str(a.output), runs=runs, numerical_gate=report["numerical_gate"])))
    if not a.tf32 and not passed:
        raise AssertionError("real mesh fused data cost or gradient failed FP32 numerical gate")


if __name__ == "__main__":
    main()
