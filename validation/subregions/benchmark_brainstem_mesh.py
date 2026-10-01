"""Compare the saved and current brainstem loops on one real subject's mesh.

The reference module is supplied as a file and is not redistributed. Both
loops use the currently imported raster/deformation modules, so this measures
the brainstem call-path migration rather than an entire historical package.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import sys
from time import monotonic

import nibabel as nib
import numpy as np
from scipy import ndimage
import torch

from fnit.gems import brainstem
from fnit.gems.atlas import GEMSAtlas
from fnit.gems.deformation import (ashburner_prior, prepare_current_geometry,
                                   prepare_deformation_reference)
from fnit.gems.initialize import estimate_mask_affine
from fnit.gems.rasterize import (build_block_index, rasterize_priors,
                                 rasterize_priors_compact)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate_objective(atlas, coarse, device, *, compact, vertices_override=None):
    """Evaluate the original BCE mask and prior at exactly the same geometry."""
    atlas = replace(atlas, vertices=atlas.reference_vertices.copy(), stiffness=.05)
    low = np.maximum(np.floor(atlas.vertices.min(0)).astype(int) - 8, 0)
    high = np.minimum(np.ceil(atlas.vertices.max(0)).astype(int) + 9, coarse.shape)
    crop = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
    shift = np.eye(4)
    shift[:3, 3] = -low
    local = atlas.transformed(shift, transform_reference=True)
    shape = coarse[crop].shape
    vertices_array = local.vertices if vertices_override is None else vertices_override - low
    vertices = torch.tensor(vertices_array, device=device, dtype=torch.float32,
                            requires_grad=True)
    reference = torch.tensor(local.reference_vertices, device=device, dtype=torch.float32)
    tetrahedra = torch.as_tensor(local.tetrahedra, device=device)
    group = np.isin(local.label_ids, (175, 174, 178, 173, 28, 7, 8, 15))
    alphas = torch.as_tensor(np.stack((local.alphas[:, group].sum(1),
                                     local.alphas[:, ~group].sum(1)), 1), device=device)
    index = build_block_index(local.vertices, local.tetrahedra, shape,
                              block_size=12, margin=10)
    with torch.no_grad():
        mask_vertices = torch.as_tensor(local.vertices, device=device, dtype=torch.float32)
        _, covered = rasterize_priors(mask_vertices, tetrahedra, alphas, shape,
                                      block_index=index, background_channel=1)
    xyz = np.stack(np.meshgrid(*[np.arange(-5, 6)] * 3, indexing="ij"), -1)
    interior = ndimage.binary_erosion(covered.cpu().numpy(),
                                     structure=np.square(xyz).sum(-1) <= 25,
                                     border_value=1)
    mask = torch.as_tensor(interior, device=device)
    target = torch.as_tensor(np.isin(coarse[crop], (16, 7, 8, 15, 28, 46, 47, 60)),
                             device=device)[mask]
    reference_geometry = prepare_deformation_reference(reference, tetrahedra)
    synchronize(device)
    started = monotonic()
    geometry = prepare_current_geometry(vertices, tetrahedra) if compact else None
    if compact:
        priors, _ = rasterize_priors_compact(
            vertices, tetrahedra, alphas, shape, valid_mask=mask, block_index=index,
            background_channel=1, current_geometry=geometry)
        probability = priors[0]
    else:
        priors, _ = rasterize_priors(vertices, tetrahedra, alphas, shape,
                                     block_index=index, background_channel=1)
        probability = priors[0][mask]
    probability = probability.clamp(1e-5, 1 - 1e-5)
    data_cost = -torch.where(target, probability.log(), torch.log1p(-probability)).sum()
    regularizer, jacobian = ashburner_prior(
        vertices, reference, tetrahedra, .05,
        reference_geometry=reference_geometry if compact else None,
        current_geometry=geometry, analytic_gradient=compact)
    objective = data_cost + regularizer
    objective.backward()
    vertices.grad.mul_(torch.as_tensor(local.can_move, device=device))
    synchronize(device)
    row = {"seconds": monotonic() - started, "objective": float(objective.detach()),
           "data_cost": float(data_cost.detach()), "prior_cost": float(regularizer.detach()),
           "min_jacobian": float(jacobian.min().detach()),
           "valid_voxels": int(mask.sum()), "shape": list(shape)}
    return row, vertices.grad.detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t1", type=Path, required=True)
    parser.add_argument("--coarse", type=Path, required=True)
    parser.add_argument("--atlas-directory", type=Path, required=True)
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--optimizer", choices=("adam", "lbfgs"))
    parser.add_argument("--memory-limit-gib", type=float, default=18.626)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            min(1., args.memory_limit_gib * 2**30 / torch.cuda.get_device_properties(device).total_memory),
            device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    reference_spec = importlib.util.spec_from_file_location(
        "fnit.gems._brainstem_reference", args.reference_source)
    reference_module = importlib.util.module_from_spec(reference_spec)
    sys.modules[reference_spec.name] = reference_module
    reference_spec.loader.exec_module(reference_module)
    image = nib.load(args.t1)
    coarse_image = nib.load(args.coarse)
    if coarse_image.shape != image.shape or not np.allclose(coarse_image.affine, image.affine):
        raise ValueError("T1 and coarse must have identical geometry")
    coarse = np.asarray(coarse_image.dataobj, dtype=np.int32)
    directory = args.atlas_directory
    config = json.loads((directory / "config.json").read_text())
    atlas = GEMSAtlas.from_freesurfer(directory / "AtlasMesh.gz",
                                     directory / "compressionLookupTable.txt")
    matrix, alignment_dice = estimate_mask_affine(
        nib.load(directory / "AtlasDump.mgz"), image, coarse,
        config.get("alignment_target_label_ids", (16,)), device=device)
    atlas = atlas.transformed(matrix, transform_reference=True)
    fit_alphas = (np.load(directory / config["segmentation_alpha_file"])
                  if config.get("segmentation_alpha_file") else None)
    if fit_alphas is not None:
        raise ValueError("objective differential currently requires the original grouped alphas")
    optimizer = args.optimizer or config.get("segmentation_fit_optimizer", "adam")
    dense, dense_gradient = evaluate_objective(atlas, coarse, device, compact=False)
    compact, compact_gradient = evaluate_objective(atlas, coarse, device, compact=True)
    gradient_difference = compact_gradient - dense_gradient
    report = {"kind": "real_brainstem_coarse_mesh_comparison", "host": socket.gethostname(),
              "scope": "40-step coarse mesh component; excludes raw T1 preprocessing, intensity fit and native outputs",
              "dependency_scope": "saved brainstem function with current shared raster/deformation dependencies",
              "device": str(device), "threads": args.threads, "iterations": args.iterations,
              "optimizer": optimizer, "dtype": "float32", "tf32": device.type == "cuda",
              "alignment_dice": alignment_dice, "alignment_matrix": matrix.tolist(),
              "input_sha256": {str(path): sha256(path) for path in
                                (args.t1, args.coarse, directory / "AtlasMesh.gz",
                                 directory / "compressionLookupTable.txt",
                                 directory / "AtlasDump.mgz", directory / "config.json")},
              "source_sha256": {str(path): sha256(path) for path in
                                 (args.reference_source, Path(brainstem.__file__),
                                  Path(sys.modules[ashburner_prior.__module__].__file__),
                                  Path(sys.modules[rasterize_priors.__module__].__file__),
                                  Path(sys.modules[brainstem.CachedLBFGS.__module__].__file__))},
              "same_geometry": {"dense": dense, "compact": compact,
                  "gradient_max_abs_difference": float(np.abs(gradient_difference).max()),
                  "gradient_relative_l2_difference": float(np.linalg.norm(gradient_difference)
                        / max(np.linalg.norm(dense_gradient), 1e-15))}, "full_fits": {}}
    fitted = {}
    for name, function in (("reference", reference_module.fit_brainstem_segmentation),
                           ("optimized", brainstem.fit_brainstem_segmentation)):
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        synchronize(device)
        started = monotonic()
        fitted[name], row = function(atlas, coarse, device=device, iterations=args.iterations,
                                     fit_alphas=fit_alphas, optimizer_name=optimizer)
        synchronize(device)
        row["wall_seconds"] = monotonic() - started
        row["peak_gpu_gib"] = (torch.cuda.max_memory_allocated(device) / 2**30
                               if device.type == "cuda" else None)
        terminal, _ = evaluate_objective(atlas, coarse, device, compact=False,
                                         vertices_override=fitted[name].vertices)
        row["terminal_dense_objective"] = terminal
        report["full_fits"][name] = row
        print(json.dumps({"fit": name, **row}), flush=True)
    delta = fitted["optimized"].vertices - fitted["reference"].vertices
    report["terminal_vertex_difference"] = {"max_abs": float(np.abs(delta).max()),
        "mean_l2": float(np.linalg.norm(delta, axis=1).mean()),
        "max_l2": float(np.linalg.norm(delta, axis=1).max())}
    report["wall_ratio_reference_over_optimized"] = (
        report["full_fits"]["reference"]["wall_seconds"] /
        report["full_fits"]["optimized"]["wall_seconds"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(args.output.with_suffix(".vertices.npz"),
                        reference=fitted["reference"].vertices,
                        optimized=fitted["optimized"].vertices)
    print(json.dumps({"output": str(args.output),
                      "wall_ratio": report["wall_ratio_reference_over_optimized"]}), flush=True)


if __name__ == "__main__":
    main()
