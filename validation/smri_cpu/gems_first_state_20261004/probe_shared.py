"""Bounded same-state GEMS autograd probe, with no mesh update."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from fnit.gems.deformation import (ashburner_prior, prepare_current_geometry,
    prepare_deformation_reference, prepare_vertex_reduction, sliding_boundary_projectors)
from fnit.gems.gaussian import GaussianParameters, gaussian_log_likelihood
from fnit.gems import rasterize as raster_module
from fnit.gems.rasterize import build_block_index, rasterize_priors_compact


def difference(left, right):
    delta = left.astype(np.float64) - right.astype(np.float64)
    return {"different": int(np.count_nonzero(delta)),
            "max_abs": float(np.abs(delta).max()),
            "rms": float(np.sqrt(np.mean(delta ** 2))),
            "relative_l2": float(np.linalg.norm(delta) / np.linalg.norm(right)),
            "p99_abs": float(np.quantile(np.abs(delta), .99))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--modes", nargs="+", default=["baseline", "epsilon", "epsilon_native_mixture", "epsilon_native_mixture_fp64_geometry", "epsilon_cpu64_internal", "epsilon_cpu64_interp_fp32_owner"])
    args = parser.parse_args()
    output = args.output or args.run / "shared_probe"
    output.mkdir(exist_ok=False)
    torch.set_num_threads(8)
    import numba
    numba.set_num_threads(8)
    data = np.load(args.run / "fnit/shared_input.npz")
    image = torch.from_numpy(data["image"])
    valid = image != 0
    reference = torch.from_numpy(data["reference"])
    tetra = torch.from_numpy(data["tetrahedra"].astype(np.int64))
    alphas = torch.from_numpy(data["alphas"])
    flags = torch.from_numpy(data["can_move"])
    projection = sliding_boundary_projectors(flags, torch.as_tensor(data["boundary_transform"], dtype=torch.float32))
    parameters = GaussianParameters(torch.from_numpy(data["means"].astype(np.float32)),
                                   torch.from_numpy(data["variances"].astype(np.float32)))
    likelihood = gaussian_log_likelihood(image[valid].reshape(-1, 1, 1), parameters).reshape(len(alphas[0]), -1)
    # Official filter evaluates Gaussian arithmetic in double, stores each
    # mixture in the FP32 image, then accumulates the weighted sum in double.
    x = image[valid].double()
    means = torch.from_numpy(data["means"][:, 0])
    variances = torch.from_numpy(data["variances"][:, 0, 0])
    native_mixture = (torch.exp(-.5 * (x[None] - means[:, None]).square() / variances[:, None])
                      / torch.sqrt(2 * torch.pi * variances[:, None])).float()
    reduction = prepare_vertex_reduction(tetra.reshape(-1), len(reference))
    saved_gradient = np.load(args.run / "fnit/fnit_gradient.npy")
    native = args.native or args.run / "native"
    official_gradient = np.load(native / "native_total_gradient.npy")
    official_cost = json.loads((native / "report.public.json").read_text())["cost"]["total"]
    report = {"scope": "same saved real state; independent objectives and their complete autograd gradients; no mesh update",
              "affinity": sorted(os.sched_getaffinity(0)), "threads": torch.get_num_threads(),
              "official_cost": official_cost, "variants": {}}
    for mode in args.modes:
        precision = torch.float64 if mode.endswith("fp64_geometry") else torch.float32
        vertices = torch.from_numpy(data["vertices"].copy()).to(precision).requires_grad_(True)
        geometry_vertices = vertices.double() if mode.startswith("epsilon_cpu64") else vertices
        geometry_precision = geometry_vertices.dtype
        reference_geometry = prepare_deformation_reference(reference.to(geometry_precision), tetra)
        index = build_block_index(vertices.detach().numpy(), data["tetrahedra"], tuple(image.shape), 8, margin=3.)
        started = perf_counter()
        geometry = prepare_current_geometry(geometry_vertices, tetra, deterministic_gradient=True, vertex_reduction=reduction)
        original_lookup = raster_module._compact_lookup
        if mode.endswith("fp32_owner"):
            # Retain the real production owner scan and its discrete result.
            # Only interpolation and the prior use double in this diagnostic.
            owner_geometry = prepare_current_geometry(vertices.detach(), tetra)
            _, _, selected, points, coverage, reorder = original_lookup(
                vertices.detach(), tetra, valid, index, owner_geometry, 2e-5)
            def common_owner(*positional, **keywords):
                return geometry.origins, geometry.inverse_edges, selected, points.double(), coverage, reorder
            raster_module._compact_lookup = common_owner
        try:
            priors, covered = rasterize_priors_compact(geometry_vertices, tetra, alphas.to(geometry_precision), tuple(image.shape),
                valid_mask=valid, block_index=index, background_channel=int(json.loads((args.run/"fnit/report.public.json").read_text())["background_class"]),
                current_geometry=geometry)
        finally:
            raster_module._compact_lookup = original_lookup
        if mode.startswith("epsilon_native_mixture"):
            # This is an arithmetic diagnostic, not the proposed epsilon-only fix.
            point_cost = -(priors.double() * native_mixture.double()).sum(0).add(1e-15).log()
        else:
            joint = priors.clamp_min(torch.finfo(priors.dtype).tiny).log() + likelihood
            log_density = joint.logsumexp(0)
            if mode.startswith("epsilon"):
                log_density = torch.logaddexp(log_density, log_density.new_tensor(np.log(1e-15)))
            point_cost = -log_density
        data_cost = point_cost.sum(dtype=torch.float64)
        prior_cost, jacobian = ashburner_prior(geometry_vertices, reference.to(geometry_precision), tetra, float(data["stiffness"]),
            reference_geometry=reference_geometry, current_geometry=geometry,
            analytic_gradient=True, double_accumulation=True)
        total = data_cost + prior_cost
        total.backward()
        vertices.grad.copy_(torch.bmm(projection.to(precision), vertices.grad[..., None]).squeeze(-1))
        gradient = vertices.grad.detach().numpy()
        elapsed = perf_counter()-started
        np.save(output / (mode+"_gradient.npy"), gradient)
        report["variants"][mode] = {"data_cost": float(data_cost), "prior_cost": float(prior_cost),
            "total_cost": float(total), "cost_minus_official": float(total)-official_cost,
            "coverage": int(covered.sum()), "min_jacobian": float(jacobian.min()),
            "single_evaluation_observation_seconds": elapsed,
            "vertex_storage_dtype": str(vertices.dtype), "gradient_storage_dtype": str(vertices.grad.dtype),
            "geometry_dtype": str(geometry_precision),
            "gradient_vs_saved_fnit": difference(gradient, saved_gradient),
            "gradient_vs_official": difference(gradient, official_gradient)}
    report["outputs"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}
    (output / "report.public.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report["variants"], indent=2))


if __name__ == "__main__":
    main()
