"""Check a real atlas against the earlier FNIT rasterizer, including gradients.

This is a local numerical check; whole-subject benchmarks use run_unified.py.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from time import monotonic

import torch

from fnit.gems.atlas import GEMSAtlas
from fnit.gems.rasterize import build_block_index, rasterize_priors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", required=True, type=Path)
    parser.add_argument("--reference-raster", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--offset", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("fnit_reference_raster", args.reference_raster)
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(.23, args.device)
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
    atlas = GEMSAtlas.from_freesurfer(args.atlas / "AtlasMesh.gz",
                                     args.atlas / "compressionLookupTable.txt")
    shape = (48, 48, 48)
    positions = atlas.reference_vertices + args.offset
    index = build_block_index(positions, atlas.tetrahedra, shape, 8, margin=1)
    tetrahedra = torch.as_tensor(atlas.tetrahedra, device=args.device)
    alphas = torch.as_tensor(atlas.alphas, device=args.device)
    values, gradients, assignments, times = [], [], [], []
    for raster in (reference.rasterize_priors, rasterize_priors):
        vertices = torch.tensor(positions, device=args.device,
                                dtype=torch.float32, requires_grad=True)
        measured = []
        for _ in range(2):
            vertices.grad = None
            if args.device.startswith("cuda"):
                torch.cuda.synchronize(args.device)
            started = monotonic()
            priors, _, cells, _ = raster(vertices, tetrahedra, alphas, shape,
                                         block_index=index, return_assignment=True)
            priors.square().sum().backward()
            if args.device.startswith("cuda"):
                torch.cuda.synchronize(args.device)
            measured.append(monotonic() - started)
        times.append(measured)
        values.append(priors.detach())
        gradients.append(vertices.grad.detach())
        assignments.append(cells.detach())
    report = {"mode": "local_atlas_numerical_check", "shape": shape, "device": args.device,
              "forward_backward_seconds": times,
              "tf32": args.tf32, "atlas_offset_voxels": args.offset,
              "changed_tetrahedron_assignments": int(
                  (assignments[0] != assignments[1]).any(-1).sum()),
              "prior_max_abs_difference": float((values[0] - values[1]).abs().max()),
              "gradient_relative_l2_difference": float(
                  torch.linalg.norm(gradients[0] - gradients[1]) / torch.linalg.norm(gradients[0]))}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    assert report["prior_max_abs_difference"] < 1e-5
    # Full FP32 is the gradient equivalence gate; TF32 changes reductions in
    # inverse backpropagation when many blocks share the same tetrahedron.
    if not args.tf32:
        assert report["gradient_relative_l2_difference"] < 1e-4


if __name__ == "__main__":
    main()
