"""Compare full and valid-block mesh costs on saved real thalamus stage inputs.

This checks one forward/backward component, not a whole-subject benchmark.
"""
import argparse
import json
import math
from pathlib import Path
from time import monotonic

import nibabel as nib
import numpy as np
import torch

from fnit.gems.atlas import GEMSAtlas
from fnit.gems.gaussian import gaussian_log_likelihood, initialise_gaussians
from fnit.gems.rasterize import BlockIndex, build_block_index, rasterize_priors
from fnit.gems.recipes import ThalamusRecipe


def masked_index(index, voxel_mask):
    """Experimental lookup used only by this validation script."""
    bs = index.block_size
    nb = tuple(int(math.ceil(s / bs)) for s in index.shape)
    candidates = []
    for block_id, ids in enumerate(index.candidates):
        bx, rem = divmod(block_id, nb[1] * nb[2])
        by, bz = divmod(rem, nb[2])
        crop = tuple(slice(p * bs, (p + 1) * bs) for p in (bx, by, bz))
        candidates.append(ids if voxel_mask[crop].any() else np.empty(0, np.int64))
    return BlockIndex(index.shape, bs, tuple(candidates))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-stage", required=True, type=Path)
    parser.add_argument("--lut", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(.14, args.device)
    stage = args.prepared_stage
    image = nib.load(stage / "processedImageMasked.mgz")
    aligned = nib.load(stage / "alignedAtlasImage.mgz")
    atlas = GEMSAtlas.from_freesurfer(stage / "warpedOriginalMesh.txt.gz", args.lut)
    atlas = atlas.transformed(np.linalg.inv(image.affine) @ aligned.affine,
                              transform_reference=True)
    classes = ThalamusRecipe("thalamus", args.lut.parent).intensity_groups(atlas, 0)
    alphas = np.zeros((len(atlas.vertices), int(classes.max()) + 1), np.float32)
    for channel, group in enumerate(classes):
        alphas[:, group] += atlas.alphas[:, channel]
    alphas = torch.as_tensor(alphas, device=args.device)
    tetrahedra = torch.as_tensor(atlas.tetrahedra, device=args.device)
    data = torch.as_tensor(np.asarray(image.dataobj, np.float32).squeeze(), device=args.device)
    valid = data > 0
    index = build_block_index(atlas.vertices, atlas.tetrahedra, tuple(data.shape), margin=3)
    masked = masked_index(index, valid.cpu().numpy())
    with torch.no_grad():
        priors, _ = rasterize_priors(torch.as_tensor(atlas.vertices, device=args.device, dtype=torch.float32),
                                     tetrahedra, alphas, tuple(data.shape), block_index=index)
        parameters = initialise_gaussians(data[valid].reshape(-1, 1, 1),
                         priors[:, valid].reshape(len(alphas[0]), -1, 1, 1),
                         torch.arange(len(alphas[0]), device=args.device))
        likelihood = gaussian_log_likelihood(data, parameters)[:, valid]
    values, gradients, costs, times = [], [], [], []
    for current in (index, masked) * 3:
        vertices = torch.tensor(atlas.vertices, device=args.device, dtype=torch.float32,
                                 requires_grad=True)
        measured = []
        for _ in range(2):
            vertices.grad = None
            if args.device.startswith("cuda"):
                torch.cuda.synchronize(args.device)
            started = monotonic()
            priors, _ = rasterize_priors(vertices, tetrahedra, alphas, tuple(data.shape),
                                         block_index=current)
            selected = priors[:, valid]
            cost = -(selected.clamp_min(torch.finfo(selected.dtype).tiny).log() + likelihood).logsumexp(0).sum()
            cost.backward()
            if args.device.startswith("cuda"):
                torch.cuda.synchronize(args.device)
            measured.append(monotonic() - started)
        values.append(selected.detach()); gradients.append(vertices.grad.detach())
        costs.append(float(cost.detach())); times.append(measured)
    report = {"mode": "real_thalamus_mesh_cost_component", "shape": list(data.shape),
              "valid_voxels": int(valid.sum()), "costs": costs, "forward_backward_seconds": times,
              "active_blocks": [sum(bool(len(ids)) for ids in item.candidates) for item in (index, masked)],
              "prior_max_abs_difference": float((values[0] - values[1]).abs().max()),
              "gradient_relative_l2_difference": float(
                  torch.linalg.vector_norm(gradients[0] - gradients[1]) / torch.linalg.vector_norm(gradients[0]))}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    assert report["prior_max_abs_difference"] < 1e-5
    assert report["gradient_relative_l2_difference"] < 1e-4


if __name__ == "__main__":
    main()
