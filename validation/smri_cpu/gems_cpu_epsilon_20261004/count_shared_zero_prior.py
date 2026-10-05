"""Count zero-prior boundaries in one saved real capture; no fit or update."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import torch

from fnit.gems.deformation import prepare_current_geometry
from fnit.gems.rasterize import _compact_lookup, build_block_index


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('use a fresh result path')
    torch.set_num_threads(8)
    import numba
    numba.set_num_threads(8)
    with np.load(args.capture / 'shared_input.npz') as arrays:
        image = torch.from_numpy(arrays['image'])
        vertices = torch.from_numpy(arrays['vertices'])
        tetra = torch.from_numpy(arrays['tetrahedra'].astype(np.int64))
        alphas = torch.from_numpy(arrays['alphas'])
    saved = torch.from_numpy(np.load(args.capture / 'fnit_priors.npy'))
    if vertices.dtype != torch.float32 or saved.dtype != torch.float64:
        raise RuntimeError('expected the bound CPU mixed-precision candidate capture')
    capture_report = json.loads((args.capture / 'report.public.json').read_text())
    background = capture_report['background_class']
    valid = image != 0
    shape = tuple(image.shape)
    index = build_block_index(vertices.numpy(), tetra.numpy(), shape, 8, margin=3.)
    owner = prepare_current_geometry(vertices, tetra)
    geometry = prepare_current_geometry(vertices.double(), tetra, deterministic_gradient=True)
    origins, inverses, selected, points, covered, reorder = _compact_lookup(
        vertices, tetra, valid, index, geometry, owner_geometry=owner)
    weights123 = torch.einsum('pij,pj->pi', inverses[selected], points - origins[selected])
    weights = torch.cat((1 - weights123.sum(-1, keepdim=True), weights123), dim=-1)
    selected_alpha = alphas[tetra[selected]]
    raw = (selected_alpha * weights[..., None]).sum(1)
    values = raw.clamp_min(0)
    values = values / values.sum(-1, keepdim=True).clamp_min(torch.finfo(values.dtype).eps)
    values = torch.where(covered[:, None], values, 0)
    priors = torch.cat((values, torch.zeros((1, values.shape[1]), dtype=values.dtype)))[reorder]
    restored_coverage = torch.cat((covered, torch.zeros(1, dtype=torch.bool)))[reorder]
    priors[:, background] = torch.where(~restored_coverage,
        torch.ones_like(priors[:, background]), priors[:, background])
    exact = bool(torch.equal(priors.T.contiguous(), saved))
    coverage_exact = bool(torch.equal(restored_coverage,
        torch.from_numpy(np.load(args.capture / 'fnit_coverage.npy'))))
    if not exact or not coverage_exact:
        raise RuntimeError('reconstructed priors differ from actual saved candidate capture')
    nonuniform = selected_alpha.amax(1) != selected_alpha.amin(1)
    zero = (values == 0) & covered[:, None]
    categories = {
        'zero_with_all_vertex_alphas_zero': zero & (selected_alpha.amax(1) == 0),
        'zero_after_negative_raw_prior_clipping': zero & (raw < 0),
        'exact_raw_zero_with_nonuniform_vertex_alphas': zero & (raw == 0) & nonuniform,
    }
    output = {'scope': 'one saved real first state; only owner/interpolation reconstruction and counts; no objective gradient or optimizer update',
              'structure': capture_report['structure'], 'priors_exact_to_saved_capture': exact,
              'coverage_exact_to_saved_capture': coverage_exact,
              'covered_voxels': int(restored_coverage.sum()),
              'saved_zero_class_voxel_pairs': int((saved == 0).sum()),
              'covered_zero_class_voxel_pairs': int(zero.sum()),
              'class_counts': {name: values.sum(0).tolist() for name, values in categories.items()},
              'total_counts': {name: int(values.sum()) for name, values in categories.items()},
              'threads': torch.get_num_threads(),
              'input_sha256': {name: digest(args.capture / name) for name in (
                  'shared_input.npz', 'fnit_priors.npy', 'fnit_coverage.npy', 'report.public.json')},
              'program_sha256': digest(Path(__file__)),
              'actual_module_sha256': {name: digest(Path(importlib.import_module(name).__file__))
                                      for name in ('fnit.gems.rasterize', 'fnit.gems.deformation',
                                                   'fnit.gems._raster_cpu_compact')},
              'interpretation': 'The exact-raw-zero/nonuniform category can expose the retained log-prior clamp derivative. All-zero alpha fields have zero spatial derivative. A negative raw prior is already clipped by rasterization; it is a separate boundary condition. These counts cover this captured state only.'}
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: output[key] for key in ('structure', 'priors_exact_to_saved_capture', 'total_counts')}))


if __name__ == '__main__':
    main()
