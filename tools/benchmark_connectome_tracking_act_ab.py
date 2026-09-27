"""Real ds004666 fixed-FOD/5TT ACT tracking comparison against three MRtrix runs.

Original commands: ``tckgen -algorithm iFOD2 -seed_gmwmi gmwmi.mif -act
5tt.mif -seeds 10000 -select 0 -maxlength 250 -cutoff 0.1 -samples 3
-power 0.5 fod.mif tracks.tck``; then ``tcksift2`` and four
``tck2connectome -symmetric -assignment_radial_search 4`` commands.
This measures the same public inputs with a chosen PyTorch RNG seed, not
row-wise identity to the independent MRtrix stochastic tractograms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch

from fnit.connectome.assignment import build_connectomes
from fnit.connectome.sift2 import estimate_sift2_weights
from fnit.connectome.tcksample_precise import sample_streamline_mean_precise
from fnit.connectome.tracking import probabilistic_tractography, _five_tissue_values
import fnit.connectome.tracking as tracking_module


def _sha(path: Path) -> str:
    """Return SHA-256 of one image, reference matrix, or source input file."""
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _sync(device: torch.device) -> None:
    """Wait for CUDA stages on ``device`` before recording wall time."""
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _stats(candidate: dict[str, np.ndarray], official: dict[str, np.ndarray]) -> dict:
    """Compare four 20x20 matrices on all upper and common-support edges."""
    upper = np.triu_indices(20, 1)
    c = candidate['count'][upper]
    o = official['count'][upper]
    cs, os = c > 0, o > 0
    common = cs & os
    result = {'count_support_dice': float(2 * common.sum() / (cs.sum() + os.sum())),
              'candidate_edges': int(cs.sum()), 'official_edges': int(os.sum()),
              'common_edges': int(common.sum()),
              'candidate_assigned': int(np.diag(candidate['count']).sum() + c.sum()),
              'official_assigned': int(np.diag(official['count']).sum() + o.sum()),
              'candidate_self_connections': int(np.diag(candidate['count']).sum()),
              'official_self_connections': int(np.diag(official['count']).sum())}
    for name in ('count', 'sift2_fbc', 'mean_length', 'mean_fa'):
        x, y = candidate[name][upper], official[name][upper]
        r = np.corrcoef(x, y)[0, 1]
        result[name] = {'pearson_upper': float(r),
                        'nmae_upper': float(np.abs(x-y).sum() / np.abs(y).sum()),
                        'mae_upper': float(np.abs(x-y).mean())}
        if common.sum() > 1:
            result[name]['pearson_common'] = float(np.corrcoef(x[common], y[common])[0, 1])
            result[name]['mae_common'] = float(np.abs(x[common]-y[common]).mean())
    return result


def main() -> None:
    """Run fixed real inputs once; save time, memory, three reference comparisons, PNG."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fod', 'five-tissue', 'gmwmi', 'processing-mask', 'fa', 'atlas', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--official-dir', action='append', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n-seeds', type=int, default=10000)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--figure', type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
    images = {name: nib.load(getattr(args, name.replace('-', '_')))
              for name in ('fod', 'five-tissue', 'gmwmi', 'processing-mask', 'fa', 'atlas')}
    def tensor(name: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Load named NIfTI voxel array to the chosen device with explicit dtype."""
        return torch.as_tensor(np.asarray(images[name].dataobj).copy(), device=device, dtype=dtype)
    fod, five, gmwmi = tensor('fod'), tensor('five-tissue'), tensor('gmwmi')
    processing, fa, atlas = tensor('processing-mask'), tensor('fa'), tensor('atlas', torch.int32)
    affine = {name: torch.as_tensor(image.affine, device=device, dtype=torch.float64)
              for name, image in images.items()}
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    start = time.perf_counter()
    tracks = probabilistic_tractography(
        fod, affine['fod'], five, affine['five-tissue'], gmwmi,
        n_seeds=args.n_seeds, seed=args.seed, lmax=8, batch_size=8192,
        cutoff=.1, power=.5,
    )
    _sync(device)
    tracking_time = time.perf_counter() - start
    step_mm = float(torch.linalg.vector_norm(affine['fod'][:3, :3], dim=0).min()) / 2
    start = time.perf_counter()
    weights = estimate_sift2_weights(
        tracks.paths, fod, affine['fod'], five, affine['five-tissue'],
        step_size_mm=step_mm, processing_mask=processing,
    )
    _sync(device)
    sift2_time = time.perf_counter() - start
    start = time.perf_counter()
    fa_values = sample_streamline_mean_precise(tracks.paths, fa, affine['fa'])
    _sync(device)
    fa_time = time.perf_counter() - start
    start = time.perf_counter()
    matrices = build_connectomes(
        tracks.endpoints, atlas, affine['atlas'], weights=weights,
        lengths=tracks.lengths_mm, fa=fa_values,
    )
    _sync(device)
    assignment_time = time.perf_counter() - start
    candidate = {name: value.cpu().numpy() for name, value in matrices.items()}
    references = {}
    for directory in args.official_dir:
        official = {name: np.loadtxt(directory / f'{name}.csv', delimiter=',')
                    for name in candidate}
        references[directory.name] = _stats(candidate, official)
    endpoint_values = _five_tissue_values(
        five, tracks.endpoints.reshape(-1, 3),
        torch.linalg.inv(affine['five-tissue']),
    ).cpu().numpy()
    cg, sg, wm, csf, path = endpoint_values.T
    is_gm = (cg+sg >= wm) & (cg+sg > csf) & (cg+sg > path)
    lengths = tracks.lengths_mm.cpu().numpy()
    report = {
        'dataset': 'OpenNeuro ds004666 sub-01 ses-2mm fixed official WM FOD, 5TT, GMWMI, FA, atlas',
        'tracking_source_sha256': _sha(Path(tracking_module.__file__)),
        'input_sha256': {name: _sha(getattr(args, name.replace('-', '_')))
                         for name in ('fod', 'five-tissue', 'gmwmi', 'processing-mask', 'fa', 'atlas')},
        'rng': {'torch_seed': args.seed, 'n_seed_attempts': args.n_seeds,
                'official_runs': [str(p) for p in args.official_dir]},
        'device': str(device), 'tf32_enabled': bool(torch.backends.cuda.matmul.allow_tf32),
        'accepted_streamlines': len(tracks.paths),
        'length_mm_quantiles': np.quantile(lengths, [0,.05,.25,.5,.75,.95,1]).tolist(),
        'length_mm_mean': float(lengths.mean()),
        'endpoint_tissue_counts': {'cortical_gm': int((is_gm & (cg>=sg)).sum()),
                                   'subcortical_gm': int((is_gm & (sg>cg)).sum()),
                                   'other': int((~is_gm).sum())},
        'time_seconds': {'tracking': tracking_time, 'sift2': sift2_time,
                         'mean_fa': fa_time, 'assignment': assignment_time,
                         'total_core': tracking_time+sift2_time+fa_time+assignment_time},
        'peak_torch_allocated_gib': torch.cuda.max_memory_allocated(device)/2**30
                                    if device.type=='cuda' else None,
        'reference_comparisons': references,
        'comparison_limit': 'MRtrix and PyTorch RNG streams differ; compare distributions and matrices, not track row identity.',
    }
    if args.figure:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        ref_dir = args.official_dir[0]
        ref_tracks = nib.streamlines.load(ref_dir/'tracks_10000.tck').streamlines
        ref_lengths = np.asarray([np.linalg.norm(np.diff(p,axis=0),axis=1).sum() for p in ref_tracks])
        official = np.loadtxt(ref_dir/'count.csv',delimiter=',')
        u=np.triu_indices(20,1)
        fig, axes = plt.subplots(1,2,figsize=(9,4),constrained_layout=True)
        bins=np.arange(0,251,5)
        axes[0].hist(ref_lengths,bins=bins,density=True,histtype='step',label='MRtrix')
        axes[0].hist(lengths,bins=bins,density=True,histtype='step',label='PyTorch')
        axes[0].set(xlabel='Track length (mm)',ylabel='Density',title='Same FOD + 5TT')
        axes[0].legend(frameon=False)
        axes[1].scatter(official[u],candidate['count'][u],s=8,alpha=.5)
        axes[1].plot([0,170],[0,170],'r--',lw=1)
        axes[1].set(xlabel='MRtrix edge count',ylabel='PyTorch edge count',title='20-region connectome')
        args.figure.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(args.figure,dpi=170)
        plt.close(fig)
        report['figure']=args.figure.name
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
