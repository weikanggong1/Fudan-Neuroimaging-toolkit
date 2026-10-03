"""真实已完成 CON03 官方 fixed-TCK：权重精度与四矩阵 CPU/CUDA 配对。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time

import nibabel as nib
import numpy as np
import torch


NAMES = ('count', 'sift2_fbc', 'mean_length', 'mean_fa')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def checked(path, expected):
    path = Path(path)
    if not path.is_file() or sha(path) != expected:
        raise ValueError(f'actual byte identity mismatch: {path}')
    return path


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def metrics(actual, expected):
    finite = np.isfinite(actual) & np.isfinite(expected)
    difference = np.abs(actual[finite].astype(np.float64) - expected[finite])
    return {'different_cells': int(np.count_nonzero(actual[finite] != expected[finite])),
            'max_abs_error': float(difference.max(initial=0)),
            'mae': float(difference.mean()) if difference.size else None,
            'rmse': float(np.sqrt(np.mean(difference ** 2))) if difference.size else None,
            'nonfinite_mismatch': int(np.count_nonzero(np.isfinite(actual) != np.isfinite(expected)))}


class MemorySampler:
    def __init__(self):
        self.done = threading.Event(); self.rows = []; self.failures = 0
    def __enter__(self):
        self.thread = threading.Thread(target=self.run, daemon=True); self.thread.start(); return self
    def run(self):
        while not self.done.is_set():
            stamp = time.perf_counter()
            try:
                rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory,gpu_uuid',
                    '--format=csv,noheader,nounits'], text=True, timeout=5).splitlines()
                own = [r.split(',') for r in rows if r.split(',')[0].strip() == str(os.getpid())]
                self.rows.append({'t': stamp, 'own_bytes': sum(int(r[1]) * 1024**2 for r in own),
                                  'all_compute_processes': rows})
            except Exception:
                self.failures += 1
            self.done.wait(.1)
    def __exit__(self, *args):
        self.done.set(); self.thread.join()
    def report(self):
        return {'pid': os.getpid(), 'samples': self.rows, 'failed_samples': self.failures,
                'peak_bytes': max((r['own_bytes'] for r in self.rows), default=None),
                'max_interval_s': float(max(np.diff([r['t'] for r in self.rows]), default=0)),
                'gpu_children': 'none: matrix function starts no subprocess'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bindings', required=True, type=Path)
    p.add_argument('--bindings-sha256', required=True)
    p.add_argument('--baseline', required=True, type=Path)
    p.add_argument('--candidate', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--device', default='cpu')
    p.add_argument('--cases', nargs='+', default=['sub-CON03'])
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    p.add_argument('--abba-rounds', type=int, default=4)
    args = p.parse_args()
    if args.output.exists(): raise ValueError('fresh output directory required')
    args.output.mkdir(parents=True)
    bindings = json.loads(checked(args.bindings, args.bindings_sha256).read_text())
    paths = {'baseline': args.baseline, 'candidate': args.candidate}
    sources = {name: {'path': str(path), 'sha256': sha(path)} for name, path in paths.items()}
    modules = {name: module(path, 'assignment_' + name) for name, path in paths.items()}
    device = torch.device(args.device); torch.set_num_threads(8)
    if device.type == 'cuda': torch.backends.cuda.matmul.allow_tf32 = True
    def sync():
        if device.type == 'cuda': torch.cuda.synchronize(device)
    report = {'scope': 'same independently completed official TCK/atlas/Double weights/Float32 length and FA; matrix component only',
        'host': socket.gethostname(), 'torch': torch.__version__, 'device': str(device),
        'sources': sources, 'harness_sha256': sha(__file__), 'bindings_sha256': args.bindings_sha256,
        'precision': 'unchanged Float32 geometry/length/FA; candidate weights Double; output Int64 count and Float32 three matrices',
        'stat_definition': 'symmetric retained diagonal; FBC=sum(w); means=sum(w*value)/sum(w); unassigned omitted',
        'timing_policy': 'same resident inputs; warm both versions; four rounds ABBA; synchronization at boundaries; load excluded and separately logged',
        'cases': {}}
    sampler = MemorySampler() if device.type == 'cuda' else None
    if sampler: sampler.__enter__()
    try:
        for case in args.cases:
            entry = bindings['cases'][case]
            reference_path = checked(entry['official_reference_manifest']['path'], entry['official_reference_manifest']['sha256'])
            reference = json.loads(reference_path.read_text()); root = reference_path.parent
            if reference.get('state') != 'completed' or reference.get('execution_completed') is not True:
                raise ValueError('actual completed official producer required')
            if len(reference['completed_commands']) != 198 or any(c.get('returncode') != 0 for c in reference['completed_commands']):
                raise ValueError('all 198 official commands must have succeeded')
            case_report = {'reference_manifest': entry['official_reference_manifest'], 'reference_programs': reference['programs'],
                'native_matrix_commands': [c for c in reference['completed_commands'] if c['stage'].startswith('matrix_')],
                'seeds': {}}
            for seed in args.seeds:
                production = reference['outputs'][str(seed)]; seed_root = root / f'seed-{seed}'
                tracks = checked(seed_root/'tracks.tck', production['tracks_sha256'])
                scalar_paths = {name: checked(seed_root/name, info['sha256']) for name, info in production['scalars'].items()}
                started = time.perf_counter()
                tractogram = nib.streamlines.load(str(tracks)).tractogram
                endpoints_np = np.stack([(path[0], path[-1]) for path in tractogram.streamlines])
                if len(endpoints_np) != production['accepted_tracks']: raise ValueError('TCK count mismatch')
                endpoints = torch.as_tensor(endpoints_np, device=device, dtype=torch.float32)
                weights = torch.as_tensor(np.loadtxt(scalar_paths['sift2_weights.txt']), device=device, dtype=torch.float64)
                lengths = torch.as_tensor(np.loadtxt(scalar_paths['lengths.txt']), device=device, dtype=torch.float32)
                fa = torch.as_tensor(np.loadtxt(scalar_paths['mean_fa.txt']), device=device, dtype=torch.float32)
                if any(v.shape != (len(endpoints),) for v in (weights, lengths, fa)): raise ValueError('track/scalar order lengths mismatch')
                if not all(torch.isfinite(v).all() for v in (weights, lengths, fa)): raise ValueError('finite fixed scalar inputs required')
                sync(); seed_report = {'tracks': {'path': str(tracks), 'sha256': sha(tracks)}, 'accepted_tracks': len(endpoints),
                    'scalar_inputs': {name: {'path': str(path), 'sha256': sha(path)} for name, path in scalar_paths.items()},
                    'scalar_load_s': time.perf_counter()-started, 'profiles': {}}
                for profile, info in production['profiles'].items():
                    atlas_path = checked(root/'inputs/atlases'/profile/'atlas_dwi.nii.gz', info['atlas_sha256'])
                    started = time.perf_counter(); image = nib.load(atlas_path)
                    atlas = torch.as_tensor(np.asarray(image.dataobj).astype(np.int32), device=device)
                    affine = torch.as_tensor(image.affine, device=device, dtype=torch.float64)
                    expected = {name: np.loadtxt(checked(Path(info['directory'])/(name+'.csv'), info['matrix_sha256'][name]), delimiter=',') for name in NAMES}
                    if any(v.shape != (info['nodes'], info['nodes']) for v in expected.values()): raise ValueError('native matrix geometry mismatch')
                    sync(); load_s = time.perf_counter()-started
                    kwargs = dict(endpoints=endpoints, atlas=atlas, affine=affine, weights=weights,
                                  lengths=lengths, fa=fa, node_count=info['nodes'])
                    for mod in modules.values():
                        mod.build_connectomes(**kwargs); sync()
                    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
                    runs = []; arrays = {}
                    order = ['baseline', 'candidate', 'candidate', 'baseline'] * args.abba_rounds
                    for position, name in enumerate(order):
                        sync(); started = time.perf_counter(); result = modules[name].build_connectomes(**kwargs); sync()
                        runs.append({'sequence': position, 'version': name, 'wall_s': time.perf_counter()-started})
                        if name not in arrays: arrays[name] = {k: result[k].cpu().numpy() for k in NAMES}
                    output = args.output/case/f'seed-{seed}'; output.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(output/(profile+'.npz'), **{n+'_'+k: v for n,a in arrays.items() for k,v in a.items()})
                    profile_report = {'atlas': {'path': str(atlas_path), 'sha256': sha(atlas_path)},
                        'atlas_affine': image.affine.tolist(), 'atlas_shape': list(image.shape), 'load_s': load_s, 'runs': runs,
                        'accuracy': {name: {key: metrics(actual, expected[key]) for key, actual in values.items()} for name, values in arrays.items()},
                        'baseline_candidate_count_identical': bool(np.array_equal(arrays['baseline']['count'],arrays['candidate']['count'])),
                        'median_wall_s': {name: float(np.median([r['wall_s'] for r in runs if r['version']==name])) for name in modules}}
                    if device.type == 'cuda':
                        profile_report.update(allocated_peak_bytes=torch.cuda.max_memory_allocated(device), reserved_peak_bytes=torch.cuda.max_memory_reserved(device))
                    seed_report['profiles'][profile] = profile_report
                    print(case, seed, profile, 'FBC',profile_report['accuracy']['baseline']['sift2_fbc']['max_abs_error'],
                          profile_report['accuracy']['candidate']['sift2_fbc']['max_abs_error'],flush=True)
                    del atlas, result
                checked(tracks,production['tracks_sha256'])
                for name,path in scalar_paths.items(): checked(path,production['scalars'][name]['sha256'])
                case_report['seeds'][str(seed)] = seed_report
            checked(reference_path,entry['official_reference_manifest']['sha256'])
            report['cases'][case] = case_report
        checked(args.bindings,args.bindings_sha256)
        for name,path in paths.items(): checked(path,sources[name]['sha256'])
    finally:
        if sampler: sampler.__exit__()
    if sampler:
        report['nvml_process']=sampler.report()
        peaks=[profile[k] for c in report['cases'].values() for s in c['seeds'].values() for profile in s['profiles'].values() for k in ['allocated_peak_bytes','reserved_peak_bytes']]
        report['sampled_memory_budget_passed']=bool(peaks and max(peaks)<20_000_000_000 and report['nvml_process']['peak_bytes'] is not None and report['nvml_process']['peak_bytes']<20_000_000_000)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
