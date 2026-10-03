"""只读审计已完成的独立官方 raw 五重复，报告官方自身矩阵范围与真实组件时间。

不调用官方程序或 GPU，不消费 FNIT 衍生图像，不评判尚未提供的 FNIT 结果。
"""
from __future__ import annotations
import argparse
from collections import Counter
import importlib.util
import itertools
import json
import os
from pathlib import Path
import platform
import sys
import time

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_connectome_raw_cohort_envelope import FIELDS, source_binding
from connectome_repeat_common import (METRIC_POLICY, check_metadata, envelope,
    load_profiles, profile_metrics, sha256)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_worker(path):
    spec = importlib.util.spec_from_file_location('actual_raw_reference_readonly_audit', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit(manifest_path, raw_manifest, raw_sha256, case_id, worker_path):
    started = time.perf_counter()
    manifest_path, worker_path = Path(manifest_path), Path(worker_path)
    digest = sha256(manifest_path)
    _, report, _, raw = source_binding(raw_manifest, raw_sha256, case_id, manifest_path, [])
    require(sha256(worker_path) == report['script_sha256'], 'actual frozen worker source changed')
    worker = source_worker(worker_path)
    require(sha256(worker_path.with_name('benchmark_connectome_repeats_official.py')) ==
            report['reference_command_helper_sha256'] and
            sha256(worker_path.parents[1] / 'connectome_repeat_common.py') == report['matrix_helper_sha256'],
            'actual frozen command or matrix helper changed')
    require(report['seeds'] == [0, 1, 2, 3, 4] and report['parameters'] == worker.parameters(100000) and
            report['downstream_threads'] == 8, 'actual frozen scientific configuration differs')
    commands = report['completed_commands']
    require(all(Path(item['log']).is_file() and Path(item['time_file']).is_file()
                for item in commands), 'actual command log or time artifact missing')
    for entry in report['programs'].values():
        require(sha256(entry['path']) == entry['sha256'], 'actual official executable changed')
    origin = worker.preflight(report['source']['anatomy_contract']['path'],
                              report['source']['official_dwi_contract']['path'], case_id)
    require(origin == report['source'], 'official producer files or contracts changed')
    input_contracts = {}
    for item in commands:
        if item['stage'] != 'input_readback':
            continue
        name = item['input']
        source = (origin['profiles'][name.removeprefix('atlas:')]['image']
                  if name.startswith('atlas:') else origin['images'][name])
        reader = worker.native_readback(item, source)
        require(reader == report['input_readbacks'][name], 'actual MRtrix reader record changed')
        input_contracts[name] = reader
    require(len(input_contracts) == 5 + len(worker.ATLASES), 'actual input readers incomplete')
    runs, profiles = {}, []
    for seed in report['seeds']:
        directory = manifest_path.parent / f'seed-{seed}'
        expected = report['outputs'][str(seed)]
        loaded = load_profiles(directory)
        require({name: meta for name, (_, meta) in loaded.items()} == expected['profiles'],
                'actual matrix, atlas or node output changed')
        require(set(loaded) == set(worker.ATLASES), 'actual atlas output set incomplete')
        profiles.append(loaded)
        tracks = directory / 'tracks.tck'
        require(sha256(tracks) == expected['tracks_sha256'], 'actual TCK bytes changed')
        count = int(nib.streamlines.load(tracks, lazy_load=True).header['count'])
        require(count == expected['accepted_tracks'] and expected['attempted_seeds'] == 100000,
                'actual track count or attempted seeds differs')
        vectors = {}
        for name in ('sift2_weights.txt', 'lengths.txt', 'mean_fa.txt'):
            path = directory / name
            require(sha256(path) == expected['scalars'][name]['sha256'], 'actual scalar bytes changed')
            values = np.loadtxt(path, ndmin=1)
            require(values.shape == (count,), 'actual scalar count differs from TCK')
            finite = values[np.isfinite(values)]
            nonfinite = int(len(values) - len(finite))
            require(nonfinite == expected['scalars'][name]['nonfinite_count'], 'actual scalar finite QC differs')
            vectors[name] = {'path': str(path.resolve()), 'sha256': sha256(path), 'count': count,
                'nonfinite_count': nonfinite, 'finite_count': len(finite),
                'finite_summary': {'minimum': float(finite.min()), 'mean': float(finite.mean()),
                    'maximum': float(finite.max()), 'p05_p50_p95': np.percentile(finite, [5, 50, 95]).tolist()}
                    if len(finite) else None,
                'scope': 'finite-subset descriptive QC; source samples remain unchanged; no filtering in matrix computation'}
        actual = [item for item in commands if item.get('seed') == seed]
        runs[str(seed)] = {'accepted_tracks': count, 'attempted_seeds': 100000,
            'accepted_fraction': count / 100000, 'tracks_sha256': expected['tracks_sha256'],
            'actual_tck_header': expected['actual_tck_header'], 'scalars': vectors,
            'commands': len(actual), 'stage_seconds_inclusive': {
                stage: sum(item['seconds_inclusive'] for item in actual if item['stage'] == stage)
                for stage in sorted({item['stage'] for item in actual})}}
    own_ranges = {}
    for name in worker.ATLASES:
        items = [item[name] for item in profiles]
        identity = check_metadata(items, name)
        pairs = [{'first_seed': report['seeds'][a], 'second_seed': report['seeds'][b],
                  'metrics': profile_metrics(items[a][0], items[b][0])}
                 for a, b in itertools.combinations(range(5), 2)]
        own_ranges[name] = {'nodes': items[0][1]['nodes'], 'input_identity': identity,
            'official_pair_count': len(pairs), 'official_pairs': pairs,
            'ranges': {field: envelope([item['metrics'][field] for item in pairs], [],
                similarity=field in ('count_support_dice', 'count_pearson')) for field in FIELDS},
            'cross_arm_status': 'not_assessed', 'fnit_reproducibility_status': 'not_assessed'}
    require(sha256(manifest_path) == digest, 'actual completed manifest changed during audit')
    return {'schema_version': 1, 'case_id': case_id, 'dataset': report['dataset'],
        'snapshot': report['snapshot'], 'scope': 'read-only independently official raw repeat artifact audit',
        'contract_status': 'passed', 'scientific_parity': 'not_assessed',
        'reason': 'no FNIT results supplied; official own observed ranges alone do not establish equivalence',
        'manifest': {'path': str(manifest_path.resolve()), 'sha256': digest},
        'raw_case_binding': raw, 'actual_worker': {'path': str(worker_path.resolve()), 'sha256': sha256(worker_path)},
        'programs': report['programs'], 'mrtrix_version': report['mrtrix_version'],
        'parameters': report['parameters'], 'seeds': report['seeds'],
        'actual_commands': len(commands), 'all_exit_zero': True,
        'command_stage_counts': dict(Counter(item['stage'] for item in commands)),
        'native_input_contracts': input_contracts, 'official_runs': runs,
        'profiles': own_ranges, 'metric_policy': METRIC_POLICY,
        'official_observed_ranges_scope': 'finite five-repeat empirical sample, not a population confidence interval',
        'timing': {'worker_wall_seconds': report['total_wall_seconds'],
            'stage_seconds_inclusive': {stage: sum(item['seconds_inclusive'] for item in commands if item['stage'] == stage)
                for stage in sorted({item['stage'] for item in commands})},
            'official_eddy': report['raw_case_binding']['official_eddy_solver'],
            'scope': report['timing_scope']},
        'audit_environment': {'host': platform.node(), 'python': platform.python_version(),
            'numpy': np.__version__, 'nibabel': nib.__version__, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')},
        'audit_script_sha256': sha256(__file__), 'cpu_readonly_audit_seconds': time.perf_counter() - started}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--raw-manifest', type=Path, required=True)
    parser.add_argument('--raw-manifest-sha256', required=True)
    parser.add_argument('--case-id', required=True)
    parser.add_argument('--worker', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists(), 'fresh immutable audit output required')
    report = audit(args.manifest, args.raw_manifest, args.raw_manifest_sha256, args.case_id, args.worker)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'case_id': args.case_id, 'contract_status': report['contract_status'],
                      'actual_commands': report['actual_commands'], 'scientific_parity': report['scientific_parity']}))


if __name__ == '__main__':
    main()
