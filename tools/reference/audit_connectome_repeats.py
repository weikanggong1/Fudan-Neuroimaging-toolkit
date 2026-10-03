"""CPU 只读复核已有官方重复命令、输入序列化与输出摘要；不运行官方程序。"""
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path
import time
import os
import platform

import nibabel as nib
import numpy as np
import torch


def _reference():
    path = Path(__file__).with_name('benchmark_connectome_repeats_official.py')
    spec = importlib.util.spec_from_file_location('official_repeat_readonly_audit', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require(condition, label):
    if not condition:
        raise ValueError(label)


def audit(manifest_path, checkpoint, fnit_dir, reference_script=None):
    reference = _reference()
    started = time.perf_counter()
    manifest_path, checkpoint, fnit_dir = map(Path, (manifest_path, checkpoint, fnit_dir))
    manifest_hash = reference.sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    _require(manifest['execution_completed'], 'official execution is not complete')
    _require(len(manifest['seeds']) >= 3 and len(set(manifest['seeds'])) == len(manifest['seeds']),
             'at least three unique official seed labels are required')
    planned, completed = manifest['commands'], manifest['completed_commands']
    _require(len(planned) == len(completed), 'planned/completed command counts differ')
    _require(all(all(actual.get(key) == value for key, value in plan.items())
                 and actual['returncode'] == 0 for plan, actual in zip(planned, completed)),
             'actual command/argv or exit code differs from plan')
    _require(all(Path(item['log']).is_file() and Path(item['time_file']).is_file()
                 for item in completed), 'command log/time artifact missing')
    for record in manifest['programs'].values():
        _require(reference.sha256(Path(record['path'])) == record['sha256'], 'reference executable changed')
    if reference_script is not None:
        _require(reference.sha256(Path(reference_script)) == manifest['script_sha256'],
                 'executed reference script hash differs from supplied script')
        _require(reference.sha256(Path(reference_script).resolve().parents[1] / 'connectome_repeat_common.py') ==
                 manifest['helper_sha256'], 'executed reference helper changed')
    payload = torch.load(checkpoint / 'tracking_inputs.pt', map_location='cpu', weights_only=True)
    _require(reference.sha256(checkpoint / 'tracking_inputs.pt') == manifest['tracking_input_sha256'],
             'actual tracking PT changed')
    readiness = reference.source_readiness(checkpoint, payload)
    _require(readiness == manifest['source_readiness'], 'core artifacts/readiness changed')
    _require(json.loads(json.dumps(reference.effective_parameters(
        payload, manifest['parameters']['n_seed_attempts']), allow_nan=False)) == manifest['parameters'], 'effective actual tracking parameters differ')
    profiles = reference.load_profiles(fnit_dir)
    _require({name: metadata for name, (_, metadata) in profiles.items()} ==
             manifest['source_profile_metadata'], 'source matrix/atlas/node metadata changed')
    fa = payload.get('fa')
    if fa is None:
        fa = np.asarray(nib.load(checkpoint / 'fa.nii.gz').dataobj)
    sources = {
        'wm_fod': (payload['wm_sh'], payload['fod_affine']),
        'five_tissue_act': (payload['five_tissue'], payload['five_tissue_affine']),
        'five_tissue_sift2': (payload['five_tissue'], payload['five_tissue_affine']),
        'gmwmi': (payload['gmwmi'], payload['five_tissue_affine']),
        'fa': (fa, payload['fod_affine']),
    }
    sources.update({f'atlas:{name}': (np.asarray(nib.load(Path(metadata['directory']) /
                        'atlas_dwi.nii.gz').dataobj), payload['fod_affine'])
                    for name, (_, metadata) in profiles.items()})
    exports = {name: item for name, item in manifest['input_exports'].items() if name != 'atlases'}
    exports.update({f'atlas:{name}': item for name, item in manifest['input_exports']['atlases'].items()})
    _require(set(exports) == set(sources) == set(manifest['input_readbacks']), 'input set changed')
    readers = {item['input']: item for item in completed if item['stage'] == 'input_readback'}
    input_audits = {}
    for name, (source, affine) in sources.items():
        source = source.numpy() if isinstance(source, torch.Tensor) else np.asarray(source)
        affine = affine.numpy() if isinstance(affine, torch.Tensor) else np.asarray(affine)
        exported = exports[name]
        image = nib.load(exported['path'])
        actual = np.asarray(image.dataobj)
        _require(reference.sha256(Path(exported['path'])) == exported['sha256'], f'{name}: exported file changed')
        _require(isinstance(image, nib.Nifti2Image) and actual.dtype == source.dtype and
                 actual.shape == source.shape, f'{name}: image format/shape/dtype differs')
        voxel_bits = np.array_equal(np.ascontiguousarray(source).view(np.uint8),
                                    np.ascontiguousarray(actual).view(np.uint8))
        sform_bits = np.array_equal(np.ascontiguousarray(affine[:3]).view(np.uint8),
                                    np.ascontiguousarray(image.affine[:3]).view(np.uint8))
        _require(voxel_bits and sform_bits, f'{name}: voxel/sform3x4 raw bits differ')
        contract = reference._nifti_affine_contract(affine)
        _require(contract == exported['affine_serialization'], f'{name}: source affine contract changed')
        readback = reference.input_readback(readers[name], exported)
        _require(readback == manifest['input_readbacks'][name], f'{name}: actual reader record changed')
        input_audits[name] = {
            'voxel_bits_equal': voxel_bits, 'sform_3x4_bits_equal': sform_bits,
            'source_affine_serialization': contract,
            'nonfinite_count': int((~np.isfinite(actual)).sum()),
            'readback_status': readback['status'],
            'reader_corner_difference_from_source_mm': readback['maximum_corner_difference_from_source_affine_mm'],
            'reader_corner_difference_from_fnit_operator_mm': readback['maximum_corner_difference_from_fnit_operator_affine_mm'],
            'geometry_scope': readback['geometry_assessment'],
            **{key: readback[key] for key in ('source_voxel_to_world_affine', 'file_voxel_to_world_affine',
                'fnit_effective_voxel_to_world_affine', 'decoded_voxel_to_world_affine',
                'expected_reader_affine', 'axis_mapping_to_source', 'mrinfo_json', 'mrinfo_json_sha256')},
        }
    runs = {}
    for seed in manifest['seeds']:
        directory = manifest_path.parent / f'seed-{seed}'
        expected = manifest['outputs'][str(seed)]
        _require(all(reference.sha256(directory / name) == digest
                     for name, digest in expected['file_sha256'].items()), f'seed{seed}: output changed')
        current = reference.load_profiles(directory)
        _require({name: metadata for name, (_, metadata) in current.items()} == expected['profiles'],
                 f'seed{seed}: matrix/atlas/node metadata changed')
        count = int(nib.streamlines.load(directory / 'tracks.tck', lazy_load=True).header['count'])
        vectors = {}
        for name in ('sift2_weights.txt', 'lengths.txt', 'mean_fa.txt'):
            values = np.loadtxt(directory / name, ndmin=1)
            _require(values.shape == (count,), f'seed{seed}: {name} count differs from TCK')
            vectors[name] = {'shape': list(values.shape), 'nonfinite_count': int((~np.isfinite(values)).sum())}
        records = [item for item in completed if item.get('seed') == seed]
        tracking = next(item for item in records if item['stage'] == 'tracking')
        argv = tracking['argv']
        _require(argv[argv.index('-nthreads') + 1] == '0' and
                 argv[argv.index('-select') + 1] == '0' and
                 int(argv[argv.index('-seeds') + 1]) == manifest['parameters']['n_seed_attempts'],
                 f'seed{seed}: tracking seed/selection/thread contract differs')
        runs[str(seed)] = {'commands': len(records), 'all_exit_zero': True, 'accepted_tracks': count,
            'attempted_seeds': manifest['parameters']['n_seed_attempts'],
            'accepted_fraction': count / manifest['parameters']['n_seed_attempts'], 'vectors': vectors,
            'stage_seconds_inclusive': {stage: sum(item['seconds_inclusive'] for item in records if item['stage'] == stage)
                                       for stage in sorted({item['stage'] for item in records})},
            'track_sha256': expected['file_sha256']['tracks.tck']}
    _require(reference.sha256(manifest_path) == manifest_hash, 'manifest changed during read-only audit')
    return {'dataset': manifest['dataset'], 'contract_status': 'passed', 'scientific_parity': 'not_assessed_by_contract_audit',
            'scope': 'CPU read-only existing artifacts; no official command or GPU operation; no geometry/scientific tolerance added',
            'manifest_path': str(manifest_path.resolve()), 'manifest_sha256': manifest_hash,
            'tracking_input_sha256': manifest['tracking_input_sha256'],
            'parameters': manifest['parameters'], 'mrtrix_version': manifest['mrtrix_version'],
            'program_sha256_verified': True, 'programs': manifest['programs'], 'reference_script_sha256_verified': reference_script is not None,
            'reference_helper_sha256_verified': reference_script is not None,
            'planned_commands': len(planned), 'completed_commands': len(completed), 'all_exit_zero': True,
            'command_stages': dict(Counter(item['stage'] for item in completed)),
            'input_contracts': input_audits, 'official_runs': runs,
            'execution_wall_seconds': manifest['execution_wall_seconds'],
            'preflight_seconds': manifest['preflight_seconds'], 'total_wall_seconds': manifest['total_wall_seconds'],
            'timing_scope': manifest['timing_scope'], 'original_host': manifest['environment']['host'],
            'audit_environment': {'host': platform.node(), 'python': platform.python_version(),
                'numpy': np.__version__, 'nibabel': nib.__version__, 'torch': torch.__version__,
                'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')},
            'audit_script_sha256': reference.sha256(Path(__file__)),
            'cpu_readonly_audit_seconds': time.perf_counter() - started}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--fnit-dir', type=Path, required=True)
    parser.add_argument('--reference-script', type=Path, help='可选：核对执行时冻结的 reference script SHA')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    report = audit(args.manifest, args.checkpoint_dir, args.fnit_dir, args.reference_script)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'contract_status': report['contract_status'],
                      'completed_commands': report['completed_commands'],
                      'input_contracts': len(report['input_contracts'])}, allow_nan=False))


if __name__ == '__main__':
    main()
