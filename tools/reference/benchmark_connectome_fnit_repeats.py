"""真实已有 FNIT 轨迹的冻结 GPU 后处理；不重新追踪，不运行官方软件。"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

LIMIT_BYTES = 20_000_000_000
ARRAY_NAMES = ('points', 'offsets', 'endpoints', 'lengths_mm', 'accepted_seeds')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic_json(path, report):
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def verify_source(config):
    root = Path(config['source_root']).resolve()
    require(sha256(config['source_archive']) == config['source_archive_sha256'], 'frozen source archive changed')
    require(sha256(config['source_identity']) == config['source_identity_sha256'], 'source identity file changed')
    actual_python = {str(p.relative_to(root)) for p in (root / 'src').rglob('*.py')}
    expected_python = {name for name in config['source_files'] if name.endswith('.py')}
    require(actual_python == expected_python, 'frozen source Python file set changed')
    for name, expected in config['source_files'].items():
        path = (root / name).resolve()
        require(path.is_relative_to(root) and sha256(path) == expected, f'frozen source/resource changed: {name}')
    require(sha256(config['monitor_script']) == config['monitor_sha256'], 'frozen monitor changed')
    require(sha256(config['official_manifest']) == config['official_manifest_sha256'], 'official manifest changed')
    for path, expected in config.get('tool_files', {}).items():
        require(sha256(path) == expected, f'frozen postprocessing tool changed: {path}')
    return {'status': 'archive_python_and_required_resource_sha_verified',
            'source_archive_sha256': config['source_archive_sha256'],
            'source_identity_sha256': config['source_identity_sha256'],
            'declared_equivalent_local_commit': config['equivalent_local_commit'],
            'remote_git_commit': None, 'files': config['source_files'],
            'monitor_sha256': config['monitor_sha256']}


def memory_gate(peaks, failed_samples):
    return 'passed' if all(value is not None and 0 <= value < LIMIT_BYTES for value in peaks) and failed_samples == 0 else 'not_passed'


def array_contract(arrays, np):
    points, offsets, endpoints, lengths, seeds = (arrays[name] for name in ARRAY_NAMES)
    count = len(offsets) - 1
    require(points.dtype == np.float32 and points.ndim == 2 and points.shape[1] == 3,
            'points require original float32 [P,3]')
    require(offsets.dtype == np.int64 and offsets.ndim == 1 and count > 0 and
            offsets[0] == 0 and offsets[-1] == len(points) and (np.diff(offsets) >= 2).all(),
            'offsets require int64 boundaries with >=2 points per track')
    require(endpoints.dtype == np.float32 and endpoints.shape == (count, 2, 3) and
            lengths.dtype == np.float32 and lengths.shape == (count,) and
            seeds.dtype == np.float32 and seeds.shape == (count, 3), 'metric shape/dtype differs')
    require(all(np.isfinite(value).all() for value in arrays.values()) and (lengths >= 0).all(),
            'trajectory arrays contain invalid values')
    original = np.stack((points[offsets[:-1]], points[offsets[1:] - 1]), axis=1)
    require(np.array_equal(np.ascontiguousarray(original).view(np.uint8), np.ascontiguousarray(endpoints).view(np.uint8)),
            'saved endpoints bits differ from stored track boundary points')
    return count


def preflight(config, seed, directory):
    import nibabel as nib
    import numpy as np
    import torch
    source = verify_source(config)
    require(not torch.cuda.is_initialized(), 'CUDA initialized before shared lock')
    checkpoint = Path(config['checkpoint_dir'])
    require(sha256(checkpoint / 'tracking_inputs.pt') == config['tracking_input_sha256'], 'actual PT changed')
    ref = load_module(Path(__file__).with_name('benchmark_connectome_repeats_official.py'), 'fnit_repeat_contract_reference')
    payload = torch.load(checkpoint / 'tracking_inputs.pt', map_location='cpu', weights_only=True)
    readiness = ref.source_readiness(checkpoint, payload)
    official = json.loads(Path(config['official_manifest']).read_text())
    require(readiness == official['source_readiness'], 'core source/readiness differs from fixed official inputs')
    profiles = ref.load_profiles(Path(config['fnit_dir']))
    require({name: meta for name, (_, meta) in profiles.items()} == official['source_profile_metadata'],
            'atlas/node/matrix identities differ from fixed official source')
    require(set(profiles) == set(config['atlases']), 'atlas set differs')
    report_path = directory / 'report.json'
    tracking = json.loads(report_path.read_text())
    kwargs = json.loads(json.dumps(payload['tracking_kwargs'], allow_nan=False))
    kwargs['seed'] = seed
    require(tracking['checkpoint_sha256'] == config['tracking_input_sha256'] and
            tracking['manifest_sha256'] == config['tracking_manifest_sha256'] and
            tracking['tracking_kwargs'] == kwargs and tracking['seed'] == seed and
            tracking['harness_sha256'] == config['tracking_worker_sha256'][str(seed)] and
            tracking['source_sha256'] == config['tracking_source_sha256'] and
            tracking['budget_pass'] is True and not tracking.get('error'),
            'completed tracking seed/source/parameters/readiness differ')
    arrays = {name: np.load(directory / f'{name}.npy', allow_pickle=False) for name in ARRAY_NAMES}
    count = array_contract(arrays, np)
    require(count == tracking['accepted'] and len(arrays['points']) == tracking['points'], 'tracking counts differ')
    fa = payload.get('fa')
    fa = np.asarray(nib.load(checkpoint / 'fa.nii.gz').dataobj) if fa is None else fa.numpy()
    images = {'wm_fod': payload['wm_sh'].numpy(), 'five_tissue_act': payload['five_tissue'].numpy(), 'fa': fa}
    verified = {}
    for name, values in images.items():
        exported = official['input_exports'][name]
        require(sha256(exported['path']) == exported['sha256'], f'fixed official {name} changed')
        actual = np.asarray(nib.load(exported['path']).dataobj)
        require(values.dtype == actual.dtype and values.shape == actual.shape and
                np.array_equal(np.ascontiguousarray(values).view(np.uint8), np.ascontiguousarray(actual).view(np.uint8)),
                f'{name}: original source voxel bits differ')
        verified[name] = {'dtype': str(values.dtype), 'shape': list(values.shape), 'bits_equal': True,
                          'nonfinite_count': int((~np.isfinite(values)).sum())}
    require(payload['fod_affine'].dtype == torch.float64 and payload['five_tissue_affine'].dtype == torch.float64,
            'original source affines must remain Float64')
    atlas_arrays = {}
    for name, (_, meta) in profiles.items():
        image = nib.load(Path(meta['directory']) / 'atlas_dwi.nii.gz')
        atlas = np.asarray(image.dataobj)
        require(np.array_equal(image.affine.view(np.uint64), payload['fod_affine'].numpy().view(np.uint64)),
                f'{name}: actual atlas affine differs from DWI affine bits')
        exported = official['input_exports']['atlases'][name]
        require(sha256(exported['path']) == exported['sha256'], f'{name}: official atlas changed')
        expected = np.asarray(nib.load(exported['path']).dataobj)
        require(atlas.dtype == expected.dtype and np.array_equal(np.ascontiguousarray(atlas).view(np.uint8), np.ascontiguousarray(expected).view(np.uint8)),
                f'{name}: actual atlas bits differ from fixed reference')
        atlas_arrays[name] = atlas
    inputs = {'tracking_directory': str(directory.resolve()), 'tracking_report_sha256': sha256(report_path),
        'tracking_array_sha256': {name: sha256(directory / f'{name}.npy') for name in ARRAY_NAMES},
        'tracking_input_sha256': config['tracking_input_sha256'], 'tracking_kwargs': kwargs,
        'tracking_harness_sha256': tracking['harness_sha256'], 'tracking_source_sha256': tracking['source_sha256'],
        'source_readiness': readiness, 'source_profiles': {name: meta for name, (_, meta) in profiles.items()},
        'source_affines': {'fod': payload['fod_affine'].tolist(), 'five_tissue': payload['five_tissue_affine'].tolist()},
        'original_images': verified, 'tracks': count, 'points': len(arrays['points']),
        'fa_nonfinite_coordinates': np.argwhere(~np.isfinite(fa)).tolist()}
    require(not torch.cuda.is_initialized(), 'preflight unexpectedly initialized CUDA')
    return torch, np, nib, source, inputs, payload, arrays, fa, atlas_arrays


def gpu_postprocess(config, torch, np, payload, arrays, fa_array, atlas_arrays, inputs, report):
    verify_source(config)  # recheck after possibly waiting for the shared lock
    torch.set_num_threads(8)
    sys.path.insert(0, str(Path(config['source_root']) / 'src'))
    from fnit.connectome.sift2 import estimate_sift2_weights
    from fnit.connectome.tcksample_precise import sample_streamline_mean_precise
    from fnit.connectome.assignment import build_connectomes
    for function in (estimate_sift2_weights, sample_streamline_mean_precise, build_connectomes):
        module = Path(sys.modules[function.__module__].__file__).resolve()
        require(module.is_relative_to(Path(config['source_root']).resolve()), 'loaded mutable candidate module')
    helper = load_module(Path(config['monitor_script']), 'frozen_repeat_gpu_monitor')
    monitor = helper.GPUProcessMonitor(torch, 'cuda:0', interval=.25, gpu_uuid=config['gpu_uuid'])
    monitor.start()
    gpu, matrices, tensor = {}, None, None
    result = {}
    try:
        stage = time.perf_counter()
        torch.cuda.init()
        uuid = torch.cuda.get_device_properties(0).uuid
        uuid = uuid.decode() if isinstance(uuid, bytes) else str(uuid)
        require(uuid == config['gpu_uuid'], 'actual CUDA GPU differs from fixed UUID')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats(0)
        report['timings_seconds']['cuda_initialization'] = time.perf_counter() - stage
        device = torch.device('cuda:0')
        stage = time.perf_counter()
        for name, key in [('wm', 'wm_sh'), ('five', 'five_tissue'),
                          ('affine', 'fod_affine'), ('five_affine', 'five_tissue_affine')]:
            gpu[name] = payload[key].to(device)
        gpu['fa'] = torch.from_numpy(np.ascontiguousarray(fa_array)).to(device)
        for name in ('points', 'endpoints', 'lengths_mm'):
            gpu[name] = torch.from_numpy(arrays[name]).to(device)
        offsets = arrays['offsets']
        gpu['paths'] = tuple(gpu['points'][int(a):int(b)] for a, b in zip(offsets[:-1], offsets[1:]))
        torch.cuda.synchronize()
        report['timings_seconds']['h2d_inputs_and_path_views'] = time.perf_counter() - stage
        # Exact production expression on the actual GPU, not rounded CPU/JSON.
        step = float(torch.linalg.vector_norm(gpu['affine'][:3, :3], dim=0).prod().pow(1 / 3)) / 2
        report['actual_gpu_step_size_mm'], report['actual_gpu_step_size_hex'] = step, step.hex()
        report['precision'] = {'tf32': True, 'image_dtype': str(gpu['wm'].dtype),
            'affine_dtype': str(gpu['affine'].dtype),
            'step_expression': 'float(torch.linalg.vector_norm(dwi_affine[:3,:3],dim=0).prod().pow(1/3))/2'}
        stage = time.perf_counter()
        gpu['weights'] = estimate_sift2_weights(gpu['paths'], gpu['wm'], gpu['affine'],
            gpu['five'], gpu['five_affine'], step_size_mm=step)
        torch.cuda.synchronize()
        require(gpu['weights'].dtype == torch.float64, 'SIFT2 output lost Float64')
        report['timings_seconds']['sift2'] = time.perf_counter() - stage
        stage = time.perf_counter()
        gpu['mean_fa'] = sample_streamline_mean_precise(gpu['paths'], gpu['fa'], gpu['affine'])
        torch.cuda.synchronize()
        report['timings_seconds']['precise_fa'] = time.perf_counter() - stage
        stage = time.perf_counter()
        result['weights'], result['mean_fa'] = gpu['weights'].cpu().numpy(), gpu['mean_fa'].cpu().numpy()
        report['timings_seconds']['scalar_d2h'] = time.perf_counter() - stage
        report['scalar_nonfinite_counts'] = {name: int((~np.isfinite(result[name])).sum())
                                            for name in ('weights', 'mean_fa')}
        report['atlas_timings_seconds'] = {}
        result['matrices'] = {}
        for name in config['atlases']:
            stage = time.perf_counter()
            meta = inputs['source_profiles'][name]
            gpu['atlas'] = torch.from_numpy(np.ascontiguousarray(atlas_arrays[name])).to(device)
            matrices = build_connectomes(gpu['endpoints'], gpu['atlas'], gpu['affine'],
                weights=gpu['weights'], lengths=gpu['lengths_mm'], fa=gpu['mean_fa'], node_count=meta['nodes'])
            torch.cuda.synchronize()
            compute_seconds = time.perf_counter() - stage
            values = {kind: tensor.cpu().numpy() for kind, tensor in matrices.items()}
            require(all(np.isfinite(value).all() for value in values.values()), f'{name}: nonfinite final matrix')
            result['matrices'][name] = values
            report['atlas_timings_seconds'][name] = {'gpu_with_atlas_h2d': compute_seconds,
                'including_matrix_d2h': time.perf_counter() - stage, 'nodes': meta['nodes'],
                'source_atlas_sha256': meta['atlas_sha256']}
            matrices = None
            del gpu['atlas']
        report['gpu_uuid'] = uuid
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
        raise
    finally:
        if torch.cuda.is_initialized():
            report['cuda_memory'] = {'peak_allocated_bytes': torch.cuda.max_memory_allocated(0),
                                    'peak_reserved_bytes': torch.cuda.max_memory_reserved(0)}
            matrices, tensor = None, None
            gpu.clear()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        report['process_memory'] = monitor.finish()
        peaks = [report.get('cuda_memory', {}).get('peak_allocated_bytes'),
                 report.get('cuda_memory', {}).get('peak_reserved_bytes'),
                 report['process_memory']['peak_process_tree_bytes']]
        report['memory_gate'] = memory_gate(peaks, report['process_memory']['failed_samples'])
    return result


def run(config, seed, directory, output, dry_run=False):
    start = time.perf_counter()
    torch, np, nib, source, inputs, payload, arrays, fa, atlases = preflight(config, seed, directory)
    report = {'dataset': config['dataset'], 'seed': seed, 'status': 'cpu_preflight_passed',
        'scope': 'frozen GPU postprocessing of existing exported tracks; not continuous raw pipeline or full runtime memory',
        'script_sha256': sha256(Path(__file__)), 'config_sha256': config['_actual_config_sha256'],
        'source': source, 'inputs': inputs, 'timings_seconds': {'cpu_preflight': time.perf_counter() - start},
        'memory_limit_bytes': LIMIT_BYTES, 'memory_gate': 'not_assessed', 'scientific_parity': 'not_assessed',
        'environment': {'host': platform.node(), 'python': platform.python_version(), 'torch': torch.__version__,
            'numpy': np.__version__, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'allocator': os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}}
    if dry_run:
        print(json.dumps(report, indent=2, allow_nan=False))
        return report
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == config['gpu_uuid'] and
            os.environ.get('PYTORCH_CUDA_ALLOC_CONF') == 'expandable_segments:True',
            'single GPU UUID/allocator environment differs from frozen binding')
    require(not output.exists(), 'output directory must be new')
    output.mkdir(parents=True)
    atomic_json(output / 'report.json', report)
    waiting = time.perf_counter()
    with Path(config['gpu_lock']).open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        report['timings_seconds']['shared_lock_wait'] = time.perf_counter() - waiting
        report['lock'] = {'path': config['gpu_lock'], 'pid': os.getpid(), 'acquired': True}
        require(not torch.cuda.is_initialized(), 'CUDA initialized before lock acquisition')
        try:
            result = gpu_postprocess(config, torch, np, payload, arrays, fa, atlases, inputs, report)
        finally:
            atomic_json(output / 'report.json', report)
            fcntl.flock(lock, fcntl.LOCK_UN)
    # CPU serialization after GPU tensors and cache are released inside lock.
    stage = time.perf_counter()
    for name, matrices in result['matrices'].items():
        target = output / 'atlases' / name
        target.mkdir(parents=True)
        report['atlas_timings_seconds'][name]['matrix_sha256'] = {}
        for kind, value in matrices.items():
            path = target / f'connectome_{kind}.csv'
            np.savetxt(path, value, delimiter=',')
            report['atlas_timings_seconds'][name]['matrix_sha256'][kind] = sha256(path)
        meta = inputs['source_profiles'][name]
        for filename in ('atlas_dwi.nii.gz', 'nodes.tsv', 'region_labels.csv'):
            source_file = Path(meta['directory']) / filename
            if source_file.is_file():
                shutil.copyfile(source_file, target / filename)
        (target / 'atlas.sha256').write_text(meta['atlas_sha256'] + '\n')
    tracks = [arrays['points'][int(a):int(b)] for a, b in zip(arrays['offsets'][:-1], arrays['offsets'][1:])]
    nib.streamlines.save(nib.streamlines.Tractogram(tracks, affine_to_rasmm=np.eye(4)), output / 'tracks.tck')
    actual = list(nib.streamlines.load(output / 'tracks.tck', lazy_load=False).streamlines)
    require(len(actual) == len(tracks) and all(np.array_equal(a.view(np.uint8), b.view(np.uint8))
                                             for a, b in zip(tracks, actual)), 'saved TCK changed point bits')
    np.savez_compressed(output / 'track_metrics.npz', weights=result['weights'], mean_fa=result['mean_fa'],
                        lengths=arrays['lengths_mm'], endpoints=arrays['endpoints'])
    for filename, values in [('sift2_weights.txt', result['weights']), ('mean_fa.txt', result['mean_fa']),
                             ('lengths.txt', arrays['lengths_mm'])]:
        np.savetxt(output / filename, values)
    report['timings_seconds']['cpu_matrix_tck_scalar_io'] = time.perf_counter() - stage
    require(all(sha256(directory / f'{name}.npy') == digest for name, digest in inputs['tracking_array_sha256'].items())
            and sha256(directory / 'report.json') == inputs['tracking_report_sha256'], 'tracking inputs changed')
    verify_source(config)
    report['output_sha256'] = {name: sha256(output / name) for name in
        ('tracks.tck', 'track_metrics.npz', 'sift2_weights.txt', 'mean_fa.txt', 'lengths.txt')}
    report['tck_coordinate_bits_verified_equal'] = True
    report['timings_seconds']['total_worker_with_queue'] = time.perf_counter() - start
    report['status'] = 'completed' if report['memory_gate'] == 'passed' else 'completed_memory_gate_not_passed'
    report['repeat_policy'] = 'frozen production functions; original atomic floating reductions may differ; no new numeric tolerance'
    atomic_json(output / 'report.json', report)
    require(report['memory_gate'] == 'passed', 'actual postprocessing memory gate did not pass')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--tracking-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true', help='actual CPU contracts only; no CUDA/output mutation')
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    config['_actual_config_sha256'] = sha256(args.config)
    require(str(args.seed) in config['tracking_worker_sha256'], 'seed outside frozen repeat plan')
    report = run(config, args.seed, args.tracking_dir, args.output_dir, args.dry_run)
    if not args.dry_run:
        print(json.dumps({'status': report['status'], 'memory_gate': report['memory_gate'], 'seed': args.seed}))


if __name__ == '__main__':
    main()
