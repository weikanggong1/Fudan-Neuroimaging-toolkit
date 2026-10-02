"""只在 CPU 将已验证的实际 FNIT packed tracks 无损写入独立人口分布诊断 TCK。"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import time


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--post-worker', type=Path, required=True, help='已冻结 GPU 后处理工具；这里只复用 CPU source/track 合同检查')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--tracking-root', type=Path, required=True, help='包含实际 seed0…seed4 packed numpy 产物的根目录')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--output-dir', type=Path, required=True, help='新的独立 CPU 诊断目录；不是 GPU 后处理完成报告')
    args = parser.parse_args(argv)
    start = time.perf_counter()
    spec = importlib.util.spec_from_file_location('frozen_existing_track_cpu_contract', args.post_worker)
    worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)
    worker.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU preparation must hide GPUs')
    worker.require(worker.sha256(args.config) == args.config_sha256, 'frozen config changed')
    worker.require(len(args.seeds) >= 2 and len(set(args.seeds)) == len(args.seeds) and min(args.seeds) >= 0,
                   'at least two distinct actual seeds required')
    config = json.loads(args.config.read_text())
    worker.require(str(args.post_worker.resolve()) in config['tool_files'] and
                   worker.sha256(args.post_worker) == config['tool_files'][str(args.post_worker.resolve())],
                   'actual frozen contract worker changed')
    worker.require(not args.output_dir.exists(), 'CPU population namespace must be new')
    args.output_dir.mkdir(parents=True)
    report = {'schema_version': 1, 'state': 'cpu_preparation_started', 'scope': 'existing FNIT tracks only; no tracking/GPU/SIFT2/matrix computation',
        'scientific_parity': 'not_assessed', 'config_sha256': args.config_sha256,
        'script_sha256': worker.sha256(Path(__file__)), 'post_contract_worker_sha256': worker.sha256(args.post_worker),
        'seeds': args.seeds, 'cases': {}, 'environment': {'host': platform.node(), 'cuda_visible_devices': ''}}
    try:
        for seed in args.seeds:
            before = time.perf_counter()
            directory = args.tracking_root / f'seed{seed}'
            torch, np, nib, source, inputs, payload, arrays, fa, atlases = worker.preflight(config, seed, directory)
            worker.require(not torch.cuda.is_initialized(), 'CPU input checks initialized CUDA')
            target = args.output_dir / f'seed-{seed}'; target.mkdir()
            tracks = [arrays['points'][int(a):int(b)] for a, b in zip(arrays['offsets'][:-1], arrays['offsets'][1:])]
            output = target / 'tracks.tck'
            nib.streamlines.save(nib.streamlines.Tractogram(tracks, affine_to_rasmm=np.eye(4)), output)
            actual = list(nib.streamlines.load(output, lazy_load=False).streamlines)
            worker.require(len(actual) == len(tracks) and all(a.dtype == b.dtype == np.float32 and
                np.array_equal(a.view(np.uint8), b.view(np.uint8)) for a, b in zip(tracks, actual)),
                'CPU diagnostic TCK changed original point dtype/track order/point bits')
            worker.require(all(worker.sha256(directory / f'{name}.npy') == digest
                for name, digest in inputs['tracking_array_sha256'].items()) and
                worker.sha256(directory / 'report.json') == inputs['tracking_report_sha256'],
                'original tracking files changed during CPU serialization')
            report['cases'][str(seed)] = {'source': source, 'inputs': inputs,
                'tck': {'path': str(output.resolve()), 'size_bytes': output.stat().st_size, 'sha256': worker.sha256(output)},
                'readback': {'dtype': 'float32', 'original_point_bits_equal': True, 'offset_order_preserved': True,
                             'tracks': len(tracks), 'points': sum(len(x) for x in actual)},
                'cpu_seconds': time.perf_counter() - before, 'gpu_postprocessing': 'not_assessed'}
            del payload, arrays, fa, atlases, tracks, actual
        worker.require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA')
        report['state'] = 'actual_existing_tracks_cpu_tck_readback_completed'
        report['cuda_initialized'] = False
    except Exception as error:
        report['state'] = 'failed'
        report['error'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        report['total_cpu_seconds'] = time.perf_counter() - start
        worker.atomic_json(args.output_dir / 'preparation_report.json', report)
    print(json.dumps({'state': report['state'], 'seeds': args.seeds, 'cpu_seconds': report['total_cpu_seconds']}))


if __name__ == '__main__':
    main()
