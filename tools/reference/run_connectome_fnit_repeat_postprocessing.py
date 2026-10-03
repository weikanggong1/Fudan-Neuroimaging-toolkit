"""CPU 控制器：等待真实已有 seed 轨迹，再逐 seed 启动冻结单 GPU 后处理。"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--tracking-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--python', required=True)
    args = parser.parse_args()
    if sha256(args.config) != args.config_sha256:
        raise ValueError('frozen configuration changed')
    config = json.loads(args.config.read_text())
    worker = args.config.parent / 'tools/reference/benchmark_connectome_fnit_repeats.py'
    worker_sha = config['tool_files'][str(worker)]
    args.output_root.mkdir(parents=True, exist_ok=False)
    state = {'status': 'waiting_for_actual_tracking', 'controller_sha256': sha256(__file__),
             'config_sha256': args.config_sha256, 'worker_sha256': worker_sha,
             'gpu_uuid': config['gpu_uuid'], 'gpu_lock': config['gpu_lock'],
             'scope': 'existing actual tracks only; no extra tracking; per-seed shared GPU lock in worker',
             'seeds': {}, 'controller_pid': os.getpid()}
    path = args.output_root / 'controller.json'
    try:
        for seed in range(5):
            tracking = args.tracking_root / f'seed{seed}'
            state['seed'] = seed
            state['status'] = 'waiting_for_actual_tracking'
            while not all((tracking / name).is_file() for name in
                          ('report.json', 'points.npy', 'offsets.npy', 'endpoints.npy', 'lengths_mm.npy', 'accepted_seeds.npy')):
                state['utc'] = datetime.now(timezone.utc).isoformat()
                atomic(path, state)
                time.sleep(15)
            if sha256(args.config) != args.config_sha256 or sha256(worker) != worker_sha:
                raise ValueError('frozen config/worker changed')
            output = args.output_root / f'seed-{seed}'
            command = [args.python, str(worker), '--config', str(args.config), '--seed', str(seed),
                       '--tracking-dir', str(tracking), '--output-dir', str(output)]
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': config['gpu_uuid'],
                   'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'PYTHONDONTWRITEBYTECODE': '1',
                   'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8'}
            state['status'] = 'worker_dispatched_waiting_for_shared_lock_or_compute'
            state['seeds'][str(seed)] = {'argv': command, 'tracking_directory': str(tracking.resolve())}
            state['utc'] = datetime.now(timezone.utc).isoformat()
            atomic(path, state)
            with (args.output_root / f'seed-{seed}.log').open('w') as stream:
                child = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
                state['worker_pid'] = child.pid
                atomic(path, state)
                code = child.wait()
            state['seeds'][str(seed)]['returncode'] = code
            if code:
                raise RuntimeError(f'actual postprocessing seed{seed} failed with {code}; original outputs retained')
            report = json.loads((output / 'report.json').read_text())
            if report['status'] != 'completed' or report['memory_gate'] != 'passed':
                raise RuntimeError(f'seed{seed} actual completion/memory contract did not pass')
            state['seeds'][str(seed)].update({'status': report['status'], 'memory_gate': report['memory_gate'],
                'report_sha256': sha256(output / 'report.json')})
        state['status'] = 'five_seed_postprocessing_completed'
    except Exception as error:
        state['status'] = 'failed'
        state['error'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        state['utc'] = datetime.now(timezone.utc).isoformat()
        atomic(path, state)


if __name__ == '__main__':
    main()
