"""CPU 控制器：等待明确指定的独立官方 DWI/anatomy 合同，再逐例执行官方五重复。"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-manifest', type=Path, required=True)
    parser.add_argument('--raw-manifest-sha256', required=True)
    parser.add_argument('--official-dwi-root', type=Path, required=True)
    parser.add_argument('--official-anatomy-root', type=Path, required=True)
    parser.add_argument('--verified-reference-manifest', type=Path, required=True)
    parser.add_argument('--verified-reference-manifest-sha256', required=True)
    parser.add_argument('--mrtrix-bin', type=Path, required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--case-ids', nargs='+')
    parser.add_argument('--n-seeds', type=int, default=100000)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument('--poll-seconds', type=float, default=15.)
    args = parser.parse_args(argv)
    if sha256(args.raw_manifest) != args.raw_manifest_sha256 or args.output_root.exists():
        parser.error('fixed original raw manifest and fresh output root required')
    source = Path(__file__).with_name('benchmark_connectome_raw_official.py')
    if not source.is_file() or args.poll_seconds < 1 or args.n_seeds < 1 or len(args.seeds) < 3 or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error('existing worker, positive poll/count, and >=3 unique seeds required')
    manifest = json.loads(args.raw_manifest.read_text())
    cases = args.case_ids or [case['case_id'] for case in manifest['cases']]
    if len(set(cases)) != len(cases) or not set(cases).issubset({case['case_id'] for case in manifest['cases']}):
        parser.error('unique exact canonical case IDs required')
    args.output_root.mkdir(parents=True, exist_ok=False)
    state = {'state': 'waiting_official_contracts', 'execution_completed': False,
        'scope': 'independent official wholechain tracking/downstream CPU only; no production/GPU/extra recon-all',
        'raw_manifest_sha256': args.raw_manifest_sha256, 'dataset': manifest['dataset'], 'snapshot': manifest['snapshot'],
        'official_dwi_root': str(args.official_dwi_root.resolve()),
        'official_anatomy_root': str(args.official_anatomy_root.resolve()),
        'workers': 1, 'downstream_threads': 8, 'tracking_threads': 0,
        'controller_pid': os.getpid(), 'controller_sha256': sha256(__file__), 'worker_sha256': sha256(source),
        'helper_sha256': sha256(source.with_name('benchmark_connectome_repeats_official.py')),
        'matrix_helper_sha256': sha256(source.parents[1] / 'connectome_repeat_common.py'),
        'verified_reference_manifest_sha256': args.verified_reference_manifest_sha256,
        'cases': {case: {'state': 'waiting'} for case in cases}}
    report_path = args.output_root / 'cohort_status.json'
    try:
        while True:
            dispatched = False
            for case in cases:
                if state['cases'][case]['state'] in ('completed', 'failed'):
                    continue
                dwi = args.official_dwi_root / case / 'consumer_contract.json'
                anatomy = args.official_anatomy_root / case / 'consumer_contract.json'
                if not dwi.is_file() or not anatomy.is_file():
                    continue
                # Atomic producer contracts appear only after real completion.
                records = [json.loads(path.read_text()) for path in (dwi, anatomy)]
                if any(item.get('state') != 'completed' for item in records):
                    continue
                if any(item.get('case_id') != case for item in records):
                    raise ValueError('producer contract case identity differs from explicit path')
                if sha256(source) != state['worker_sha256'] or sha256(args.raw_manifest) != args.raw_manifest_sha256:
                    raise ValueError('frozen worker/raw manifest changed while waiting')
                if sha256(source.with_name('benchmark_connectome_repeats_official.py')) != state['helper_sha256'] or sha256(source.parents[1] / 'connectome_repeat_common.py') != state['matrix_helper_sha256']:
                    raise ValueError('verified command/matrix helper changed while waiting')
                command = [args.python, str(source), '--anatomy-contract', str(anatomy), '--dwi-contract', str(dwi),
                    '--raw-manifest', str(args.raw_manifest), '--raw-manifest-sha256', args.raw_manifest_sha256,
                    '--verified-reference-manifest', str(args.verified_reference_manifest),
                    '--verified-reference-manifest-sha256', args.verified_reference_manifest_sha256,
                    '--case-id', case, '--mrtrix-bin', str(args.mrtrix_bin),
                    '--output-dir', str(args.output_root / case), '--n-seeds', str(args.n_seeds),
                    '--downstream-threads', '8', '--seeds', *map(str, args.seeds)]
                row = state['cases'][case]
                row.update(state='running', argv=command, dwi_contract_sha256=sha256(dwi),
                           anatomy_contract_sha256=sha256(anatomy))
                state['state'] = 'running'
                atomic(report_path, state)
                env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1',
                       'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8'}
                started = time.perf_counter()
                with (args.output_root / (case + '.log')).open('x') as stream:
                    child = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
                    row['pid'] = child.pid; atomic(report_path, state)
                    row['returncode'] = child.wait()
                row['worker_wall_seconds'] = time.perf_counter() - started
                result_path = args.output_root / case / 'reference_manifest.json'
                result = json.loads(result_path.read_text()) if result_path.is_file() else {}
                row['state'] = 'completed' if row['returncode'] == 0 and result.get('execution_completed') is True else 'failed'
                if row['state'] == 'completed':
                    row['reference_manifest_sha256'] = sha256(result_path)
                else:
                    row['reason'] = 'actual worker failed; original log/report retained, no fallback to old producer directories'
                state['utc'] = datetime.now(timezone.utc).isoformat(); atomic(report_path, state)
                dispatched = True
                break
            if all(item['state'] in ('completed', 'failed') for item in state['cases'].values()):
                state['state'] = 'completed' if all(item['state'] == 'completed' for item in state['cases'].values()) else 'failed'
                state['execution_completed'] = state['state'] == 'completed'
                break
            if not dispatched:
                state['state'] = 'waiting_official_contracts'
                state['utc'] = datetime.now(timezone.utc).isoformat(); atomic(report_path, state)
                time.sleep(args.poll_seconds)
    except Exception as error:
        state['state'] = 'failed'
        state['error'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        state['utc'] = datetime.now(timezone.utc).isoformat(); atomic(report_path, state)


if __name__ == '__main__':
    main()
