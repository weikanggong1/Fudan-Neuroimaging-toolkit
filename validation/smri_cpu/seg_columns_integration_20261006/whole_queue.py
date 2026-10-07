"""Prepared CPU ABBA or GPU AB; no execution before whole-stage approval."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import signal
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def own_gpu_sample(pid, uuid):
    # Only return this worker tree's aggregate; never store other process identities.
    pending, own = [pid], set()
    while pending and len(own) < 256:
        current = pending.pop()
        if current in own:
            continue
        own.add(current)
        try:
            pending.extend(map(int, Path('/proc/' + str(current) + '/task/' + str(current) + '/children').read_text().split()))
        except (OSError, ValueError):
            pass
    result = {'monotonic': time.monotonic(), 'status': 'unavailable', 'own_process_bytes': None,
              'own_process_count': None}
    try:
        output = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
        if output.returncode:
            result['status'] = 'command_failed'
        else:
            memory = []
            for line in output.stdout.splitlines():
                fields = [value.strip() for value in line.split(',')]
                if len(fields) == 3 and fields[0].lower() == uuid.lower() and int(fields[1]) in own:
                    memory.append(int(fields[2]) * 1024**2)
            result.update(status='ok', own_process_bytes=sum(memory), own_process_count=len(memory))
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'workspace', 'baseline', 'candidate', 'plan', 'run'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), required=True)
    parser.add_argument('--approved-whole', action='store_true')
    args = parser.parse_args()
    assert args.approved_whole, 'prepared only; separate coordinator approval required'
    os.umask(0o077)
    plan = json.loads(args.plan.read_text())
    assert sha(__file__) == plan['validation_sources']['whole_queue.py']
    worker = args.workspace / 'whole_worker.py'
    assert sha(worker) == plan['validation_sources']['whole_worker.py']
    cpu = args.device == 'cpu'
    group = plan['whole_CPU'] if cpu else plan['whole_GPU']
    assert group['status'] == 'prepared_requires_separate_approval'
    args.run.mkdir(mode=0o700)
    cache = args.run / 'fresh_whole_cache'
    assert not cache.exists(), 'cold cache is a new private path; keep contract cache intact'
    environment = {**os.environ, 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8',
        'OPENBLAS_NUM_THREADS': '8', 'NUMBA_NUM_THREADS': '8', 'PYTHONDONTWRITEBYTECODE': '1',
        'CUDA_VISIBLE_DEVICES': '' if cpu else '1', 'FNIT_SYNTHSEG_CPU_CACHE': str(cache)}
    if cpu:
        environment['CXX'] = plan['Conda_CXX']
    for name in ('PYTHONPATH', 'LD_PRELOAD', 'LD_LIBRARY_PATH', 'OPENBLAS_CORETYPE'):
        environment.pop(name, None)
    receipt = {'schema': 'fnit_columns_complete33_queue/v1', 'status': 'waiting_lock',
        'PLAN_sha256': sha(args.plan), 'queue_sha256': sha(__file__), 'worker_sha256': sha(worker),
        'device': args.device, 'started_monotonic': time.monotonic(), 'jobs': [],
        'deadline_seconds': 23000, 'lock_scope': 'one fresh arm; release between arms'}
    record = args.run / 'QUEUE.json'
    def save():
        record.write_text(json.dumps(receipt, indent=2) + '\n')
    deadline = time.monotonic() + 23000
    for name, arm in group['arm_order']:
        receipt.update(status='waiting_lock', next_arm=name)
        save()
        with (args.root / group['common_lock']).open('a+b') as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() + group['worker_timeout_seconds'] + 30 >= deadline:
                        receipt.update(status='deadline_before_arm'); save(); return 1
                    time.sleep(5)
            assert time.monotonic() + group['worker_timeout_seconds'] + 30 < deadline
            if cpu and name == 'B1_cold':
                assert not cache.exists(), 'short-contract cache must be separate'
            if cpu and name == 'B2_warm':
                assert cache.is_dir(), 'warm uses B1 successful cache'
            source = args.baseline if arm == 'baseline' else args.candidate
            command = ['/usr/bin/taskset', '-c', ','.join(map(str, group['affinity'])),
                       '/usr/bin/timeout', '--kill-after=30', str(group['worker_timeout_seconds']),
                       str(args.root / 'envs/default/bin/python'), str(worker), '--root', str(args.root),
                       '--source', str(source), '--plan', str(args.plan), '--output', str(args.run / name),
                       '--arm', arm, '--device', args.device, '--approved-whole']
            receipt.update(status='running', next_arm=name); save()
            tick = time.perf_counter()
            with (args.run / (name + '.log')).open('x') as log:
                child = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, start_new_session=True)
                samples = []
                try:
                    if cpu:
                        rc = child.wait(timeout=group['worker_timeout_seconds'] + 35)
                    else:
                        while child.poll() is None:
                            if time.perf_counter() - tick > group['worker_timeout_seconds'] + 35:
                                raise subprocess.TimeoutExpired(command, group['worker_timeout_seconds'] + 35)
                            samples.append(own_gpu_sample(child.pid, plan['GPU_uuid']))
                            time.sleep(.5)
                        rc = child.returncode
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
                    rc = 124
            receipt['jobs'].append({'name': name, 'arm': arm, 'returncode': rc,
                                    'cold_process_seconds': time.perf_counter() - tick})
            if not cpu:
                observed = [sample for sample in samples if sample.get('own_process_count')]
                receipt['jobs'][-1]['gpu_sampling'] = {
                    'sample_count': len(samples), 'nonzero_own_samples': len(observed),
                    'failed_samples': sum(sample['status'] != 'ok' for sample in samples),
                    'requested_interval_seconds': .5,
                    'maximum_gap_seconds': max((second['monotonic'] - first['monotonic']
                        for first, second in zip(samples, samples[1:])), default=None),
                    'sampled_own_process_tree_peak_bytes': max((sample['own_process_bytes'] for sample in observed), default=None),
                    'status': ('observed_within_20e9' if observed and all(sample['own_process_bytes'] <= 20_000_000_000 for sample in observed)
                               else 'observed_exceeded_20e9' if observed else 'not_assessed'),
                    'limitation': 'Driver sampling is not an absolute peak; zero/unavailable cannot prove budget compliance.'}
            if rc:
                receipt.update(status='failed_stop_remaining'); save(); return rc
            try:
                observation_path = args.run / name / 'WHOLE.json'
                observation = json.loads(observation_path.read_text())
                receipt['jobs'][-1]['WHOLE_sha256'] = sha(observation_path)
                receipt['jobs'][-1]['saved_outputs'] = observation['saved_outputs']
                assert observation['valid_complete_arm'], 'worker after-gates failed'
                if cpu and arm == 'candidate':
                    assert observation['compile_calls'] == (1 if name == 'B1_cold' else 0), 'cold/warm compiler gate failed'
                if not cpu:
                    assert receipt['jobs'][-1]['gpu_sampling']['status'] == 'observed_within_20e9', 'own-GPU sampling gate failed or unassessed'
                reference_name = group['arm_order'][0][0]
                actual_outputs = {file: {'bytes': (args.run / name / file).stat().st_size,
                                        'sha256': sha(args.run / name / file)}
                                  for file in ('segmentation.nii.gz', 'volumes.csv')}
                assert actual_outputs == observation['saved_outputs'], 'saved output changed after worker'
                receipt['jobs'][-1]['comparison_executed'] = name != reference_name
                if name != reference_name:
                    reference = json.loads((args.run / reference_name / 'WHOLE.json').read_text())
                    reference_actual = {file: {'bytes': (args.run / reference_name / file).stat().st_size,
                                             'sha256': sha(args.run / reference_name / file)}
                                        for file in actual_outputs}
                    assert reference_actual == reference['saved_outputs'], 'reference changed'
                    assert actual_outputs == reference_actual, 'full same-name gzip/CSV SHA gate failed'
                    if not cpu:
                        assert observation['gpu'] == reference['gpu'], 'GPU UUID/allocated/reserved gate failed'
                    receipt['jobs'][-1]['reference_arm'] = reference_name
                receipt['jobs'][-1]['immediate_full_file_SHA_gate_passed'] = True
            except (OSError, ValueError, KeyError, TypeError, AssertionError) as error:
                receipt['jobs'][-1]['postcondition_error_type'] = type(error).__name__
                receipt['jobs'][-1]['postcondition_error'] = str(error)
                receipt.update(status='failed_postcondition_stop_remaining', failed_arm=name)
                save()
                return 1
            receipt.update(status='between_arms_lock_released'); save()
    receipt.update(status='complete', ended_monotonic=time.monotonic()); save()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
