"""Prepared controller: single rigid/affine/score, CPU lock and hard bounds.

Does not dispatch itself. A caller must supply the reviewed frozen PLAN SHA.
No official command, second attempt, retry or GEMS invocation is possible.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode()
    fd = os.open(path, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--plan', required=True)
    args.add_argument('--approved-plan-sha', required=True)
    options = args.parse_args()
    if sha(options.plan) != options.approved_plan_sha:
        raise ValueError('explicit frozen PLAN SHA differs')
    plan = json.loads(Path(options.plan).read_text())
    run = Path(plan['run_directory'])
    if not run.is_dir() or (run/'controller.private.json').exists():
        raise ValueError('prepared new run with no prior controller required')
    os.umask(0o077)
    env = os.environ.copy()
    for key in ('LD_LIBRARY_PATH','LD_PRELOAD','PYTHONPATH','OPENBLAS_CORETYPE',
                'FS_SetVoxToRasXform_Change_VoxSize'):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8',
               NUMEXPR_NUM_THREADS='8')
    worker = Path(plan['workspace_directory'])/'run_real_worker.py'
    item = plan['bindings']['new:run_real_worker.py']
    if worker.stat().st_size != item['bytes'] or sha(worker) != item['sha256']:
        raise ValueError('frozen worker SHA differs')
    report = {'status': 'waiting', 'PID': os.getpid(), 'PGID': os.getpgrp(),
              'UID': os.getuid(), 'plan_sha256': options.approved_plan_sha,
              'scope': 'one new FNIT CPU rigid + own MGH reload + affine + score',
              'official_commands_executed': 0, 'GEMS': 0, 'GPU': 0,
              'phases': []}
    started = time.monotonic(); code = 1
    try:
        with Path(plan['common_CPU_lock']).open('a+') as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic()-started >= plan['limits']['lock_wait_seconds']:
                        raise TimeoutError('common CPU lock wait exhausted')
                    time.sleep(.1)
            report['lock_wait_seconds'] = time.monotonic()-started
            deadline = time.monotonic()+plan['limits']['after_lock_total_seconds']
            for phase in ('rigid','affine','score'):
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('declared 300s scientific total exhausted')
                seconds = min(plan['limits']['each_API_child_seconds'] if phase != 'score'
                              else plan['limits']['score_child_seconds'], remaining)
                command = [plan['python'], str(worker), '--plan', options.plan,
                           '--approved-plan-sha', options.approved_plan_sha,
                           '--phase', phase]
                arm_started = time.monotonic()
                with (run/(phase+'.stdout.private.txt')).open('xb') as out, \
                     (run/(phase+'.stderr.private.txt')).open('xb') as err:
                    child = subprocess.Popen(command, env=env, stdout=out, stderr=err,
                                             start_new_session=True)
                    timed_out = False
                    try:
                        rc = child.wait(timeout=seconds)
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=2)
                        rc = 124
                report['phases'].append({'phase': phase, 'PID': child.pid,
                    'wall_seconds': time.monotonic()-arm_started,
                    'timeout_seconds': seconds, 'timed_out': timed_out,
                    'returncode': rc})
                if rc:
                    code = rc; report['status'] = 'stopped_first_failure'
                    break
            else:
                code = 0; report['status'] = 'completed'
            report['locked_scope_seconds'] = (plan['limits']['after_lock_total_seconds']-
                                             (deadline-time.monotonic()))
    except BaseException as error:
        report['status'] = 'failed'
        report['exception'] = {'type': type(error).__name__, 'message': str(error)}
        (run/'controller.exception.private.txt').write_text(traceback.format_exc())
    finally:
        report['wall_seconds'] = time.monotonic()-started
        report['returncode'] = code
        save(run/'controller.private.json', report)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
