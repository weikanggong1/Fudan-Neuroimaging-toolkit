"""One new score-only arm using the exact original frozen scientific worker.

Original controller/failed affine receipt/source/output bytes stay unchanged.
There is no registration function or source/header modification here.
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


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def check(bindings):
    result = {}
    for key, item in bindings.items():
        p = Path(item['path'])
        value = {'bytes': p.stat().st_size, 'sha256': digest(p)}
        if value != {n:item[n] for n in ('bytes','sha256')}:
            raise ValueError('saved source/output/reference changed: '+key)
        result[key] = value
    return result


def save(path, value):
    data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode()
    fd = os.open(path, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--plan', required=True)
    args.add_argument('--approved-recovery-sha', required=True)
    a = args.parse_args()
    if digest(a.plan) != a.approved_recovery_sha:
        raise ValueError('explicit frozen recovery plan SHA differs')
    plan = json.loads(Path(a.plan).read_text())
    run = Path(plan['recovery_directory'])
    if not run.is_dir() or (run/'controller.private.json').exists():
        raise ValueError('new prepared recovery directory required')
    source_plan = json.loads(Path(plan['old_source_plan']).read_text())
    if Path(plan['expected_score_report']).parent.exists():
        raise ValueError('the original score arm already started')
    os.umask(0o077)
    env = os.environ.copy()
    for key in ('LD_LIBRARY_PATH','LD_PRELOAD','PYTHONPATH','OPENBLAS_CORETYPE',
                'FS_SetVoxToRasXform_Change_VoxSize'):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8',
               NUMEXPR_NUM_THREADS='8')
    report = {'status':'waiting', 'PID':os.getpid(), 'PGID':os.getpgrp(),
              'UID':os.getuid(), 'recovery_PLAN_sha256':a.approved_recovery_sha,
              'original_PLAN_sha256':plan['old_plan_sha256'],
              'scope':'one saved-output score; zero API/official/GEMS/GPU',
              'new_registration_API_calls':0, 'new_official_commands':0,
              'original_controller_not_relabelled':True}
    started = time.monotonic(); code = 1
    try:
        with Path(plan['common_CPU_lock']).open('a+') as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB); break
                except BlockingIOError:
                    if time.monotonic()-started >= 120:
                        raise TimeoutError('120s common-lock deadline; zero science')
                    time.sleep(.1)
            report['lock_wait_seconds'] = time.monotonic()-started
            report['bindings_before'] = check(plan['bindings'])
            report['original_518bindings_before'] = check(source_plan['bindings'])
            command = [plan['python'],plan['original_worker'],'--plan',plan['old_source_plan'],
                       '--approved-plan-sha',plan['old_plan_sha256'],'--phase','score']
            with (run/'score.stdout.private.txt').open('xb') as out, \
                 (run/'score.stderr.private.txt').open('xb') as err:
                child = subprocess.Popen(command,env=env,stdout=out,stderr=err,
                                         start_new_session=True)
                report['child_PID'] = child.pid
                worker_started = time.monotonic()
                try:
                    code = child.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=2)
                    code = 124
                report['score_child_wall_seconds'] = time.monotonic()-worker_started
            report['status'] = 'completed' if code==0 else 'score_failed_or_gates_failed'
            report['bindings_after'] = check(plan['bindings'])
            report['original_518bindings_after'] = check(source_plan['bindings'])
            result = Path(plan['expected_score_report'])
            if result.exists():
                report['saved_score_report'] = {'path':str(result),'bytes':result.stat().st_size,
                                               'sha256':digest(result)}
    except BaseException as error:
        report['status'] = 'failed';code = 1
        report['exception'] = {'type':type(error).__name__,'message':str(error)}
        (run/'exception.private.txt').write_text(traceback.format_exc())
    finally:
        report['wall_seconds'] = time.monotonic()-started
        report['returncode'] = code
        save(run/'controller.private.json',report)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
