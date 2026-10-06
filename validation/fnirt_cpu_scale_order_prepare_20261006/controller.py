"""One CPU8 scalar controller. Metadata and source files alone are prepared."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import resource
import signal
import stat
import subprocess
import sys
import time

from contracts import bound, check_bindings, check_freeze, git_head, require, write_json


def pid_record(pid):
    try:
        text = Path('/proc', str(pid), 'stat').read_text()
    except FileNotFoundError:
        return {'pid': int(pid), 'process_group': int(pid), 'start_ticks': None, 'already_exited_before_proc_read': True}
    fields = text[text.rfind(')') + 2:].split()
    return {'pid': int(pid), 'process_group': int(fields[2]), 'start_ticks': int(fields[19])}


def terminate_owned(child, grace_seconds=5):
    """Never raise during cleanup; preserve every signal/wait failure."""
    record = {'child_present':child is not None,'SIGTERM_sent':False,'SIGKILL_sent':False,
              'returncode':None,'still_running':False,'errors':[]}
    if child is None:
        return record
    def error(stage, value):
        record['errors'].append({'stage':stage,'type':type(value).__name__,'message':str(value)})
    def poll():
        try:
            result = child.poll()
            if result is not None:
                record['returncode'] = int(result)
            return result
        except BaseException as value:
            error('poll',value)
            return None
    if poll() is not None:
        return record
    if grace_seconds > 0:
        try:
            os.killpg(child.pid,signal.SIGTERM)
            record['SIGTERM_sent'] = True
        except ProcessLookupError:
            pass
        except BaseException as value:
            error('SIGTERM',value)
        try:
            record['returncode'] = int(child.wait(timeout=grace_seconds))
            return record
        except subprocess.TimeoutExpired:
            pass
        except BaseException as value:
            error('grace_wait',value)
    # poll/wait owns the unreaped child identity; only its created process group
    # can be signalled. A concurrent natural exit yields ESRCH, which is benign.
    if poll() is None:
        try:
            os.killpg(child.pid,signal.SIGKILL)
            record['SIGKILL_sent'] = True
        except ProcessLookupError:
            pass
        except BaseException as value:
            error('SIGKILL',value)
    try:
        record['returncode'] = int(child.wait(timeout=2))
    except BaseException as value:
        error('final_wait',value)
    record['still_running'] = poll() is None
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--canonical-main-commit', required=True)
    parser.add_argument('--mode', choices=['enqueue', 'controller'], required=True)
    parser.add_argument('--approved-scale-control', action='store_true')
    args = parser.parse_args()
    require(args.approved_scale_control, 'root-approved finite execution required')
    invocation_started = time.monotonic()
    def invocation_deadline(signum, frame):
        raise TimeoutError('finite controller invocation deadline or termination')
    signal.signal(signal.SIGALRM, invocation_deadline)
    signal.signal(signal.SIGTERM, invocation_deadline)
    signal.setitimer(signal.ITIMER_REAL, 300)
    os.umask(0o077)
    expected = json.loads((args.workspace / 'EXPECTED.public.json').read_text())
    plan = json.loads((args.workspace / 'PLAN.public.json').read_text())
    wanted = args.root / 'runs' / plan['canonical_leaf_proposed']
    workspace_wanted = args.root / 'workspaces' / plan['canonical_leaf_proposed']
    require(args.run.resolve() == wanted.resolve() and args.workspace.resolve() == workspace_wanted.resolve(), 'wrong unique canonical leaf')
    before, failures = check_bindings(args.root, expected)
    frozen, bad = check_freeze(args.workspace)
    require(not failures and not bad, 'source/input/freeze preflight failed')
    require(git_head(args.root/'repo') == args.canonical_main_commit, 'canonical commit preflight failed')
    run_stat = args.run.stat()
    require(stat.S_ISDIR(run_stat.st_mode) and run_stat.st_uid == os.getuid()
            and stat.S_IMODE(run_stat.st_mode) == 0o700, 'root must register an owned700 run before enqueue')
    freeze_record = bound(args.workspace/'freeze.public.json')
    if args.mode == 'enqueue':
        marker = args.run/'enqueue.once.private.json'
        fd = os.open(marker, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'w') as stream:
            json.dump({'status':'unique_enqueue_claim_no_retry','freeze':freeze_record,
                       'canonical_main_commit':args.canonical_main_commit}, stream)
            stream.write('\n');stream.flush();os.fsync(stream.fileno())
        command = [sys.executable, '-B', str(args.workspace/'controller.py'), '--root', str(args.root),
                   '--workspace', str(args.workspace), '--run', str(args.run), '--canonical-main-commit', args.canonical_main_commit,
                   '--mode', 'controller', '--approved-scale-control']
        with (args.run/'controller.log').open('xb') as log:
            controller = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        launch = {'status':'one_controller_enqueued','controller':pid_record(controller.pid),
                  'freeze':freeze_record,'command':command,'controller_seconds':300,'worker_seconds':60,
                  'CPU8_lock_wait_seconds':120,'canonical_main_commit':args.canonical_main_commit,
                  'numeric_worker_started_at_enqueue':'unobserved','retry':False}
        write_json(args.run/'launch.private.json',launch)
        print(json.dumps(launch),flush=True)
        return 0
    marker = json.loads((args.run/'enqueue.once.private.json').read_text())
    require(marker['freeze'] == freeze_record and marker['canonical_main_commit'] == args.canonical_main_commit, 'own enqueue claim changed')
    started = invocation_started
    report = {'status':'started','controller':pid_record(os.getpid()),'freeze':freeze_record,
              'worker_started':False,'worker_returncode':None,'exit_code':1,'retry':False,
              'unreaped_child_retains_lock':False,
              'bindings_before':before,'harness_before':frozen,'source_inputs_unchanged':False,'postcheck_errors':[]}
    deadline = started + 300
    lock_fd, child, code = None, None, 1
    def stopped(signum, frame):
        raise TimeoutError('finite controller deadline or termination')
    signal.signal(signal.SIGALRM,stopped)
    signal.signal(signal.SIGTERM,stopped)
    signal.setitimer(signal.ITIMER_REAL,max(.1,deadline-time.monotonic()))
    try:
        lock_path = args.root/'runs/smri_cpu_20261004/nodecw7.gems.cpu8.lock'
        lock_fd = os.open(lock_path,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
        lock_stat = os.fstat(lock_fd)
        require(stat.S_ISREG(lock_stat.st_mode) and lock_stat.st_uid == os.getuid(),'CPU8 lock must be owned regular file')
        lock_deadline = min(deadline,time.monotonic()+120)
        while True:
            try:
                fcntl.flock(lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= lock_deadline:
                    raise TimeoutError('CPU8 lock deadline; zero numeric worker')
                time.sleep(.1)
        report['lock_wait_seconds'] = float(time.monotonic()-started)
        after_lock, failures = check_bindings(args.root,expected)
        after_lock_frozen, bad = check_freeze(args.workspace)
        require(after_lock == before and after_lock_frozen == frozen and not failures and not bad,'source/input changed while queued')
        require(git_head(args.root/'repo') == args.canonical_main_commit,'canonical commit changed while queued')
        env = dict(os.environ)
        for name in ('PYTHONPATH','LD_LIBRARY_PATH','LD_PRELOAD'):
            env.pop(name,None)
        env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',
                   NUMBA_NUM_THREADS='8',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
        def child_limits():
            os.sched_setaffinity(0,{32,36,40,44,48,52,56,60})
            resource.setrlimit(resource.RLIMIT_AS,(8_000_000_000,8_000_000_000))
        command = [sys.executable,'-B',str(args.workspace/'scalar_control.py'),'--root',str(args.root),
                   '--workspace',str(args.workspace),'--output',str(args.run/'scalar'),
                   '--canonical-main-commit',args.canonical_main_commit,'--approved-scale-control']
        with (args.run/'scalar.log').open('xb') as log:
            child = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,preexec_fn=child_limits,start_new_session=True,pass_fds=(lock_fd,))
            report['worker_started'] = True
            report['worker'],report['command'] = pid_record(child.pid),command
            write_json(args.run/'worker_launch.private.json',report)
            try:
                code = int(child.wait(timeout=min(60,max(.1,deadline-time.monotonic()))))
            except subprocess.TimeoutExpired:
                report['timeout_cleanup'] = terminate_owned(child)
                code = 124
                report['worker_timeout'] = True
        report['worker_returncode'] = code
        report['status'] = 'completed' if code == 0 else 'failed_stopped_no_retry'
    except Exception as error:
        report['exception_cleanup'] = terminate_owned(child, grace_seconds=0)
        report['status'] = 'failed_stopped_no_retry'
        report['exception'] = {'type':type(error).__name__,'message':str(error)}
        code = 124 if isinstance(error,TimeoutError) else 1
    finally:
        report['final_cleanup'] = terminate_owned(child, grace_seconds=0)
        report['unreaped_child_retains_lock'] = bool(lock_fd is not None and child is not None
                                                   and report['final_cleanup']['still_running'])
        report['final_cleanup']['unreaped_child_retains_lock'] = report['unreaped_child_retains_lock']
        if report['final_cleanup']['still_running'] or report['final_cleanup']['errors']:
            code = code or 1
            report['status'] = 'failed_cleanup_stopped_no_retry'
        try:
            write_json(args.run/'cleanup.private.json', report['final_cleanup'])
        except Exception as error:
            report['postcheck_errors'].append({'type':type(error).__name__,'message':str(error),'stage':'cleanup_receipt'})
            code = code or 1
        try:
            after, failures = check_bindings(args.root,expected)
            frozen_after, bad = check_freeze(args.workspace)
            report['bindings_after'],report['harness_after'] = after,frozen_after
            report['source_inputs_unchanged'] = before == after and frozen == frozen_after and not failures and not bad
            require(report['source_inputs_unchanged'],'source/input/freeze changed after finite child')
            require(git_head(args.root/'repo') == args.canonical_main_commit,'canonical commit changed after child')
        except Exception as error:
            report['postcheck_errors'].append({'type':type(error).__name__,'message':str(error)})
            code = code or 1
        if lock_fd is not None:
            try:
                os.close(lock_fd)
            except Exception as error:
                report['postcheck_errors'].append({'type':type(error).__name__,'message':str(error),'stage':'lock_close'})
                code = code or 1
        signal.setitimer(signal.ITIMER_REAL,0)
        report['exit_code'],report['controller_seconds'] = int(code),float(time.monotonic()-started)
        write_json(args.run/'controller.private.json',report)
        (args.run/'controller.exitcode').write_text(str(code)+'\n')
    print(json.dumps({'status':report['status'],'worker_started':report['worker_started'],'exit_code':code}),flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
