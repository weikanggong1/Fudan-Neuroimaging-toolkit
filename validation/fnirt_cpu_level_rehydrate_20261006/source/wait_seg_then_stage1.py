"""Finite scheduling gate only; wait for the already queued Seg controller."""
import argparse
import json
import os
from pathlib import Path
import time

from stage1_io import bound, check_freeze, write_json


def process_identity(pid):
    try:
        text = (Path('/proc') / str(pid) / 'stat').read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    # Field22 is starttime; comm may contain spaces or parentheses.
    fields = text.rsplit(')', 1)[1].split()
    return None if fields[0] == 'Z' else int(fields[19])


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--seg-controller', type=int, required=True)
    parser.add_argument('--seg-start-ticks', type=int, required=True)
    args = parser.parse_args()
    freeze, failures = check_freeze(args.workspace)
    if failures:
        raise RuntimeError('frozen stage1 source mismatch')
    if (args.run / 'stage1').exists() or (args.run / 'preflight_before.public.json').exists():
        raise RuntimeError('scientific worker already started; no scheduling repeat')
    report = {'phase': 'waiting_for_preexisting_Seg_controller', 'scientific_worker_started': False,
              'seg_controller_pid': args.seg_controller, 'seg_start_ticks': args.seg_start_ticks,
              'seg_terminal_exit_code_known': False, 'pid_reuse_is_not_followed': True,
              'parent_controller_timeout_seconds': 19600, 'started_unix': time.time(),
              'scheduler_source': bound(__file__),
              'freeze_sha256': bound(args.workspace / 'freeze.public.json')['sha256'],
              'frozen_harness_bindings': freeze,
              'scope': 'Scheduling only; no numerical imports/computation and no control of another process.'}
    write_json(args.run / 'controller_phase.public.json', report)
    print('Waiting for the preexisting Seg controller; no scientific worker or CPU lock acquired.', flush=True)
    while process_identity(args.seg_controller) == args.seg_start_ticks:
        time.sleep(5)
    report['phase'] = 'Seg_controller_finished_then_waiting_common_CPU_lock'
    report['prerequisite_finished_observed_unix'] = time.time()
    write_json(args.run / 'controller_phase.public.json', report)
    print('Preexisting Seg controller finished; enter the frozen shared-lock stage1 runner once.', flush=True)
    os.execv('/bin/bash', ['/bin/bash', str(args.workspace / 'run_stage1.sh'), 'stage1-authorized'])


if __name__ == '__main__':
    main()
