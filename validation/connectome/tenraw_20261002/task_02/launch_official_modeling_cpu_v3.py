"""Standby launcher: create a fresh CPU modeling freeze only after real CPU readiness."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
from official_modeling_cohort_cpu_v3 import cpu_activation_preflight, sha, save, utc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    proof = cpu_activation_preflight(config)  # no output or launch before actual readiness
    out = Path(config['output_root'])
    if out.exists():
        raise ValueError('Fresh CPU modeling namespace required; preserve any previous attempt')
    previous = Path(config['previous_modeling_root'])
    retired = {'previous_namespace':str(previous),'retired_waiter':False}
    if (previous/'cohort_status.json').is_file():
        status = json.loads((previous/'cohort_status.json').read_text())
        retired['previous_status_sha256'] = sha(previous/'cohort_status.json')
        selected = {case['subject'] for case in config['subjects'] if case.get('binding_ready') is True}
        if any(case['state']=='running' or (subject in selected and case['state']=='completed') for subject,case in status['subjects'].items()):
            raise ValueError('Previous modeling running or selected case completed; refuse duplicate/rebinding')
        if status['actual_host'] != socket.gethostname():
            raise ValueError('Retire own old waiter on its actual host before CPU activation')
        pid = int((previous/'coordinator.pid').read_text())
        cmdline = Path(f'/proc/{pid}/cmdline')
        if cmdline.is_file():
            if b'official_modeling_cohort_v2.py' not in cmdline.read_bytes():
                raise ValueError('Old PID no longer belongs to this modeling waiter')
            os.kill(pid, signal.SIGTERM)
            retired.update(retired_waiter=True, retired_PID=pid)
    out.mkdir(parents=True)
    tool = Path(__file__).with_name('official_modeling_cohort_cpu_v3.py')
    command = [sys.executable,'-u',str(tool),'--config',str(config_path)]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MKL_NUM_THREADS='8')
    freeze = {'actual_host':socket.gethostname(),'created_utc':utc(),'GPU':False,
              'upstream_EDDY_solver':'cpu','configuration':config,'config_sha256':sha(config_path),
              'harness_sha256':sha(tool),'launcher_sha256':sha(__file__),
              'CPU_activation_proof':proof,'old_waiter':retired,'command':command,
              'CPU_threads_per_subject':8,'CPU_concurrency':1}
    save(out/'freeze.json',freeze)
    with (out/'coordinator.log').open('w') as log:
        process = subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL,start_new_session=True)
    (out/'coordinator.pid').write_text(str(process.pid)+'\n')
    print(json.dumps({'PID':process.pid,'host':socket.gethostname(),'output_root':str(out),'GPU':False}))


if __name__ == '__main__':
    main()
