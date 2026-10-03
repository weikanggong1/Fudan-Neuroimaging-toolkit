"""CPU-only standby dispatcher; no modeling namespace before verified completed input."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from official_modeling_cohort_cpu_v3 import cpu_activation_preflight, sha, save, utc


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--state-root',required=True)
    args=p.parse_args();config_path=Path(args.config).resolve();config=json.loads(config_path.read_text())
    state_root=Path(args.state_root);state_root.mkdir(parents=True,exist_ok=False)
    tools=[config_path,Path(__file__),Path(__file__).with_name('official_modeling_cohort_cpu_v3.py'),Path(__file__).with_name('launch_official_modeling_cpu_v3.py')]
    pinned={str(path):sha(path) for path in tools}
    record={'state':'waiting_first_true_verified_CPU_upstream','actual_host':socket.gethostname(),'PID':os.getpid(),
            'started_utc':utc(),'source_sha256':pinned,'configuration':config,'models_launched':False,'GPU':False}
    save(state_root/'freeze.json',record)
    while True:
        try:
            if any(sha(path)!=digest for path,digest in pinned.items()):raise ValueError('Standby dispatcher frozen source changed')
            proof=cpu_activation_preflight(config)
        except ValueError as error:
            if str(error).startswith('No actual completed CPU contract'):
                record.update(state='waiting_first_true_verified_CPU_upstream',last_check_utc=utc(),reason=str(error));save(state_root/'status.json',record);time.sleep(30);continue
            record.update(state='failed_guard',last_check_utc=utc(),error=repr(error));save(state_root/'status.json',record);raise
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
        command=[sys.executable,str(Path(__file__).with_name('launch_official_modeling_cpu_v3.py')),'--config',str(config_path)]
        result=subprocess.run(command,env=env,capture_output=True,text=True)
        record.update(state='modeling_launched' if result.returncode==0 else 'failed_launch',activation_proof=proof,
                      completed_utc=utc(),models_launched=result.returncode==0,command=command,returncode=result.returncode,
                      stdout=result.stdout,stderr=result.stderr)
        save(state_root/'status.json',record)
        print(json.dumps(record))
        if result.returncode:raise RuntimeError('Official CPU modeling launch failed; inspect status')
        return


if __name__=='__main__':main()
