"""冻结任务5固定TCK组件controller；共享锁内串行GPU，不占等待SSH通道。"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--python',required=True)
    args=p.parse_args();root=args.root.resolve()
    status=root/'cuda_controller_v1';status.mkdir(exist_ok=False)
    worker=root/'current/validation/connectome/accuracy_20261003/task_05/compare_common_matrix_reference.py'
    helper=worker.with_name('benchmark_assignment_weights.py')
    source=root/'current/src/fnit/connectome/assignment.py'
    reference=root/'common_matrix_reference_cpu_v1/report.json'
    files={str(path):sha(path) for path in [worker,helper,source,reference]}
    command=['flock','/tmp/fnit-recon-five-20261002-gongwk.gpu.lock','env',
        'CUDA_VISIBLE_DEVICES=GPU-e25cac06-0ce8-a833-abf9-09ab18c9c9ba',
        'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True','OMP_NUM_THREADS=8','MKL_NUM_THREADS=8','OPENBLAS_NUM_THREADS=8',
        args.python,str(worker),'--source',str(source),'--reference',str(reference),'--device','cuda:0',
        '--output',str(root/'current_common_matrix_cuda_v1')]
    record={'host':socket.gethostname(),'controller_pid':os.getpid(),'controller_sha256':sha(__file__),
        'command':command,'frozen_sources':files,'state':'queued_or_running','started_unix':time.time()}
    (status/'status.json').write_text(json.dumps(record,indent=2)+'\n')
    with open(status/'worker.log','w') as log:process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    if any(sha(path)!=expected for path,expected in files.items()):raise RuntimeError('frozen source changed')
    record.update(state='completed' if process.returncode==0 else 'failed',returncode=process.returncode,
                  ended_unix=time.time(),worker_log_sha256=sha(status/'worker.log'))
    (status/'status.json').write_text(json.dumps(record,indent=2)+'\n')
    raise SystemExit(process.returncode)


if __name__=='__main__':main()
