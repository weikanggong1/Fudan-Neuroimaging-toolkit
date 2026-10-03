"""Wait only for saved GPU reports, then audit/display them with CUDA hidden."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode=True
ROOT=Path('/cwStorage/home/gongwk/Notebook_code/FNIT/runs/connectome-accuracy-memory-recovery-20261003-v1')
PYTHON='/home1/gongwk/anaconda3/bin/python3.11'
environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
def save(value):
    temp=ROOT/'.CPU_status.tmp';temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(ROOT/'CPU_status.json')
status={'status':'waiting_saved_reports','start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'pid':os.getpid(),
    'python':sys.executable,'CUDA_VISIBLE_DEVICES':'','source_helper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'scope':'CPU saved-artifact audit only; never launches tracking or GPU scientific work'}
try:
    while True:
        if (ROOT/'controller_failure.json').exists():raise ValueError('GPU controller failed; do not infer successful recovery')
        actual=json.loads((ROOT/'status.json').read_bytes());status['GPU_status_observed']=actual['status'];status['last_check_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());save(status)
        if actual['status']=='GPU_execution_completed_CPU_comparison_pending':break
        time.sleep(45)
    status['status']='running_CPU_saved_array_audit';save(status)
    commands=[]
    for helper in ('compare_original_recovery.py','make_real_array_figure.py'):
        command=[PYTHON,str(ROOT/helper)];commands.append({'argv':command,'helper_sha256':hashlib.sha256((ROOT/helper).read_bytes()).hexdigest()})
        result=subprocess.run(command,env=environment,cwd=ROOT,stdout=sys.stdout,stderr=sys.stderr)
        if result.returncode:raise RuntimeError(f'{helper} exited {result.returncode}')
    status.update(status='CPU_evidence_completed',end_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),commands=commands);save(status)
except Exception as error:
    status.update(status='failed',error=f'{type(error).__name__}: {error}');save(status);raise
