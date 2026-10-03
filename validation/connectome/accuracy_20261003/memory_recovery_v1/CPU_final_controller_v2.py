"""Run the two saved-artifact CPU helpers after actual GPU completion."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode=True
ROOT=Path('/cwStorage/home/gongwk/Notebook_code/FNIT/runs/connectome-accuracy-memory-recovery-20261003-v1')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert json.loads((ROOT/'status.json').read_bytes())['status']=='GPU_execution_completed_CPU_comparison_pending'
environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
status={'status':'running_CPU_strict_comparison','CUDA_VISIBLE_DEVICES':'','commands':[],
        'controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'prior_CPU_launcher_error_retained':'CPU_final.log',
        'start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
def save():
    temp=ROOT/'.CPU_final_status.tmp';temp.write_text(json.dumps(status,indent=2)+'\n');temp.replace(ROOT/'CPU_final_status.json')
save()
try:
    for name in ('compare_original_recovery.py','make_real_array_figure.py'):
        args=[sys.executable,str(ROOT/name)]
        status['commands'].append({'argv':args,'source_sha256':hashlib.sha256((ROOT/name).read_bytes()).hexdigest()});save()
        subprocess.run(args,env=environment,check=True)
    status.update(status='CPU_evidence_completed',end_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
except Exception as error:
    status.update(status='failed',error=f'{type(error).__name__}: {error}');raise
finally:save()
