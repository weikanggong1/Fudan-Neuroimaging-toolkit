"""Start metadata-only CON11 readiness waiter and explicit modeling collector once."""
import argparse,os,subprocess,sys,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();own=a.root/'task_02';env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MKL_NUM_THREADS='8');launch=own/'CON11_subset_metadata_launch_v1.json'
with launch.open('x') as receipt:
    result={'GPU':False,'CPU_threads':8,'scientific_modeling_started':False,'children':{}}
    for key,tool,flag,out in [('CON11_waiter','orchestrate_CON11_CPU_subset_v1.py','--state-root',own/'CON11_subset_dispatcher_v1'),('explicit_collector','collect_explicit_ten_modeling_v1.py','--output',own/'official_modeling_explicit_ten_delivery_v1')]:
        if out.exists():raise ValueError('Metadata namespace already exists; refuse duplicate')
        command=[sys.executable,'-u',str(own/tool),'--root',str(a.root),flag,str(out)]
        with (own/(key+'_v1.log')).open('x') as log:child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        result['children'][key]={'PID':child.pid,'command':command,'log':str(own/(key+'_v1.log')),'state_namespace':str(out)}
    receipt.write(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
