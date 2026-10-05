import argparse,datetime,hashlib,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('phase',choices=('official','candidate'));args=p.parse_args()
r=Path('/cwStorage/home/gongwk/Notebook_code/FNIT');w=r/'workspaces/fmri_reference_alignment_20261004';code=w/'bold_reference_v1_code';run=r/'runs/fmri_reference_alignment_20261004/bold_reference_v1'
driver=code/'run_reference_benchmark.py';manifest=w/'input_manifest.private.json';source=w/'source_v1';lock=r/'workspaces/fnit_surface_ten_public_20261003/.fnit_physical_gpu_0_benchmark.lock'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(driver)=='a4a794fbfb26d20cd393521aa19bc510ff1bcbff2f4e5db6e39cbaec8a06b4aa'
assert sha(manifest)=='40ecbdf2cce9ef4e5fabcce71231347ea5bcdde85ad2d668ef02efecb143c9d1'
assert sha(source/'src/fnit/fmri/reference.py')=='ac885355a286ff6799aaeafc9735de1d0c0264b8afba55041ea4e94b1ddc3484'
state=run/(args.phase+'.queue.private.json')
queue={'status':'running','phase':args.phase,'pid':os.getpid(),'runner_sha256':sha(Path(__file__)),'driver_sha256':sha(driver),'manifest_sha256':sha(manifest),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'order':['CON01','CON06'],'completed':[]}
def write():
 queue['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();tmp=state.with_suffix('.tmp');tmp.write_text(json.dumps(queue,indent=2)+'\n');tmp.replace(state)
write()
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0' if args.phase=='candidate' else '',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');env.pop('PYTORCH_NO_CUDA_MEMORY_CACHING',None)
for case in queue['order']:
 output=run/case/(args.phase+'_attempt01');log=run/(case+'.'+args.phase+'.process.private.log')
 command=[sys.executable,str(driver),'--manifest',str(manifest),'--case',case,'--phase',args.phase,'--output',str(output),'--allowed-run-root',str(run),'--source-root',str(source),'--threads','4']
 if args.phase=='candidate':command+=['--device','cuda:0','--gpu-lock',str(lock)]
 queue['current_case']=case;queue['current_command']=command;queue['status']='waiting_for_gpu_lock_or_running' if args.phase=='candidate' else 'running';write()
 queue['gpu_sample_before']=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout.strip();write()
 started=time.perf_counter()
 with log.open('x') as stream:
  proc=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT);queue['child_pid']=proc.pid;write();rc=proc.wait()
 item={'case':case,'exit_code':rc,'driver_process_wall_seconds':time.perf_counter()-started,'output':str(output),'log':str(log),'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 if (output/'report.public.json').exists():item['report_sha256']=sha(output/'report.public.json');item['report_status']=json.loads((output/'report.public.json').read_text())['status']
 queue['completed'].append(item);write()
 if rc:queue['status']='failed';write();sys.exit(rc)
queue['status']='complete';queue['current_case']=None;queue['child_pid']=None;queue['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();write()
