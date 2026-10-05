from pathlib import Path
import datetime,json,subprocess,os,time,hashlib
r=Path('/cwStorage/home/gongwk/Notebook_code/FNIT');w=r/'workspaces/fmri_reference_alignment_20261004';code=w/'bold_reference_v3_code';run=r/'runs/fmri_reference_alignment_20261004/bold_reference_v3'
state=run/'posthoc.queue.private.json';q={'status':'waiting','pid':os.getpid(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
def write():q['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();t=state.with_suffix('.tmp');t.write_text(json.dumps(q,indent=2)+'\n');t.replace(state)
write()
while True:
 phases=[json.loads((r/'runs/fmri_reference_alignment_20261004/bold_reference_v1/official.queue.private.json').read_text())['status'],json.loads((run/'candidate.queue.private.json').read_text())['status']]
 if 'failed' in phases:q['status']='blocked_by_retained_benchmark_failure';write();raise SystemExit(2)
 if phases==['complete','complete'] and (run/'candidate.gpu_observation.public.json').exists():break
 time.sleep(10)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
command=['/home1/gongwk/anaconda3/bin/python',str(code/'collect_reference_results.py'),'--run-root',str(run),'--official-run-root',str(r/'runs/fmri_reference_alignment_20261004/bold_reference_v1'),'--manifest',str(w/'input_manifest.private.json'),'--source-root',str(w/'source_v1'),'--driver',str(code/'run_reference_benchmark.py'),'--output',str(run/'summary_attempt01')]
q['status']='running';q['command']=command;write();start=time.perf_counter()
with (run/'posthoc.process.private.log').open('x') as stream:
 proc=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT);q['child_pid']=proc.pid;write();rc=proc.wait()
q.update(status='complete' if rc==0 else 'failed',exit_code=rc,posthoc_process_seconds=time.perf_counter()-start);write()
