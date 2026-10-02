"""两例冻结nu/brainmask：完整原生EM基线与局部缓存后端配对，不运行整例。"""
import argparse,csv,hashlib,json,os,platform,re,subprocess,threading,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--case',choices=('whole_sub01_candidate_retry1','whole_sub02_candidate_retry2'));p.add_argument('--same-build',action='store_true',help='Use the candidate executable for both original and cpu_cached; one case per lock window');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lta(path):
 lines=Path(path).read_text().splitlines();i=lines.index('1 4 4');return np.array([[float(v) for v in line.split()] for line in lines[i+1:i+5]])
snapshot=Path(__file__).resolve().parents[5]/'source_commit.txt'
actual_commit=snapshot.read_text().strip() if snapshot.exists() else a.commit
report={'commit':actual_commit,'dispatch_commit':a.commit,'host':platform.node(),'tolerance_declared':{'paired_lta_matrix_atol':0},'scope':'frozen_same_input_full_native_GCA_not_continuous_chain','overall_equivalence':'not_assessed','timing_includes_io_and_spawn':True,'cpu_threads':4,'cpu_affinity':sorted(os.sched_getaffinity(0)),'same_build_control':a.same_build,'pid':os.getpid(),'rows':[],'external_load':[]}
stop=threading.Event()
def monitor():
 while not stop.is_set():
  try:
   report['external_load'].append({'time':time.time(),'loadavg':os.getloadavg(),'processes':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)})
  except Exception as error:report['external_load'].append({'error':str(error)})
  stop.wait(1)
watcher=threading.Thread(target=monitor);watcher.start()
try:
 baseline=a.root/'serial_20261001/native_bundle/bin/mri_em_register';atlas=a.root/'assets/average/RB_all_2020-01-02.gca'
 if a.same_build: baseline=a.candidate
 report['binary_sha256']={'baseline':sha(baseline),'candidate':sha(a.candidate)}
 report['build']=json.loads((a.candidate.parent/'build.json').read_text())
 for index,case in enumerate((a.case,) if a.case else ('whole_sub01_candidate_retry1','whole_sub02_candidate_retry2')):
  mri=a.root/'serial_20261001'/case/'mri'; row={'case':case,'inputs':{p.name:sha(p) for p in (mri/'nu.mgz',mri/'brainmask.mgz',atlas)},'runs':[]}
  order=('baseline','candidate') if case=='whole_sub01_candidate_retry1' else ('candidate','baseline')
  for mode in order:
   output=a.output/(case+'_'+mode+'.lta');command=[str(baseline if mode=='baseline' else a.candidate),'-uns','3','-mask',str(mri/'brainmask.mgz'),str(mri/'nu.mgz'),str(atlas),str(output.resolve())]
   env=dict(os.environ,FREESURFER_HOME=str(a.root/'assets'),FNIT_GCA_DIAGNOSTICS='1');env.pop('FNIT_GCA_SCORER',None);env.pop('FNIT_GCA_QUERY_CAPABILITIES',None)
   if mode=='candidate':env['FNIT_GCA_SCORER']='cpu_cached'
   with (a.output/(case+'_'+mode+'.log')).open('w') as log:
    tick=time.perf_counter();process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    samples=[];runtime_libraries=[]
    while process.poll() is None:
     try:
      status=Path('/proc')/str(process.pid)/'status'
      if not runtime_libraries:
       lines=(Path('/proc')/str(process.pid)/'maps').read_text().splitlines()
       runtime_libraries=sorted({line.split()[-1] for line in lines if any(key in line for key in ('libomp','libgomp','libblas','libopenblas','libitkvnl'))})
      values={key:value.strip() for line in status.read_text().splitlines() if ':' in line for key,value in [line.split(':',1)]}
      samples.append({'time':time.time(),'threads':int(values['Threads']),'rss_kib':int(values['VmRSS'].split()[0])})
     except (FileNotFoundError,KeyError):pass
     time.sleep(.5)
    returncode=process.wait()
   row['runs'].append({'backend':mode,'seconds_including_io':time.perf_counter()-tick,'returncode':returncode,'native_pid':process.pid,'GCA_call_diagnostics':[line for line in (a.output/(case+'_'+mode+'.log')).read_text().splitlines() if line.startswith('FNIT_GCA_CALLS')],'mapped_runtime_libraries':runtime_libraries,'thread_rss_samples':samples,'environment_threads':{key:env.get(key) for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS')},'output_sha256':sha(output) if output.exists() else None})
   if returncode:raise RuntimeError('native stage failed; inspect task log')
   (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
  before=lta(a.output/(case+'_baseline.lta'));after=lta(a.output/(case+'_candidate.lta'))
  delta=np.abs(before-after)
  row['comparison']={'exact_matrix':bool(np.array_equal(before,after)),'different_elements':int(np.count_nonzero(before!=after)),'max_error':float(delta.max()),'p99_error':float(np.quantile(delta,.99)),'baseline_matrix':before.tolist(),'candidate_matrix':after.tolist()}
  report['rows'].append(row);(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
finally:
 stop.set();watcher.join();report['script_sha256']=sha(__file__);(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
with (a.output/'timing.csv').open('w') as out:
 writer=csv.writer(out);writer.writerow(['case','backend','seconds_including_io','exact_matrix'])
 for row in report['rows']:
  for run in row['runs']:writer.writerow([row['case'],run['backend'],run['seconds_including_io'],row['comparison']['exact_matrix']])
