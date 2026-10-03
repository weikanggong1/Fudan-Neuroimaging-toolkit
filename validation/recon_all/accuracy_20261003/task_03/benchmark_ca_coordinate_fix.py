"""两例冻结相同输入 baseline/candidate ABBA 配对；不启动整例。"""
import argparse,fcntl,importlib.util,json,os,platform,time
from pathlib import Path
import numpy as np
from volume_probe import sha,compare
import fnit.recon_all.ca_normalize_python as candidate
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--baseline-module',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();c=json.loads(a.config.read_text());a.output.mkdir(parents=True,exist_ok=False)
spec=importlib.util.spec_from_file_location('fnit.recon_all.ca_normalize_baseline_probe',a.baseline_module);baseline=importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name]=baseline;spec.loader.exec_module(baseline)
report={'scope':'two_old_real_cases_frozen_same_input_ABBA_not_raw_T1_whole','baseline_commit':c['commit'],'candidate_source_sha256':sha(candidate.__file__),'baseline_source_sha256':sha(a.baseline_module),'script_sha256':sha(__file__),'host':platform.node(),'pid':os.getpid(),'overall_equivalence':'not_assessed','gpu_used_by_this_component':False,'precision':'FP32 coordinate operator; all other CA operators unchanged; no global TF32 change','threads':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS')},'asset_sha256':sha(c['atlas']),'cases':[]}
def save():(a.output/'report.json').write_text(json.dumps(report,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')
for case in c['cases']:
 mri=Path(case['official'])/'mri';item={'id':case['id'],'input_sha256':{n:sha(mri/n) for n in ('nu.mgz','brainmask.mgz','transforms/talairach.lta')},'runs':[]};report['cases'].append(item);save()
 for k,kind in enumerate(('baseline','candidate','candidate','baseline')):
  out=a.output/case['id']/(str(k)+'_'+kind);out.mkdir(parents=True);lock=open('/tmp/fnit-shared-benchmark.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)
  try:
   module=baseline if kind=='baseline' else candidate;start=time.perf_counter();details=module.run_ca_normalize(mri/'nu.mgz',mri/'brainmask.mgz',c['atlas'],mri/'transforms/talairach.lta',out/'norm.mgz',out/'ctrl_pts.mgz');elapsed=time.perf_counter()-start
   item['runs'].append({'kind':kind,'order':k,'seconds_including_load_compute_io':elapsed,'details':details,'norm_vs_official':compare(out/'norm.mgz',mri/'norm.mgz'),'controls_vs_official':compare(out/'ctrl_pts.mgz',mri/'ctrl_pts.mgz')});save()
  finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
report['complete']=True;save()
