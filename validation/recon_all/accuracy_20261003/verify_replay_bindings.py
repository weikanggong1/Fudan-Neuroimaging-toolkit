"""只读复核原冻结资源与源码，供资源失败的原始T1重跑使用。"""
from pathlib import Path
import argparse,hashlib,json,tarfile,datetime,subprocess,os

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def _verify(*, cohort_root, expected, report_path):
 R=Path(cohort_root);report_path=Path(report_path)
 S=Path('/cwStorage/home/gongwk/Notebook_code/FNIT')
 index_sha={}
 for n in ('README.md','INDEX.md','INDEX.json'):
  raw=(S/n).read_bytes();raw.decode('utf-8');index_sha[n]=hashlib.sha256(raw).hexdigest()
  if n=='INDEX.json':json.loads(raw)
 cases=('ds000030_sub-10189','ds000114_sub-06','ds000114_sub-07','ds000114_sub-08')
 out={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':os.uname().nodename,'scope':'original frozen baseline resources; no GPU inference','code_commit':expected['code_commit'],'index_sha256':index_sha,'mismatches':[],'cases':{}}
 for case in cases:
  path=R/'configs_v1'/('baseline_'+case+'.json');c=json.loads(path.read_text())
  launch_path=Path(c['diagnostic_root'])/'launch.json';launch=json.loads(launch_path.read_text())
  mismatch=[k for k,v in c.items() if launch.get(k)!=v]
  if mismatch:out['mismatches'].append({'case':case,'config_vs_launch':mismatch})
  if c['code_commit']!=expected['code_commit'] or c['source_archive_sha256']!=expected['source_archive_sha256']:
   out['mismatches'].append({'case':case,'source_binding':'different'})
  if digest(c['input'])!=c['input_sha256'] or digest(c['input'])!=expected['inputs'][case]['sha256']:
   out['mismatches'].append({'case':case,'input':'different'})
  if digest(Path(c['code_root'])/'src/fnit/recon_all/native_free.py')!=launch.get('candidate_native_free_sha256'):
   out['mismatches'].append({'case':case,'original_native_free_sha256':'different'})
  out['cases'][case]={'config':c,'config_sha256':digest(path),'original_launch_sha256':digest(launch_path),'input_sha256':digest(c['input'])}
 first=next(iter(out['cases'].values()))['config']
 archive=R/'baseline_816e5610.tar.gz'
 out['source_archive_sha256']=digest(archive)
 if out['source_archive_sha256']!=expected['source_archive_sha256']:out['mismatches'].append({'source_archive':'different'})
 source=Path(first['code_root']);out['source_checked_files']=0
 with tarfile.open(archive,'r:gz') as tar:
  for member in tar:
   if not member.isfile():continue
   name=Path(member.name)
   if name.is_absolute() or '..' in name.parts:raise ValueError('unsafe archive entry')
   h=hashlib.sha256();stream=tar.extractfile(member)
   for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
   path=source/name
   if not path.is_file() or digest(path)!=h.hexdigest():out['mismatches'].append({'source_file':member.name})
   out['source_checked_files']+=1
 out['resources']={}
 for group,key in (('weights','weights'),('assets','assets'),('binaries','native_bin_dir')):
  rows={};base=Path(first[key])
  for name,entry in expected[group].items():
   path=base/name;actual=digest(path)
   rows[name]={'sha256':actual,'bytes':path.stat().st_size,'resolved_path':str(path.resolve())}
   if actual!=entry['sha256'] or path.stat().st_size!=entry['size_bytes']:out['mismatches'].append({'group':group,'name':name})
  out['resources'][group]=rows
 for case,row in out['cases'].items():
  for key in ('weights','assets','native_bin_dir','python','code_root','threads','device','gpu_uuid'):
   if row['config'][key]!=first[key]:out['mismatches'].append({'case':case,'inconsistent_config':key})
 out['gpu_snapshot']=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10).stdout.strip().splitlines()
 import torch,numpy,scipy,nibabel,numba
 out['runtime_versions']={'torch':torch.__version__,'torch_cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'numpy':numpy.__version__,'scipy':scipy.__version__,'nibabel':nibabel.__version__,'numba':numba.__version__}
 out['version_differences']={k:{'original':expected.get(k),'current':v} for k,v in out['runtime_versions'].items() if v!=expected.get(k)}
 out['status']='passed' if not out['mismatches'] and not out['version_differences'] else 'failed'
 report_path.parent.mkdir(parents=True,exist_ok=True);report_path.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
 return out

def verify(*, cohort_root, expected, report_path):
 report_path=Path(report_path)
 report_path.parent.mkdir(parents=True,exist_ok=True)
 with report_path.open('x') as stream:stream.write('{}\n')
 try:
  return _verify(cohort_root=cohort_root,expected=expected,report_path=report_path)
 except Exception as error:
  out={'status':'failed','scope':'original frozen baseline resources; no GPU inference',
       'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'host':os.uname().nodename,'error':repr(error),'verification_complete':False}
  report_path.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
  return out

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cohort-root',type=Path,required=True);p.add_argument('--expected',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
 result=verify(cohort_root=a.cohort_root,expected=json.loads(a.expected.read_text()),report_path=a.report)
 print(json.dumps(result,ensure_ascii=False));raise SystemExit(0 if result['status']=='passed' else 1)
