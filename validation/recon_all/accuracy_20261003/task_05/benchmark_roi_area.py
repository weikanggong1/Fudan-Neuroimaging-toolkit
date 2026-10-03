"""隔离同输入脑区面积基线/候选/官方对照，历史被试仅诊断。"""
import argparse, hashlib, importlib.util, json, os, platform, shutil, subprocess, sys, time
from pathlib import Path
import nibabel.freesurfer.io as fs
import numpy as np
import torch
from fnit.recon_all.surface_stats_cache import SurfaceStatsCache

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--subject',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
p.add_argument('--baseline-source',type=Path,required=True); p.add_argument('--official-home',type=Path,required=True)
p.add_argument('--hemi',choices=('lh','rh'),required=True); p.add_argument('--device',default='cuda:0')
a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False); torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=True
files={k:a.subject/v for k,v in {'surface':f'surf/{a.hemi}.white','pial':f'surf/{a.hemi}.pial',
    'thickness':f'surf/{a.hemi}.thickness','annotation':f'label/{a.hemi}.aparc.annot'}.items()}
report={'scope':'isolated_same_input_roi; historical_subject_only_for_diagnosis',
 'overall_equivalence':'not_assessed','host':platform.node(),'threads':2,'device':a.device,
 'baseline_commit':'816e5610417a4c587caf321049438a9554139016','benchmark_sha256':sha(__file__),
 'input_sha256':{k:sha(v) for k,v in files.items()},'runs':{}}
spec=importlib.util.spec_from_file_location('fnit.recon_all._frozen_stats_cache',a.baseline_source)
b=importlib.util.module_from_spec(spec); sys.modules[spec.name]=b; spec.loader.exec_module(b)
import fnit.recon_all.surface_stats_cache as candidate
report['source_sha256']={'baseline':sha(a.baseline_source),'candidate':sha(candidate.__file__)}
results={}
for index,order in enumerate((('baseline','candidate'),('candidate','baseline'))):
 for name in order:
  cls=b.SurfaceStatsCache if name=='baseline' else SurfaceStatsCache
  torch.cuda.synchronize(a.device); torch.cuda.reset_peak_memory_stats(a.device); tick=time.perf_counter()
  with cls(device=a.device) as cache:
   rows,volumes=cache.roi_base(files['surface'],files['annotation'],files['thickness'],white=files['surface'],pial=files['pial'])
   counters=dict(cache.counters)
  torch.cuda.synchronize(a.device)
  results[name]=rows
  report['runs'][f'{index}_{name}']={'seconds':time.perf_counter()-tick,'roi':rows,'volumes_mm3':volumes,
   'counters':counters,'allocated_peak_bytes':torch.cuda.max_memory_allocated(a.device),'reserved_peak_bytes':torch.cuda.max_memory_reserved(a.device)}
# 独立上游面分摊定义；同一GPU叉乘避免将面计算差异误归因于ROI归约。
xyz,faces=fs.read_geometry(str(files['surface'])); labels,_,names=fs.read_annot(str(files['annotation']))
v=torch.as_tensor(xyz,dtype=torch.float32,device=a.device); tri=torch.as_tensor(faces.astype(np.int64),device=a.device)
v0,v1,v2=(v[tri[:,i]] for i in range(3))
shares=(torch.linalg.vector_norm(torch.cross(v1-v0,v2-v0,dim=1),dim=1)*.5/3).cpu().numpy()
oracle=np.zeros(len(names),dtype=np.float64)
for i,face in enumerate(faces):
 for vertex in face:
  if labels[vertex]>=0: oracle[labels[vertex]]+=float(shares[i])
expected={name.decode():float(oracle[i]) for i,name in enumerate(names)}
report['same_face_area_oracle']={}
for name,rows in results.items():
 errors={region:{'actual_mm2':row[1],'reference_mm2':expected[region],
    'absolute_mm2':abs(row[1]-expected[region]),'relative':abs(row[1]-expected[region])/expected[region] if expected[region] else None} for region,row in rows.items()}
 values=np.array([x['absolute_mm2'] for x in errors.values()])
 report['same_face_area_oracle'][name]={'regions':errors,'p99_mm2':float(np.quantile(values,.99)),'max_mm2':float(values.max())}
# 官方仅在独立目录执行，源被试不写回。
isolated=a.output/'official_subjects'/'diagnostic'
for folder in ('surf','mri','label','stats'): shutil.copytree(a.subject/folder,isolated/folder,symlinks=False)
binary=a.official_home/'bin/mris_anatomical_stats'
command=[str(binary),'-noglobal','-no-th3','-b','-f',str(a.output/'official.stats'),'-a',str(isolated/f'label/{a.hemi}.aparc.annot'),'-cortex',str(isolated/f'label/{a.hemi}.cortex.label'),'diagnostic',a.hemi,'white']
env=dict(os.environ,FREESURFER_HOME=str(a.official_home),SUBJECTS_DIR=str(isolated.parent)); tick=time.perf_counter()
with (a.output/'official.log').open('w') as log: process=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
report['official']={'command':command,'returncode':process.returncode,'seconds':time.perf_counter()-tick,'binary_sha256':sha(binary)}
if process.returncode==0:
 reference={}
 for line in (a.output/'official.stats').read_text().splitlines():
  cols=line.split()
  if len(cols)==10 and not line.startswith('#'): reference[cols[0]]=float(cols[2])
 report['official']['area_difference_mm2']={name:{region:row[1]-reference[region] for region,row in rows.items() if region in reference} for name,rows in results.items()}
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
raise SystemExit(process.returncode)
