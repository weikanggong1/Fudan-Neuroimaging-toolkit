import pathlib,json,shutil,time,hashlib
import torch
import numpy as np
from nibabel.freesurfer.io import read_morph_data
from fnit.recon_all.native_free import _run_surface_metrics
p=pathlib.Path('/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/volume_parity_20260930')
while True:
 f=p/'full_sub01_279e09f'/'fnit-native-free-run.json'
 if f.exists():
  d=json.loads(f.read_text())
  if d['stages'] and d['stages'][-1]['name']=='SynthSeg':break
  if any(s['name']=='mri_em_register' for s in d.get('stages',[])):raise RuntimeError('EM CPU window already passed')
 time.sleep(2)
root=p/'surface_metrics_wiring_tf32_279e09f';(root/'surf').mkdir(parents=True,exist_ok=False)
base=p/'full_sub01_b8cd';torch.set_num_threads(4)
for name in ['white','pial']:
 shutil.copyfile(base/'surf'/('lh.'+name),root/'surf'/('lh.'+name))
t=torch.cuda; t.init();t.synchronize('cuda:0');t.reset_peak_memory_stats('cuda:0')
torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.allow_tf32=True
timings=_run_surface_metrics(pathlib.Path('/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/fnit_main_env/bin/mris_place_surface'),root,'lh',pathlib.Path('/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/assets'),device='cuda:0')
rows={}
for name in timings:
 a=read_morph_data(base/'surf'/('lh.'+name));b=read_morph_data(root/'surf'/('lh.'+name));error=np.abs(a.astype(np.float64)-b)
 outliers=error>(.001 if name.startswith('area') else .005)+.001*np.abs(a)
 rows[name]={'maximum_absolute_error':float(error.max()),'p99_absolute_error':float(np.quantile(error,.99)),'outliers':int(outliers.sum()),'pass':not bool(outliers.any())}
source=pathlib.Path(__import__('fnit.recon_all.native_free',fromlist=['__file__']).__file__)
report={'code_commit':'279e09f0d2a166237871b3d683a6be75bd5e99b4','scope':'frozen FNIT white/pial; real production GPU wrapper','seconds_including_io':timings,'comparison_to_baseline_conda':rows,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'input_sha256':{name:hashlib.sha256((root/'surf'/('lh.'+name)).read_bytes()).hexdigest() for name in ['white','pial']},'precision':{'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32,'fp16_or_bf16':False},'gpu_memory_mode':'torch_cuda_allocator','gpu_peak_allocated_bytes':t.max_memory_allocated('cuda:0'),'gpu_peak_reserved_bytes':t.max_memory_reserved('cuda:0')}
(root/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
assert all(row['pass'] for row in rows.values())
