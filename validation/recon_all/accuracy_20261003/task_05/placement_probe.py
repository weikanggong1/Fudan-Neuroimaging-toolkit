"""完整同输入white/pial隔离诊断，Python白质仅有prefix，不伪造完整第三方。"""
from __future__ import annotations
import argparse, hashlib, json, os, platform, re, shutil, subprocess, time
from pathlib import Path
import nibabel.freesurfer.io as fs
import numpy as np
import torch
from fnit.recon_all.place_pial_python import place_pial_t1

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--subject',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
p.add_argument('--kind',choices=('prewhite','white','pial'),required=True)
p.add_argument('--backend',choices=('official','conda','python'),required=True)
p.add_argument('--hemi',choices=('lh','rh'),required=True); p.add_argument('--binary',type=Path)
p.add_argument('--assets',type=Path,required=True); p.add_argument('--device',default='cuda:0')
a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False); h=a.hemi
if a.kind!='pial' and a.backend=='python': raise ValueError('Complete Python white placement unavailable; prefix is not full algorithm')
paths=[f'surf/{h}.white',f'surf/{h}.white.preaparc',f'surf/{h}.orig',f'surf/{h}.orig.premesh',
       f'surf/autodet.gw.stats.{h}.dat',f'label/{h}.cortex.label',f'label/{h}.cortex+hipamyg.label',f'label/{h}.aparc.annot',
       'mri/brain.finalsurfs.mgz','mri/wm.mgz','mri/aseg.presurf.mgz']
subject=a.output/'subject'
for path in paths:
 target=subject/path; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(a.subject/path,target)
(subject/'scripts').mkdir(exist_ok=True)
report={'scope':'isolated_same_complete_input; old_frozen_FNIT_subject_for_diagnosis_only','kind':a.kind,'backend':a.backend,
 'hemi':h,'host':platform.node(),'threads':2,'target_gpu_uuid':os.environ.get('CUDA_VISIBLE_DEVICES'),'coordinate_space':'surface RAS','coordinate_unit':'mm',
 'overall_equivalence':'not_assessed','input_sha256':{path:sha(subject/path) for path in paths},
 'script_sha256':sha(__file__),'resource_manifest_sha256':sha(Path('/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/accuracy_20261003/resources_verified_816e5610.json')) if (Path('/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/accuracy_20261003/resources_verified_816e5610.json')).is_file() else None,'runtime_sources_sha256':{str(path.name):sha(path) for path in Path(__import__('fnit.recon_all.place_pial_python',fromlist=['']).__file__).parent.glob('place*.py')},
 'complete_python_white':'unavailable; first-pass prefix intentionally excluded'}
output=subject/f'surf/{h}.probe'; started=time.perf_counter()
if a.backend=='python':
 torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=True
 torch.backends.cudnn.allow_tf32=True; torch.cuda.synchronize(a.device)
 trace=[]
 def sink(step,outer,vertices,state):
  trace.append({'step':step,'pass_index':outer,'state':state,'coordinates_sha256':hashlib.sha256(vertices.tobytes()).hexdigest()})
  if state['stop']: fs.write_geometry(str(a.output/f'pass{outer}.surface'),vertices,fs.read_geometry(str(subject/f'surf/{h}.white'))[1])
 report['stage']=place_pial_t1(subject=subject,hemisphere=h,output=output,max_steps=400,
     sampling_backend='triton',candidate_backend='snapshot',device=a.device,trace_callback=sink)
 torch.cuda.synchronize(a.device)
 report['trace']=trace
 report['precision']={'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32,'autocast':False}
 report['cuda_allocated_peak_bytes']=torch.cuda.max_memory_allocated(a.device)
 report['cuda_reserved_peak_bytes']=torch.cuda.max_memory_reserved(a.device)
else:
 common=[str(a.binary),'--adgws-in',str(subject/f'surf/autodet.gw.stats.{h}.dat'),'--seg',str(subject/'mri/aseg.presurf.mgz'),
 '--threads','2','--wm',str(subject/'mri/wm.mgz'),'--invol',str(subject/'mri/brain.finalsurfs.mgz'),f'--{h}',
 '--o',str(output)]
 if a.kind=='prewhite':
  extra=['--i',str(subject/f'surf/{h}.orig'),'--white','--nsmooth','5','--rip-bg-no-annot','--rip-bg','--rip-bg-lof',
         '--restore-255','--restore-255','--outvol',str(subject/'mri/probe.mgz')]
 elif a.kind=='white':
  extra=['--i',str(subject/f'surf/{h}.white.preaparc'),'--white','--nsmooth','0','--rip-label',str(subject/f'label/{h}.cortex.label'),
         '--rip-bg','--rip-surf',str(subject/f'surf/{h}.white.preaparc'),'--aparc',str(subject/f'label/{h}.aparc.annot'),
         '--restore-255','--restore-255','--outvol',str(subject/'mri/probe.mgz'),'--rip-bg-lof']
 else:
  extra=['--i',str(subject/f'surf/{h}.white'),'--pial','--nsmooth','0','--rip-label',str(subject/f'label/{h}.cortex+hipamyg.label'),
         '--pin-medial-wall',str(subject/f'label/{h}.cortex.label'),'--aparc',str(subject/f'label/{h}.aparc.annot'),
         '--repulse-surf',str(subject/f'surf/{h}.white'),'--white-surf',str(subject/f'surf/{h}.white'),'--restore-255']
 command=common+extra; report['command']=command; report['program_sha256']=sha(a.binary)
 env=dict(os.environ,FREESURFER_HOME=str(a.assets),SUBJECTS_DIR=str(subject.parent))
 with (a.output/'native.log').open('w') as log: result=subprocess.run(command,cwd=subject/'scripts',env=env,stdout=log,stderr=subprocess.STDOUT)
 report['returncode']=result.returncode
 text=(a.output/'native.log').read_text(errors='replace')
 report['native_rounds']=[]
 for match in re.finditer(r'Iteration (\d+) =+\n(.*?)(?=Iteration \d+ =+|\Z)',text,re.S):
  outer=int(match.group(1)); chunk=match.group(2)
  accepted=[{'step':int(x[0]),'dt':float(x[1]),'sse':float(x[2]),'rms':float(x[3])} for x in re.findall(r'(?m)^(\d+): dt: ([\d.]+), sse=([\d.e+-]+), rms=([\d.e+-]+)',chunk)]
  report['native_rounds'].append({'pass_index':outer,'completed_log_steps':[row for row in accepted if row['step']!=0],'initial_metrics':[row for row in accepted if row['step']==0],'rejected_trials':chunk.count('RMS increased, rejecting step'),
                                'ended_on_reduction_limit':'maximum number of reductions reached' in chunk})
 report['native_cleanup_messages']=[line for line in text.splitlines() if any(term in line.lower() for term in ('intersection','pinning','medial wall'))]
 report['native_round_coordinate_snapshots']='not_available_from_fixed_official_CLI; logs are scalar diagnostic only; terminal rejected step metrics may describe rejected trial while coordinates are restored'
 if result.returncode:
  report['seconds_including_io']=time.perf_counter()-started; (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n'); raise SystemExit(result.returncode)
report['seconds_including_io']=time.perf_counter()-started
xyz,faces=fs.read_geometry(str(output)); initial_path=subject/f'surf/{h}.white'
initial,initial_faces=fs.read_geometry(str(initial_path))
report['output']={'path':str(output),'sha256':sha(output),'vertices':len(xyz),'faces':len(faces),'finite':bool(np.isfinite(xyz).all()),
                  'ordered_faces_unchanged':bool(np.array_equal(faces,initial_faces)),'input_ancestry':'copied same ordered mesh; placement cannot remesh',
                  'max_displacement_from_final_white_mm':float(np.linalg.norm(xyz-initial,axis=1).max()) if xyz.shape==initial.shape else None}
report['input_sha256_after']={path:sha(subject/path) for path in paths}
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
